// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_cache_define.vh"

// Word-sliced, way-indexed data array.
//
// The line is split into CS_WORDS_PER_LINE independent word slices, and the
// way dimension is folded into the array address as {way, line_idx} rather
// than replicated as NUM_WAYS parallel full-line arrays. The way is resolved
// at read-issue (the hit way for a core read/write, the victim way for a
// fill/flush), so the array is addressed directly:
//
//   * a load reads only the slice selected by word_idx -> one CS_WORD_WIDTH read
//   * a store writes only that slice (byte-enabled)
//   * a fill writes all slices in parallel (full line)
//   * a writeback/flush reads all slices in parallel (full line)
//
// This removes both the all-ways data read and the late NUM_WAYS:1 line mux of
// the previous parallel-access design: read_data carries the selected way's
// line directly.

module VX_cache_data import VX_gpu_pkg::*; #(
    // Size of cache in bytes
    parameter CACHE_SIZE        = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE         = 16,
    // Number of banks
    parameter NUM_BANKS         = 1,
    // Number of associative ways
    parameter NUM_WAYS          = 1,
    // Size of a word in bytes
    parameter WORD_SIZE         = 1,
    // Size of a sector in bytes (fill granule); = LINE_SIZE => 1 sector
    parameter SECTOR_SIZE       = LINE_SIZE,
    // Enable cache writeable
    parameter WRITE_ENABLE      = 1,
    // Enable cache writeback
    parameter WRITEBACK         = 0,
    // Enable dirty bytes on writeback
    parameter DIRTY_BYTES       = 0
) (
    input wire                          clk,
    input wire                          reset,
    // inputs
    input wire                          init,
    input wire                          fill,
    input wire                          flush,
    input wire                          read,
    input wire                          write,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_WAY_SEL_WIDTH-1:0]  evict_way,
    input wire [NUM_WAYS-1:0]           tag_matches,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] fill_data,
    input wire [`CS_WORD_WIDTH-1:0]     write_word,
    input wire [WORD_SIZE-1:0]          write_byteen,
    input wire [`UP(`CS_WORD_SEL_BITS)-1:0] word_idx,
    input wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx, // sector being filled
    input wire [`CS_WAY_SEL_WIDTH-1:0]  way_idx_r,
    // outputs
    output wire [`CS_LINE_WIDTH-1:0]    read_data,
    output wire [LINE_SIZE-1:0]         evict_byteen
);
    // The main data array resolves the way at read-issue and folds it into the
    // array address, so the S1 output-way register is only used by the (narrow,
    // off-path) dirty-byte mask below.

    localparam WAY_SEL_BITS    = `CS_WAY_SEL_BITS;
    localparam DATA_RAM_DEPTH  = `CS_LINES_PER_BANK * NUM_WAYS;
    localparam DATA_ADDR_WIDTH = `LOG2UP(DATA_RAM_DEPTH);

    // Resolve the access way: hit way for core read/write, victim way for
    // fill/flush. Encoded combinationally from tag_matches so the data array
    // is addressed at read-issue (S0). This is the tag-compare -> way ->
    // data-address path (see redesign proposal, timing section).
    wire [`CS_WAY_SEL_WIDTH-1:0] hit_way;
    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) hit_way_enc (
        .data_in  (tag_matches),
        .data_out (hit_way),
        `UNUSED_PIN (valid_out)
    );

    wire is_evict = fill || flush;
    wire hit_any  = (| tag_matches);
    wire [`CS_WAY_SEL_WIDTH-1:0] way_sel = is_evict ? evict_way : hit_way;

    // {way, line} -> physical address
    wire [DATA_ADDR_WIDTH-1:0] data_addr;
    if (NUM_WAYS > 1) begin : g_way_addr
        assign data_addr = DATA_ADDR_WIDTH'({way_sel[WAY_SEL_BITS-1:0], line_idx});
    end else begin : g_no_way_addr
        `UNUSED_VAR (way_sel)
        assign data_addr = line_idx;
    end

    // Per-byte dirty mask (writeback only). Unlike the main data array this is
    // kept as one line-indexed array per way (evict way selected at S1): it is
    // narrow and off the load path, and -- crucially -- it must be cleared for
    // every way during the line-only init walk, which the way-folded data
    // layout cannot do in a single pass.
    if (DIRTY_BYTES != 0) begin : g_dirty_bytes
        wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin : g_write_mask
            wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == i);
            assign write_mask[i] = write_byteen & {WORD_SIZE{word_en}};
        end

        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] byteen_rdata;
        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_byteen_store
            wire [LINE_SIZE-1:0] byteen_wdata = {LINE_SIZE{write}}; // only asserted on writes
            wire [LINE_SIZE-1:0] byteen_wren  = {LINE_SIZE{init || fill || flush}} | write_mask;
            wire byteen_write = ((fill || flush) && ((NUM_WAYS == 1) || (evict_way == i)))
                             || (write && tag_matches[i])
                             || init;
            wire byteen_read  = fill || flush;

            // The dirty mask has 1-bit (per-byte) write granularity, which
            // block RAM cannot do -- inferring BRAM shatters it into dozens of
            // tiny RAMB18 per way. It is small, so map it to distributed RAM.
            VX_sp_ram #(
                .DATAW   (LINE_SIZE),
                .WRENW   (LINE_SIZE),
                .SIZE    (`CS_LINES_PER_BANK),
                .OUT_REG (1),
                .LUTRAM  (1),
                .RDW_MODE ("R")
            ) byteen_store (
                .clk   (clk),
                .reset (reset),
                .read  (byteen_read),
                .write (byteen_write),
                .wren  (byteen_wren),
                .addr  (line_idx),
                .wdata (byteen_wdata),
                .rdata (byteen_rdata[i])
            );
        end
        assign evict_byteen = byteen_rdata[way_idx_r];
    end else begin : g_no_dirty_bytes
        `UNUSED_VAR (init)
        `UNUSED_VAR (flush)
        `UNUSED_VAR (way_idx_r)
        assign evict_byteen = '1; // update whole line
    end

    wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_rdata;

    for (genvar s = 0; s < `CS_WORDS_PER_LINE; ++s) begin : g_data_slice

        localparam WRENW = WRITE_ENABLE ? WORD_SIZE : 1;
        // sector this slice belongs to (high bits of its in-line word index).
        localparam SLICE_SECTOR = s / `CS_WORDS_PER_SECTOR;

        wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == s);
        // a fill installs only the fetched sector's slices (whole line when 1
        // sector/line, since every slice maps to sector 0).
        wire fill_sec_en = (`CS_SECTORS_PER_LINE == 1)
                        || (sector_idx == `UP(`CS_SECTOR_SEL_BITS)'(SLICE_SECTOR));

        // load reads the selected slice; writeback/flush reads all slices.
        wire slice_read = (read && word_en) || ((fill || flush) && WRITEBACK);

        wire slice_write;
        wire [WRENW-1:0]          slice_wren;
        wire [`CS_WORD_WIDTH-1:0] slice_wdata;

        if (WRITE_ENABLE) begin : g_wren
            // fill writes the fetched sector's slices; a store writes only the hit slice.
            assign slice_write = (fill && fill_sec_en) || (write && hit_any && word_en);
            assign slice_wren  = fill ? {WORD_SIZE{1'b1}} : write_byteen;
            assign slice_wdata = fill ? fill_data[s] : write_word;
        end else begin : g_no_wren
            `UNUSED_VAR (write)
            `UNUSED_VAR (write_word)
            `UNUSED_VAR (write_byteen)
            `UNUSED_VAR (hit_any)
            assign slice_write = fill && fill_sec_en;
            assign slice_wren  = 1'b1;
            assign slice_wdata = fill_data[s];
        end

        VX_sp_ram #(
            .DATAW   (`CS_WORD_WIDTH),
            .WRENW   (WRENW),
            .SIZE    (DATA_RAM_DEPTH),
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) data_store (
            .clk   (clk),
            .reset (reset),
            .read  (slice_read),
            .write (slice_write),
            .wren  (slice_wren),
            .addr  (data_addr),
            .wdata (slice_wdata),
            .rdata (line_rdata[s])
        );
    end

    assign read_data = line_rdata;

endmodule
