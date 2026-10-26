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
    parameter CACHE_SIZE        = 1024,       // Size of cache in bytes
    parameter LINE_SIZE         = 16,         // Size of line inside a bank in bytes
    parameter NUM_BANKS         = 1,          // Number of banks
    parameter NUM_WAYS          = 1,          // Number of associative ways
    parameter WORD_SIZE         = 1,          // Size of a word in bytes
    parameter SECTOR_SIZE       = LINE_SIZE,  // Size of a sector in bytes (fill granule); = LINE_SIZE => 1 sector
    parameter WRITE_ENABLE      = 1,          // Enable cache writeable
    parameter WRITEBACK         = 0,          // Enable cache writeback
    parameter DIRTY_BYTES       = 0,          // Enable dirty bytes on writeback
    parameter LAZY_FETCH        = 0           // Fills may be data-less or merge under the dirty bytes
) (
    input wire                          clk,
    input wire                          reset,
    // inputs
    input wire                          fill,
    input wire                          fill_nodata, // data-less fill: allocate only, no array write
    // dirty-byte mask access at the commit stage (stC), one stage after the
    // array access: a store sets its bytes, a fill/flush reads and clears
    input wire                          mask_write,
    input wire                          mask_set,
    input wire                          mask_clr_line,
    input wire                          mask_clr_sector,
    input wire [`CS_WAY_SEL_WIDTH-1:0]  mask_way,
    input wire [`CS_LINE_SEL_BITS-1:0]  mask_line,
    input wire [`UP(`CS_WORD_SEL_BITS)-1:0] mask_word,
    input wire [WORD_SIZE-1:0]          mask_byteen,
    input wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] mask_sector,
    // merge of a refetched sector into its resident copy at the commit stage
    // (the stage behind it is empty): only the clean bytes
    input wire                          merge_write,
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
    input wire [`CS_WORDS_PER_LINE-1:0] read_words, // slices a read returns (word_idx's, plus a merged read's)
    input wire [`UP(`CS_SECTOR_SEL_BITS)-1:0] sector_idx, // sector being filled
    // outputs
    output wire [`CS_LINE_WIDTH-1:0]    read_data,
    output wire [LINE_SIZE-1:0]         evict_byteen
);
    `STATIC_ASSERT(!LAZY_FETCH || (DIRTY_BYTES && WRITEBACK), ("invalid parameter: lazy fetch requires write-back dirty bytes"))

    localparam WAY_SEL_BITS    = `CS_WAY_SEL_BITS;
    localparam DATA_RAM_DEPTH  = `CS_LINES_PER_BANK * NUM_WAYS;
    localparam DATA_ADDR_WIDTH = `LOG2UP(DATA_RAM_DEPTH);

    // Resolve the access way: hit way for core read/write, victim way for
    // fill/flush. Encoded combinationally from tag_matches so the data array
    // is addressed at read-issue (S0). This is the timing-critical
    // tag-compare -> way -> data-address path.
    wire [`CS_WAY_SEL_WIDTH-1:0] hit_way;
    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) hit_way_enc (
        .data_in  (tag_matches),
        .data_out (hit_way),
        `UNUSED_PIN (valid_out)
    );

    wire fill_write = fill && ~((LAZY_FETCH != 0) && fill_nodata);
    wire is_evict = fill || flush;
    wire hit_any  = (| tag_matches);
    wire [`CS_WAY_SEL_WIDTH-1:0] way_sel = is_evict ? evict_way : hit_way;

    // {way, line} -> physical address
    wire [DATA_ADDR_WIDTH-1:0] data_addr;
    if (NUM_WAYS > 1) begin : g_way_addr
        assign data_addr = merge_write ? DATA_ADDR_WIDTH'({mask_way[WAY_SEL_BITS-1:0], mask_line})
                                       : DATA_ADDR_WIDTH'({way_sel[WAY_SEL_BITS-1:0], line_idx});
    end else begin : g_no_way_addr
        `UNUSED_VAR ({way_sel, mask_way})
        assign data_addr = merge_write ? mask_line : line_idx;
    end

    // Per-byte dirty mask (writeback only), one mask word per {way, line}. It
    // is read and updated only at the commit stage, so each access sees every
    // older update with no bypass. A line's mask is cleared when it is
    // installed, which leaves nothing for the reset walk to clear.
    if (DIRTY_BYTES != 0) begin : g_dirty_bytes
        wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] set_word;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin : g_set_word
            wire word_en = (`CS_WORDS_PER_LINE == 1) || (mask_word == i);
            assign set_word[i] = mask_byteen & {WORD_SIZE{mask_set && word_en}};
        end
        // A sector refill clears only the installed sector: the other resident
        // sectors may hold dirty bytes not yet written back.
        wire [`CS_SECTORS_PER_LINE-1:0][SECTOR_SIZE-1:0] clr_sector;
        for (genvar i = 0; i < `CS_SECTORS_PER_LINE; ++i) begin : g_clr_sector
            wire sec_en = (`CS_SECTORS_PER_LINE == 1) || (mask_sector == `UP(`CS_SECTOR_SEL_BITS)'(i));
            assign clr_sector[i] = {SECTOR_SIZE{mask_clr_line || (mask_clr_sector && sec_en)}};
        end
        wire [LINE_SIZE-1:0] set_mask = set_word;
        wire [LINE_SIZE-1:0] clr_mask = clr_sector;

        wire [DATA_ADDR_WIDTH-1:0] mask_addr;
        if (NUM_WAYS > 1) begin : g_mask_way_addr
            assign mask_addr = DATA_ADDR_WIDTH'({mask_way[WAY_SEL_BITS-1:0], mask_line});
        end else begin : g_mask_line_addr
            assign mask_addr = mask_line;
        end

        // A whole-word read-modify-write shares one write enable across the
        // mask, so it maps onto wide distributed RAM rather than one cell per
        // byte (or a shattered block RAM).
        wire [LINE_SIZE-1:0] mask_rdata;
        VX_sp_ram #(
            .DATAW       (LINE_SIZE),
            .SIZE        (DATA_RAM_DEPTH),
            .OUT_REG     (0),
            .LUTRAM      (1),
            .RDW_MODE    ("W"),
            .INIT_ENABLE (1),
            .INIT_VALUE  (0)
        ) byteen_store (
            .clk   (clk),
            .reset (reset),
            .read  (1'b1),
            .write (mask_write),
            .wren  (1'b1),
            .addr  (mask_addr),
            .wdata ((mask_rdata & ~clr_mask) | set_mask),
            .rdata (mask_rdata)
        );
        assign evict_byteen = mask_rdata;
    end else begin : g_no_dirty_bytes
        `UNUSED_VAR ({mask_write, mask_set, mask_clr_line, mask_clr_sector})
        `UNUSED_VAR ({mask_line, mask_word, mask_byteen, mask_sector})
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

        // a load reads its words' slices; writeback/flush reads all slices.
        wire slice_read = (read && read_words[s]) || ((fill || flush) && WRITEBACK);

        wire slice_write;
        wire [WRENW-1:0]          slice_wren;
        wire [`CS_WORD_WIDTH-1:0] slice_wdata;

        if (WRITE_ENABLE) begin : g_wren
            // fill writes the fetched sector's slices; a store writes only the hit slice.
            wire merge_sec_en = (`CS_SECTORS_PER_LINE == 1)
                             || (mask_sector == `UP(`CS_SECTOR_SEL_BITS)'(SLICE_SECTOR));
            assign slice_write = (fill_write && fill_sec_en) || (write && hit_any && word_en)
                              || (merge_write && merge_sec_en);
            assign slice_wren  = merge_write ? ~evict_byteen[s * WORD_SIZE +: WORD_SIZE]
                                             : (fill ? {WORD_SIZE{1'b1}} : write_byteen);
            assign slice_wdata = (fill || merge_write) ? fill_data[s] : write_word;
        end else begin : g_no_wren
            `UNUSED_VAR (write)
            `UNUSED_VAR (write_word)
            `UNUSED_VAR (write_byteen)
            `UNUSED_VAR (hit_any)
            `UNUSED_VAR ({fill_write, merge_write})
            `UNUSED_VAR (word_en)
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
