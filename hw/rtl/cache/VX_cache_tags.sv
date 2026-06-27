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

// Single-array tag store with per-way write-enable.
//
// All NUM_WAYS tags for a set live in one block-RAM word (read in parallel for
// the hit compare); a per-way write-enable updates a single way on a
// fill/write/invalidate without a read-modify-write. This replaces the
// previous NUM_WAYS separate tag arrays with one BRAM.

module VX_cache_tags import VX_gpu_pkg::*; #(
    // Size of cache in bytes
    parameter CACHE_SIZE    = 1024,
    // Size of line inside a bank in bytes
    parameter LINE_SIZE     = 16,
    // Number of banks
    parameter NUM_BANKS     = 1,
    // Number of associative ways
    parameter NUM_WAYS      = 1,
    // Size of a word in bytes
    parameter WORD_SIZE     = 1,
    // Enable cache writeback
    parameter WRITEBACK     = 0,
    // Enable the AMO-passthrough line invalidate (non-LLC banks only)
    parameter AMO_ENABLE    = 0
) (
    input wire                          clk,
    input wire                          reset,

    // inputs
    input wire                          stall,
    input wire                          init,
    input wire                          flush,
    input wire                          fill,
    input wire                          read,
    input wire                          write,
    input wire                          invalidate, // clear valid on the hit way
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx_n,
    input wire [`CS_TAG_SEL_BITS-1:0]   line_tag,
    input wire [`CS_WAY_SEL_WIDTH-1:0]  evict_way,

    // outputs
    output wire [NUM_WAYS-1:0]          tag_matches,
    output wire                         evict_dirty,
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag
);
    //                    valid,          tag    (dirty held in a separate array)
    localparam TAG_ENTRYW = 1 + `CS_TAG_SEL_BITS;
    `UNUSED_VAR (read)

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0] read_valid;
    wire [NUM_WAYS-1:0] read_dirty;

    if (WRITEBACK) begin : g_evict_tag_wb
        assign evict_dirty = read_dirty[evict_way];
        assign evict_tag = read_tag[evict_way];
    end else begin : g_evict_tag_wt
        `UNUSED_VAR (read_dirty)
        assign evict_dirty = 1'b0;
        assign evict_tag = '0;
    end

    // Per-way decoded strobes. At most one operation type fires per cycle
    // (input arbitration in the bank). The tag word is rewritten on
    // init/fill/flush/invalidate; a write hit only updates the dirty bit, so it
    // is excluded from the tag write-enable and the tag BRAM never waits on the
    // tag compare. The dirty bit lives in a separate array updated in parallel.
    wire [NUM_WAYS-1:0] tag_write;
    wire [NUM_WAYS-1:0][TAG_ENTRYW-1:0] line_wdata;
    wire [NUM_WAYS-1:0][TAG_ENTRYW-1:0] tag_rdata;

    wire [NUM_WAYS-1:0] dirty_write;
    wire [NUM_WAYS-1:0] dirty_wdata;
    wire [NUM_WAYS-1:0] dirty_rdata;

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_way_decode
        // raw valid tag match, excluding the just-filled (rdw_fill) case.
        wire raw_hit = read_valid[i] && (line_tag == read_tag[i]);

        wire way_en   = (NUM_WAYS == 1) || (evict_way == i);
        wire do_init  = init; // init all ways
        wire do_fill  = fill && way_en;
        wire do_flush = flush && (!WRITEBACK || way_en); // flush all ways in writethrough mode
        wire do_write = WRITEBACK && write && tag_matches[i]; // only write on tag hit
        // AMO passthrough invalidate: clear valid on the resident hit way.
        // Using raw_hit (not tag_matches) skips a line being filled this
        // cycle, so an in-flight fill's replay still finds its line.
        wire do_inval = (AMO_ENABLE != 0) && invalidate && raw_hit;

        wire line_valid = (fill || write) && ~do_inval;

        // Tag word updated on init/fill/flush/invalidate only (not a write hit).
        assign tag_write[i] = do_init || do_fill || do_flush || do_inval;
        assign line_wdata[i] = {line_valid, line_tag};

        // Dirty bit: set by a write hit, cleared by init/fill/flush/invalidate.
        assign dirty_write[i] = tag_write[i] || do_write;
        assign dirty_wdata[i] = write;

        // Read-First arrays: Read-During-Write hazard not supported. Fills are
        // followed by MSHR replays that hit; a dirty bit set by a write must be
        // visible to a fill/flush reading the same line the next cycle.
        wire rdw_fill, rdw_write;
        `BUFFER(rdw_fill, do_fill);
        `BUFFER(rdw_write, do_write && (line_idx == line_idx_n));

        wire [TAG_ENTRYW-1:0] rdata_i = tag_rdata[i];
        assign {read_valid[i], read_tag[i]} = rdata_i;

        if (WRITEBACK) begin : g_dirty
            assign read_dirty[i] = dirty_rdata[i] || rdw_write;
        end else begin : g_dirty
            `UNUSED_VAR (rdw_write)
            assign read_dirty[i] = 1'b0;
        end

        assign tag_matches[i] = raw_hit || rdw_fill;
    end

    // Single tag array: {valid, tag} per way; per-way write-enable updates a
    // single way. Read at line_idx_n (one cycle ahead), written at line_idx,
    // read-first to match the pipeline's fill/replay ordering.
    VX_dp_ram #(
        .DATAW (NUM_WAYS * TAG_ENTRYW),
        .WRENW (NUM_WAYS),
        .SIZE  (`CS_LINES_PER_BANK),
        .OUT_REG (1),
        .RDW_MODE ("R")
    ) tag_store (
        .clk   (clk),
        .reset (reset),
        .read  (~stall),
        .write (| tag_write),
        .wren  (tag_write),
        .waddr (line_idx),
        .raddr (line_idx_n),
        .wdata (line_wdata),
        .rdata (tag_rdata)
    );

    // Dirty-state array: 1 bit/way in LUTRAM, decoupled from the tag BRAM so a
    // write hit's dirty update never gates the tag store's write-enable. Same
    // look-ahead/read-first access as the tag store keeps the two aligned.
    if (WRITEBACK) begin : g_dirty_store
        VX_dp_ram #(
            .DATAW (NUM_WAYS),
            .WRENW (NUM_WAYS),
            .SIZE  (`CS_LINES_PER_BANK),
            .OUT_REG (1),
            .LUTRAM (1),
            .RDW_MODE ("R")
        ) dirty_store (
            .clk   (clk),
            .reset (reset),
            .read  (~stall),
            .write (| dirty_write),
            .wren  (dirty_write),
            .waddr (line_idx),
            .raddr (line_idx_n),
            .wdata (dirty_wdata),
            .rdata (dirty_rdata)
        );
    end else begin : g_no_dirty_store
        `UNUSED_VAR (dirty_write)
        `UNUSED_VAR (dirty_wdata)
        assign dirty_rdata = '0;
        `UNUSED_VAR (dirty_rdata)
    end

endmodule
