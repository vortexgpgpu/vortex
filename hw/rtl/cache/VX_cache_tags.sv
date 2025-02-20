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

module VX_cache_tags #(
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
    parameter WRITEBACK     = 0
) (
    input wire                          clk,
    input wire                          reset,

    // inputs
    input wire                          init,
    input wire                          flush,
    input wire                          fill,
    input wire                          read,
    input wire                          write,
    input wire [`CS_LINE_SEL_BITS-1:0]  line_idx,
    input wire [`CS_TAG_SEL_BITS-1:0]   line_tag,
    input wire [`CS_WAY_SEL_WIDTH-1:0]  evict_way,

    // outputs
    output wire [NUM_WAYS-1:0]          tag_matches,
    output wire                         evict_dirty,
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag
);
    //                   valid,  dirty,          tag
    localparam TAG_WIDTH = 1 + WRITEBACK + `CS_TAG_SEL_BITS;

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0] read_valid;
    wire [NUM_WAYS-1:0] read_dirty;
    `UNUSED_VAR (read)

    if (WRITEBACK) begin : g_evict_tag_wb
        assign evict_dirty = read_dirty[evict_way];
        assign evict_tag = read_tag[evict_way];
    end else begin : g_evict_tag_wt
        `UNUSED_VAR (read_dirty)
        assign evict_dirty = 1'b0;
        assign evict_tag = '0;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_tag_store
        wire way_en   = (NUM_WAYS == 1) || (evict_way == i);
        wire do_init  = init; // init all ways
        wire do_fill  = fill && way_en;
        wire do_flush = flush && (!WRITEBACK || way_en); // flush the whole line in writethrough mode
        wire do_write = WRITEBACK && write && tag_matches[i]; // only write on tag hit

        wire line_read  = read || write || (WRITEBACK && (fill || flush));
        wire line_write = do_init || do_fill || do_flush || do_write;
        wire line_valid = fill || write;

        wire [TAG_WIDTH-1:0] line_wdata;
        wire [TAG_WIDTH-1:0] line_rdata;

        if (WRITEBACK) begin : g_wdata
            assign line_wdata = {line_valid, write, line_tag};
            assign {read_valid[i], read_dirty[i], read_tag[i]} = line_rdata;
        end else begin : g_wdata
            assign line_wdata = {line_valid, line_tag};
            assign {read_valid[i], read_tag[i]} = line_rdata;
            assign read_dirty[i] = 1'b0;
        end

        VX_sp_ram #(
            .DATAW (TAG_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .RDW_MODE ("W"),
            .RADDR_REG (1)
        ) tag_store (
            .clk   (clk),
            .reset (reset),
            .read  (line_read),
            .write (line_write),
            .wren  (1'b1),
            .addr  (line_idx),
            .wdata (line_wdata),
            .rdata (line_rdata)
        );
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_tag_matches
        assign tag_matches[i] = read_valid[i] && (line_tag == read_tag[i]);
    end

endmodule
