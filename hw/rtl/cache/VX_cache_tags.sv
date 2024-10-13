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
    input wire                          stall,

    // inputs
    input wire                          init,
    input wire                          flush,
    input wire                          fill,
    input wire                          lookup,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] line_addr,
    input wire [NUM_WAYS-1:0]           flush_way,

    // outputs
    output wire [NUM_WAYS-1:0]          tag_matches_r,
    output wire [`CS_TAG_SEL_BITS-1:0]  line_tag_r,
    output wire [NUM_WAYS-1:0]          evict_way,
    output wire [NUM_WAYS-1:0]          evict_way_r,
    output wire [`CS_TAG_SEL_BITS-1:0]  evict_tag_r
);
    //                   valid,       tag
    localparam TAG_WIDTH = 1 + `CS_TAG_SEL_BITS;

    wire [`CS_LINE_SEL_BITS-1:0] line_idx = line_addr[`CS_LINE_SEL_BITS-1:0];
    wire [`CS_TAG_SEL_BITS-1:0] line_tag = `CS_LINE_ADDR_TAG(line_addr);

    wire [NUM_WAYS-1:0][`CS_TAG_SEL_BITS-1:0] read_tag;
    wire [NUM_WAYS-1:0] read_valid;

    if (NUM_WAYS > 1) begin : g_evict_way
        reg [NUM_WAYS-1:0] victim_way;
        // cyclic assignment of replacement way
        always @(posedge clk) begin
            if (reset) begin
                victim_way <= 1;
            end else if (~stall) begin
                victim_way <= {victim_way[NUM_WAYS-2:0], victim_way[NUM_WAYS-1]};
            end
        end
        assign evict_way = fill ? victim_way : flush_way;
        `BUFFER_EX(evict_way_r, evict_way, ~stall, 1);
    end else begin : g_evict_way_0
        `UNUSED_VAR (flush_way)
        assign evict_way   = 1'b1;
        assign evict_way_r = 1'b1;
    end

    if (WRITEBACK) begin : g_evict_tag_wb
        VX_onehot_mux #(
            .DATAW (`CS_TAG_SEL_BITS),
            .N     (NUM_WAYS)
        ) evict_tag_sel (
            .data_in  (read_tag),
            .sel_in   (evict_way_r),
            .data_out (evict_tag_r)
        );
    end else begin : g_evict_tag_wt
        assign evict_tag_r = '0;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_tag_store

        wire do_fill    = fill && evict_way[i];
        wire do_flush   = flush && (!WRITEBACK || evict_way[i]); // flush the whole line in writethrough mode

        wire line_read  = lookup || (WRITEBACK && (fill || flush));
        wire line_write = init || do_fill || do_flush;
        wire line_valid = fill;

        wire [TAG_WIDTH-1:0] line_wdata;
        wire [TAG_WIDTH-1:0] line_rdata;

        assign line_wdata = {line_valid, line_tag};
        assign {read_valid[i], read_tag[i]} = line_rdata;

        VX_sp_ram #(
            .DATAW (TAG_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .OUT_REG (1)
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

    `BUFFER_EX(line_tag_r, line_tag, ~stall, 1);

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_tag_matches
        assign tag_matches_r[i] = read_valid[i] && (line_tag_r == read_tag[i]);
    end

endmodule
