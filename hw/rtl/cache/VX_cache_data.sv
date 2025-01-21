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

module VX_cache_data #(
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
    input wire [`CS_WAY_SEL_WIDTH-1:0]  way_idx_r,
    // outputs
    output wire [`CS_LINE_WIDTH-1:0]    read_data,
    output wire [LINE_SIZE-1:0]         evict_byteen
);
    `UNUSED_PARAM (WORD_SIZE)

    wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
    for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin : g_write_mask
        wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == i);
        assign write_mask[i] = write_byteen & {WORD_SIZE{word_en}};
    end

    if (DIRTY_BYTES != 0) begin : g_dirty_bytes

        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] byteen_rdata;

        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_byteen_store
            wire [LINE_SIZE-1:0] byteen_wdata = {LINE_SIZE{write}}; // only asserted on writes
            wire [LINE_SIZE-1:0] byteen_wren = {LINE_SIZE{init || fill || flush}} | write_mask;
            wire byteen_write = ((fill || flush) && ((NUM_WAYS == 1) || (evict_way == i)))
                             || (write && tag_matches[i])
                             || init;
            wire byteen_read  = fill || flush;

            VX_sp_ram #(
                .DATAW (LINE_SIZE),
                .WRENW (LINE_SIZE),
                .SIZE  (`CS_LINES_PER_BANK),
                .OUT_REG (1),
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
        assign evict_byteen = '1; // update whole line
    end

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_rdata;

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_data_store

        localparam WRENW = WRITE_ENABLE ? LINE_SIZE : 1;

        wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
        wire [WRENW-1:0] line_wren;

        if (WRITE_ENABLE) begin : g_wren
            assign line_wdata = fill ? fill_data : {`CS_WORDS_PER_LINE{write_word}};
            assign line_wren  = {LINE_SIZE{fill}} | write_mask;
        end else begin : g_no_wren
            `UNUSED_VAR (write_word)
            `UNUSED_VAR (write_mask)
            assign line_wdata = fill_data;
            assign line_wren  = 1'b1;
        end

        wire line_write = (fill && ((NUM_WAYS == 1) || (evict_way == i)))
                       || (write && tag_matches[i] && WRITE_ENABLE);

        wire line_read = read || ((fill || flush) && WRITEBACK);

        VX_sp_ram #(
            .DATAW (`CS_LINE_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .WRENW (WRENW),
            .OUT_REG (1),
            .RDW_MODE ("R")
        ) data_store (
            .clk   (clk),
            .reset (reset),
            .read  (line_read),
            .write (line_write),
            .wren  (line_wren),
            .addr  (line_idx),
            .wdata (line_wdata),
            .rdata (line_rdata[i])
        );
    end

    assign read_data = line_rdata[way_idx_r];

endmodule
