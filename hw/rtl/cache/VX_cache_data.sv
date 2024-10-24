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
    input wire                          stall,
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
    `UNUSED_VAR (stall)

    if (DIRTY_BYTES != 0) begin : g_dirty_bytes

        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] byteen_rdata;
        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] byteen_wdata;
        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] byteen_wren;

        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_byteen_wdata
            wire evict = fill || flush;
            wire evict_way_en = (NUM_WAYS == 1) || (evict_way == i);
            wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
            for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_write_mask
                wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == j);
                assign write_mask[j] = write_byteen & {WORD_SIZE{word_en}};
            end
            assign byteen_wdata[i] = {LINE_SIZE{write}}; // only asserted on writes
            assign byteen_wren[i]  = {LINE_SIZE{init}}
                                   | {LINE_SIZE{evict && evict_way_en}}
                                   | ({LINE_SIZE{write && tag_matches[i]}} & write_mask);
        end

        wire byteen_read = fill || flush;
        wire byteen_write = init || write || fill || flush;

        VX_sp_ram #(
            .DATAW (LINE_SIZE * NUM_WAYS),
            .WRENW (LINE_SIZE * NUM_WAYS),
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
            .rdata (byteen_rdata)
        );

        assign evict_byteen = byteen_rdata[way_idx_r];
    end else begin : g_no_dirty_bytes
        `UNUSED_VAR (init)
        `UNUSED_VAR (flush)
        assign evict_byteen = '1; // update whole line
    end

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_rdata;

    if (WRITE_ENABLE) begin : g_data_store
        // create a single write-enable block ram to reduce area overhead
        wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] line_wren;
        wire line_write;
        wire line_read;

        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_wdata
            wire fill_way_en = (NUM_WAYS == 1) || (evict_way == i);
            wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
            for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_write_mask
                wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == j);
                assign write_mask[j] = write_byteen & {WORD_SIZE{word_en}};
            end
            assign line_wdata[i] = fill ? fill_data : {`CS_WORDS_PER_LINE{write_word}};
            assign line_wren[i] = {LINE_SIZE{fill && fill_way_en}}
                                | ({LINE_SIZE{write && tag_matches[i]}} & write_mask);
        end

        assign line_read = read || ((fill || flush) && WRITEBACK);
        assign line_write = fill || (write && WRITE_ENABLE);

        VX_sp_ram #(
            .DATAW (NUM_WAYS * `CS_LINE_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .WRENW (NUM_WAYS * LINE_SIZE),
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
            .rdata (line_rdata)
        );
    end else begin : g_data_store
        `UNUSED_VAR (write)
        `UNUSED_VAR (write_byteen)
        `UNUSED_VAR (write_word)
        `UNUSED_VAR (word_idx)
        `UNUSED_VAR (tag_matches)

        // we don't merge the ways into a single block ram due to WREN overhead
        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_ways
            wire fill_way_en = (NUM_WAYS == 1) || (evict_way == i);
            VX_sp_ram #(
                .DATAW (`CS_LINE_WIDTH),
                .SIZE  (`CS_LINES_PER_BANK),
                .OUT_REG (1),
                .RDW_MODE ("R")
            ) data_store (
                .clk   (clk),
                .reset (reset),
                .read  (read),
                .write (fill && fill_way_en),
                .wren  (1'b1),
                .addr  (line_idx),
                .wdata (fill_data),
                .rdata (line_rdata[i])
            );
        end
    end

    assign read_data = line_rdata[way_idx_r];

endmodule
