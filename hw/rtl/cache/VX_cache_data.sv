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
    input wire [NUM_WAYS-1:0]           evict_way,
    input wire [NUM_WAYS-1:0]           tag_matches,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] fill_data,
    input wire [`CS_WORD_WIDTH-1:0]     write_word,
    input wire [WORD_SIZE-1:0]          write_byteen,
    input wire [`UP(`CS_WORD_SEL_BITS)-1:0] word_idx,
    // outputs
    output wire [`CS_WORD_WIDTH-1:0]    read_data,
    output wire                         line_dirty,
    output wire [`CS_LINE_WIDTH-1:0]    evict_data,
    output wire [LINE_SIZE-1:0]         evict_byteen
);
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (stall)

    localparam BYTEENW = (WRITE_ENABLE != 0 || NUM_WAYS != 1) ? (LINE_SIZE * NUM_WAYS) : 1;

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_rdata;

    if (WRITEBACK != 0) begin : g_writeback
        localparam BYTEEN_DATAW = 1 + ((DIRTY_BYTES != 0) ? LINE_SIZE : 0);
        wire [`LOG2UP(NUM_WAYS)-1:0] evict_way_idx, evict_way_idx_r;

        VX_onehot_encoder #(
            .N (NUM_WAYS)
        ) fill_way_enc (
            .data_in  (evict_way),
            .data_out (evict_way_idx),
            `UNUSED_PIN (valid_out)
        );

        `BUFFER_EX(evict_way_idx_r, evict_way_idx, ~stall, 1);

        wire [NUM_WAYS-1:0][BYTEEN_DATAW-1:0] byteen_rdata;
        wire [NUM_WAYS-1:0][BYTEEN_DATAW-1:0] byteen_wdata;
        wire [NUM_WAYS-1:0][BYTEEN_DATAW-1:0] byteen_wren;

        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_byteen_wdata
            wire evict = fill || flush;
            wire evict_way_en = (NUM_WAYS == 1) || evict_way[i];
            wire dirty_data = write; // only asserted on writes
            wire dirty_wren = init || (evict && evict_way_en) || (write && tag_matches[i]);
            if (DIRTY_BYTES != 0) begin : g_dirty_bytes
                wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
                for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_write_mask
                    wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == j);
                    assign write_mask[j] = write_byteen & {WORD_SIZE{word_en}};
                end
                wire [LINE_SIZE-1:0] bytes_data = {LINE_SIZE{write}}; // only asserted on writes
                wire [LINE_SIZE-1:0] bytes_wren = {LINE_SIZE{init}}
                                                | {LINE_SIZE{evict && evict_way_en}}
                                                | ({LINE_SIZE{write && tag_matches[i]}} & write_mask);
                assign byteen_wdata[i] = {dirty_data, bytes_data};
                assign byteen_wren[i] = {dirty_wren, bytes_wren};
            end else begin : g_no_dirty_bytes
                assign byteen_wdata[i] = dirty_data;
                assign byteen_wren[i] = dirty_wren;
            end
        end

        wire byteen_read = fill || flush;
        wire byteen_write = init || write || fill || flush;

        VX_sp_ram #(
            .DATAW (BYTEEN_DATAW * NUM_WAYS),
            .WRENW (BYTEEN_DATAW * NUM_WAYS),
            .SIZE  (`CS_LINES_PER_BANK),
            .OUT_REG (1)
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

        if (DIRTY_BYTES != 0) begin : g_line_dirty_and_byteen
            assign {line_dirty, evict_byteen} = byteen_rdata[evict_way_idx_r];
        end else begin : g_line_dirty
            assign line_dirty = byteen_rdata[evict_way_idx_r];
            assign evict_byteen = '1;
        end

        assign evict_data = line_rdata[evict_way_idx_r];

    end else begin : g_no_writeback
        `UNUSED_VAR (init)
        `UNUSED_VAR (flush)
        assign line_dirty = 0;
        assign evict_data = '0;
        assign evict_byteen = '0;
    end

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
    wire [BYTEENW-1:0] line_wren;
    wire line_write;
    wire line_read;

    if (BYTEENW != 1) begin : g_wdata
        wire [NUM_WAYS-1:0][LINE_SIZE-1:0] line_wren_w;
        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_ways
            wire fill_way_en = (NUM_WAYS == 1) || evict_way[i];
            if (WRITE_ENABLE != 0) begin : g_we
                wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_mask;
                for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_write_mask
                    wire word_en = (`CS_WORDS_PER_LINE == 1) || (word_idx == j);
                    assign write_mask[j] = write_byteen & {WORD_SIZE{word_en}};
                end
                assign line_wdata[i] = (fill && fill_way_en) ? fill_data : {`CS_WORDS_PER_LINE{write_word}};
                assign line_wren_w[i] = {LINE_SIZE{fill && fill_way_en}}
                                      | ({LINE_SIZE{write && tag_matches[i]}} & write_mask);
            end else begin : g_ro
                `UNUSED_VAR (write)
                `UNUSED_VAR (write_byteen)
                `UNUSED_VAR (write_word)
                `UNUSED_VAR (word_idx)
                assign line_wdata[i] = fill_data;
                assign line_wren_w[i] = {LINE_SIZE{fill_way_en}};
            end
        end
        assign line_wren = line_wren_w;
    end else begin : g_ro_1w_wdata
        `UNUSED_VAR (write)
        `UNUSED_VAR (evict_way)
        `UNUSED_VAR (write_byteen)
        `UNUSED_VAR (write_word)
        assign line_wdata = fill_data;
        assign line_wren = 1'b1;
    end

    assign line_write = fill || (write && WRITE_ENABLE);
    assign line_read = read || ((fill || flush) && WRITEBACK);

    VX_sp_ram #(
        .DATAW (NUM_WAYS * `CS_LINE_WIDTH),
        .SIZE  (`CS_LINES_PER_BANK),
        .WRENW (BYTEENW),
        .OUT_REG (1)
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

    wire [`LOG2UP(NUM_WAYS)-1:0] hit_way_idx;
    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) hit_idx_enc (
        .data_in  (tag_matches),
        .data_out (hit_way_idx),
        `UNUSED_PIN (valid_out)
    );

    if (`CS_WORDS_PER_LINE > 1) begin : g_read_data
        // order the data layout to perform ways multiplexing last.
        // this allows converting way index to binary in parallel with BRAM read and word indexing.
        wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] transposed_rdata;
        VX_transpose #(
            .DATAW (`CS_WORD_WIDTH),
            .N (NUM_WAYS),
            .M (`CS_WORDS_PER_LINE)
        ) transpose (
            .data_in  (line_rdata),
            .data_out (transposed_rdata)
        );
        assign read_data = transposed_rdata[word_idx][hit_way_idx];
    end else begin : g_read_data_1w
        `UNUSED_VAR (word_idx)
        assign read_data = line_rdata[hit_way_idx];
    end

endmodule
