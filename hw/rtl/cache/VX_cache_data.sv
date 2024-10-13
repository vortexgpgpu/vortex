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
    input wire [`CS_WORD_WIDTH-1:0]     write_data,
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

    localparam BYTEENW = (WRITE_ENABLE != 0) ? LINE_SIZE : 1;

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
            wire dirty_data = write; // only asserted on writes
            wire dirty_wren = init || (write ? tag_matches[i] : evict_way[i]);

            if (DIRTY_BYTES != 0) begin : g_dirty_bytes
                wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] bytes_data;
                wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] bytes_wren;
                for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_j
                    wire word_sel = tag_matches[i] && ((WORD_SIZE == 1) || (word_idx == j));
                    wire [WORD_SIZE-1:0] word_en = write_byteen & {WORD_SIZE{word_sel}};
                    assign bytes_data[j] = {WORD_SIZE{write}}; // only asserted on writes
                    assign bytes_wren[j] = {WORD_SIZE{init}} | (write ? word_en : {WORD_SIZE{evict_way[i]}});
                end
                assign byteen_wdata[i] = {dirty_data, bytes_data};
                assign byteen_wren[i] = {dirty_wren, bytes_wren};
                assign {line_dirty, evict_byteen} = byteen_rdata[evict_way_idx_r];
            end else begin : g_no_dirty_bytes
                assign byteen_wdata[i] = dirty_data;
                assign byteen_wren[i] = dirty_wren;
                assign line_dirty = byteen_rdata[evict_way_idx_r];
                assign evict_byteen = '1;
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

        assign evict_data = line_rdata[evict_way_idx_r];

    end else begin : g_no_writeback
        `UNUSED_VAR (init)
        assign line_dirty = 0;
        assign evict_data = '0;
        assign evict_byteen = '0;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_data_store

        wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
        wire [BYTEENW-1:0] line_wren;
        wire line_write;
        wire line_read;

        if (WRITE_ENABLE != 0) begin : g_line_data
            wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wren_w;
            for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_j
                wire word_en = (WORD_SIZE == 1) || (word_idx == j);
                assign line_wdata[j] = write ? write_data : fill_data[j];
                assign wren_w[j] = write ? (write_byteen & {WORD_SIZE{word_en}}) : {WORD_SIZE{1'b1}};
            end
            assign line_wren  = wren_w;
            assign line_write = (fill && ((NUM_WAYS == 1) || evict_way[i]))
                             || (write && tag_matches[i]);
            assign line_read = read || ((fill || flush) && WRITEBACK);
        end else begin : g_line_data_ro
            `UNUSED_VAR (write)
            `UNUSED_VAR (flush)
            `UNUSED_VAR (write_byteen)
            `UNUSED_VAR (write_data)
            `UNUSED_VAR (word_idx)
            assign line_wdata = fill_data;
            assign line_wren  = 1'b1;
            assign line_write = fill && ((NUM_WAYS == 1) || evict_way[i]);
            assign line_read  = read;
        end

        VX_sp_ram #(
            .DATAW (`CS_LINE_WIDTH),
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
            .rdata (line_rdata[i])
        );
    end

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
