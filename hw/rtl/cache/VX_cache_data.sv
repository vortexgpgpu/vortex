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
    parameter `STRING INSTANCE_ID= "",
    parameter BANK_ID           = 0,
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
    parameter DIRTY_BYTES       = 0,
    // Request debug identifier
    parameter UUID_WIDTH        = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

    input wire                          init,
    input wire                          read,
    input wire                          fill,
    input wire                          flush,
    input wire                          write,
    input wire [`CS_LINE_ADDR_WIDTH-1:0] line_addr,
    input wire [`UP(`CS_WORD_SEL_BITS)-1:0] word_idx,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] fill_data,
    input wire [`CS_WORD_WIDTH-1:0]     write_data,
    input wire [WORD_SIZE-1:0]          write_byteen,
    input wire [NUM_WAYS-1:0]           way_idx,
    output wire [`CS_WORD_WIDTH-1:0]    read_data,
    output wire [`CS_LINE_WIDTH-1:0]    dirty_data,
    output wire [LINE_SIZE-1:0]         dirty_byteen
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (stall)
    `UNUSED_VAR (line_addr)
    `UNUSED_VAR (init)
    `UNUSED_VAR (read)
    `UNUSED_VAR (flush)

    localparam BYTEENW = (WRITE_ENABLE != 0) ? LINE_SIZE : 1;

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_rdata;
    wire [`LOG2UP(NUM_WAYS)-1:0] way_idx_bin;
    wire [`CS_LINE_SEL_BITS-1:0] line_idx;

    assign line_idx = line_addr[`CS_LINE_SEL_BITS-1:0];

    VX_encoder #(
        .N (NUM_WAYS)
    ) way_idx_enc (
        .data_in  (way_idx),
        .data_out (way_idx_bin),
        `UNUSED_PIN (valid_out)
    );

    if (WRITEBACK) begin : g_dirty_data
        assign dirty_data = line_rdata[way_idx_bin];
    end else begin : g_dirty_data_0
        assign dirty_data = '0;
    end

    if (DIRTY_BYTES) begin : g_dirty_byteen
        wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] bs_rdata;
        wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] bs_wdata;

        for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_bs_wdata
            for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_j
                wire [WORD_SIZE-1:0] word_mask = {WORD_SIZE{(WORD_SIZE == 1) || (word_idx == j)}};
                wire [WORD_SIZE-1:0] wdata = write ? (bs_rdata[i][j] | (write_byteen & word_mask)) : ((fill || flush) ? '0 : bs_rdata[i][j]);
                assign bs_wdata[i][j] = init ? '0 : (way_idx[i] ? wdata : bs_rdata[i][j]);
            end
        end

        wire bs_read  = write || fill || flush;
        wire bs_write = init || write || fill || flush;

        VX_sp_ram #(
            .DATAW (LINE_SIZE * NUM_WAYS),
            .SIZE  (`CS_LINES_PER_BANK)
        ) byteen_store (
            .clk   (clk),
            .reset (reset),
            .read  (bs_read && ~stall),
            .write (bs_write && ~stall),
            .wren  (1'b1),
            .addr  (line_idx),
            .wdata (bs_wdata),
            .rdata (bs_rdata)
        );

        assign dirty_byteen = bs_rdata[way_idx_bin];
    end else begin : g_dirty_byteen_0
        assign dirty_byteen = '1;
    end

    for (genvar i = 0; i < NUM_WAYS; ++i) begin : g_data_store

        wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
        wire [BYTEENW-1:0] line_wren;
        wire line_write;
        wire line_read;

        wire way_en = (NUM_WAYS == 1) || way_idx[i];

        if (WRITE_ENABLE != 0) begin : g_line_data
            wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] wren_w;
            for (genvar j = 0; j < `CS_WORDS_PER_LINE; ++j) begin : g_j
                wire word_en = (WORD_SIZE == 1) || (word_idx == j);
                assign line_wdata[j] = fill ? fill_data[j] : write_data;
                assign wren_w[j] = {WORD_SIZE{fill}} | (write_byteen & {WORD_SIZE{word_en}});
            end
            assign line_wren  = wren_w;
            assign line_write = (fill || write) && way_en;
            if (WRITEBACK) begin : g_line_read_wb
                assign line_read = (read || fill || flush);
            end else begin : g_line_read_wt
                assign line_read = read;
            end
        end else begin : g_line_data_ro
            `UNUSED_VAR (write)
            `UNUSED_VAR (write_byteen)
            `UNUSED_VAR (write_data)
            assign line_wdata = fill_data;
            assign line_wren  = 1'b1;
            assign line_write = fill && way_en;
            assign line_read  = read;
        end

        VX_sp_ram #(
            .DATAW (`CS_LINE_WIDTH),
            .SIZE  (`CS_LINES_PER_BANK),
            .WRENW (BYTEENW),
            .NO_RWCHECK (1),
            .RW_ASSERT (1)
        ) data_store (
            .clk   (clk),
            .reset (reset),
            .read  (line_read && ~stall),
            .write (line_write && ~stall),
            .wren  (line_wren),
            .addr  (line_idx),
            .wdata (line_wdata),
            .rdata (line_rdata[i])
        );
    end

    if (`CS_WORDS_PER_LINE > 1) begin : g_read_data
        // order the data layout to perform ways multiplexing last.
        // this allows converting way index to binary in parallel with BRAM readaccess  and way selection.
        wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] transposed_rdata;
        VX_transpose #(
            .DATAW (`CS_WORD_WIDTH),
            .N (NUM_WAYS),
            .M (`CS_WORDS_PER_LINE)
        ) transpose (
            .data_in  (line_rdata),
            .data_out (transposed_rdata)
        );
        assign read_data = transposed_rdata[word_idx][way_idx_bin];
    end else begin : g_read_data_1w
        `UNUSED_VAR (word_idx)
        assign read_data = line_rdata[way_idx_bin];
    end

`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%t: %s fill: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_idx, line_idx, fill_data))
        end
        if (flush && ~stall) begin
            `TRACE(3, ("%t: %s flush: addr=0x%0h, way=%b, blk_addr=%0d, byteen=0x%h, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_idx, line_idx, dirty_byteen, dirty_data))
        end
        if (read && ~stall) begin
            `TRACE(3, ("%t: %s read: addr=0x%0h, way=%b, blk_addr=%0d, wsel=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_idx, line_idx, word_idx, read_data, req_uuid))
        end
        if (write && ~stall) begin
            `TRACE(3, ("%t: %s write: addr=0x%0h, way=%b, blk_addr=%0d, wsel=%0d, byteen=0x%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_idx, line_idx, word_idx, write_byteen, write_data, req_uuid))
        end
    end
`endif

endmodule
