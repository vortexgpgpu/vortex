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
    input wire [`UP(`CS_WORD_SEL_BITS)-1:0] wsel,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] fill_data,
    input wire [`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] write_data,
    input wire [`CS_WORDS_PER_LINE-1:0][WORD_SIZE-1:0] write_byteen,
    input wire [NUM_WAYS-1:0]           way_sel,
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

    localparam BYTEENW = (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) ? (LINE_SIZE * NUM_WAYS) : 1;

    wire [`CS_LINE_SEL_BITS-1:0] line_sel = line_addr[`CS_LINE_SEL_BITS-1:0];

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] line_rdata;
    wire [`LOG2UP(NUM_WAYS)-1:0] way_idx;

    if (WRITEBACK) begin
        if (DIRTY_BYTES) begin
            wire [NUM_WAYS-1:0][LINE_SIZE-1:0] bs_rdata;
            wire [NUM_WAYS-1:0][LINE_SIZE-1:0] bs_wdata;

            for (genvar i = 0; i < NUM_WAYS; ++i) begin
                wire [LINE_SIZE-1:0] wdata = write ? (bs_rdata[i] | write_byteen) : ((fill || flush) ? '0 : bs_rdata[i]);
                assign bs_wdata[i] = init ? '0 : (way_sel[i] ? wdata : bs_rdata[i]);
            end

            VX_sp_ram #(
                .DATAW (LINE_SIZE * NUM_WAYS),
                .SIZE  (`CS_LINES_PER_BANK)
            ) byteen_store (
                .clk   (clk),
                .reset (reset),
                .read  (write || fill || flush),
                .write (init || write || fill || flush),
                .wren  (1'b1),
                .addr  (line_sel),
                .wdata (bs_wdata),
                .rdata (bs_rdata)
            );

            assign dirty_byteen = bs_rdata[way_idx];
        end else begin
            assign dirty_byteen = {LINE_SIZE{1'b1}};
        end

        wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] flipped_rdata;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
            for (genvar j = 0; j < NUM_WAYS; ++j) begin
                assign flipped_rdata[j][i] = line_rdata[i][j];
            end
        end
        assign dirty_data = flipped_rdata[way_idx];
    end else begin
        assign dirty_byteen = '0;
        assign dirty_data = '0;
    end

    // order the data layout to perform ways multiplexing last.
    // this allows converting way index to binary in parallel with BRAM readaccess  and way selection.

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] line_wdata;
    wire [BYTEENW-1:0] line_wren;

    if (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) begin
        wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][WORD_SIZE-1:0] wren_w;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
            for (genvar j = 0; j < NUM_WAYS; ++j) begin
                assign line_wdata[i][j] = (fill || !WRITE_ENABLE) ? fill_data[i] : write_data[i];
                assign wren_w[i][j] = ((fill || !WRITE_ENABLE) ? {WORD_SIZE{1'b1}} : write_byteen[i])
                                    & {WORD_SIZE{(way_sel[j] || (NUM_WAYS == 1))}};
            end
        end
        assign line_wren = wren_w;
    end else begin
        `UNUSED_VAR (write)
        `UNUSED_VAR (write_byteen)
        `UNUSED_VAR (write_data)
        assign line_wdata = fill_data;
        assign line_wren  = fill;
    end

    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) way_enc (
        .data_in  (way_sel),
        .data_out (way_idx),
        `UNUSED_PIN (valid_out)
    );

    wire line_read = (read && ~stall)
                  || (WRITEBACK && (fill || flush));

    wire line_write = write || fill;

    VX_sp_ram #(
        .DATAW (`CS_LINE_WIDTH * NUM_WAYS),
        .SIZE  (`CS_LINES_PER_BANK),
        .WRENW (BYTEENW),
        .NO_RWCHECK (1),
        .RW_ASSERT (1)
    ) data_store (
        .clk   (clk),
        .reset (reset),
        .read  (line_read),
        .write (line_write),
        .wren  (line_wren),
        .addr  (line_sel),
        .wdata (line_wdata),
        .rdata (line_rdata)
    );

    wire [NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] per_way_rdata;
    if (`CS_WORDS_PER_LINE > 1) begin
        assign per_way_rdata = line_rdata[wsel];
    end else begin
        `UNUSED_VAR (wsel)
        assign per_way_rdata = line_rdata;
    end
    assign read_data = per_way_rdata[way_idx];

`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s fill: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, fill_data));
        end
        if (flush && ~stall) begin
            `TRACE(3, ("%d: %s flush: addr=0x%0h, way=%b, blk_addr=%0d, byteen=%h, data=0x%h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, dirty_byteen, dirty_data));
        end
        if (read && ~stall) begin
            `TRACE(3, ("%d: %s read: addr=0x%0h, way=%b, blk_addr=%0d, wsel=%0d, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, wsel, read_data, req_uuid));
        end
        if (write && ~stall) begin
            `TRACE(3, ("%d: %s write: addr=0x%0h, way=%b, blk_addr=%0d, wsel=%0d, byteen=%h, data=0x%h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, wsel, write_byteen[wsel], write_data[wsel], req_uuid));
        end
    end
`endif

endmodule
