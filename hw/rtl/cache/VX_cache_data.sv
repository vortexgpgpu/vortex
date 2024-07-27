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
    // Request debug identifier
    parameter UUID_WIDTH        = 0
) (
    input wire                          clk,
    input wire                          reset,

`IGNORE_UNUSED_BEGIN
    input wire[`UP(UUID_WIDTH)-1:0]     req_uuid,
`IGNORE_UNUSED_END

    input wire                          stall,

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
    output wire                         dirty_valid,
    output wire [`CS_LINE_WIDTH-1:0]    dirty_data,
    output wire [LINE_SIZE-1:0]         dirty_byteen
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_PARAM (BANK_ID)
    `UNUSED_PARAM (WORD_SIZE)
    `UNUSED_VAR (reset)
    `UNUSED_VAR (stall)
    `UNUSED_VAR (line_addr)
    `UNUSED_VAR (read)
    `UNUSED_VAR (flush)

    localparam BYTEENW = (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) ? (LINE_SIZE * NUM_WAYS) : 1;

    if (WRITEBACK) begin
        reg [`CS_LINES_PER_BANK * NUM_WAYS-1:0][LINE_SIZE-1:0] dirty_bytes_r;
        reg [`CS_LINES_PER_BANK * NUM_WAYS-1:0] dirty_blocks_r;

        wire [`CLOG2(`CS_LINES_PER_BANK * NUM_WAYS)-1:0] way_addr;
        if (NUM_WAYS > 1) begin
            assign way_addr = {line_sel, way_idx};
        end else begin
            assign way_addr = line_sel;
        end

        always @(posedge clk) begin
            if (fill) begin
                dirty_bytes_r[way_addr] <= '0;
            end else if (write) begin
                dirty_bytes_r[way_addr] <= dirty_bytes_r[way_addr] | write_byteen;
            end
        end

        always @(posedge clk) begin
            if (reset) begin
                for (integer i = 0; i < `CS_LINES_PER_BANK * NUM_WAYS; ++i) begin
                    dirty_blocks_r[i] <= 0;
                end
            end else begin
                if (fill) begin
                    dirty_blocks_r[way_addr] <= 0;
                end else if (write) begin
                    dirty_blocks_r[way_addr] <= 1;
                end
            end
        end

        assign dirty_byteen = dirty_bytes_r[way_addr];
        assign dirty_valid  = dirty_blocks_r[way_addr];
    end else begin
        assign dirty_byteen = '0;
        assign dirty_valid  = 0;
    end

    // order the data layout to perform ways multiplexing last.
    // this allows converting way index to binary in parallel with BRAM read.

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] wdata;
    wire [BYTEENW-1:0] wren;

    if (WRITE_ENABLE != 0 || (NUM_WAYS > 1)) begin
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
            assign wdata[i] = (fill || !WRITE_ENABLE) ? {NUM_WAYS{fill_data[i]}} : {NUM_WAYS{write_data[i]}};
        end

        wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][WORD_SIZE-1:0] wren_w;
        for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
            for (genvar j = 0; j < NUM_WAYS; ++j) begin
                assign wren_w[i][j] = ((fill || !WRITE_ENABLE) ? {WORD_SIZE{1'b1}} : write_byteen[i])
                                    & {WORD_SIZE{(way_sel[j] || (NUM_WAYS == 1))}};
            end
        end
        assign wren = wren_w;
    end else begin
        `UNUSED_VAR (write)
        `UNUSED_VAR (write_byteen)
        `UNUSED_VAR (write_data)
        assign wdata = fill_data;
        assign wren  = fill;
    end

    wire [`LOG2UP(NUM_WAYS)-1:0] way_idx;

    VX_onehot_encoder #(
        .N (NUM_WAYS)
    ) way_enc (
        .data_in  (way_sel),
        .data_out (way_idx),
        `UNUSED_PIN (valid_out)
    );

    wire [`CS_WORDS_PER_LINE-1:0][NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] rdata;

    wire [`CS_LINE_SEL_BITS-1:0] line_sel = line_addr[`CS_LINE_SEL_BITS-1:0];

    VX_sp_ram #(
        .DATAW (`CS_LINE_WIDTH * NUM_WAYS),
        .SIZE  (`CS_LINES_PER_BANK),
        .WRENW (BYTEENW),
        .NO_RWCHECK (1)
    ) data_store (
        .clk   (clk),
        .read  (1'b1),
        .write (write || fill),
        .wren  (wren),
        .addr  (line_sel),
        .wdata (wdata),
        .rdata (rdata)
    );

    wire [NUM_WAYS-1:0][`CS_WORD_WIDTH-1:0] per_way_rdata;
    if (`CS_WORDS_PER_LINE > 1) begin
        assign per_way_rdata = rdata[wsel];
    end else begin
        `UNUSED_VAR (wsel)
        assign per_way_rdata = rdata;
    end
    assign read_data = per_way_rdata[way_idx];

    wire [NUM_WAYS-1:0][`CS_WORDS_PER_LINE-1:0][`CS_WORD_WIDTH-1:0] dirty_data_w;
    for (genvar i = 0; i < `CS_WORDS_PER_LINE; ++i) begin
        for (genvar j = 0; j < NUM_WAYS; ++j) begin
            assign dirty_data_w[j][i] = rdata[i][j];
        end
    end
    assign dirty_data = dirty_data_w[way_idx];

`ifdef DBG_TRACE_CACHE
    always @(posedge clk) begin
        if (fill && ~stall) begin
            `TRACE(3, ("%d: %s fill: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, fill_data));
        end
        if (flush && ~stall) begin
            `TRACE(3, ("%d: %s flush: addr=0x%0h, way=%b, blk_addr=%0d, dirty=%b, byteen=%b\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, dirty_valid, dirty_byteen));
        end
        if (read && ~stall) begin
            `TRACE(3, ("%d: %s read: addr=0x%0h, way=%b, blk_addr=%0d, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, read_data, req_uuid));
        end
        if (write && ~stall) begin
            `TRACE(3, ("%d: %s write: addr=0x%0h, way=%b, blk_addr=%0d, byteen=%b, data=0x%0h (#%0d)\n", $time, INSTANCE_ID, `CS_LINE_TO_FULL_ADDR(line_addr, BANK_ID), way_sel, line_sel, write_byteen, write_data, req_uuid));
        end
    end
`endif

endmodule
