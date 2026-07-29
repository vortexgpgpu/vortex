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

`include "VX_define.vh"

`ifdef VX_CFG_DXA_SBAR_ENABLE

module VX_dxa_sbar_completion import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 1,
    parameter ATTR_WIDTH = DXA_LMEM_ATTR_W
) (
    input wire                         clk,
    input wire                         reset,
    input wire [NUM_BANKS-1:0]         bank_wr_fire,
    input wire [ATTR_WIDTH-1:0]        bank_wr_attr,
    VX_mem_bus_if.master               lmem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam WORD32_COUNT = LSU_WORD_SIZE / 4;
    localparam WORD32_INDEX_W = `LOG2UP(WORD32_COUNT);

    wire terminal_write =
        (|bank_wr_fire) && bank_wr_attr[ATTR_WIDTH-1];
    wire [DXA_COMPLETION_REF_W-1:0] completion_ref =
        bank_wr_attr[DXA_COMPLETION_REF_W-1:0];

    localparam FIFO_DEPTH =
        (`VX_CFG_NUM_WARPS < 2) ? 2 : `VX_CFG_NUM_WARPS;
    wire [DXA_COMPLETION_REF_W-1:0] fifo_data;
    wire fifo_empty;
    wire fifo_full;
    wire fifo_pop;

    VX_fifo_queue #(
        .DATAW  (DXA_COMPLETION_REF_W),
        .DEPTH  (FIFO_DEPTH),
        .LUTRAM (1)
    ) completion_fifo (
        .clk        (clk),
        .reset      (reset),
        .push       (terminal_write),
        .pop        (fifo_pop),
        .data_in    (completion_ref),
        .data_out   (fifo_data),
        .empty      (fifo_empty),
        .full       (fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    reg amo_inflight_r;
    wire request_valid = ~fifo_empty && ~amo_inflight_r;
    wire request_fire = request_valid && lmem_if.req_ready;
    assign fifo_pop = request_fire;

    wire [WORD32_INDEX_W-1:0] word32_index =
        WORD32_INDEX_W'(fifo_data >> 2);
    wire [LSU_WORD_SIZE*8-1:0] request_data;
    wire [LSU_WORD_SIZE-1:0] request_byteen;
    for (genvar i = 0; i < WORD32_COUNT; ++i) begin : g_request_word
        wire selected = (word32_index == WORD32_INDEX_W'(i));
        assign request_data[i * 32 +: 32] =
            selected ? 32'hffffffff : '0;
        assign request_byteen[i * 4 +: 4] =
            {4{selected}};
    end

    amo_req_t amo;
    assign amo.amo_valid = 1;
    assign amo.amo_op = AMO_OP_ADD;
    assign amo.amo_unsigned = 0;
    assign amo.hart_id = '0;

    mem_bus_attr_t attr;
    assign attr = mem_bus_attr_t'({
        amo,
        1'b1,
        1'b0,
        1'b0
    });

    assign lmem_if.req_valid = request_valid;
    assign lmem_if.req_data.rw = 1'b1;
    assign lmem_if.req_data.addr =
        $bits(lmem_if.req_data.addr)'(
            fifo_data >> `CLOG2(LSU_WORD_SIZE));
    assign lmem_if.req_data.data = request_data;
    assign lmem_if.req_data.byteen = request_byteen;
    assign lmem_if.req_data.attr = attr;
    assign lmem_if.req_data.tag = '0;
    assign lmem_if.rsp_ready = 1'b1;

    always @(posedge clk) begin
        if (reset) begin
            amo_inflight_r <= 0;
        end else begin
            if (request_fire)
                amo_inflight_r <= 1;
            if (lmem_if.rsp_valid)
                amo_inflight_r <= 0;
        end
    end

    `RUNTIME_ASSERT(!(terminal_write && fifo_full),
        ("DXA software-barrier completion FIFO overflow"))
    `RUNTIME_ASSERT(!terminal_write || (completion_ref[1:0] == 0),
        ("DXA software-barrier completion reference is misaligned"))
    `RUNTIME_ASSERT(!lmem_if.rsp_valid || amo_inflight_r,
        ("unexpected software-barrier AMO response"))

endmodule

`endif
