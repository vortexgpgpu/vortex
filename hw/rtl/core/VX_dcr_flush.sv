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

// Arbitrates a flush request (from dcr_flush_if) with a real dcache request
// (core_bus_if) into a single merged output (dcache_bus_if).
// The output tag is TAG_WIDTH+1 wide (VX_mem_arb adds a 1-bit sel field at
// bit 0, i.e. TAG_SEL_IDX=0).

module VX_dcr_flush import VX_gpu_pkg::*; #(
    parameter WORD_SIZE = 4,
    parameter TAG_WIDTH = 1    // pre-arb (core-side) tag width
) (
    input wire clk,
    input wire reset,

    VX_dcr_flush_if.slave  dcr_flush_if,
    VX_mem_bus_if.slave    core_bus_if,   // TAG_WIDTH
    VX_mem_bus_if.master   dcache_bus_if  // TAG_WIDTH+1
);
    // Synthetic flush request bus
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) flush_bus_if();

    // Prevent re-injection while a flush request is already in flight
    reg flush_inflight_r;
    wire flush_req_fire = flush_bus_if.req_valid && flush_bus_if.req_ready;
    always @(posedge clk) begin
        if (reset) begin
            flush_inflight_r <= 1'b0;
        end else if (flush_req_fire) begin
            flush_inflight_r <= 1'b1;
        end else if (flush_bus_if.rsp_valid) begin
            flush_inflight_r <= 1'b0;
        end
    end

    assign flush_bus_if.req_valid = dcr_flush_if.req && !flush_inflight_r;
    assign flush_bus_if.req_data  = '{
        rw:     1'b0,
        addr:   '0,
        data:   '0,
        byteen: '0,
        flags:  MEM_FLAGS_WIDTH'(1 << MEM_REQ_FLAG_FLUSH),
        tag:    '0
    `ifdef EXT_A_ENABLE
        ,
        amo:    amo_req_t'('0)
    `endif
    };
    assign dcr_flush_if.done      = flush_bus_if.rsp_valid;
    assign flush_bus_if.rsp_ready = 1'b1;

    // 2-to-1 arb: input 0 = flush, input 1 = real dcache port 0
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) dcache_arb_in_if[2]();

    `ASSIGN_VX_MEM_BUS_IF (dcache_arb_in_if[0], flush_bus_if);
    `ASSIGN_VX_MEM_BUS_IF (dcache_arb_in_if[1], core_bus_if);

    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH + 1)
    ) dcache_arb_out_if[1]();

    VX_mem_arb #(
        .NUM_INPUTS  (2),
        .NUM_OUTPUTS (1),
        .DATA_SIZE   (WORD_SIZE),
        .TAG_WIDTH   (TAG_WIDTH),
        .TAG_SEL_IDX (0),
        .ARBITER     ("P")
    ) dcache_flush_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcache_arb_in_if),
        .bus_out_if (dcache_arb_out_if)
    );

    `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if, dcache_arb_out_if[0]);

endmodule
