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

// Arbitrates a flush request (from dcr_flush_if) with a real cache request
// (core_bus_if) into a single merged output (cache_bus_if). Usable for both
// the dcache and the icache invalidation.
//
// The interface is one-shot per dcr_flush_if.req assertion: once the flush
// has completed, `done` is held high and we refuse to re-fire the synthetic
// request until the initiator drops `req`. Without this latch, a shared req
// across multiple flush instances (e.g. one per cache) would cause the first
// instance to keep re-flushing while waiting on the slower instance's done.

module VX_dcr_flush import VX_gpu_pkg::*; #(
    parameter WORD_SIZE = 4,
    parameter TAG_WIDTH = 1    // pre-arb (core-side) tag width
) (
    input wire clk,
    input wire reset,

    VX_dcr_flush_if.slave  dcr_flush_if,
    VX_mem_bus_if.slave    core_bus_if,   // TAG_WIDTH
    VX_mem_bus_if.master   cache_bus_if   // TAG_WIDTH+1
);
    // Synthetic flush request bus
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) flush_bus_if();

    // Prevent re-injection while a flush request is already in flight, AND
    // hold "done" stably high after the first completion until req drops.
    reg flush_inflight_r;
    reg flush_done_r;
    wire flush_req_fire = flush_bus_if.req_valid && flush_bus_if.req_ready;
    always @(posedge clk) begin
        if (reset) begin
            flush_inflight_r <= 1'b0;
            flush_done_r     <= 1'b0;
        end else if (!dcr_flush_if.req) begin
            // Initiator released the request — re-arm for the next cycle.
            flush_inflight_r <= 1'b0;
            flush_done_r     <= 1'b0;
        end else begin
            if (flush_req_fire) begin
                flush_inflight_r <= 1'b1;
            end else if (flush_bus_if.rsp_valid) begin
                flush_inflight_r <= 1'b0;
            end
            if (flush_bus_if.rsp_valid) begin
                flush_done_r <= 1'b1;
            end
        end
    end

    assign flush_bus_if.req_valid = dcr_flush_if.req && !flush_inflight_r && !flush_done_r;
    assign flush_bus_if.req_data  = '{
        rw:      1'b0,
        addr:    '0,
        data:    '0,
        byteen:  '0,
        attr:    MEM_ATTR_WIDTH'(1 << MEM_ATTR_FLUSH_OFFS),
        tag:     '0
    };
    // The flush request only consumes MEM_ATTR_FLUSH_OFFS downstream; the
    // remaining req_data fields are tied off and intentionally unused.
    `UNUSED_VAR (flush_bus_if.req_data.rw)
    `UNUSED_VAR (flush_bus_if.req_data.addr)
    `UNUSED_VAR (flush_bus_if.req_data.data)
    `UNUSED_VAR (flush_bus_if.req_data.byteen)
    `UNUSED_VAR (flush_bus_if.req_data.tag)
    // Level-held done so a shared dcr_flush_if can AND multiple instances.
    assign dcr_flush_if.done      = flush_done_r;
    assign flush_bus_if.rsp_ready = 1'b1;

    // 2-to-1 arb: input 0 = flush, input 1 = real dcache port 0
    VX_mem_bus_if #(
        .DATA_SIZE (WORD_SIZE),
        .TAG_WIDTH (TAG_WIDTH)
    ) dcache_arb_in_if[2]();

    // Core traffic on the priority input so it wins arbitration as long as
    // the LSU has any in-flight stores to push downstream. The flush waits
    // until upstream is quiescent — the host issues cache_flush only after
    // `busy` drops to 0, so once the LSU adapter buffers drain, the core
    // input goes idle and the flush wins. STICKY=1 hardens against any
    // 1-cycle gap in core valid during the drain phase.
    `ASSIGN_VX_MEM_BUS_IF (dcache_arb_in_if[0], core_bus_if);
    `ASSIGN_VX_MEM_BUS_IF (dcache_arb_in_if[1], flush_bus_if);

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
        .ARBITER     ("P"),
        .STICKY      (1)
    ) dcache_flush_arb (
        .clk        (clk),
        .reset      (reset),
        .bus_in_if  (dcache_arb_in_if),
        .bus_out_if (dcache_arb_out_if)
    );

    `ASSIGN_VX_MEM_BUS_IF (cache_bus_if, dcache_arb_out_if[0]);

endmodule
