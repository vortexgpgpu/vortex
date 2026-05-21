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

module VX_dxa_completion import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    parameter NUM_BANKS = 1,
    parameter ATTR_WIDTH = DXA_LMEM_ATTR_W
) (
    input  wire                                clk,
    input  wire                                reset,

    // DXA bank writes (from dedicated DMA port, all are DXA by definition)
    input  wire [NUM_BANKS-1:0]                bank_wr_fire,
    input  wire [ATTR_WIDTH-1:0]               bank_wr_attr,  // completion info: {last_pkt, bar_addr}, packed into the bus attr field

    VX_txbar_bus_if.master                     txbar_bus_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire is_last = bank_wr_attr[ATTR_WIDTH-1];
    wire any_dxa_wr = |bank_wr_fire;

    wire done_fire = any_dxa_wr && is_last;
    wire [BAR_ADDR_W-1:0] done_fire_bar = bank_wr_attr[BAR_ADDR_W-1:0];

    // ════════════════════════════════════════════════════════════════════
    // Release-event FIFO.
    //
    // Multicast can fire popcount(cta_mask) back-to-back release events on
    // the LAST drain word — one per receiver. A single pending slot would
    // drop events if downstream (wctl_unit) is slower than the burst rate.
    //
    // Depth = NUM_WARPS handles the worst-case intra-core multicast burst
    // (NUM_WARPS receivers in NUM_WARPS consecutive cycles). The DXA worker
    // never fires releases faster than 1/cycle, so a per-cycle drain rate
    // ≥ 1/2 keeps the FIFO bounded. An assertion catches overflow.
    // ════════════════════════════════════════════════════════════════════
    wire [BAR_ADDR_W-1:0] fifo_dout;
    wire fifo_empty;
    wire fifo_full;
    wire fifo_push = done_fire;
    wire fifo_pop  = txbar_bus_if.valid && txbar_bus_if.ready;

    // DEPTH must be a power of 2 and >= 2 (the queue's ALM_FULL defaults to
    // DEPTH-1, which fails its `ALM_FULL > 0` static assert at DEPTH=1).
    // Bump to 2 when NUM_WARPS=1 — the depth doesn't affect correctness, just
    // backpressure granularity.
    localparam FIFO_DEPTH = (NUM_WARPS < 2) ? 2 : NUM_WARPS;
    VX_fifo_queue #(
        .DATAW  (BAR_ADDR_W),
        .DEPTH  (FIFO_DEPTH),
        .LUTRAM (1)
    ) compl_fifo (
        .clk        (clk),
        .reset      (reset),
        .push       (fifo_push),
        .pop        (fifo_pop),
        .data_in    (done_fire_bar),
        .data_out   (fifo_dout),
        .empty      (fifo_empty),
        .full       (fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    assign txbar_bus_if.valid        = ~fifo_empty;
    assign txbar_bus_if.data.addr    = fifo_dout;
    assign txbar_bus_if.data.is_done = 1'b1;

    `RUNTIME_ASSERT(~(fifo_push && fifo_full),
        ("%t: %s overflow — multicast release FIFO full, event for bar=0x%0h would be dropped",
         $time, INSTANCE_ID, done_fire_bar))

`ifdef DBG_TRACE_DXA
    always @(posedge clk) begin
        if (any_dxa_wr && !reset) begin
            `TRACE(2, ("%t: %s: bank_wr_fire=%b, attr=0x%0h, is_last=%b, done_fire=%b, fifo_empty=%b, valid=%b, ready=%b\n",
                $time, INSTANCE_ID, bank_wr_fire, bank_wr_attr, is_last, done_fire,
                fifo_empty, txbar_bus_if.valid, txbar_bus_if.ready))
        end
        if (fifo_pop && !reset) begin
            `TRACE(2, ("%t: %s: ACCEPTED bar_addr=0x%0h\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr))
        end
        if (~reset && fifo_pop) begin
            $write("DXA_TL,%0d,DONE_DETECT,%s,bar=%0d\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr);
        end
    end
`endif
endmodule
