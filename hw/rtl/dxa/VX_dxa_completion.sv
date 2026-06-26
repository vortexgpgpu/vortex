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
    input  wire [ATTR_WIDTH-1:0]               bank_wr_attr,  // completion info packed into the bus attr field

    VX_txbar_bus_if.master                     txbar_bus_if,
    VX_mem_bus_if.master                       softbar_lmem_if
);
    `UNUSED_SPARAM (INSTANCE_ID)

    wire is_last = bank_wr_attr[DXA_LMEM_ATTR_LAST_OFF];
    wire any_dxa_wr = |bank_wr_fire;

    wire done_fire = any_dxa_wr && is_last;
    wire [DXA_BAR_RAW_W-1:0] done_fire_raw =
        bank_wr_attr[DXA_LMEM_ATTR_BAR_OFF +: DXA_BAR_RAW_W];
    wire done_fire_soft = done_fire_raw[DXA_SOFT_BAR_BIT_IDX];

    wire [BAR_ADDR_W-1:0] done_fire_bar;
    if (`VX_CFG_NUM_WARPS > 1) begin : g_done_bar_w
        assign done_fire_bar = {done_fire_raw[0 +: NW_BITS],
                                done_fire_raw[BAR_ID_SHIFT +: NB_BITS]};
    end else begin : g_done_bar_wo
        assign done_fire_bar = done_fire_raw[BAR_ID_SHIFT +: NB_BITS];
    end

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
    wire fifo_push = done_fire && ~done_fire_soft;
    wire fifo_pop  = txbar_bus_if.valid && txbar_bus_if.ready;

    // DEPTH must be a power of 2 and >= 2 (the queue's ALM_FULL defaults to
    // DEPTH-1, which fails its `ALM_FULL > 0` static assert at DEPTH=1).
    // Bump to 2 when NUM_WARPS=1 — the depth doesn't affect correctness, just
    // backpressure granularity.
    localparam FIFO_DEPTH = (`VX_CFG_NUM_WARPS < 2) ? 2 : `VX_CFG_NUM_WARPS;
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

    wire [DXA_SOFT_BAR_OFFSET_W-1:0] soft_fifo_dout;
    wire soft_fifo_empty;
    wire soft_fifo_full;
    wire soft_fifo_push = done_fire && done_fire_soft;
    wire soft_fifo_pop  = softbar_lmem_if.req_valid && softbar_lmem_if.req_ready;

    VX_fifo_queue #(
        .DATAW  (DXA_SOFT_BAR_OFFSET_W),
        .DEPTH  (FIFO_DEPTH),
        .LUTRAM (1)
    ) soft_compl_fifo (
        .clk        (clk),
        .reset      (reset),
        .push       (soft_fifo_push),
        .pop        (soft_fifo_pop),
        .data_in    (done_fire_raw[0 +: DXA_SOFT_BAR_OFFSET_W]),
        .data_out   (soft_fifo_dout),
        .empty      (soft_fifo_empty),
        .full       (soft_fifo_full),
        `UNUSED_PIN (alm_empty),
        `UNUSED_PIN (alm_full),
        `UNUSED_PIN (size)
    );

    localparam SOFT_WORD_OFF_BITS = `CLOG2(LSU_WORD_SIZE);
    localparam SOFT_REQ_ADDR_W = `VX_CFG_MEM_ADDR_WIDTH - SOFT_WORD_OFF_BITS;
    wire [SOFT_WORD_OFF_BITS-1:0] soft_byte_off =
        soft_fifo_dout[0 +: SOFT_WORD_OFF_BITS];
    wire [SOFT_REQ_ADDR_W-1:0] soft_word_addr =
        SOFT_REQ_ADDR_W'(soft_fifo_dout >> SOFT_WORD_OFF_BITS);
    wire [LSU_WORD_SIZE-1:0] softbar_byteen =
        LSU_WORD_SIZE'(4'hf) << soft_byte_off;
    wire [LSU_WORD_SIZE*8-1:0] softbar_data =
        (LSU_WORD_SIZE*8)'(32'hffff_ffff) << ({soft_byte_off, 3'b000});
    wire [MEM_ATTR_WIDTH-1:0] softbar_attr =
        MEM_ATTR_WIDTH'({{HART_ID_WIDTH{1'b1}}, 1'b0, AMO_OP_ADD, 1'b1}) << MEM_ATTR_AMO_OFFS;

    assign softbar_lmem_if.req_valid       = ~soft_fifo_empty;
    assign softbar_lmem_if.req_data.rw     = 1'b1;
    assign softbar_lmem_if.req_data.addr   = soft_word_addr;
    assign softbar_lmem_if.req_data.data   = softbar_data;
    assign softbar_lmem_if.req_data.byteen = softbar_byteen;
    assign softbar_lmem_if.req_data.attr   = softbar_attr;
    assign softbar_lmem_if.req_data.tag    = '0;
    assign softbar_lmem_if.rsp_ready       = 1'b1;

    `RUNTIME_ASSERT(~(fifo_push && fifo_full),
        ("%t: %s overflow — multicast release FIFO full, event for bar=0x%0h would be dropped",
         $time, INSTANCE_ID, done_fire_bar))
    `RUNTIME_ASSERT(~(soft_fifo_push && soft_fifo_full),
        ("%t: %s overflow — soft-barrier completion FIFO full, event for raw=0x%0h would be dropped",
         $time, INSTANCE_ID, done_fire_raw))

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
        if (soft_fifo_pop && !reset) begin
            `TRACE(2, ("%t: %s: SOFT_ACCEPTED offset=0x%0h\n",
                $time, INSTANCE_ID, soft_fifo_dout))
        end
        if (~reset && fifo_pop) begin
            $write("DXA_TL,%0d,DONE_DETECT,%s,bar=%0d\n",
                $time, INSTANCE_ID, txbar_bus_if.data.addr);
        end
    end
`endif
endmodule
