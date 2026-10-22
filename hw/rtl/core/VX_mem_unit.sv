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

module VX_mem_unit import VX_gpu_pkg::*; #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    output lmem_perf_t      lmem_perf,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
    VX_txbar_bus_if.master  dxa_txbar_bus_if,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // TCU LMEM read port
    VX_mem_bus_if.slave     tcu_lmem_if,
`endif

    // the LSU blocks' memory paths
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if.slave     lmem_bus_if [LSU_NUM_REQS],
`endif
    VX_mem_bus_if.slave     lsu_dcache_bus_if [DCACHE_NUM_REQS],

    VX_dcr_flush_if.slave   dcr_flush_if,
    input wire              flush_l1_only,
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
);
`ifndef VX_CFG_LMEM_ENABLE
    `UNUSED_SPARAM (INSTANCE_ID)
`endif
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("TCU_WGMMA_ENABLE requires LMEM_ENABLE"))
    // The WGMMA shared-memory descriptor carries the LMEM byte offset in a 16-bit
    // field, so a WGMMA-visible LMEM cannot exceed 64KB; beyond that the offset
    // wraps into the adjacent stride field.
    `STATIC_ASSERT(`VX_CFG_LMEM_LOG_SIZE <= 16, ("LMEM exceeds the 16-bit smem descriptor offset field"))
`endif

`ifdef VX_CFG_LMEM_ENABLE

    `STATIC_ASSERT(`IS_DIVISBLE((1 << `VX_CFG_LMEM_LOG_SIZE), `VX_CFG_MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`VX_MEM_LMEM_BASE_ADDR % (1 << `VX_CFG_LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

    // DMA arbiter: merge DXA writes and/or TCU reads onto single DMA port.

    VX_mem_bus_if #(
        .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
    ) lmem_dma_if();

    localparam LMEM_DMA_IN_TAG_W = LMEM_DMA_IN_TAG_MAX;

    if (LMEM_DMA_INPUTS > 0) begin : g_lmem_dma

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_in_if[LMEM_DMA_INPUTS]();

        VX_mem_bus_if #(
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
        ) dma_arb_out_if[1]();

        // Wire DXA and/or TCU into the arbiter input array.
    `ifdef VX_CFG_EXT_DXA_ENABLE
        // DXA issues LMEM-relative word addresses directly (issuer_lmem_base + intra,
        // with multicast replay adding r×smem_stride per beat), so no per-slot
        // address translation is needed here.
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_DXA_IDX], dxa_lmem_bus_if, LMEM_DMA_IN_TAG_W, DXA_LMEM_OUT_TAG_W, UUID_WIDTH);
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_TCU_IDX], tcu_lmem_if, LMEM_DMA_IN_TAG_W, TCU_LMEM_TAG_W, UUID_WIDTH);
    `endif

        // Round-robin, so a DXA write stream cannot starve the TCU's operand reads.
        VX_mem_bus_arb #(
            .NUM_INPUTS  (LMEM_DMA_INPUTS),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH),
            .ARBITER     ("R")
        ) lmem_dma_arb (
            .clk         (clk),
            .reset       (reset),
            .bus_in_if   (dma_arb_in_if),
            .bus_out_if  (dma_arb_out_if)
        );

        `ASSIGN_VX_MEM_BUS_IF (lmem_dma_if, dma_arb_out_if[0]);

    `ifdef VX_CFG_EXT_DXA_ENABLE
        wire [`VX_CFG_LMEM_NUM_BANKS-1:0] dxa_bank_wr_fire;
        for (genvar i = 0; i < `VX_CFG_LMEM_NUM_BANKS; ++i) begin : g_dxa_bank_wr_fire
            assign dxa_bank_wr_fire[i] = lmem_dma_if.req_valid
                                      && lmem_dma_if.req_ready
                                      && lmem_dma_if.req_data.rw
                                      && (|lmem_dma_if.req_data.byteen[i*LSU_WORD_SIZE +: LSU_WORD_SIZE]);
        end

        VX_dxa_completion #(
            .INSTANCE_ID (`SFORMATF(("%s-lmem-dma-compl_det", INSTANCE_ID))),
            .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
            .ATTR_WIDTH  (DXA_LMEM_ATTR_W)
        ) dxa_completion_detect (
            .clk           (clk),
            .reset         (reset),
            .bank_wr_fire  (dxa_bank_wr_fire),
            .bank_wr_attr  (DXA_LMEM_ATTR_W'(lmem_dma_if.req_data.attr)),
            .txbar_bus_if  (dxa_txbar_bus_if)
        );

        `ifdef DBG_TRACE_DXA
        always @(posedge clk) begin
            if (dxa_txbar_bus_if.valid && !reset) begin
                `TRACE(2, ("%t: %s-lmem-dma: dxa_txbar valid=1, bar_addr=0x%0h, ready=%b\n",
                    $time, INSTANCE_ID, dxa_txbar_bus_if.data.addr, dxa_txbar_bus_if.ready))
            end
        end
        `endif

    `endif

    end else begin : g_no_lmem_dma

        assign lmem_dma_if.req_valid = 1'b0;
        assign lmem_dma_if.req_data  = '0;
        `UNUSED_VAR (lmem_dma_if.req_ready)

        `UNUSED_VAR (lmem_dma_if.rsp_valid)
        `UNUSED_VAR (lmem_dma_if.rsp_data)
        assign lmem_dma_if.rsp_ready = 1'b0;

    end

    VX_local_mem #(
        .INSTANCE_ID (`SFORMATF(("%s-lmem", INSTANCE_ID))),
        .SIZE        (1 << `VX_CFG_LMEM_LOG_SIZE),
        .NUM_REQS    (LSU_NUM_REQS),
        .NUM_BANKS   (`VX_CFG_LMEM_NUM_BANKS),
        .WORD_SIZE   (LSU_WORD_SIZE),
        .ADDR_WIDTH  (LMEM_ADDR_WIDTH),
        .TAG_WIDTH   (LSU_TAG_WIDTH),
        .DMA_ENABLE  (LMEM_DMA_EN),
        .DMA_TAG_WIDTH (LMEM_DMA_TAG_WIDTH),
        .AMO_ENABLE  (`VX_CFG_EXT_A_ENABLED),
        .OUT_BUF     (3)
    ) local_mem (
        .clk         (clk),
        .reset       (reset),
    `ifdef PERF_ENABLE
        .lmem_perf   (lmem_perf),
    `endif
        .dma_bus_if  (lmem_dma_if),
        .lsu_bus_if  (lmem_bus_if)
    );

`else

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif

`endif

    // A private dcache takes the lanes straight from the adapter, whose drain
    // store_drained already waits for, so one token suffices. Otherwise (a
    // shared dcache, or none) the lanes are buffered independently before the
    // next flush point, so a store on one lane can still be queued when another
    // lane's token arrives: every lane then carries its own token behind its
    // stores, and the flush is done when every lane's is.
    localparam DCACHE_PRIVATE = (`VX_CFG_SOCKET_SIZE == `VX_CFG_NUM_DCACHES);
    localparam FLUSH_LANES = DCACHE_PRIVATE ? 1 : DCACHE_NUM_REQS;
    VX_dcr_flush_if lane_flush_if [FLUSH_LANES] ();
    wire [FLUSH_LANES-1:0] lane_flush_done;
    for (genvar k = 0; k < FLUSH_LANES; ++k) begin : g_lane_flush
        assign lane_flush_if[k].req = dcr_flush_if.req;
        assign lane_flush_done[k] = lane_flush_if[k].done;
    end
    assign dcr_flush_if.done = (& lane_flush_done);

    // A block's D-cache request carries a clear tag LSB, the flush arbiter's
    // select bit; the block takes the cache response directly and drops the
    // flush token's.
    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin : g_dcache_bus_if
        if (i < FLUSH_LANES) begin : g_flush_lane
            VX_mem_bus_if #(
                .DATA_SIZE (DCACHE_WORD_SIZE),
                .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
            ) flush_core_if();

            assign flush_core_if.req_valid       = lsu_dcache_bus_if[i].req_valid;
            assign flush_core_if.req_data.rw     = lsu_dcache_bus_if[i].req_data.rw;
            assign flush_core_if.req_data.addr   = lsu_dcache_bus_if[i].req_data.addr;
            assign flush_core_if.req_data.data   = lsu_dcache_bus_if[i].req_data.data;
            assign flush_core_if.req_data.byteen = lsu_dcache_bus_if[i].req_data.byteen;
            assign flush_core_if.req_data.attr   = lsu_dcache_bus_if[i].req_data.attr;
            assign flush_core_if.req_data.tag    = lsu_dcache_bus_if[i].req_data.tag[DCACHE_TAG_WIDTH-1:1];
            assign lsu_dcache_bus_if[i].req_ready = flush_core_if.req_ready;
            `UNUSED_VAR (lsu_dcache_bus_if[i].req_data.tag[0])

            assign lsu_dcache_bus_if[i].rsp_valid = dcache_bus_if[i].rsp_valid;
            assign lsu_dcache_bus_if[i].rsp_data  = dcache_bus_if[i].rsp_data;
            assign flush_core_if.rsp_ready        = lsu_dcache_bus_if[i].rsp_ready;
            `UNUSED_VAR (flush_core_if.rsp_valid)
            `UNUSED_VAR (flush_core_if.rsp_data)

            VX_dcr_flush #(
                .WORD_SIZE (DCACHE_WORD_SIZE),
                .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH),
                .REQ_OUT_BUF (3) // register core dcache-request boundary; rsp already registered by L1 CORE_OUT_BUF
            ) dcr_flush (
                .clk          (clk),
                .reset        (reset),
                .dcr_flush_if (lane_flush_if[i]),
                .l1_only      (flush_l1_only),
                .core_bus_if  (flush_core_if),
                .cache_bus_if (dcache_bus_if[i])
            );
        end else begin : g_passthru_lane
            // the cache sees the select bit as a clear tag MSB
            assign dcache_bus_if[i].req_valid       = lsu_dcache_bus_if[i].req_valid;
            assign dcache_bus_if[i].req_data.rw     = lsu_dcache_bus_if[i].req_data.rw;
            assign dcache_bus_if[i].req_data.addr   = lsu_dcache_bus_if[i].req_data.addr;
            assign dcache_bus_if[i].req_data.data   = lsu_dcache_bus_if[i].req_data.data;
            assign dcache_bus_if[i].req_data.byteen = lsu_dcache_bus_if[i].req_data.byteen;
            assign dcache_bus_if[i].req_data.attr   = lsu_dcache_bus_if[i].req_data.attr;
            assign dcache_bus_if[i].req_data.tag    = {1'b0, lsu_dcache_bus_if[i].req_data.tag[DCACHE_TAG_WIDTH-1:1]};
            assign lsu_dcache_bus_if[i].req_ready   = dcache_bus_if[i].req_ready;
            `UNUSED_VAR (lsu_dcache_bus_if[i].req_data.tag[0])

            assign lsu_dcache_bus_if[i].rsp_valid     = dcache_bus_if[i].rsp_valid;
            assign lsu_dcache_bus_if[i].rsp_data.data = dcache_bus_if[i].rsp_data.data;
            assign lsu_dcache_bus_if[i].rsp_data.tag  = {dcache_bus_if[i].rsp_data.tag[DCACHE_TAG_WIDTH-2:0], 1'b0};
            assign dcache_bus_if[i].rsp_ready         = lsu_dcache_bus_if[i].rsp_ready;
            `UNUSED_VAR (dcache_bus_if[i].rsp_data.tag[DCACHE_TAG_WIDTH-1])
        end
    end

endmodule
