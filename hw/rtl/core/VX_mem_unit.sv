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
    output coalescer_perf_t coalescer_perf,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
    VX_txbar_bus_if.master  dxa_txbar_bus_if,
    // CTA slot-table view (driven by the local dispatcher).
    VX_cta_table_if.slave   cta_table_if,
`endif

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // TCU LMEM read port
    VX_mem_bus_if.slave     tcu_lmem_if,
`endif

    VX_lsu_mem_if.slave     lsu_mem_if [`VX_CFG_NUM_LSU_BLOCKS],
    VX_dcr_flush_if.slave   dcr_flush_if,
    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS]
);
    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_dcache_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    `STATIC_ASSERT(`VX_CFG_LMEM_ENABLED, ("TCU_WGMMA_ENABLE requires LMEM_ENABLE"))
`endif

`ifdef VX_CFG_LMEM_ENABLE

    `STATIC_ASSERT(`IS_DIVISBLE((1 << `VX_CFG_LMEM_LOG_SIZE), `VX_CFG_MEM_BLOCK_SIZE), ("invalid parameter"))
    `STATIC_ASSERT(0 == (`VX_MEM_LMEM_BASE_ADDR % (1 << `VX_CFG_LMEM_LOG_SIZE)), ("invalid parameter"))

    localparam LMEM_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE);

    VX_lsu_mem_if #(
        .NUM_LANES (`VX_CFG_NUM_LSU_LANES),
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_if[`VX_CFG_NUM_LSU_BLOCKS]();

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_switches
        VX_lmem_switch #(
            .GLOBAL_OUT_BUF(1),
            .LOCAL_OUT_BUF(1),
            .RSP_OUT_BUF  (1),
            .ARBITER      ("P")
        ) lmem_switch (
            .clk          (clk),
            .reset        (reset),
            .lsu_in_if    (lsu_mem_if[i]),
            .global_out_if(lsu_dcache_if[i]),
            .local_out_if (lsu_lmem_if[i])
        );
    end

    // Per-block local memory adapters to avoid deadlock when NUM_LSU_BLOCKS > 1.
    // Each block gets dedicated ports into local memory, eliminating the circular
    // dependency that arises from sharing a single adapter's unpack/pack buffers.

    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lmem_adapt_if[LSU_NUM_REQS]();

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lmem_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (LSU_WORD_SIZE),
            .TAG_WIDTH (LSU_TAG_WIDTH)
        ) lmem_block_if[`VX_CFG_NUM_LSU_LANES]();

        VX_lsu_adapter #(
            .NUM_LANES    (`VX_CFG_NUM_LSU_LANES),
            .DATA_SIZE    (LSU_WORD_SIZE),
            .TAG_WIDTH    (LSU_TAG_WIDTH),
            .TAG_SEL_BITS (LSU_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (3),
            .RSP_OUT_BUF  (0)
        ) lmem_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (lsu_lmem_if[i]),
            .mem_bus_if (lmem_block_if)
        );

        for (genvar j = 0; j < `VX_CFG_NUM_LSU_LANES; ++j) begin : g_lmem_adapt_if
            `ASSIGN_VX_MEM_BUS_IF (lmem_adapt_if[i * `VX_CFG_NUM_LSU_LANES + j], lmem_block_if[j]);
        end
    end

    // DMA arbiter: merge DXA writes and/or TCU reads onto single DMA port.

    VX_mem_bus_if #(
        .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
        .TAG_WIDTH   (LMEM_DMA_TAG_WIDTH),
        .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH)
    ) lmem_dma_if();

    localparam LMEM_DMA_IN_TAG_W = `MAX(DXA_LMEM_OUT_TAG_W, TCU_LMEM_TAG_W);

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
        // DXA multicast address resolution (receiver-side).
        //
        // The DXA SMEM bus carries:
        //   - addr  = CTA-relative intra-offset (LMEM-word units)
        //   - attr  = {is_last, bar_addr}; bar_addr[upper NW_WIDTH] = recv_wid
        //
        // Translate to an absolute LMEM word address using THIS core's
        // cta_table (slot-table state never leaves the core), with two
        // timing-conscious choices:
        //
        //   Fix A: read wid_to_lmem_base[recv_wid] — a single registered
        //          MUX, not the cascaded cta_slot_per_warp → slot_to_lmem_base
        //          chain. Critical at NUM_WARPS ≥ 16 / U55C 300 MHz where
        //          two 32:1 indexed MUXes in series + adder + arb cone
        //          would not close.
        //
        //   Fix B: register the translated request between the translator
        //          and the lmem_dma arbiter (1-cycle skid). DXA writes are
        //          bulk traffic; +1 cycle latency is negligible, but a
        //          guaranteed register boundary makes the path closure
        //          insensitive to PnR variation.
        localparam SMEM_OFF_W = `CLOG2(`VX_CFG_LMEM_NUM_BANKS * LSU_WORD_SIZE);
        wire [BAR_ADDR_W-1:0]            dxa_bar_addr = dxa_lmem_bus_if.req_data.attr[BAR_ADDR_W-1:0];
        wire [NW_WIDTH-1:0]              dxa_recv_wid = dxa_bar_addr[BAR_ADDR_W-1 -: NW_WIDTH];
        // Fix A — single registered indexed MUX:
        wire [`VX_CFG_LMEM_LOG_SIZE-1:0] dxa_recv_base_bytes = cta_table_if.wid_to_lmem_base[dxa_recv_wid];
        wire [LMEM_DMA_ADDR_WIDTH-1:0]   dxa_recv_base_words = LMEM_DMA_ADDR_WIDTH'(dxa_recv_base_bytes >> SMEM_OFF_W);
        wire [LMEM_DMA_ADDR_WIDTH-1:0]   dxa_translated_addr = dxa_recv_base_words + dxa_lmem_bus_if.req_data.addr;

        // bar_id-portion of bar_addr is meaningful to the downstream
        // completion path, but only the wid-portion is needed for address
        // translation here.
        `UNUSED_VAR (dxa_bar_addr)
        `UNUSED_VAR (cta_table_if.slot_to_lmem_base)
        `UNUSED_VAR (cta_table_if.cta_slot_per_warp)

        // ── Fix B: 1-cycle skid between translator and arbiter ──────────
        // Pack the post-translation request through a small elastic buffer
        // so the translator's combinational MUX + adder is the entire
        // src-to-flop path (no arb cone tacked on combinationally).
        localparam DXA_REQ_DATAW =
              1                                  // rw
            + LMEM_DMA_ADDR_WIDTH                // addr
            + LMEM_DMA_DATA_SIZE * 8             // data
            + LMEM_DMA_DATA_SIZE                 // byteen
            + DXA_LMEM_ATTR_W                    // attr
            + DXA_LMEM_OUT_TAG_W;                // tag (pre-widen)

        wire [DXA_REQ_DATAW-1:0] dxa_skid_in_data = {
            dxa_lmem_bus_if.req_data.rw,
            dxa_translated_addr,
            dxa_lmem_bus_if.req_data.data,
            dxa_lmem_bus_if.req_data.byteen,
            DXA_LMEM_ATTR_W'(dxa_lmem_bus_if.req_data.attr),
            DXA_LMEM_OUT_TAG_W'(dxa_lmem_bus_if.req_data.tag)
        };

        wire                                       dxa_skid_out_valid;
        wire                                       dxa_skid_out_rw;
        wire [LMEM_DMA_ADDR_WIDTH-1:0]             dxa_skid_out_addr;
        wire [LMEM_DMA_DATA_SIZE*8-1:0]            dxa_skid_out_data;
        wire [LMEM_DMA_DATA_SIZE-1:0]              dxa_skid_out_byteen;
        wire [DXA_LMEM_ATTR_W-1:0]                 dxa_skid_out_attr;
        wire [DXA_LMEM_OUT_TAG_W-1:0]              dxa_skid_out_tag;
        wire [DXA_REQ_DATAW-1:0]                   dxa_skid_out_data_flat;

        assign {
            dxa_skid_out_rw,
            dxa_skid_out_addr,
            dxa_skid_out_data,
            dxa_skid_out_byteen,
            dxa_skid_out_attr,
            dxa_skid_out_tag
        } = dxa_skid_out_data_flat;

        VX_elastic_buffer #(
            .DATAW (DXA_REQ_DATAW),
            .SIZE  (2)
        ) dxa_xlat_skid (
            .clk      (clk),
            .reset    (reset),
            .valid_in (dxa_lmem_bus_if.req_valid),
            .ready_in (dxa_lmem_bus_if.req_ready),
            .data_in  (dxa_skid_in_data),
            .valid_out(dxa_skid_out_valid),
            .ready_out(dma_arb_in_if[LMEM_DMA_DXA_IDX].req_ready),
            .data_out (dxa_skid_out_data_flat)
        );

        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_valid       = dxa_skid_out_valid;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.rw     = dxa_skid_out_rw;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.addr   = dxa_skid_out_addr;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.data   = dxa_skid_out_data;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.byteen = dxa_skid_out_byteen;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.attr   = LMEM_DMA_ATTR_W'(dxa_skid_out_attr);
        // Tag width widening (DXA → DMA) — UUID-aware, matching the
        // ASSIGN_VX_MEM_BUS_IF_EX pattern when LMEM_DMA_IN_TAG_W > DXA_LMEM_OUT_TAG_W.
        if (LMEM_DMA_IN_TAG_W > DXA_LMEM_OUT_TAG_W) begin : g_dxa_tag_widen
            assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.tag = {
                dxa_skid_out_tag[DXA_LMEM_OUT_TAG_W-1 -: UUID_WIDTH],
                {(LMEM_DMA_IN_TAG_W - DXA_LMEM_OUT_TAG_W){1'b0}},
                dxa_skid_out_tag[DXA_LMEM_OUT_TAG_W-UUID_WIDTH-1:0]
            };
        end else begin : g_dxa_tag_passthru
            assign dma_arb_in_if[LMEM_DMA_DXA_IDX].req_data.tag = dxa_skid_out_tag;
        end
        // DXA never reads SMEM (write-only path) — tie off the rsp side.
        assign dxa_lmem_bus_if.rsp_valid = 1'b0;
        assign dxa_lmem_bus_if.rsp_data  = '0;
        assign dma_arb_in_if[LMEM_DMA_DXA_IDX].rsp_ready = 1'b1;
        `UNUSED_VAR (dma_arb_in_if[LMEM_DMA_DXA_IDX].rsp_valid)
        `UNUSED_VAR (dma_arb_in_if[LMEM_DMA_DXA_IDX].rsp_data)
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        `ASSIGN_VX_MEM_BUS_IF_EX (dma_arb_in_if[LMEM_DMA_TCU_IDX], tcu_lmem_if, LMEM_DMA_IN_TAG_W, TCU_LMEM_TAG_W, UUID_WIDTH);
    `endif

        VX_mem_arb #(
            .NUM_INPUTS  (LMEM_DMA_INPUTS),
            .NUM_OUTPUTS (1),
            .DATA_SIZE   (LMEM_DMA_DATA_SIZE),
            .TAG_WIDTH   (LMEM_DMA_IN_TAG_W),
            .ATTR_WIDTH  (LMEM_DMA_ATTR_W),
            .ADDR_WIDTH  (LMEM_DMA_ADDR_WIDTH),
            .ARBITER     ("P")
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
        .OUT_BUF     (3)
    ) local_mem (
        .clk         (clk),
        .reset       (reset),
    `ifdef PERF_ENABLE
        .lmem_perf   (lmem_perf),
    `endif
        .dma_bus_if  (lmem_dma_if),
        .lsu_bus_if  (lmem_adapt_if)
    );

`else

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_lsu_dcache_if
        `ASSIGN_VX_MEM_BUS_IF (lsu_dcache_if[i], lsu_mem_if[i]);
    end

`ifdef PERF_ENABLE
    assign lmem_perf = '0;
`endif

`endif

    VX_lsu_mem_if #(
        .NUM_LANES (DCACHE_CHANNELS),
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
    ) dcache_coalesced_if[`VX_CFG_NUM_LSU_BLOCKS]();

`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] per_block_coalescer_misses;
    wire [PERF_CTR_BITS-1:0] coalescer_misses;
    VX_reduce_tree #(
        .IN_W (PERF_CTR_BITS),
        .N    (`VX_CFG_NUM_LSU_BLOCKS),
        .OP   ("+")
    ) coalescer_reduce (
        .data_in  (per_block_coalescer_misses),
        .data_out (coalescer_misses)
    );
    `BUFFER(coalescer_perf.misses, coalescer_misses);
`endif

    if ((`VX_CFG_NUM_LSU_LANES > 1) && (DCACHE_WORD_SIZE > LSU_WORD_SIZE)) begin : g_coalescing

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_coalescers

            VX_mem_coalescer #(
                .INSTANCE_ID    (`SFORMATF(("%s-coalescer%0d", INSTANCE_ID, i))),
                .NUM_REQS       (`VX_CFG_NUM_LSU_LANES),
                .DATA_IN_SIZE   (LSU_WORD_SIZE),
                .DATA_OUT_SIZE  (DCACHE_WORD_SIZE),
                .ADDR_WIDTH     (LSU_ADDR_WIDTH),
                .USER_WIDTH     (MEM_ATTR_WIDTH),
                .TAG_WIDTH      (LSU_TAG_WIDTH),
                .UUID_WIDTH     (UUID_WIDTH),
                .QUEUE_SIZE     (`VX_CFG_LSUQ_OUT_SIZE),
                .PERF_CTR_BITS  (PERF_CTR_BITS)
            ) mem_coalescer (
                .clk            (clk),
                .reset          (reset),

            `ifdef PERF_ENABLE
                .misses         (per_block_coalescer_misses[i]),
            `else
                `UNUSED_PIN (misses),
            `endif

                // Input request
                .in_req_valid   (lsu_dcache_if[i].req_valid),
                .in_req_mask    (lsu_dcache_if[i].req_data.mask),
                .in_req_rw      (lsu_dcache_if[i].req_data.rw),
                .in_req_byteen  (lsu_dcache_if[i].req_data.byteen),
                .in_req_addr    (lsu_dcache_if[i].req_data.addr),
                .in_req_user    (lsu_dcache_if[i].req_data.user),
                .in_req_no_merge(lsu_dcache_if[i].req_data.user[0][MEM_ATTR_AMO_OFFS]),
                .in_req_data    (lsu_dcache_if[i].req_data.data),
                .in_req_tag     (lsu_dcache_if[i].req_data.tag),
                .in_req_ready   (lsu_dcache_if[i].req_ready),

                // Input response
                .in_rsp_valid   (lsu_dcache_if[i].rsp_valid),
                .in_rsp_mask    (lsu_dcache_if[i].rsp_data.mask),
                .in_rsp_data    (lsu_dcache_if[i].rsp_data.data),
                .in_rsp_tag     (lsu_dcache_if[i].rsp_data.tag),
                .in_rsp_ready   (lsu_dcache_if[i].rsp_ready),

                // Output request
                .out_req_valid  (dcache_coalesced_if[i].req_valid),
                .out_req_mask   (dcache_coalesced_if[i].req_data.mask),
                .out_req_rw     (dcache_coalesced_if[i].req_data.rw),
                .out_req_byteen (dcache_coalesced_if[i].req_data.byteen),
                .out_req_addr   (dcache_coalesced_if[i].req_data.addr),
                .out_req_user   (dcache_coalesced_if[i].req_data.user),
                .out_req_data   (dcache_coalesced_if[i].req_data.data),
                .out_req_tag    (dcache_coalesced_if[i].req_data.tag),
                .out_req_ready  (dcache_coalesced_if[i].req_ready),

                // Output response
                .out_rsp_valid  (dcache_coalesced_if[i].rsp_valid),
                .out_rsp_mask   (dcache_coalesced_if[i].rsp_data.mask),
                .out_rsp_data   (dcache_coalesced_if[i].rsp_data.data),
                .out_rsp_tag    (dcache_coalesced_if[i].rsp_data.tag),
                .out_rsp_ready  (dcache_coalesced_if[i].rsp_ready)
            );
        end

    end else begin : g_no_coalescing

        for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_coalesced_if
            `ASSIGN_VX_MEM_BUS_IF (dcache_coalesced_if[i], lsu_dcache_if[i]);
        `ifdef PERF_ENABLE
            assign per_block_coalescer_misses[i] = '0;
        `endif
        end

    end

    for (genvar i = 0; i < `VX_CFG_NUM_LSU_BLOCKS; ++i) begin : g_dcache_adapters

        VX_mem_bus_if #(
            .DATA_SIZE (DCACHE_WORD_SIZE),
            .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
        ) dcache_bus_tmp_if[DCACHE_CHANNELS]();

        VX_lsu_adapter #(
            .NUM_LANES    (DCACHE_CHANNELS),
            .DATA_SIZE    (DCACHE_WORD_SIZE),
            .TAG_WIDTH    (DCACHE_CORE_TAG_WIDTH),
            .TAG_SEL_BITS (DCACHE_CORE_TAG_WIDTH - UUID_WIDTH),
            .ARBITER      ("P"),
            .REQ_OUT_BUF  (0),
            .RSP_OUT_BUF  (0)
        ) dcache_adapter (
            .clk        (clk),
            .reset      (reset),
            .lsu_mem_if (dcache_coalesced_if[i]),
            .mem_bus_if (dcache_bus_tmp_if)
        );

        for (genvar j = 0; j < DCACHE_CHANNELS; ++j) begin : g_dcache_bus_if
            if (i == 0 && j == 0) begin : g_flush_port
                // Port 0: route through VX_dcr_flush to inject flush requests
                VX_dcr_flush #(
                    .WORD_SIZE (DCACHE_WORD_SIZE),
                    .TAG_WIDTH (DCACHE_CORE_TAG_WIDTH)
                ) dcr_flush (
                    .clk          (clk),
                    .reset        (reset),
                    .dcr_flush_if (dcr_flush_if),
                    .core_bus_if  (dcache_bus_tmp_if[j]),
                    .cache_bus_if (dcache_bus_if[0])
                );
            end else begin : g_passthru_port
                // Ports 1+: pass through; tag is zero-extended from
                // DCACHE_CORE_TAG_WIDTH to DCACHE_TAG_WIDTH on request,
                // and MSB-stripped on response.
                `ASSIGN_VX_MEM_BUS_IF_EX (dcache_bus_if[i * DCACHE_CHANNELS + j], dcache_bus_tmp_if[j], DCACHE_TAG_WIDTH_BASE, DCACHE_CORE_TAG_WIDTH, 0);
            end
        end

    end

endmodule
