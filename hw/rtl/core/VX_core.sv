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

`ifdef VX_CFG_EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_core import VX_gpu_pkg::*, VX_tlb_pkg::*;
`ifdef VX_CFG_EXT_TCU_ENABLE
import VX_tcu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = ""
) (
    `SCOPE_IO_DECL

    // Clock
    input wire              clk,
    input wire              reset,

    input wire [CORE_ID_WIDTH-1:0] core_id,

`ifdef PERF_ENABLE
    input sysmem_perf_t     sysmem_perf,
`endif

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    icache_bus_if [ICACHE_NUM_REQS],

`ifdef VX_CFG_VM_ENABLE
    // Translation happens at the socket (one MMU per L1 cache): the core emits
    // virtual addresses, and busy/barrier logic waits on the socket MMU draining.
    input wire              mmu_drained,
`endif

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if.master dxa_req_bus_if,
    VX_mem_bus_if.slave     dxa_lmem_bus_if,
`endif

`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if.master    tex_bus_if,
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if.master    rtu_bus_if,
`endif

`ifdef EXT_GFX_ANY_ENABLE
    VX_dcr_flush_if.master  cluster_flush_if,
`endif

    // KMU bus
    VX_kmu_bus_if.slave     kmu_bus_if,

    // Global barrier
    VX_gbar_bus_if.master   gbar_bus_if,

    // Status: the core's own busy, and each LSU block's memory path busy
    output wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_busy,
    output wire             busy
);
    VX_warp_shared_if   warp_shared_if[`VX_CFG_NUM_SFU_BLOCKS]();

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_txbar_bus_if     dxa_txbar_bus_if();
`endif

    // the LSU blocks' memory paths
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_sched_empty;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0] lsu_store_empty;
`ifdef VX_CFG_LMEM_ENABLE
    VX_mem_bus_if #(
        .DATA_SIZE (LSU_WORD_SIZE),
        .TAG_WIDTH (LSU_TAG_WIDTH)
    ) lsu_lmem_bus_if[LSU_NUM_REQS]();
`endif
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE),
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) lsu_dcache_bus_if[DCACHE_NUM_REQS]();
`ifdef PERF_ENABLE
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rd_req_fire;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_wr_req_fire;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][`VX_CFG_NUM_LSU_LANES-1:0] lsu_perf_rsp_fire;
    wire [`VX_CFG_NUM_LSU_BLOCKS-1:0][PERF_CTR_BITS-1:0] lsu_coalescer_misses;
`endif

`ifdef TCU_META_ENABLE
    // The TCU metadata AGU is the second client of the LSU schedulers.
    VX_lsu_sched_if lsu_tcu_mem_if [`VX_CFG_NUM_LSU_BLOCKS]();
`endif

    // frontend-slice icache requests; port 0 also carries the injected flush
    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE),
        .TAG_WIDTH (ICACHE_FETCH_TAG_WIDTH)
    ) fetch_icache_if[ICACHE_NUM_REQS]();

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    localparam TCU_LMEM_BANK_ADDR_W = `VX_CFG_LMEM_LOG_SIZE - `CLOG2(LSU_WORD_SIZE) - `CLOG2(`VX_CFG_LMEM_NUM_BANKS);
    VX_mem_bus_if #(
        .DATA_SIZE  (`VX_CFG_LMEM_NUM_BANKS * LSU_WORD_SIZE),
        .TAG_WIDTH  (TCU_LMEM_TAG_W),
        .ATTR_WIDTH (LMEM_DMA_ATTR_W),
        .ADDR_WIDTH (TCU_LMEM_BANK_ADDR_W)
    ) tcu_lmem_if();
`endif

`ifdef PERF_ENABLE
    lmem_perf_t lmem_perf;
    coalescer_perf_t coalescer_perf;
    pipeline_perf_t pipeline_perf;
`ifdef VX_CFG_EXT_TCU_ENABLE
    tcu_perf_t tcu_perf;
    assign pipeline_perf.tcu = tcu_perf;
`endif
    sysmem_perf_t sysmem_perf_tmp;
    always @(*) begin
        sysmem_perf_tmp = sysmem_perf;
        sysmem_perf_tmp.lmem = lmem_perf;
        sysmem_perf_tmp.coalescer = coalescer_perf;
    end
`endif

    VX_dcr_csr_if dcr_csr_if();

    // One DCR flush request invalidates both L1s: a kernel re-loaded at the same
    // address must not execute stale icache lines.
    VX_dcr_flush_if dcr_flush_if();
    VX_dcr_flush_if dcr_flush_dcache_if();
    VX_dcr_flush_if dcr_flush_icache_if();
    wire dcr_flush_l1_only;

    // the in-core global-store path is empty
    wire mem_unit_empty = (& lsu_store_empty);
    wire mem_unit_busy;

`ifdef VX_CFG_VM_ENABLE
`ifdef PERF_ENABLE
    // the MMU perf counters are kept at the socket
    assign pipeline_perf.mmu = '0;
`endif
    // Translation drains at the socket MMU; the socket returns its empty status.
    wire mmu_empty = mmu_drained;
`else
    wire mmu_empty = 1'b1;
`endif


    // Stores are ack-less and lag warp exit, so the dcache flush waits until every
    // store of this core has reached L1.
    wire store_drained = (&lsu_sched_empty) && mem_unit_empty;
    assign dcr_flush_dcache_if.req = dcr_flush_if.req && store_drained;
    // Only the dcache flush walks the shared levels (the icache does not
    // forward its flush), so the icache flushes alongside it.
    assign dcr_flush_icache_if.req = dcr_flush_if.req;
    // Each flush holds done until its req drops, so the dones AND into the combined
    // completion (with the cluster graphics caches when present).
`ifdef EXT_GFX_ANY_ENABLE
    assign cluster_flush_if.req  = dcr_flush_if.req;
    assign dcr_flush_if.done = dcr_flush_dcache_if.done & dcr_flush_icache_if.done & cluster_flush_if.done;
`else
    assign dcr_flush_if.done = dcr_flush_dcache_if.done & dcr_flush_icache_if.done;
`endif

    VX_dcr_data #(
        .INSTANCE_ID (`SFORMATF(("%s-dcr_data", INSTANCE_ID)))
    ) dcr_data (
        .clk        (clk),
        .reset      (reset),
        .core_id    (core_id),
        .dcr_bus_if (dcr_bus_if),
        .dcr_csr_if (dcr_csr_if),
        .dcr_flush_if(dcr_flush_if),
        .flush_l1_only(dcr_flush_l1_only)
    );

    wire sched_busy;
    wire [PERF_CTR_BITS-1:0] sched_cycles;
    wire [PERF_CTR_BITS-1:0] sched_instret;

    VX_csr_mpm #(
        .INSTANCE_ID (`SFORMATF(("%s-mpm", INSTANCE_ID)))
    ) csr_mpm (
    `ifdef PERF_ENABLE
        .sysmem_perf   (sysmem_perf_tmp),
        .pipeline_perf (pipeline_perf),
    `endif
        .cycles        (sched_cycles),
        .instret       (sched_instret),
        .dcr_csr_if    (dcr_csr_if)
    );

    `SCOPE_IO_SWITCH (3);

    // Port 0 injects the cache flush into its request stream; the cache flushes
    // every bank. The merged request leaves the core registered.
    VX_dcr_flush #(
        .WORD_SIZE   (ICACHE_WORD_SIZE),
        .TAG_WIDTH   (ICACHE_FETCH_TAG_WIDTH),
        .REQ_OUT_BUF (3)
    ) icache_dcr_flush (
        .clk          (clk),
        .reset        (reset),
        .dcr_flush_if (dcr_flush_icache_if),
        .l1_only      (dcr_flush_l1_only),
        .core_bus_if  (fetch_icache_if[0]),
        .cache_bus_if (icache_bus_if[0])
    );

    // The other ports carry their frontend slice's registered requests, with the
    // flush arbiter's select bit, the tag's LSB, clear.
    for (genvar i = 1; i < ICACHE_NUM_REQS; ++i) begin : g_icache_ports
        assign icache_bus_if[i].req_valid       = fetch_icache_if[i].req_valid;
        assign icache_bus_if[i].req_data.rw     = fetch_icache_if[i].req_data.rw;
        assign icache_bus_if[i].req_data.addr   = fetch_icache_if[i].req_data.addr;
        assign icache_bus_if[i].req_data.data   = fetch_icache_if[i].req_data.data;
        assign icache_bus_if[i].req_data.byteen = fetch_icache_if[i].req_data.byteen;
        assign icache_bus_if[i].req_data.attr   = fetch_icache_if[i].req_data.attr;
        assign icache_bus_if[i].req_data.tag    = {fetch_icache_if[i].req_data.tag, 1'b0};
        assign fetch_icache_if[i].req_ready     = icache_bus_if[i].req_ready;

        assign fetch_icache_if[i].rsp_valid     = icache_bus_if[i].rsp_valid;
        assign fetch_icache_if[i].rsp_data.data = icache_bus_if[i].rsp_data.data;
        assign fetch_icache_if[i].rsp_data.tag  = icache_bus_if[i].rsp_data.tag[ICACHE_TAG_WIDTH-1:1];
        assign icache_bus_if[i].rsp_ready       = fetch_icache_if[i].rsp_ready;
        `UNUSED_VAR (icache_bus_if[i].rsp_data.tag[0])
    end

    // scheduling: the core scheduler and its per-frontend-slice interfaces

    localparam FW = `VX_CFG_FETCH_WIDTH;

    VX_slice_sched_if   slice_sched_if[FW]();

    wire [`VX_CFG_NUM_WARPS-1:0] committed_warps;
    wire [`VX_CFG_NUM_WARPS-1:0] sched_active_warps;
`ifdef VX_CFG_VM_ENABLE
    wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] sched_satp;
    wire [`VX_CFG_NUM_SFU_BLOCKS-1:0] satp_wr_valid;
    wire [`VX_CFG_NUM_SFU_BLOCKS-1:0][`VX_CFG_XLEN-1:0] satp_wr_data;
`endif

    cta_bcast_t cta_bcast;

    VX_scheduler #(
        .INSTANCE_ID (`SFORMATF(("%s-scheduler", INSTANCE_ID)))
    ) scheduler (
        .clk            (clk),
        .reset          (reset),
        .core_id        (core_id),
    `ifdef PERF_ENABLE
        .sched_perf     (pipeline_perf.sched),
    `endif
        .slice_if       (slice_sched_if),
        .warp_shared_if (warp_shared_if),
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .txbar_bus_if   (dxa_txbar_bus_if),
    `endif
        .committed_warps(committed_warps),
        .cta_bcast      (cta_bcast),
        .kmu_bus_if     (kmu_bus_if),
        .gbar_bus_if    (gbar_bus_if),
        .cycles         (sched_cycles),
        .instret        (sched_instret),
        .active_warps   (sched_active_warps),
    `ifdef VX_CFG_VM_ENABLE
        .satp           (sched_satp),
        .satp_wr_valid  (satp_wr_valid),
        .satp_wr_data   (satp_wr_data),
    `endif
        .busy           (sched_busy)
    );

`ifdef PERF_ENABLE
    fetch_perf_t per_slice_fetch_perf [FW];
    `PERF_COUNTER_ADD (pipeline_perf.fetch, per_slice_fetch_perf, stalls, PERF_CTR_BITS, FW, (FW > 2))
`endif

    // core-wide services, reached through one port per SFU or TCU block

`ifdef VX_CFG_EXT_DXA_ENABLE
    VX_dxa_req_bus_if block_dxa_req_bus_if[`VX_CFG_NUM_SFU_BLOCKS]();
`endif
`ifdef VX_CFG_EXT_TEX_ENABLE
    VX_tex_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (TEX_REQ_TAG_WIDTH)
    ) block_tex_bus_if[`VX_CFG_NUM_SFU_BLOCKS]();
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
    VX_rtu_bus_if #(
        .NUM_LANES (`VX_CFG_NUM_SFU_LANES),
        .TAG_WIDTH (RTU_REQ_TAG_WIDTH)
    ) block_rtu_bus_if[`VX_CFG_NUM_SFU_BLOCKS]();
`endif
`ifdef VX_CFG_TCU_WGMMA_ENABLE
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_exec_valid, tcu_exec_ready;
    tcu_execute_t                     tcu_exec_data [`VX_CFG_NUM_TCU_BLOCKS];
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tcu_tbuf_rs1_data;
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0]   tcu_tbuf_rs2_data;
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_tbuf_ready;
`endif
`ifdef TCU_META_ENABLE
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_ld_valid, tcu_agu_ld_ready;
    tcu_execute_t                     tcu_agu_ld_data [`VX_CFG_NUM_TCU_BLOCKS];
    wire [`VX_CFG_NUM_TCU_BLOCKS-1:0] tcu_agu_result_valid, tcu_agu_result_ready;
    tcu_result_t                      tcu_agu_result_data [`VX_CFG_NUM_TCU_BLOCKS];
    wire                              tcu_meta_wr_en;
    wire [NW_WIDTH-1:0]               tcu_meta_wr_wid;
    wire [4:0]                        tcu_meta_wr_idx;
    wire [TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0] tcu_meta_wr_data;
`endif

    VX_sfu_shared #(
        .INSTANCE_ID (`SFORMATF(("%s-sfu-shared", INSTANCE_ID)))
    ) sfu_shared (
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .block_dxa_req_bus_if (block_dxa_req_bus_if),
        .dxa_req_bus_if       (dxa_req_bus_if),
    `endif
    `ifdef VX_CFG_EXT_TEX_ENABLE
        .block_tex_bus_if     (block_tex_bus_if),
        .tex_bus_if           (tex_bus_if),
    `endif
    `ifdef VX_CFG_EXT_RTU_ENABLE
        .block_rtu_bus_if     (block_rtu_bus_if),
        .rtu_bus_if           (rtu_bus_if),
    `endif
        .clk                  (clk),
        .reset                (reset)
    );

`ifdef VX_CFG_EXT_TCU_ENABLE
    VX_tcu_shared #(
        .INSTANCE_ID (`SFORMATF(("%s-tcu-shared", INSTANCE_ID))),
        .BLOCK_SIZE  (`VX_CFG_NUM_TCU_BLOCKS)
    ) tcu_shared (
    `ifdef PERF_ENABLE
        .tcu_perf         (tcu_perf),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .exec_valid       (tcu_exec_valid),
        .exec_ready       (tcu_exec_ready),
        .exec_data        (tcu_exec_data),
        .tbuf_rs1_data    (tcu_tbuf_rs1_data),
        .tbuf_rs2_data    (tcu_tbuf_rs2_data),
        .tbuf_ready_eff   (tcu_tbuf_ready),
        .tcu_lmem_if      (tcu_lmem_if),
    `endif
    `ifdef TCU_META_ENABLE
        .agu_ld_valid     (tcu_agu_ld_valid),
        .agu_ld_data      (tcu_agu_ld_data),
        .agu_ld_ready     (tcu_agu_ld_ready),
        .agu_result_valid (tcu_agu_result_valid),
        .agu_result_data  (tcu_agu_result_data),
        .agu_result_ready (tcu_agu_result_ready),
        .meta_wr_en       (tcu_meta_wr_en),
        .meta_wr_wid      (tcu_meta_wr_wid),
        .meta_wr_idx      (tcu_meta_wr_idx),
        .meta_wr_data     (tcu_meta_wr_data),
        .tcu_mem_if       (lsu_tcu_mem_if),
    `endif
        .clk              (clk),
        .reset            (reset)
    );
`endif

    // fetch, issue, execute and commit, structured by COMPUTE_SLICES / BACKEND_SLICES

    if (COMPUTE_SLICES) begin : g_compute_slices

    `ifdef PERF_ENABLE
        issue_perf_t per_slice_issue_perf [`VX_CFG_ISSUE_WIDTH];
        `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, ibf_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
        `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, scb_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
        `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, opd_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
        for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_issue_perf_units_uses
            `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, dispatch_stalls[i], PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
            `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, dispatch_instrs[i], PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
        end
    `endif

    `ifdef SCOPE
        wire [`VX_CFG_ISSUE_WIDTH-1:0] slice_scope_bus_in, slice_scope_bus_out;
        VX_scope_switch #(
            .N (`VX_CFG_ISSUE_WIDTH)
        ) slice_scope_switch (
            .clk     (clk),
            .reset   (scope_reset_w[1]),
            .req_in  (scope_bus_in_w[1]),
            .rsp_out (scope_bus_out_w[1]),
            .req_out (slice_scope_bus_in),
            .rsp_in  (slice_scope_bus_out)
        );
    `endif
        `SCOPE_IO_UNUSED (0)
        `SCOPE_IO_UNUSED (2)

        for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_slices
            wire [PER_ISSUE_WARPS-1:0] committed_wis;

            VX_compute_slice #(
                .INSTANCE_ID (INSTANCE_ID),
                .SLICE_STR   (`SFORMATF(("%0d", i)))
            ) compute_slice (
            `ifdef SCOPE
                .scope_reset          (scope_reset_w[1]),
                .scope_bus_in         (slice_scope_bus_in[i]),
                .scope_bus_out        (slice_scope_bus_out[i]),
            `endif
                .clk                  (clk),
                .reset                (reset),
                .core_id              (core_id),
                .slice_id             (ISSUE_ISW_W'(i)),
            `ifdef PERF_ENABLE
                .fetch_perf           (per_slice_fetch_perf[i]),
                .issue_perf           (per_slice_issue_perf[i]),
            `endif
                .slice_sched_if       (slice_sched_if[i]),
                .icache_bus_if        (fetch_icache_if[i]),
                .committed_wis        (committed_wis),
                .cycles               (sched_cycles),
                .instret              (sched_instret),
                .active_warps         (sched_active_warps),
                .all_lsu_sched_empty  (lsu_sched_empty),
                .mmu_empty            (mmu_empty),
                .cta_bcast            (cta_bcast),
            `ifdef VX_CFG_VM_ENABLE
                .satp                 (sched_satp[i]),
                .satp_wr_valid        (satp_wr_valid[i]),
                .satp_wr_data         (satp_wr_data[i]),
            `endif
            `ifdef PERF_ENABLE
                .lsu_perf_rd_req_fire (lsu_perf_rd_req_fire[i]),
                .lsu_perf_wr_req_fire (lsu_perf_wr_req_fire[i]),
                .lsu_perf_rsp_fire    (lsu_perf_rsp_fire[i]),
                .lsu_coalescer_misses (lsu_coalescer_misses[i]),
            `endif
            `ifdef TCU_META_ENABLE
                .lsu_tcu_mem_if       (lsu_tcu_mem_if[i]),
            `endif
                .lsu_sched_empty      (lsu_sched_empty[i]),
                .lsu_store_empty      (lsu_store_empty[i]),
                .lsu_busy             (lsu_busy[i]),
            `ifdef VX_CFG_LMEM_ENABLE
                .lmem_bus_if          (lsu_lmem_bus_if[i * `VX_CFG_NUM_LSU_LANES +: `VX_CFG_NUM_LSU_LANES]),
            `endif
                .dcache_bus_if        (lsu_dcache_bus_if[i * DCACHE_CHANNELS +: DCACHE_CHANNELS]),
            `ifdef VX_CFG_EXT_DXA_ENABLE
                .dxa_req_bus_if       (block_dxa_req_bus_if[i]),
            `endif
            `ifdef VX_CFG_EXT_TEX_ENABLE
                .tex_bus_if           (block_tex_bus_if[i]),
            `endif
            `ifdef VX_CFG_EXT_RTU_ENABLE
                .rtu_bus_if           (block_rtu_bus_if[i]),
            `endif
            `ifdef VX_CFG_TCU_WGMMA_ENABLE
                .tcu_exec_valid       (tcu_exec_valid[i]),
                .tcu_exec_ready       (tcu_exec_ready[i]),
                .tcu_exec_data        (tcu_exec_data[i]),
                .tcu_tbuf_rs1_data    (tcu_tbuf_rs1_data[i]),
                .tcu_tbuf_rs2_data    (tcu_tbuf_rs2_data[i]),
                .tcu_tbuf_ready       (tcu_tbuf_ready[i]),
            `endif
            `ifdef TCU_META_ENABLE
                .tcu_agu_ld_valid     (tcu_agu_ld_valid[i]),
                .tcu_agu_ld_data      (tcu_agu_ld_data[i]),
                .tcu_agu_ld_ready     (tcu_agu_ld_ready[i]),
                .tcu_agu_result_valid (tcu_agu_result_valid[i]),
                .tcu_agu_result_data  (tcu_agu_result_data[i]),
                .tcu_agu_result_ready (tcu_agu_result_ready[i]),
                .tcu_meta_wr_en       (tcu_meta_wr_en && (NW_WIDTH'(tcu_meta_wr_wid % `VX_CFG_NUM_TCU_BLOCKS) == NW_WIDTH'(i))),
                .tcu_meta_wr_wid      (tcu_meta_wr_wid),
                .tcu_meta_wr_idx      (tcu_meta_wr_idx),
                .tcu_meta_wr_data     (tcu_meta_wr_data),
            `endif
                .warp_shared_if       (warp_shared_if[i])
            );

            // wid = {wis, isw}: slice i owns every ISSUE_WIDTH-th warp starting at i
            for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_committed
                assign committed_warps[w * `VX_CFG_ISSUE_WIDTH + i] = committed_wis[w];
            end
        end

    end else begin : g_crossbar

        VX_decode_if        fs_decode_if[FW]();
        VX_decode_if        is_decode_if[`VX_CFG_ISSUE_WIDTH]();
        VX_branch_ctl_if    branch_ctl_if[`VX_CFG_NUM_ALU_BLOCKS]();
        VX_warp_ctl_if      warp_ctl_if[`VX_CFG_NUM_SFU_BLOCKS]();
        VX_sched_csr_if     sched_csr_if[`VX_CFG_NUM_SFU_BLOCKS]();
    `ifdef VX_CFG_EXT_RTU_ENABLE
        VX_sched_unlock_if  sched_unlock_if[`VX_CFG_NUM_SFU_BLOCKS]();
    `endif
        wire [`VX_CFG_ISSUE_WIDTH-1:0] issue_valid;
        wire [`VX_CFG_ISSUE_WIDTH-1:0][ISSUE_WIS_W-1:0] issue_wis;

        localparam NPA = FETCH_ALU_PORTS;
        localparam NPS = FETCH_SFU_PORTS;

        sched_branch_t [FW-1:0][NPA-1:0] fs_branch;
        sched_wctl_t [FW-1:0][NPS-1:0] fs_wctl;
        wire [FW-1:0][NPS-1:0][NW_WIDTH-1:0] fs_dvstack_wid;
        wire [FW-1:0][NPS-1:0][DV_STACK_SIZEW-1:0] fs_dvstack_ptr;
        wire [FW-1:0][NPS-1:0][`VX_CFG_MEM_ADDR_WIDTH-1:0] fs_dvstack_mscratch;
        wire [FW-1:0][NPS-1:0][NW_WIDTH-1:0] fs_csr_rd_wid;
        sched_csr_rd_t [FW-1:0][NPS-1:0] fs_csr_rd;
        sched_csr_wr_t [FW-1:0][NPS-1:0] fs_csr_wr;
    `ifdef VX_CFG_EXT_RTU_ENABLE
        wire [FW-1:0][NPS-1:0] fs_rtu_unlock_valid;
        wire [FW-1:0][NPS-1:0][NW_WIDTH-1:0] fs_rtu_unlock_wid;
    `endif
        wire [FW-1:0][PER_FETCH_WARPS-1:0] fs_issued;
        wire [FW-1:0][PER_FETCH_WARPS-1:0] fs_committed;
        wire [FW-1:0][PER_FETCH_WARPS-1:0] fs_pending_alm_empty;

        VX_frontend_xbar frontend_xbar (
            .fs_decode_if         (fs_decode_if),
            .is_decode_if         (is_decode_if),
            .issue_valid          (issue_valid),
            .issue_wis            (issue_wis),
            .committed_warps      (committed_warps),
            .fs_issued            (fs_issued),
            .fs_committed         (fs_committed),
            .fs_pending_alm_empty (fs_pending_alm_empty),
            .branch_ctl_if        (branch_ctl_if),
            .warp_ctl_if          (warp_ctl_if),
            .sched_csr_if         (sched_csr_if),
        `ifdef VX_CFG_EXT_RTU_ENABLE
            .sched_unlock_if      (sched_unlock_if),
        `endif
            .cycles               (sched_cycles),
            .instret              (sched_instret),
            .active_warps         (sched_active_warps),
            .lsu_sched_empty      (lsu_sched_empty),
            .mmu_empty            (mmu_empty),
        `ifdef VX_CFG_VM_ENABLE
            .satp                 (sched_satp),
            .satp_wr_valid        (satp_wr_valid),
            .satp_wr_data         (satp_wr_data),
        `endif
            .fs_branch            (fs_branch),
            .fs_wctl              (fs_wctl),
            .fs_dvstack_wid       (fs_dvstack_wid),
            .fs_dvstack_ptr       (fs_dvstack_ptr),
            .fs_dvstack_mscratch  (fs_dvstack_mscratch),
            .fs_csr_rd_wid        (fs_csr_rd_wid),
            .fs_csr_rd            (fs_csr_rd),
        `ifdef VX_CFG_EXT_RTU_ENABLE
            .fs_rtu_unlock_valid  (fs_rtu_unlock_valid),
            .fs_rtu_unlock_wid    (fs_rtu_unlock_wid),
        `endif
            .fs_csr_wr            (fs_csr_wr)
        );

    `ifdef SCOPE
        wire [FW-1:0] frontend_scope_bus_in, frontend_scope_bus_out;
        VX_scope_switch #(
            .N (FW)
        ) frontend_scope_switch (
            .clk     (clk),
            .reset   (scope_reset_w[0]),
            .req_in  (scope_bus_in_w[0]),
            .rsp_out (scope_bus_out_w[0]),
            .req_out (frontend_scope_bus_in),
            .rsp_in  (frontend_scope_bus_out)
        );
    `endif

        for (genvar f = 0; f < FW; ++f) begin : g_frontend_slices
            VX_frontend_slice #(
                .INSTANCE_ID (INSTANCE_ID),
                .SLICE_STR   ((FW > 1) ? `SFORMATF(("%0d", f)) : "")
            ) frontend_slice (
            `ifdef SCOPE
                .scope_reset       (scope_reset_w[0]),
                .scope_bus_in      (frontend_scope_bus_in[f]),
                .scope_bus_out     (frontend_scope_bus_out[f]),
            `endif
                .clk               (clk),
                .reset             (reset),
                .core_id           (core_id),
                .slice_id          (FETCH_FSW_W'(f)),
            `ifdef PERF_ENABLE
                .fetch_perf        (per_slice_fetch_perf[f]),
            `endif
                .slice_sched_if    (slice_sched_if[f]),
                .icache_bus_if     (fetch_icache_if[f]),
                .decode_if         (fs_decode_if[f]),
                .branch            (fs_branch[f]),
                .wctl              (fs_wctl[f]),
                .dvstack_wid       (fs_dvstack_wid[f]),
                .dvstack_ptr       (fs_dvstack_ptr[f]),
                .dvstack_mscratch  (fs_dvstack_mscratch[f]),
                .csr_rd_wid        (fs_csr_rd_wid[f]),
                .csr_rd            (fs_csr_rd[f]),
                .csr_wr            (fs_csr_wr[f]),
            `ifdef VX_CFG_EXT_RTU_ENABLE
                .rtu_unlock_valid  (fs_rtu_unlock_valid[f]),
                .rtu_unlock_wid    (fs_rtu_unlock_wid[f]),
            `endif
                .issued            (fs_issued[f]),
                .committed         (fs_committed[f]),
                .pending_alm_empty (fs_pending_alm_empty[f])
            );
        end

        if (BACKEND_SLICES) begin : g_backend_slices

        `ifdef PERF_ENABLE
            issue_perf_t per_slice_issue_perf [`VX_CFG_ISSUE_WIDTH];
            `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, ibf_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
            `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, scb_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
            `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, opd_stalls, PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
            for (genvar i = 0; i < NUM_EX_UNITS; ++i) begin : g_issue_perf_units_uses
                `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, dispatch_stalls[i], PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
                `PERF_COUNTER_ADD (pipeline_perf.issue, per_slice_issue_perf, dispatch_instrs[i], PERF_CTR_BITS, `VX_CFG_ISSUE_WIDTH, (`VX_CFG_ISSUE_WIDTH > 2))
            end
        `endif

        `ifdef SCOPE
            wire [`VX_CFG_ISSUE_WIDTH-1:0] slice_scope_bus_in, slice_scope_bus_out;
            VX_scope_switch #(
                .N (`VX_CFG_ISSUE_WIDTH)
            ) slice_scope_switch (
                .clk     (clk),
                .reset   (scope_reset_w[1]),
                .req_in  (scope_bus_in_w[1]),
                .rsp_out (scope_bus_out_w[1]),
                .req_out (slice_scope_bus_in),
                .rsp_in  (slice_scope_bus_out)
            );
        `endif
            `SCOPE_IO_UNUSED (2)

            for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_slices
                wire [PER_ISSUE_WARPS-1:0] committed_wis;

                VX_backend_slice #(
                    .INSTANCE_ID (INSTANCE_ID),
                    .SLICE_STR   (`SFORMATF(("%0d", i)))
                ) backend_slice (
                `ifdef SCOPE
                    .scope_reset          (scope_reset_w[1]),
                    .scope_bus_in         (slice_scope_bus_in[i]),
                    .scope_bus_out        (slice_scope_bus_out[i]),
                `endif
                    .clk                  (clk),
                    .reset                (reset),
                    .core_id              (core_id),
                    .slice_id             (ISSUE_ISW_W'(i)),
                `ifdef PERF_ENABLE
                    .issue_perf           (per_slice_issue_perf[i]),
                `endif
                    .decode_if            (is_decode_if[i]),
                    .warp_issued          (issue_valid[i]),
                    .warp_issued_wis      (issue_wis[i]),
                    .committed_wis        (committed_wis),
                    .branch_ctl_if        (branch_ctl_if[i]),
                    .warp_ctl_if          (warp_ctl_if[i]),
                    .sched_csr_if         (sched_csr_if[i]),
                `ifdef VX_CFG_EXT_RTU_ENABLE
                    .sched_unlock_if      (sched_unlock_if[i]),
                `endif
                `ifdef PERF_ENABLE
                    .lsu_perf_rd_req_fire (lsu_perf_rd_req_fire[i]),
                    .lsu_perf_wr_req_fire (lsu_perf_wr_req_fire[i]),
                    .lsu_perf_rsp_fire    (lsu_perf_rsp_fire[i]),
                    .lsu_coalescer_misses (lsu_coalescer_misses[i]),
                `endif
                `ifdef TCU_META_ENABLE
                    .lsu_tcu_mem_if       (lsu_tcu_mem_if[i]),
                `endif
                    .lsu_sched_empty      (lsu_sched_empty[i]),
                    .lsu_store_empty      (lsu_store_empty[i]),
                    .lsu_busy             (lsu_busy[i]),
                `ifdef VX_CFG_LMEM_ENABLE
                    .lmem_bus_if          (lsu_lmem_bus_if[i * `VX_CFG_NUM_LSU_LANES +: `VX_CFG_NUM_LSU_LANES]),
                `endif
                    .dcache_bus_if        (lsu_dcache_bus_if[i * DCACHE_CHANNELS +: DCACHE_CHANNELS]),
                `ifdef VX_CFG_EXT_DXA_ENABLE
                    .dxa_req_bus_if       (block_dxa_req_bus_if[i]),
                `endif
                `ifdef VX_CFG_EXT_TEX_ENABLE
                    .tex_bus_if           (block_tex_bus_if[i]),
                `endif
                `ifdef VX_CFG_EXT_RTU_ENABLE
                    .rtu_bus_if           (block_rtu_bus_if[i]),
                `endif
                `ifdef VX_CFG_TCU_WGMMA_ENABLE
                    .tcu_exec_valid       (tcu_exec_valid[i]),
                    .tcu_exec_ready       (tcu_exec_ready[i]),
                    .tcu_exec_data        (tcu_exec_data[i]),
                    .tcu_tbuf_rs1_data    (tcu_tbuf_rs1_data[i]),
                    .tcu_tbuf_rs2_data    (tcu_tbuf_rs2_data[i]),
                    .tcu_tbuf_ready       (tcu_tbuf_ready[i]),
                `endif
                `ifdef TCU_META_ENABLE
                    .tcu_agu_ld_valid     (tcu_agu_ld_valid[i]),
                    .tcu_agu_ld_data      (tcu_agu_ld_data[i]),
                    .tcu_agu_ld_ready     (tcu_agu_ld_ready[i]),
                    .tcu_agu_result_valid (tcu_agu_result_valid[i]),
                    .tcu_agu_result_data  (tcu_agu_result_data[i]),
                    .tcu_agu_result_ready (tcu_agu_result_ready[i]),
                    .tcu_meta_wr_en       (tcu_meta_wr_en && (NW_WIDTH'(tcu_meta_wr_wid % `VX_CFG_NUM_TCU_BLOCKS) == NW_WIDTH'(i))),
                    .tcu_meta_wr_wid      (tcu_meta_wr_wid),
                    .tcu_meta_wr_idx      (tcu_meta_wr_idx),
                    .tcu_meta_wr_data     (tcu_meta_wr_data),
                `endif
                    .warp_shared_if       (warp_shared_if[i]),
                    .cta_bcast            (cta_bcast)
                );

                // wid = {wis, isw}: slice i owns every ISSUE_WIDTH-th warp starting at i
                for (genvar w = 0; w < PER_ISSUE_WARPS; ++w) begin : g_committed
                    assign committed_warps[w * `VX_CFG_ISSUE_WIDTH + i] = committed_wis[w];
                end
            end

        end else begin : g_stage_sliced

            VX_issue_sched_if   issue_sched_if[`VX_CFG_ISSUE_WIDTH]();
            VX_commit_sched_if  commit_sched_if();

            VX_dispatch_if  dispatch_if[NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH]();
            VX_commit_if    commit_if[NUM_EX_UNITS * `VX_CFG_ISSUE_WIDTH]();
            VX_writeback_if writeback_if[`VX_CFG_ISSUE_WIDTH]();

            VX_issue #(
                .INSTANCE_ID (`SFORMATF(("%s-issue", INSTANCE_ID)))
            ) issue (
                `SCOPE_IO_BIND  (1)

                .clk            (clk),
                .reset          (reset),

            `ifdef PERF_ENABLE
                .issue_perf     (pipeline_perf.issue),
            `endif

                .decode_if      (is_decode_if),
                .writeback_if   (writeback_if),
                .dispatch_if    (dispatch_if),
                .issue_sched_if (issue_sched_if)
            );

            for (genvar i = 0; i < `VX_CFG_ISSUE_WIDTH; ++i) begin : g_issue_sched
                assign issue_valid[i] = issue_sched_if[i].valid;
                assign issue_wis[i]   = issue_sched_if[i].wis;
            end

            VX_execute #(
                .INSTANCE_ID (`SFORMATF(("%s-execute", INSTANCE_ID)))
            ) execute (
                `SCOPE_IO_BIND  (2)

                .clk                  (clk),
                .reset                (reset),
                .core_id              (core_id),

            `ifdef PERF_ENABLE
                .lsu_perf_rd_req_fire (lsu_perf_rd_req_fire),
                .lsu_perf_wr_req_fire (lsu_perf_wr_req_fire),
                .lsu_perf_rsp_fire    (lsu_perf_rsp_fire),
                .lsu_coalescer_misses (lsu_coalescer_misses),
            `endif
            `ifdef TCU_META_ENABLE
                .lsu_tcu_mem_if       (lsu_tcu_mem_if),
            `endif
                .lsu_sched_empty      (lsu_sched_empty),
                .lsu_store_empty      (lsu_store_empty),
                .lsu_busy             (lsu_busy),
            `ifdef VX_CFG_LMEM_ENABLE
                .lmem_bus_if          (lsu_lmem_bus_if),
            `endif
                .dcache_bus_if        (lsu_dcache_bus_if),

                .dispatch_if          (dispatch_if),
                .commit_if            (commit_if),

                .sched_csr_if         (sched_csr_if),

            `ifdef VX_CFG_TCU_WGMMA_ENABLE
                .tcu_exec_valid       (tcu_exec_valid),
                .tcu_exec_ready       (tcu_exec_ready),
                .tcu_exec_data        (tcu_exec_data),
                .tcu_tbuf_rs1_data    (tcu_tbuf_rs1_data),
                .tcu_tbuf_rs2_data    (tcu_tbuf_rs2_data),
                .tcu_tbuf_ready       (tcu_tbuf_ready),
            `endif
            `ifdef TCU_META_ENABLE
                .tcu_agu_ld_valid     (tcu_agu_ld_valid),
                .tcu_agu_ld_data      (tcu_agu_ld_data),
                .tcu_agu_ld_ready     (tcu_agu_ld_ready),
                .tcu_agu_result_valid (tcu_agu_result_valid),
                .tcu_agu_result_data  (tcu_agu_result_data),
                .tcu_agu_result_ready (tcu_agu_result_ready),
                .tcu_meta_wr_en       (tcu_meta_wr_en),
                .tcu_meta_wr_wid      (tcu_meta_wr_wid),
                .tcu_meta_wr_idx      (tcu_meta_wr_idx),
                .tcu_meta_wr_data     (tcu_meta_wr_data),
            `endif
            `ifdef VX_CFG_EXT_DXA_ENABLE
                .dxa_req_bus_if       (block_dxa_req_bus_if),
            `endif
            `ifdef VX_CFG_EXT_TEX_ENABLE
                .tex_bus_if           (block_tex_bus_if),
            `endif
            `ifdef VX_CFG_EXT_RTU_ENABLE
                .rtu_bus_if           (block_rtu_bus_if),
                .sched_unlock_if      (sched_unlock_if),
            `endif

                .warp_shared_if       (warp_shared_if),

                .warp_ctl_if          (warp_ctl_if),
                .branch_ctl_if        (branch_ctl_if),
                .cta_bcast            (cta_bcast)
            );

            VX_commit #(
                .INSTANCE_ID (`SFORMATF(("%s-commit", INSTANCE_ID)))
            ) commit (
                .clk            (clk),
                .reset          (reset),

                .commit_if      (commit_if),

                .writeback_if   (writeback_if),

                .commit_sched_if(commit_sched_if)
            );

            assign committed_warps = commit_sched_if.committed_warps;

        end
    end

    VX_mem_unit #(
        .INSTANCE_ID (INSTANCE_ID)
    ) mem_unit (
        .clk             (clk),
        .reset           (reset),
    `ifdef PERF_ENABLE
        .lmem_perf       (lmem_perf),
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        .tcu_lmem_if     (tcu_lmem_if),
    `endif
    `ifdef VX_CFG_EXT_DXA_ENABLE
        .dxa_lmem_bus_if (dxa_lmem_bus_if),
        .dxa_txbar_bus_if(dxa_txbar_bus_if),
    `endif
    `ifdef VX_CFG_LMEM_ENABLE
        .lmem_bus_if     (lsu_lmem_bus_if),
    `endif
        .lsu_dcache_bus_if (lsu_dcache_bus_if),
        .dcr_flush_if    (dcr_flush_dcache_if),
        .flush_l1_only   (dcr_flush_l1_only),
        .dcache_bus_if   (dcache_bus_if),
        .busy            (mem_unit_busy)
    );

    // Fragment work drains at the producer (VX_raster_core.busy), not here.
    // DCR reads (cache flush, MPM readback) order on their own DCR response, so
    // busy tracks only kernel execution: warps resident or stores not yet at L1.
    // ~mmu_empty also keeps the core busy while a translation or parked store is
    // still in flight, so kernel completion cannot race un-drained accesses.
    // A store still in an LSU block's memory path shows on lsu_busy, which the
    // parent ORs in; one past a flush injector's buffer shows on mem_unit_busy.
    assign busy = sched_busy || mem_unit_busy || ~mmu_empty;

`ifdef PERF_ENABLE

    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rd_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_wr_req_per_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)-1:0] perf_dcache_rsp_per_cycle;

    wire [`CLOG2(ICACHE_NUM_REQS+1)-1:0] perf_icache_req_per_cycle;
    wire [`CLOG2(ICACHE_NUM_REQS+1)-1:0] perf_icache_rsp_per_cycle;
    wire [`CLOG2(ICACHE_NUM_REQS+1)+1-1:0] perf_icache_pending_read_cycle;
    wire [`CLOG2(LSU_NUM_REQS+1)+1-1:0] perf_dcache_pending_read_cycle;

    reg [PERF_CTR_BITS-1:0] perf_icache_pending_reads;
    reg [PERF_CTR_BITS-1:0] perf_dcache_pending_reads;

    reg [PERF_CTR_BITS-1:0] perf_ifetches;
    reg [PERF_CTR_BITS-1:0] perf_loads;
    reg [PERF_CTR_BITS-1:0] perf_stores;

    wire [ICACHE_NUM_REQS-1:0] perf_icache_req_fire;
    wire [ICACHE_NUM_REQS-1:0] perf_icache_rsp_fire;
    for (genvar i = 0; i < ICACHE_NUM_REQS; ++i) begin : g_perf_icache
        assign perf_icache_req_fire[i] = icache_bus_if[i].req_valid && icache_bus_if[i].req_ready;
        assign perf_icache_rsp_fire[i] = icache_bus_if[i].rsp_valid && icache_bus_if[i].rsp_ready;
    end
    `POP_COUNT(perf_icache_req_per_cycle, perf_icache_req_fire);
    `POP_COUNT(perf_icache_rsp_per_cycle, perf_icache_rsp_fire);

    wire [LSU_NUM_REQS-1:0] perf_dcache_rd_req_fire, perf_dcache_rd_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_wr_req_fire, perf_dcache_wr_req_fire_r;
    wire [LSU_NUM_REQS-1:0] perf_dcache_rsp_fire;

    assign perf_dcache_rd_req_fire = lsu_perf_rd_req_fire;
    assign perf_dcache_wr_req_fire = lsu_perf_wr_req_fire;
    assign perf_dcache_rsp_fire    = lsu_perf_rsp_fire;

    wire [PERF_CTR_BITS-1:0] coalescer_misses;
    VX_reduce_tree #(
        .IN_W (PERF_CTR_BITS),
        .N    (`VX_CFG_NUM_LSU_BLOCKS),
        .OP   ("+")
    ) coalescer_reduce (
        .data_in  (lsu_coalescer_misses),
        .data_out (coalescer_misses)
    );
    `BUFFER(coalescer_perf.misses, coalescer_misses);

    `BUFFER(perf_dcache_rd_req_fire_r, perf_dcache_rd_req_fire);
    `BUFFER(perf_dcache_wr_req_fire_r, perf_dcache_wr_req_fire);

    `POP_COUNT(perf_dcache_rd_req_per_cycle, perf_dcache_rd_req_fire_r);
    `POP_COUNT(perf_dcache_wr_req_per_cycle, perf_dcache_wr_req_fire_r);
    `POP_COUNT(perf_dcache_rsp_per_cycle, perf_dcache_rsp_fire);

    assign perf_icache_pending_read_cycle = perf_icache_req_per_cycle - perf_icache_rsp_per_cycle;
    assign perf_dcache_pending_read_cycle = perf_dcache_rd_req_per_cycle - perf_dcache_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_icache_pending_reads <= '0;
            perf_dcache_pending_reads <= '0;
        end else begin
            perf_icache_pending_reads <= $signed(perf_icache_pending_reads) + PERF_CTR_BITS'($signed(perf_icache_pending_read_cycle));
            perf_dcache_pending_reads <= $signed(perf_dcache_pending_reads) + PERF_CTR_BITS'($signed(perf_dcache_pending_read_cycle));
        end
    end

    reg [PERF_CTR_BITS-1:0] perf_icache_lat;
    reg [PERF_CTR_BITS-1:0] perf_dcache_lat;

    always @(posedge clk) begin
        if (reset) begin
            perf_ifetches   <= '0;
            perf_loads      <= '0;
            perf_stores     <= '0;
            perf_icache_lat <= '0;
            perf_dcache_lat <= '0;
        end else begin
            perf_ifetches   <= perf_ifetches   + PERF_CTR_BITS'(perf_icache_req_per_cycle);
            perf_loads      <= perf_loads      + PERF_CTR_BITS'(perf_dcache_rd_req_per_cycle);
            perf_stores     <= perf_stores     + PERF_CTR_BITS'(perf_dcache_wr_req_per_cycle);
            perf_icache_lat <= perf_icache_lat + perf_icache_pending_reads;
            perf_dcache_lat <= perf_dcache_lat + perf_dcache_pending_reads;
        end
    end

    assign pipeline_perf.ifetches = perf_ifetches;
    assign pipeline_perf.loads = perf_loads;
    assign pipeline_perf.stores = perf_stores;
    assign pipeline_perf.ifetch_latency = perf_icache_lat;
    assign pipeline_perf.load_latency = perf_dcache_lat;

`endif

endmodule
