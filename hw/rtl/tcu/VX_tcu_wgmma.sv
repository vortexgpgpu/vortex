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

`ifdef VX_CFG_TCU_WGMMA_ENABLE

// WGMMA orchestrator. Owns everything that exists only because WGMMA
// exists: dispatch-IF unpack into tbuf request, VX_tcu_tbuf instantiation,
// CTA lockstep gate, and WGMMA-specific perf counters.
//
// Observation-only on the dispatch path: this module does not drive
// .ready. The wrapper (VX_tcu_unit) feeds tbuf_ready_eff into tcu_core,
// which produces the actual handshake .ready going back to dispatch.
//
// tbuf_ready_eff is registered. The tile buffers and the CTA lockstep are
// evaluated for the uop each block presents NEXT cycle (the entry behind the
// head when the head fires, else the head), against the state this cycle's
// fires and LMEM responses leave behind. A same-cycle ready would otherwise
// close a loop across every block (shared-bbuf representative + key match ->
// ready -> fire -> operand-buffer enables) on the WGMMA critical path. A uop
// not yet visible in its block queue waits one cycle.

module VX_tcu_wgmma import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         BLOCK_SIZE      = `VX_CFG_NUM_TCU_BLOCKS,
    parameter         BANK_ADDR_WIDTH = `VX_CFG_LMEM_LOG_SIZE
                                      - $clog2(`VX_CFG_XLEN / 8)
                                      - $clog2(`VX_CFG_LMEM_NUM_BANKS)
) (
    input wire clk,
    input wire reset,

`ifdef PERF_ENABLE
    output tcu_perf_t                                                    tcu_perf,
`endif

    // Observation of dispatch path (read-only; does not drive .ready).
    input  wire [BLOCK_SIZE-1:0]                                         exec_valid,
    input  wire [BLOCK_SIZE-1:0]                                         exec_ready,
    input  tcu_execute_t                                                 exec_data [BLOCK_SIZE],
    // The entry queued behind each head.
    input  wire [BLOCK_SIZE-1:0]                                         exec_next_valid,
    input  tcu_execute_t                                                 exec_next_data [BLOCK_SIZE],

    // Bank-parallel LMEM read port.
    VX_mem_bus_if.master                                                 tcu_lmem_if,

    // Outputs to tcu_core (consumed by the wrapper).
    output wire [BLOCK_SIZE-1:0][TCU_WG_A_DATA_SIZE-1:0][`VX_CFG_XLEN-1:0] tbuf_rs1_data,
    output wire [BLOCK_SIZE-1:0][TCU_WG_RS2_WIDTH-1:0][`VX_CFG_XLEN-1:0] tbuf_rs2_data,
    output wire [BLOCK_SIZE-1:0]                                         tbuf_ready_eff
);
    `UNUSED_SPARAM (INSTANCE_ID)

    // -----------------------------------------------------------------------
    // CTA lockstep gate. is_first_uop / is_last_uop come from op_args.tcu,
    // set by VX_tcu_uops alongside fu_lock/fu_unlock.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0]                  cta_conflict;
    wire [BLOCK_SIZE-1:0]                  is_wgmma_b_w;
    wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0]  new_cta_b_w;
    wire [BLOCK_SIZE-1:0]                  exec_fire_b_w;
    wire [BLOCK_SIZE-1:0]                  is_first_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  is_last_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  is_setup_uop_b_w;
    wire [BLOCK_SIZE-1:0]                  needs_setup_b_w;

    logic [BLOCK_SIZE-1:0][`VX_CFG_XLEN-1:0] desc_a_r;
    logic [BLOCK_SIZE-1:0][`VX_CFG_XLEN-1:0] desc_b_r;

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_inputs
        wire is_wgmma_op = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign is_setup_uop_b_w[bi] = is_wgmma_op && !exec_data[bi].header.wb;
        assign needs_setup_b_w[bi] =
            `ifdef VX_CFG_TCU_FEDP2K
                is_wgmma_op && !exec_data[bi].op_args.tcu.a_from_smem
            `ifdef VX_CFG_TCU_SPARSE_ENABLE
                && (exec_data[bi].op_type != INST_TCU_WGMMA_SP)
            `endif
            `else
                1'b0
            `endif
                ;
        assign is_wgmma_b_w[bi]     = exec_valid[bi] && is_wgmma_op && !is_setup_uop_b_w[bi];
        assign new_cta_b_w[bi]      = exec_data[bi].header.cta_id;
        assign exec_fire_b_w[bi]    = exec_valid[bi] && exec_ready[bi];
        assign is_first_uop_b_w[bi] = is_wgmma_op && exec_data[bi].op_args.tcu.is_first_uop;
        assign is_last_uop_b_w[bi]  = is_wgmma_op && exec_data[bi].op_args.tcu.is_last_uop;
    end

    always_ff @(posedge clk) begin
        if (reset) begin
            desc_a_r <= '0;
            desc_b_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; ++bi) begin
                if (exec_valid[bi] && exec_ready[bi]
                 && (is_setup_uop_b_w[bi]
                  || (is_first_uop_b_w[bi] && !needs_setup_b_w[bi]))) begin
                    if (exec_data[bi].op_args.tcu.a_from_smem)
                        desc_a_r[bi] <= exec_data[bi].rs1_data[0];
                    desc_b_r[bi] <= exec_data[bi].rs2_data[0];
                end
            end
        end
    end

    // -----------------------------------------------------------------------
    // Next cycle's head per block.
    // -----------------------------------------------------------------------
    wire [BLOCK_SIZE-1:0] nh_valid;
    tcu_execute_t         nh_data [BLOCK_SIZE];
    wire [BLOCK_SIZE-1:0] nh_is_wgmma_op, nh_is_setup, nh_compute, nh_needs_setup, nh_is_first;
    wire [BLOCK_SIZE-1:0][NCTA_WIDTH-1:0] nh_cta;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_next_head
        assign nh_valid[bi] = exec_fire_b_w[bi] ? exec_next_valid[bi] : exec_valid[bi];
        assign nh_data[bi]  = exec_fire_b_w[bi] ? exec_next_data[bi]  : exec_data[bi];
        assign nh_is_wgmma_op[bi] = (nh_data[bi].op_type == INST_TCU_WGMMA)
                            `ifdef VX_CFG_TCU_SPARSE_ENABLE
                               || (nh_data[bi].op_type == INST_TCU_WGMMA_SP)
                            `endif
                               ;
        assign nh_is_setup[bi] = nh_is_wgmma_op[bi] && !nh_data[bi].header.wb;
        assign nh_compute[bi]  = nh_valid[bi] && nh_is_wgmma_op[bi] && !nh_is_setup[bi];
        assign nh_needs_setup[bi] =
            `ifdef VX_CFG_TCU_FEDP2K
                nh_is_wgmma_op[bi] && !nh_data[bi].op_args.tcu.a_from_smem
            `ifdef VX_CFG_TCU_SPARSE_ENABLE
                && (nh_data[bi].op_type != INST_TCU_WGMMA_SP)
            `endif
            `else
                1'b0
            `endif
                ;
        assign nh_is_first[bi] = nh_is_wgmma_op[bi] && nh_data[bi].op_args.tcu.is_first_uop;
        assign nh_cta[bi]      = nh_data[bi].header.cta_id;
    end

    wire [BLOCK_SIZE-1:0] cta_conflict_nh;
    `UNUSED_VAR (cta_conflict)

    VX_tcu_lockstep #(
        .INSTANCE_ID (`SFORMATF(("%s-lockstep", INSTANCE_ID))),
        .BLOCK_SIZE  (BLOCK_SIZE)
    ) lockstep (
        .clk            (clk),
        .reset          (reset),
        .is_wgmma_b     (is_wgmma_b_w),
        .new_cta_b      (new_cta_b_w),
        .exec_fire_b    (exec_fire_b_w),
        .is_first_uop_b (is_first_uop_b_w),
        .is_last_uop_b  (is_last_uop_b_w),
        .is_wgmma_nh_b  (nh_compute),
        .new_cta_nh_b   (nh_cta),
        .cta_conflict   (cta_conflict),
        .cta_conflict_nh(cta_conflict_nh)
    );

    // -----------------------------------------------------------------------
    // Next cycle's heads → VX_tcu_tbuf request.
    // req[b].valid is masked by the next-cycle CTA conflict so the shared bbuf
    // doesn't refill its bank-row for a CTA whose fire will be gated by
    // lockstep. setup_fire is this cycle's: a setup uop firing now drops the
    // buffers at the end of the cycle.
    // -----------------------------------------------------------------------
    tcu_tbuf_req_t [BLOCK_SIZE-1:0] req;
    wire [BLOCK_SIZE-1:0][3:0]      out_step_m;
    wire [BLOCK_SIZE-1:0]           out_is_sparse;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_inputs
        // Descriptors as the setup/first-uop latch above leaves them.
        wire latch_now = exec_fire_b_w[bi]
                      && (is_setup_uop_b_w[bi] || (is_first_uop_b_w[bi] && !needs_setup_b_w[bi]));
        wire [`VX_CFG_XLEN-1:0] desc_a_next = (latch_now && exec_data[bi].op_args.tcu.a_from_smem)
                                            ? exec_data[bi].rs1_data[0] : desc_a_r[bi];
        wire [`VX_CFG_XLEN-1:0] desc_b_next = latch_now ? exec_data[bi].rs2_data[0] : desc_b_r[bi];
        wire use_live_desc = nh_is_first[bi] && !nh_needs_setup[bi];
        assign req[bi].valid        = nh_compute[bi] && !cta_conflict_nh[bi];
        assign req[bi].uuid         = nh_data[bi].header.uuid;
        assign req[bi].wid          = nh_data[bi].header.wid;
        assign req[bi].cta_id       = nh_data[bi].header.cta_id;
        assign req[bi].step_m       = nh_data[bi].op_args.tcu.step_m;
        assign req[bi].step_k       = nh_data[bi].op_args.tcu.step_k;
        assign req[bi].step_n       = nh_data[bi].op_args.tcu.step_n;
        assign req[bi].cd_nregs     = nh_data[bi].op_args.tcu.cd_nregs;
        assign req[bi].desc_a       = use_live_desc ? nh_data[bi].rs1_data[0] : desc_a_next;
        assign req[bi].desc_b       = use_live_desc ? nh_data[bi].rs2_data[0] : desc_b_next;
        assign req[bi].a_is_smem    = nh_data[bi].op_args.tcu.a_from_smem;
        assign req[bi].is_first_uop = nh_data[bi].op_args.tcu.is_first_uop;
        assign req[bi].is_last_uop  = nh_data[bi].op_args.tcu.is_last_uop;
        assign req[bi].setup_fire   = exec_fire_b_w[bi] && is_setup_uop_b_w[bi];
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        assign req[bi].is_sparse = (nh_data[bi].op_type == INST_TCU_WGMMA_SP);
        assign out_is_sparse[bi] = (exec_data[bi].op_type == INST_TCU_WGMMA_SP);
    `else
        assign out_is_sparse[bi] = 1'b0;
    `endif
        assign out_step_m[bi] = exec_data[bi].op_args.tcu.step_m;
    end

    wire [BLOCK_SIZE-1:0] tbuf_ready;
    wire [BLOCK_SIZE-1:0] tbuf_ready_next;

`ifdef PERF_ENABLE
    wire [PERF_CTR_BITS-1:0] tbuf_stalls_w;
    wire [PERF_CTR_BITS-1:0] tbuf_cache_hits_w;
    wire [PERF_CTR_BITS-1:0] lmem_reads_w;
`endif

    VX_tcu_tbuf #(
        .INSTANCE_ID    (`SFORMATF(("%s-tbuf", INSTANCE_ID))),
        .NUM_BANKS      (`VX_CFG_LMEM_NUM_BANKS),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .BLOCK_SIZE     (BLOCK_SIZE)
    ) tbuf (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tbuf_stalls    (tbuf_stalls_w),
        .tbuf_cache_hits(tbuf_cache_hits_w),
        .lmem_reads     (lmem_reads_w),
    `endif
        .req            (req),
        .out_step_m     (out_step_m),
        .out_is_sparse  (out_is_sparse),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready     (tbuf_ready),
        .tbuf_ready_next(tbuf_ready_next)
    );
    `UNUSED_VAR (tbuf_ready)

    // Registered compute-uop ready for next cycle's head. Setup uops never
    // wait on the buffers; tcu_core passes them through on its own.
    reg [BLOCK_SIZE-1:0] tbuf_ready_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            tbuf_ready_r <= '0;
        end else begin
            tbuf_ready_r <= nh_compute & tbuf_ready_next & ~cta_conflict_nh;
        end
    end
    assign tbuf_ready_eff = tbuf_ready_r;

    // -----------------------------------------------------------------------
    // WGMMA perf counters (instrs / stalls).
    // -----------------------------------------------------------------------
`ifdef PERF_ENABLE
    assign tcu_perf.tbuf_stalls     = tbuf_stalls_w;
    assign tcu_perf.tbuf_cache_hits = tbuf_cache_hits_w;
    assign tcu_perf.lmem_reads      = lmem_reads_w;

    logic wgmma_fire_b  [BLOCK_SIZE];
    logic wgmma_stall_b [BLOCK_SIZE];
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_wgmma_perf
        wire is_wgmma_p = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign wgmma_fire_b [bi] = exec_valid[bi] && exec_ready[bi] && is_wgmma_p;
        assign wgmma_stall_b[bi] = exec_valid[bi] && !exec_ready[bi] && is_wgmma_p;
    end

    logic [PERF_CTR_BITS-1:0] wgmma_instrs_ctr_r;
    logic [PERF_CTR_BITS-1:0] wgmma_stalls_ctr_r;
    always_ff @(posedge clk) begin
        if (reset) begin
            wgmma_instrs_ctr_r <= '0;
            wgmma_stalls_ctr_r <= '0;
        end else begin
            for (int bi = 0; bi < BLOCK_SIZE; bi++) begin
                if (wgmma_fire_b[bi])  wgmma_instrs_ctr_r <= wgmma_instrs_ctr_r + PERF_CTR_BITS'(1);
                if (wgmma_stall_b[bi]) wgmma_stalls_ctr_r <= wgmma_stalls_ctr_r + PERF_CTR_BITS'(1);
            end
        end
    end
    assign tcu_perf.wgmma_instrs = wgmma_instrs_ctr_r;
    assign tcu_perf.wgmma_stalls = wgmma_stalls_ctr_r;
`endif

endmodule

`endif // VX_CFG_TCU_WGMMA_ENABLE
