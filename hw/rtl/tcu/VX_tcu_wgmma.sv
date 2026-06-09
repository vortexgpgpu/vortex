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

    // Bank-parallel LMEM read port.
    VX_mem_bus_if.master                                                 tcu_lmem_if,

    // Outputs to tcu_core (consumed by the wrapper).
    output wire [BLOCK_SIZE-1:0][TCU_BLOCK_CAP-1:0][`VX_CFG_XLEN-1:0]    tbuf_rs1_data,
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
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_lockstep_inputs
        wire is_wgmma_op = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign is_wgmma_b_w[bi]     = exec_valid[bi] && is_wgmma_op;
        assign new_cta_b_w[bi]      = exec_data[bi].header.cta_id;
        assign exec_fire_b_w[bi]    = exec_valid[bi] && exec_ready[bi];
        assign is_first_uop_b_w[bi] = exec_data[bi].op_args.tcu.is_first_uop;
        assign is_last_uop_b_w[bi]  = exec_data[bi].op_args.tcu.is_last_uop;
    end

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
        .cta_conflict   (cta_conflict)
    );

    // -----------------------------------------------------------------------
    // Per-block uop observation → VX_tcu_tbuf request.
    // req[b].valid is masked by cta_conflict so the shared bbuf doesn't
    // refill its bank-row for a CTA whose fire is currently gated by
    // lockstep.
    // -----------------------------------------------------------------------
    tcu_tbuf_req_t [BLOCK_SIZE-1:0] req;
    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_inputs
        wire is_wgmma_b = (exec_data[bi].op_type == INST_TCU_WGMMA)
                    `ifdef VX_CFG_TCU_SPARSE_ENABLE
                       || (exec_data[bi].op_type == INST_TCU_WGMMA_SP)
                    `endif
                       ;
        assign req[bi].valid        = exec_valid[bi] && is_wgmma_b && !cta_conflict[bi];
        assign req[bi].uuid         = exec_data[bi].header.uuid;
        assign req[bi].wid          = exec_data[bi].header.wid;
        assign req[bi].step_m       = exec_data[bi].op_args.tcu.step_m;
        assign req[bi].step_k       = exec_data[bi].op_args.tcu.step_k;
        assign req[bi].step_n       = exec_data[bi].op_args.tcu.step_n;
        assign req[bi].cd_nregs     = exec_data[bi].op_args.tcu.cd_nregs;
        assign req[bi].desc_a       = exec_data[bi].rs1_data[0];
        assign req[bi].desc_b       = exec_data[bi].rs2_data[0];
        assign req[bi].a_is_smem    = exec_data[bi].op_args.tcu.a_from_smem;
        assign req[bi].is_first_uop = exec_data[bi].op_args.tcu.is_first_uop;
        assign req[bi].is_last_uop  = exec_data[bi].op_args.tcu.is_last_uop;
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        assign req[bi].is_sparse = (exec_data[bi].op_type == INST_TCU_WGMMA_SP);
    `endif
    end

    wire [BLOCK_SIZE-1:0] tbuf_ready;

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
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data),
        .tbuf_ready     (tbuf_ready)
    );

    for (genvar bi = 0; bi < BLOCK_SIZE; ++bi) begin : g_tbuf_eff
        assign tbuf_ready_eff[bi] = tbuf_ready[bi] && !cta_conflict[bi];
    end

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
