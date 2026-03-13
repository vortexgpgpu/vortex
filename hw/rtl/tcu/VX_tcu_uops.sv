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

//
// TCU uop expander.
//
module VX_tcu_uops import VX_tcu_pkg::*, VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
`ifdef TCU_SPARSE_ENABLE
    localparam MAX_META_COLS = TCU_BLOCK_CAP / 2;  // worst case: 4-bit types
    localparam MAX_FUSED = NT16_SPARSE ? (TCU_UOPS + MAX_META_COLS) : (TCU_UOPS / 2 + MAX_META_COLS);
    localparam CTR_W = $clog2(MAX_FUSED > TCU_UOPS ? MAX_FUSED : TCU_UOPS);
`else
    localparam CTR_W = $clog2(TCU_UOPS);
`endif
    `STATIC_ASSERT (CTR_W <= UOP_CTR_W, ("invalid parameter"))

    localparam LG_N = $clog2(TCU_N_STEPS);
    localparam LG_M = $clog2(TCU_M_STEPS);
    localparam LG_K = $clog2(TCU_K_STEPS);

    localparam LG_A_SB = $clog2(TCU_A_SUB_BLOCKS);
    localparam LG_B_SB = $clog2(TCU_B_SUB_BLOCKS);

    `UNUSED_VAR ({clk, reset, start, advance, uop_idx})

    // Truncate the wide uop_idx to the bits this expander actually uses.
    wire [`UP(CTR_W)-1:0] ctr = `UP(CTR_W)'(uop_idx);

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_SB_SP = $clog2(TCU_B_SUB_BLOCKS_SP);

    wire is_sparse = (ibuf_in.op_type == INST_TCU_WMMA_SP);
    wire is_meta_store = (ibuf_in.op_type == INST_TCU_META_STORE);

    /* verilator lint_off UNUSEDSIGNAL */
    wire [4:0] sparse_meta_cols = meta_num_cols(ibuf_in.op_args.tcu.fmt_s);
    /* verilator lint_on UNUSEDSIGNAL */

    // Combinational meta-phase detection — comparator/subtractor absorbed
    // by the registered uop_data stage in VX_uop_sequencer.
    wire is_meta_phase = is_sparse && (ctr < `UP(CTR_W)'(sparse_meta_cols));
    wire [`UP(CTR_W)-1:0] mma_ctr = ctr - `UP(CTR_W)'(sparse_meta_cols);
    wire meta_uop = is_meta_store || is_meta_phase;
    localparam META_REG0 = TCU_RA + 4;  // f14 — fragA.data[4]
    localparam META_REG1 = TCU_RA + 5;  // f15 — fragA.data[5]

    // Fused meta+MMA uop counts
    assign uop_count = is_meta_store
        ? UOP_CTR_W'(meta_num_cols(ibuf_in.op_args.tcu.fmt_s))
        : is_sparse
            ? (NT16_SPARSE
                ? UOP_CTR_W'(TCU_UOPS + int'(meta_num_cols(ibuf_in.op_args.tcu.fmt_s)))
                : UOP_CTR_W'(TCU_UOPS / 2 + int'(meta_num_cols(ibuf_in.op_args.tcu.fmt_s))))
            : UOP_CTR_W'(TCU_UOPS);
`else
    // Dense-only: count is a compile-time constant.
    assign uop_count = UOP_CTR_W'(TCU_UOPS);
`endif

`ifdef TCU_SPARSE_ENABLE
    wire [`UP(CTR_W)-1:0] eff_ctr = (is_sparse && !is_meta_phase) ? mma_ctr : ctr;
`else
    wire [`UP(CTR_W)-1:0] eff_ctr = ctr;
`endif

    // -----------------------------------------------------------------------
    // Index extraction from uop_idx
    // -----------------------------------------------------------------------
    logic [`UP(LG_N)-1:0] n_index;
    logic [`UP(LG_M)-1:0] m_index;
    logic [`UP(LG_K)-1:0] k_index;

    if (LG_N != 0) begin : g_n_idx
        assign n_index = eff_ctr[0 +: LG_N];
    end else begin : g_n_idx0
        assign n_index = 0;
    end

    if (LG_M != 0) begin : g_m_idx
        assign m_index = eff_ctr[LG_N +: LG_M];
    end else begin : g_m_idx0
        assign m_index = 0;
    end

    if (LG_K != 0) begin : g_k_idx
        assign k_index = eff_ctr[LG_N + LG_M +: LG_K];
    end else begin : g_k_idx0
        assign k_index = 0;
    end

    // -----------------------------------------------------------------------
    // Register-offset arithmetic.
    // -----------------------------------------------------------------------
    logic [`UP(CTR_W)-1:0] rs1_offset;
    logic [`UP(CTR_W)-1:0] rs2_offset;
    logic [`UP(CTR_W)-1:0] rs3_offset;

`ifdef TCU_SPARSE_ENABLE
    if (NT16_SPARSE) begin : g_nt16_off
        wire [`UP(CTR_W)-1:0] n_sp = `UP(CTR_W)'(eff_ctr[0 +: (LG_N + LG_K)]);
        wire [`UP(CTR_W)-1:0] m_sp = `UP(CTR_W)'(eff_ctr[(LG_N + LG_K) +: LG_M]);
        assign rs1_offset = is_sparse ? `UP(CTR_W)'(m_sp)
            : ((`UP(CTR_W)'(m_index) >> LG_A_SB) << LG_K) | `UP(CTR_W)'(k_index);
        assign rs2_offset = is_sparse ? `UP(CTR_W)'(n_sp)
            : ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB;
        assign rs3_offset = is_sparse ? (`UP(CTR_W)'(eff_ctr) >> 1)
            : (`UP(CTR_W)'(m_index) << LG_N) | `UP(CTR_W)'(n_index);
    end else begin : g_def_off
        assign rs1_offset = is_sparse
            ? ((`UP(CTR_W)'(m_index) >> LG_A_SB) << (LG_K / 2)) | `UP(CTR_W)'(k_index)
            : ((`UP(CTR_W)'(m_index) >> LG_A_SB) << LG_K)       | `UP(CTR_W)'(k_index);
        assign rs2_offset = is_sparse
            ? ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB_SP
            : ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB;
        assign rs3_offset = (`UP(CTR_W)'(m_index) << LG_N) | `UP(CTR_W)'(n_index);
    end
`else
    assign rs1_offset = ((`UP(CTR_W)'(m_index) >> LG_A_SB) << LG_K) | `UP(CTR_W)'(k_index);
    assign rs2_offset = ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB;
    assign rs3_offset = (`UP(CTR_W)'(m_index) << LG_N) | `UP(CTR_W)'(n_index);
`endif

    wire [4:0] rs1 = TCU_RA + 5'(rs1_offset);
    wire [4:0] rs2 = TCU_RB + 5'(rs2_offset);
    wire [4:0] rs3 = TCU_RC + 5'(rs3_offset);

    // -----------------------------------------------------------------------
    // Output uop assembly.
    // -----------------------------------------------------------------------
`ifdef TCU_SPARSE_ENABLE
    /* verilator lint_off UNSIGNED */
    wire meta_use_rs2 = (ctr >= `UP(CTR_W)'(TCU_META_COLS_PER_LOAD));
    /* verilator lint_on UNSIGNED */
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
`ifdef TCU_SPARSE_ENABLE
        if (NT16_SPARSE) begin
            ibuf_r.tmask = is_sparse
                ? (is_meta_phase ? ibuf_in.tmask
                    : (eff_ctr[0] ? ibuf_in.tmask & 16'hCCCC
                                   : ibuf_in.tmask & 16'h3333))
                : ibuf_in.tmask;
        end

        ibuf_r.op_type = meta_uop ? INST_TCU_META_STORE : ibuf_in.op_type;
        ibuf_r.op_args.tcu.fmt_d = meta_uop ? 4'(ctr) : ibuf_in.op_args.tcu.fmt_d;

        if (NT16_SPARSE) begin
            /* verilator lint_off UNUSEDSIGNAL */
            logic [`UP(CTR_W)-1:0] n_sp_s;
            logic [`UP(CTR_W)-1:0] m_sp_s;
            /* verilator lint_on UNUSEDSIGNAL */
            n_sp_s = `UP(CTR_W)'(eff_ctr[0 +: (LG_N + LG_K)]);
            m_sp_s = `UP(CTR_W)'(eff_ctr[(LG_N + LG_K) +: LG_M]);

            ibuf_r.op_args.tcu.step_m = meta_uop ? '0 : (is_sparse ? 4'(m_sp_s) : 4'(m_index));
            ibuf_r.op_args.tcu.step_n = meta_uop ? '0 : (is_sparse ? 4'(n_sp_s) : 4'(n_index));
            ibuf_r.op_args.tcu.step_k = meta_uop ? '0 : (is_sparse ? 4'(0)      : 4'(k_index));
        end else begin
            ibuf_r.op_args.tcu.step_m = meta_uop ? '0 : 4'(m_index);
            ibuf_r.op_args.tcu.step_n = meta_uop ? '0 : 4'(n_index);
            ibuf_r.op_args.tcu.step_k = meta_uop ? '0 : 4'(k_index);
        end

        ibuf_r.wb  = meta_uop ? 1'b0 : 1'b1;
        ibuf_r.rd  = meta_uop ? '0 : make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.rs1 = meta_uop
            ? (is_meta_store
                ? (meta_use_rs2 ? ibuf_in.rs2 : ibuf_in.rs1)
                : (meta_use_rs2 ? make_reg_num(REG_TYPE_F, 5'(META_REG1))
                                : make_reg_num(REG_TYPE_F, 5'(META_REG0))))
            : make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = meta_uop ? ibuf_in.rs2 : make_reg_num(REG_TYPE_F, rs2);
        ibuf_r.rs3 = meta_uop ? '0 : make_reg_num(REG_TYPE_F, rs3);
`else
        ibuf_r.op_args.tcu.step_m = 4'(m_index);
        ibuf_r.op_args.tcu.step_n = 4'(n_index);
        ibuf_r.op_args.tcu.step_k = 4'(k_index);
        ibuf_r.wb  = 1;
        ibuf_r.rd  = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.rs1 = make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = make_reg_num(REG_TYPE_F, rs2);
        ibuf_r.rs3 = make_reg_num(REG_TYPE_F, rs3);
`endif
    end

    assign ibuf_out = ibuf_r;

endmodule
