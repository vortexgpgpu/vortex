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
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    localparam CTR_W = $clog2(TCU_UOPS);
    `STATIC_ASSERT (CTR_W <= UOP_CTR_W, ("invalid parameter"))

    localparam LG_N = $clog2(TCU_N_STEPS);
    localparam LG_M = $clog2(TCU_M_STEPS);
    localparam LG_K = $clog2(TCU_K_STEPS);

    localparam LG_A_SB = $clog2(TCU_A_SUB_BLOCKS);
    localparam LG_B_SB = $clog2(TCU_B_SUB_BLOCKS);

    // Truncate the wide uop_idx to the bits this expander actually uses.
    wire [`UP(CTR_W)-1:0] ctr = `UP(CTR_W)'(uop_idx);
    `UNUSED_VAR ({clk, reset, start, uop_idx})

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_SB_SP       = $clog2(TCU_B_SUB_BLOCKS_SP);
    localparam SPARSE_SAME_CYCLES = (TCU_BLOCK_CAP == 16);
    localparam HALF_K           = TCU_K_STEPS / 2;

    wire is_sparse = (ibuf_in.op_type == INST_TCU_WMMA_SP);
    wire is_meta_store = (ibuf_in.op_type == INST_TCU_META_STORE);

    assign uop_count = is_meta_store
        ? UOP_CTR_W'(meta_num_cols(ibuf_in.op_args.tcu.fmt_s))
        : (is_sparse && !SPARSE_SAME_CYCLES)
            ? UOP_CTR_W'(TCU_UOPS / 2)
            : UOP_CTR_W'(TCU_UOPS);
`else
    // Dense-only: count is a compile-time constant.
    assign uop_count = UOP_CTR_W'(TCU_UOPS);
`endif

    // -----------------------------------------------------------------------
    // Index extraction from uop_idx
    // -----------------------------------------------------------------------
    logic [`UP(LG_N)-1:0] n_index;
    logic [`UP(LG_M)-1:0] m_index;
    logic [`UP(LG_K)-1:0] k_index;

    if (LG_N != 0) begin : g_n_idx
        assign n_index = ctr[0 +: LG_N];
    end else begin : g_n_idx0
        assign n_index = 0;
    end

    if (LG_M != 0) begin : g_m_idx
        assign m_index = ctr[LG_N +: LG_M];
    end else begin : g_m_idx0
        assign m_index = 0;
    end

    if (LG_K != 0) begin : g_k_idx
        assign k_index = ctr[LG_N + LG_M +: LG_K];
    end else begin : g_k_idx0
        assign k_index = 0;
    end

    // -----------------------------------------------------------------------
    // Register-offset arithmetic.
    // -----------------------------------------------------------------------
`ifdef TCU_SPARSE_ENABLE
    wire [`UP(CTR_W)-1:0] rs1_offset = is_sparse
        ? ((`UP(CTR_W)'(m_index) >> LG_A_SB) << (LG_K / 2)) | `UP(CTR_W)'(k_index)
        : ((`UP(CTR_W)'(m_index) >> LG_A_SB) << LG_K)       | `UP(CTR_W)'(k_index);

    wire [`UP(CTR_W)-1:0] rs2_offset = is_sparse
        ? ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB_SP
        : ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB;
`else
    wire [`UP(CTR_W)-1:0] rs1_offset = ((`UP(CTR_W)'(m_index) >> LG_A_SB) << LG_K) | `UP(CTR_W)'(k_index);
    wire [`UP(CTR_W)-1:0] rs2_offset = ((`UP(CTR_W)'(k_index) << LG_N) | `UP(CTR_W)'(n_index)) >> LG_B_SB;
`endif

    wire [`UP(CTR_W)-1:0] rs3_offset = (`UP(CTR_W)'(m_index) << LG_N) | `UP(CTR_W)'(n_index);

    wire [4:0] rs1 = TCU_RA + 5'(rs1_offset);
    wire [4:0] rs2 = TCU_RB + 5'(rs2_offset);
    wire [4:0] rs3 = TCU_RC + 5'(rs3_offset);

    // -----------------------------------------------------------------------
    // Output uop assembly.
    // -----------------------------------------------------------------------
    assign ibuf_out.uuid = get_uop_uuid(ibuf_in.uuid, uop_idx);
`ifdef TCU_SPARSE_ENABLE
    wire sparse_k_masked = SPARSE_SAME_CYCLES && is_sparse
                        && (k_index >= `UP(LG_K)'(HALF_K));
    assign ibuf_out.tmask = sparse_k_masked ? '0 : ibuf_in.tmask;
`else
    assign ibuf_out.tmask = ibuf_in.tmask;
`endif
    assign ibuf_out.PC      = ibuf_in.PC;
    assign ibuf_out.ex_type = ibuf_in.ex_type;
    assign ibuf_out.op_type = ibuf_in.op_type;
    assign ibuf_out.op_args.tcu.fmt_s = ibuf_in.op_args.tcu.fmt_s;
`ifdef TCU_SPARSE_ENABLE
    /* verilator lint_off UNSIGNED */
    wire meta_use_rs2 = (ctr >= `UP(CTR_W)'(TCU_META_COLS_PER_LOAD));
    /* verilator lint_on UNSIGNED */
    assign ibuf_out.op_args.tcu.fmt_d  = is_meta_store ? 4'(ctr) : ibuf_in.op_args.tcu.fmt_d;
    assign ibuf_out.op_args.tcu.step_m = is_meta_store ? '0 : 4'(m_index);
    assign ibuf_out.op_args.tcu.step_n = is_meta_store ? '0 : 4'(n_index);
    assign ibuf_out.op_args.tcu.step_k = is_meta_store ? '0 : 4'(k_index);
    assign ibuf_out.wb  = is_meta_store ? 1'b0 : 1'b1;
    assign ibuf_out.rd  = is_meta_store ? '0 : make_reg_num(REG_TYPE_F, rs3);
    assign ibuf_out.rs1 = is_meta_store ? (meta_use_rs2 ? ibuf_in.rs2 : ibuf_in.rs1) : make_reg_num(REG_TYPE_F, rs1);
    assign ibuf_out.rs2 = is_meta_store ? ibuf_in.rs2 : make_reg_num(REG_TYPE_F, rs2);
    assign ibuf_out.rs3 = is_meta_store ? '0 : make_reg_num(REG_TYPE_F, rs3);
`else
    assign ibuf_out.op_args.tcu.fmt_d  = ibuf_in.op_args.tcu.fmt_d;
    assign ibuf_out.op_args.tcu.step_m = 4'(m_index);
    assign ibuf_out.op_args.tcu.step_n = 4'(n_index);
    assign ibuf_out.op_args.tcu.step_k = 4'(k_index);
    assign ibuf_out.wb  = 1;
    assign ibuf_out.rd  = make_reg_num(REG_TYPE_F, rs3);
    assign ibuf_out.rs1 = make_reg_num(REG_TYPE_F, rs1);
    assign ibuf_out.rs2 = make_reg_num(REG_TYPE_F, rs2);
    assign ibuf_out.rs3 = make_reg_num(REG_TYPE_F, rs3);
`endif
    assign ibuf_out.rd_xregs = ibuf_in.rd_xregs;
    assign ibuf_out.wr_xregs = ibuf_in.wr_xregs;
    assign ibuf_out.used_rs  = ibuf_in.used_rs;

    `UNUSED_VAR (ibuf_in.wb)
    `UNUSED_VAR (ibuf_in.rd)
`ifndef TCU_SPARSE_ENABLE
    `UNUSED_VAR (ibuf_in.rs1)
    `UNUSED_VAR (ibuf_in.rs2)
`endif
    `UNUSED_VAR (ibuf_in.rs3)

endmodule
