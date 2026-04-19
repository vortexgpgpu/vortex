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
    // Worst-case uop count for sparse: fused meta-store + MMA steps.
    localparam MAX_META_STORES = ((TCU_BLOCK_CAP + 1) / 2) * TCU_STORES_PER_COL;
    localparam MAX_UOPS = SYM_SPARSE ? (TCU_UOPS + MAX_META_STORES) : (TCU_UOPS / 2 + MAX_META_STORES);
`else
    localparam MAX_UOPS = TCU_UOPS;
`endif

`ifdef TCU_WGMMA_ENABLE
    localparam CTR_W = $clog2(MAX_UOPS > TCU_WG_UOPS ? MAX_UOPS : TCU_WG_UOPS);
`else
    localparam CTR_W = $clog2(MAX_UOPS);
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

`ifdef TCU_WGMMA_ENABLE
    wire is_wgmma = (ibuf_in.op_type == INST_TCU_WGMMA);
    wire wg_a_from_smem = ibuf_in.op_args.tcu.a_from_smem;

    // Variable NRC based on cd_nregs: 0→8, 1→16, 2→32
    // Loop order: m (inner) → n → k (outer)  [K-outer, Nvidia-style]
    // m_steps=2 and k_steps=2 are fixed; n varies (middle).
    // K-outer lets independent (m,n) tiles overlap FEDP latency.
    localparam LG_M_WG = $clog2(TCU_WG_M_STEPS);  // 1
    localparam LG_K_WG = $clog2(TCU_WG_K_STEPS);   // 1
    localparam LG_N_WG_MAX = $clog2(TCU_WG_N_STEPS); // 4 (for NRC=32)

    // Pre-computed (m×n) uop counts per k-step for each cd_nregs value
    localparam WG_MN_NR8  =  8;   // nrc=8:  n_steps=4,  mn=8
    localparam WG_MN_NR16 = 16;   // nrc=16: n_steps=8,  mn=16
    localparam WG_MN_NR32 = 32;   // nrc=32: n_steps=16, mn=32

    // Pre-computed total uop counts for each cd_nregs value (dense)
    localparam WG_UOPS_NR8  =  WG_MN_NR8  * TCU_WG_K_STEPS;
    localparam WG_UOPS_NR16 = WG_MN_NR16 * TCU_WG_K_STEPS;
    localparam WG_UOPS_NR32 = WG_MN_NR32 * TCU_WG_K_STEPS;

    // Mux uop count based on cd_nregs: 0→8, 1→16, 2→32
    reg [UOP_CTR_W-1:0] wg_uop_cnt;
    always_comb begin
        case (ibuf_in.op_args.tcu.cd_nregs)
            2'd0: wg_uop_cnt = UOP_CTR_W'(WG_UOPS_NR8);
            2'd1: wg_uop_cnt = UOP_CTR_W'(WG_UOPS_NR16);
            default: wg_uop_cnt = UOP_CTR_W'(WG_UOPS_NR32);
        endcase
    end

    // K-outer index extraction:
    //   Dense:  ctr = k * (n_steps * m_steps) + n * m_steps + m
    //   Sparse: ctr = n * m_steps + m  (k always 0)
    // Since n_steps varies by cd_nregs, k-index bit position shifts.
    // m is always bit 0 (m_steps=2). n and k extracted via mux.
    wire [`UP(LG_M_WG)-1:0] wg_m_index = ctr[0 +: `UP(LG_M_WG)];
    reg [`UP(LG_K_WG)-1:0] wg_k_index;
    reg [`UP(LG_N_WG_MAX)-1:0] wg_n_index;
`ifdef TCU_SPARSE_ENABLE
    wire wg_is_sparse = ibuf_in.op_args.tcu.is_sparse;
`endif
    always_comb begin
    `ifdef TCU_SPARSE_ENABLE
        if (wg_is_sparse) begin
            // Sparse: k always 0, n right after m
            wg_k_index = '0;
            wg_n_index = `UP(LG_N_WG_MAX)'(ctr[LG_M_WG +: `UP(LG_N_WG_MAX)]);
        end else
    `endif
        begin
            // Dense K-outer: m (inner) → n (middle) → k (outer)
            // n width varies by cd_nregs; k bit shifts accordingly.
            case (ibuf_in.op_args.tcu.cd_nregs)
                2'd0: begin // nrc=8, n_steps=4, LG_N=2
                    wg_n_index = `UP(LG_N_WG_MAX)'(ctr[LG_M_WG +: 2]);
                    wg_k_index = `UP(LG_K_WG)'(ctr[LG_M_WG + 2 +: `UP(LG_K_WG)]);
                end
                2'd1: begin // nrc=16, n_steps=8, LG_N=3
                    wg_n_index = `UP(LG_N_WG_MAX)'(ctr[LG_M_WG +: 3]);
                    wg_k_index = `UP(LG_K_WG)'(ctr[LG_M_WG + 3 +: `UP(LG_K_WG)]);
                end
                default: begin // nrc=32, n_steps=16, LG_N=4
                    wg_n_index = `UP(LG_N_WG_MAX)'(ctr[LG_M_WG +: LG_N_WG_MAX]);
                    wg_k_index = `UP(LG_K_WG)'(ctr[LG_M_WG + LG_N_WG_MAX +: `UP(LG_K_WG)]);
                end
            endcase
        end
    end

    // Accumulator register index: n * m_steps + m  (n-major layout)
    wire [4:0] wg_rs3_off = (5'(wg_n_index) << LG_M_WG) | 5'(wg_m_index);
  `ifdef TCU_ACC_ENABLE
    // K-step gating for internal accumulation (Nvidia-style):
    //   first_k: read C from RF (rs3); last_k: writeback D to RF (wb/rd)
    wire wg_is_first_k = (wg_k_index == '0);
    `ifdef TCU_SPARSE_ENABLE
    wire wg_is_last_k = wg_is_sparse
        ? (wg_k_index == `UP(LG_K_WG)'(TCU_WG_K_STEPS / 2 - 1))
        : (wg_k_index == `UP(LG_K_WG)'(TCU_WG_K_STEPS - 1));
    `else
    wire wg_is_last_k = (wg_k_index == `UP(LG_K_WG)'(TCU_WG_K_STEPS - 1));
    `endif
  `endif

    // Fixed A register base for RS mode: f24..f27
    localparam [4:0] wg_ra_base = TCU_WG_RA;

    // Register offsets for from-reg mode
    // A: rs1_off = m * k_steps + k  (NRA=4 registers starting at ra_base)
    localparam LG_WG_A_SB = $clog2(`UP(TCU_WG_A_SUB_BLOCKS));
    wire [`UP(CTR_W)-1:0] wg_rs1_reg_off = ((`UP(CTR_W)'(wg_m_index) >> LG_WG_A_SB) << `UP(LG_K_WG)) | `UP(CTR_W)'(wg_k_index);
    if (`UP(CTR_W) > 5) begin : g_unused_wg_rs1_off
        `UNUSED_VAR (wg_rs1_reg_off[`UP(CTR_W)-1 : 5])
    end
`endif

`ifdef TCU_SPARSE_ENABLE
    wire is_sparse = ibuf_in.op_args.tcu.is_sparse;
    wire is_meta_store = (ibuf_in.op_type == INST_TCU_META_STORE);

    wire [4:0] sparse_meta_total = meta_total_store_uops(ibuf_in.op_args.tcu.fmt_s);

    // Combinational meta-phase detection — comparator/subtractor absorbed
    // by the registered uop_data stage in VX_uop_sequencer.
    wire is_meta_phase = is_sparse && (ctr < `UP(CTR_W)'(sparse_meta_total));
    wire [`UP(CTR_W)-1:0] mma_ctr = ctr - `UP(CTR_W)'(sparse_meta_total);
    wire meta_uop = is_meta_store || is_meta_phase;
    localparam META_REG0 = TCU_RA + 4;  // f14 — fragA.data[4]
    localparam META_REG1 = TCU_RA + 5;  // f15 — fragA.data[5]
`endif

    assign uop_count =
`ifdef TCU_WGMMA_ENABLE
        is_wgmma ? (
    `ifdef TCU_SPARSE_ENABLE
            ibuf_in.op_args.tcu.is_sparse
                ? (wg_uop_cnt >> 1)
                :
    `endif
            wg_uop_cnt) :
`endif
`ifdef TCU_SPARSE_ENABLE
        is_meta_store ? UOP_CTR_W'(meta_total_store_uops(ibuf_in.op_args.tcu.fmt_s)) :
        is_sparse ? (SYM_SPARSE
            ? UOP_CTR_W'(TCU_UOPS + int'(meta_total_store_uops(ibuf_in.op_args.tcu.fmt_s)))
            : UOP_CTR_W'(TCU_UOPS / 2 + int'(meta_total_store_uops(ibuf_in.op_args.tcu.fmt_s)))) :
`endif
        UOP_CTR_W'(TCU_UOPS);

`ifdef TCU_SPARSE_ENABLE
    wire [`UP(CTR_W)-1:0] eff_ctr = (is_sparse && !is_meta_phase) ? mma_ctr : ctr;

    // Parametric symmetric tmask for sparse mode
    // sym_mask_lo[t] = 1 for threads where (t % tcN) < (tcN/2)
    logic [`NUM_THREADS-1:0] sym_mask_lo;
    if (SYM_SPARSE) begin : g_sym_mask
        for (genvar t = 0; t < `NUM_THREADS; ++t) begin : g_bit
            assign sym_mask_lo[t] = ((t % TCU_TC_N) < (TCU_TC_N / 2)) ? 1'b1 : 1'b0;
        end
    end else begin : g_sym_mask
        assign sym_mask_lo = '0;
    end
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

    // WMMA K-step gating for internal accumulation
    // SYM_SPARSE flattens (m, n_col, n_step) into eff_ctr with step_k=0 always
    // (k_count = K_STEPS/2 = 1), so every uop is both first-k and last-k.
    // k_index would incorrectly alias with the upper bits of m_sp_s.
`ifdef TCU_ACC_ENABLE
  `ifdef TCU_SPARSE_ENABLE
    wire wmma_is_first_k = (SYM_SPARSE && is_sparse) ? 1'b1 : (k_index == '0);
    wire wmma_is_last_k = is_sparse
        ? (SYM_SPARSE ? 1'b1 : (k_index == `UP(LG_K)'(TCU_K_STEPS / 2 - 1)))
        : (k_index == `UP(LG_K)'(TCU_K_STEPS - 1));
  `else
    wire wmma_is_first_k = (k_index == '0);
    wire wmma_is_last_k = (k_index == `UP(LG_K)'(TCU_K_STEPS - 1));
  `endif
`endif

    // -----------------------------------------------------------------------
    // Register-offset arithmetic.
    // -----------------------------------------------------------------------
    logic [`UP(CTR_W)-1:0] rs1_offset;
    logic [`UP(CTR_W)-1:0] rs2_offset;
    logic [`UP(CTR_W)-1:0] rs3_offset;

    // Bank-conflict-free register offset formulas.
    // Permutes A, B, C offsets so all three operands always land in
    // different GPR banks, eliminating all RF read-port stalls.
    // - Pattern A (LG_B_SB==0): A={m[0],~m[hi],k}, B={n^k,~k}, C={m[0],~m[hi],XNOR(m[hi],n)}
    // - Pattern B (LG_B_SB>0): A={k[0],~m,m^k[hi]}, B={k[0],k[hi]^np,~np}, C={n[0],~m,n[hi]}
    wire [`UP(CTR_W)-1:0] ra_off, rb_off, rc_off;
    if (LG_B_SB == 0) begin : g_bcfree_bsub1
        assign ra_off = `UP(CTR_W)'({m_index[0], ~m_index[`UP(LG_M)-1], k_index[0]});
        assign rb_off = `UP(CTR_W)'({n_index[0] ^ k_index[0], ~k_index[0]});
        assign rc_off = `UP(CTR_W)'({m_index[0], ~m_index[`UP(LG_M)-1], ~(m_index[`UP(LG_M)-1] ^ n_index[0])});
    end else begin : g_bcfree_bsub2
        assign ra_off = `UP(CTR_W)'({k_index[0], ~m_index[0], m_index[0] ^ k_index[`UP(LG_K)-1]});
        assign rb_off = `UP(CTR_W)'({k_index[0], k_index[`UP(LG_K)-1] ^ n_index[`UP(LG_N)-1], ~n_index[`UP(LG_N)-1]});
        assign rc_off = `UP(CTR_W)'({n_index[0], ~m_index[0], n_index[`UP(LG_N)-1]});
    end

`ifdef TCU_SPARSE_ENABLE
    if (SYM_SPARSE) begin : g_sym_off
        wire [`UP(CTR_W)-1:0] n_sp = `UP(CTR_W)'(eff_ctr[0 +: (LG_N + LG_K)]);
        wire [`UP(CTR_W)-1:0] m_sp = `UP(CTR_W)'(eff_ctr[(LG_N + LG_K) +: LG_M]);
        assign rs1_offset = is_sparse ? `UP(CTR_W)'(m_sp) : ra_off;
        assign rs2_offset = is_sparse ? `UP(CTR_W)'(n_sp) : rb_off;
        assign rs3_offset = is_sparse ? (`UP(CTR_W)'(eff_ctr) >> 1) : rc_off;
    end else begin : g_def_off
        // Bank-conflict-free sparse register offsets (non-sym sparse).
        // A stays identity; B and C are permuted for 0 stalls.
        // B_off_sp = {n[hi], ~(n[0]^k[0]), ~k[0]}
        // C_off_sp = {n[hi], m[0], ~(m[0]^n[0])}
        wire [`UP(CTR_W)-1:0] rb_off_sp = `UP(CTR_W)'({n_index[`UP(LG_N)-1], ~(n_index[0] ^ k_index[0]), ~k_index[0]});
        wire [`UP(CTR_W)-1:0] rc_off_sp = `UP(CTR_W)'({n_index[`UP(LG_N)-1], m_index[0], ~(m_index[0] ^ n_index[0])});
        assign rs1_offset = is_sparse
            ? ((`UP(CTR_W)'(m_index) >> LG_A_SB) << (LG_K / 2)) | `UP(CTR_W)'(k_index)
            : ra_off;
        assign rs2_offset = is_sparse ? rb_off_sp : rb_off;
        assign rs3_offset = is_sparse ? rc_off_sp : rc_off;
    end
`else
    assign rs1_offset = ra_off;
    assign rs2_offset = rb_off;
    assign rs3_offset = rc_off;
`endif
    // Suppress Verilator warnings for upper bits widened by WGMMA/SPARSE.
    if (`UP(CTR_W) > LG_N+LG_M+LG_K) begin : g_unused_upper_eff_ctr
        `UNUSED_VAR (eff_ctr[`UP(CTR_W)-1 : LG_N+LG_M+LG_K])
    end
    if (`UP(CTR_W) > 5) begin : g_unused_upper_offsets
        `UNUSED_VAR (rs1_offset[`UP(CTR_W)-1 : 5])
        `UNUSED_VAR (rs2_offset[`UP(CTR_W)-1 : 5])
        `UNUSED_VAR (rs3_offset[`UP(CTR_W)-1 : 5])
    end

    wire [4:0] rs1 = TCU_RA + 5'(rs1_offset);
    wire [4:0] rs2 = TCU_RB + 5'(rs2_offset);
    wire [4:0] rs3 = TCU_RC + 5'(rs3_offset);

    // -----------------------------------------------------------------------
    // Output uop assembly.
    // -----------------------------------------------------------------------
`ifdef TCU_SPARSE_ENABLE
    wire meta_use_rs2 = (ctr >= `UP(CTR_W)'(1));
`endif

`ifdef TCU_SPARSE_ENABLE
    logic [3:0] n_sp_s;
    logic [3:0] m_sp_s;
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
    `ifdef TCU_SPARSE_ENABLE
        n_sp_s = '0;
        m_sp_s = '0;
    `endif
    `ifdef TCU_WGMMA_ENABLE
        if (is_wgmma) begin
            ibuf_r.op_args.tcu.step_m = 4'(wg_m_index);
            ibuf_r.op_args.tcu.step_n = 4'(wg_n_index);
            ibuf_r.op_args.tcu.step_k = 4'(wg_k_index);
        `ifdef TCU_ACC_ENABLE
            // Only last-k writes back to RF; only first-k reads C from RF
            ibuf_r.wb  = wg_is_last_k;
            ibuf_r.rd  = wg_is_last_k ? make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off) : '0;
            ibuf_r.rs3 = wg_is_first_k ? make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off) : '0;
            ibuf_r.used_rs[2] = wg_is_first_k;
        `else
            ibuf_r.wb  = 1'b1;
            ibuf_r.rd  = make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off);
            ibuf_r.rs3 = make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off);
            ibuf_r.used_rs[2] = 1'b1;
        `endif
            // Smem descriptors are invariant across the whole WGMMA expansion,
            // so only fetch them on the first uop.
            if (wg_a_from_smem) begin
                ibuf_r.rs1 = make_reg_num(REG_TYPE_I, 5'd10);
                ibuf_r.used_rs[0] = (ctr == '0);
            end else begin
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, wg_ra_base + 5'(wg_rs1_reg_off));
                ibuf_r.used_rs[0] = 1'b1;
            end
            // B source: always smem descriptor (x11), fetched once
            ibuf_r.rs2 = make_reg_num(REG_TYPE_I, 5'd11);
            ibuf_r.used_rs[1] = (ctr == '0);
        end else
    `endif
        begin
    `ifdef TCU_SPARSE_ENABLE
        if (SYM_SPARSE) begin
            ibuf_r.tmask = is_sparse
                ? (is_meta_phase ? ibuf_in.tmask
                    : (eff_ctr[0] ? ibuf_in.tmask & ~sym_mask_lo : ibuf_in.tmask &  sym_mask_lo))
                : ibuf_in.tmask;
            n_sp_s = 4'(eff_ctr[0 +: (LG_N + LG_K)]);
            m_sp_s = 4'(eff_ctr[(LG_N + LG_K) +: LG_M]);
        end

        ibuf_r.op_type = meta_uop ? INST_TCU_META_STORE : ibuf_in.op_type;
        ibuf_r.op_args.tcu.fmt_d = meta_uop ? 4'(ctr) : ibuf_in.op_args.tcu.fmt_d;

        ibuf_r.op_args.tcu.step_m = meta_uop ? '0 : (SYM_SPARSE && is_sparse ? 4'(m_sp_s) : 4'(m_index));
        ibuf_r.op_args.tcu.step_n = meta_uop ? '0 : (SYM_SPARSE && is_sparse ? 4'(n_sp_s) : 4'(n_index));
        ibuf_r.op_args.tcu.step_k = meta_uop ? '0 : (SYM_SPARSE && is_sparse ? 4'(0)      : 4'(k_index));

    `ifdef TCU_ACC_ENABLE
        ibuf_r.wb = meta_uop ? 1'b0 : wmma_is_last_k;
        ibuf_r.rd = meta_uop ? '0 : (wmma_is_last_k ? make_reg_num(REG_TYPE_F, rs3) : '0);
    `else
        ibuf_r.wb = meta_uop ? 1'b0 : 1'b1;
        ibuf_r.rd = meta_uop ? '0 : make_reg_num(REG_TYPE_F, rs3);
    `endif
        ibuf_r.rs1 = meta_uop
            ? (is_meta_store
                ? (meta_use_rs2 ? ibuf_in.rs2 : ibuf_in.rs1)
                : (meta_use_rs2 ? make_reg_num(REG_TYPE_F, 5'(META_REG1))
                                : make_reg_num(REG_TYPE_F, 5'(META_REG0))))
            : make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = meta_uop ? ibuf_in.rs2 : make_reg_num(REG_TYPE_F, rs2);
    `ifdef TCU_ACC_ENABLE
        ibuf_r.rs3 = meta_uop ? '0 : (wmma_is_first_k ? make_reg_num(REG_TYPE_F, rs3) : '0);
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = !meta_uop && wmma_is_first_k;
    `else
        ibuf_r.rs3 = meta_uop ? '0 : make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = !meta_uop;
    `endif
    `else
        ibuf_r.op_args.tcu.step_m = 4'(m_index);
        ibuf_r.op_args.tcu.step_n = 4'(n_index);
        ibuf_r.op_args.tcu.step_k = 4'(k_index);
    `ifdef TCU_ACC_ENABLE
        ibuf_r.wb  = wmma_is_last_k;
        ibuf_r.rd  = wmma_is_last_k ? make_reg_num(REG_TYPE_F, rs3) : '0;
    `else
        ibuf_r.wb  = 1'b1;
        ibuf_r.rd  = make_reg_num(REG_TYPE_F, rs3);
    `endif
        ibuf_r.rs1 = make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = make_reg_num(REG_TYPE_F, rs2);
    `ifdef TCU_ACC_ENABLE
        ibuf_r.rs3 = wmma_is_first_k ? make_reg_num(REG_TYPE_F, rs3) : '0;
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = wmma_is_first_k;
    `else
        ibuf_r.rs3 = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = 1'b1;
    `endif
    `endif
        end
        // FU lock: 10=acquire (first), 00=middle, 01=release (last), 11=default.
    `ifdef TCU_WLOCK_ENABLE
        ibuf_r.fu_lock   = (uop_idx == UOP_CTR_W'(0));
        ibuf_r.fu_unlock = (uop_idx == (uop_count - UOP_CTR_W'(1)));
    `endif
    end

    assign ibuf_out = ibuf_r;

endmodule
