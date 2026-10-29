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
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Counter sized strictly for the MMA expansion (TCU_LD handles metadata preload).
    localparam MAX_UOPS = SYM_SPARSE ? TCU_UOPS : (TCU_UOPS / 2);
`else
    localparam MAX_UOPS = TCU_UOPS;
`endif

    localparam LG_N = $clog2(TCU_N_STEPS);
    localparam LG_M = $clog2(TCU_M_STEPS);
    localparam LG_K = $clog2(TCU_K_STEPS);

    // Index extractions below assume eff_ctr is at least
    // LG_N + LG_M + LG_K bits wide (k_index slice covers the top end).
    // For configs where the sparse-only MAX_UOPS encodes fewer bits than
    // the index range (NT=8 non-sym sparse: MAX_UOPS=16 → 4 bits, but
    // LG_N+LG_M+LG_K=5), pad CTR_W up so the part-selects stay in-range.
    // The extra bit reads as zero (the counter never reaches it under the
    // sparse uop count).
    localparam IDX_BITS = LG_N + LG_M + LG_K;
`ifdef VX_CFG_TCU_WGMMA_ENABLE
  `ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam TCU_WG_K_STEPS_SP = (TCU_WG_K_STEPS > 1) ? (TCU_WG_K_STEPS / 2) : 1;
    localparam MAX_WG_UOPS_SP = TCU_WG_M_STEPS * TCU_WG_N_STEPS * TCU_WG_K_STEPS_SP;
  `ifdef VX_CFG_TCU_FEDP2K
    localparam MAX_WG_UOPS_DENSE = TCU_WG_UOPS + 1;
  `else
    localparam MAX_WG_UOPS_DENSE = TCU_WG_UOPS;
  `endif
    localparam CTR_W_MAX_WMMA = MAX_UOPS > MAX_WG_UOPS_DENSE ? MAX_UOPS : MAX_WG_UOPS_DENSE;
    localparam CTR_W_BASE0 = $clog2(CTR_W_MAX_WMMA > MAX_WG_UOPS_SP ? CTR_W_MAX_WMMA : MAX_WG_UOPS_SP);
  `else
  `ifdef VX_CFG_TCU_FEDP2K
    localparam MAX_WG_UOPS_DENSE = TCU_WG_UOPS + 1;
  `else
    localparam MAX_WG_UOPS_DENSE = TCU_WG_UOPS;
  `endif
    localparam CTR_W_BASE0 = $clog2(MAX_UOPS > MAX_WG_UOPS_DENSE ? MAX_UOPS : MAX_WG_UOPS_DENSE);
  `endif
`else
    localparam CTR_W_BASE0 = $clog2(MAX_UOPS);
`endif
    // UMMA's own NRC ceiling (128) exceeds WGMMA's (32), so its worst-case
    // uop count (128 * TCU_WG_K_STEPS, up to 256) needs its own entry
`ifdef VX_CFG_TCU_TMEM_ENABLE
    localparam MAX_UMMA_UOPS = 128 * TCU_WG_K_STEPS;
    localparam CTR_W_UMMA = $clog2(MAX_UMMA_UOPS);
    localparam CTR_W_BASE = (CTR_W_BASE0 > CTR_W_UMMA) ? CTR_W_BASE0 : CTR_W_UMMA;
`else
    localparam CTR_W_BASE = CTR_W_BASE0;
`endif
    localparam CTR_W = (CTR_W_BASE > IDX_BITS) ? CTR_W_BASE : IDX_BITS;
    `STATIC_ASSERT (CTR_W <= UOP_CTR_W, ("invalid parameter"))

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam LG_A_SB = $clog2(TCU_A_SUB_BLOCKS);
`endif
    localparam LG_B_SB = $clog2(TCU_B_SUB_BLOCKS);

    `UNUSED_VAR ({clk, reset, start, advance, uop_idx})

    // Truncate the wide uop_idx to the bits this expander actually uses.
    wire [`UP(CTR_W)-1:0] ctr = `UP(CTR_W)'(uop_idx);

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // is_sparse is derived from op_type (INST_TCU_*_SP variants).
    wire is_wgmma = (ibuf_in.op_type == INST_TCU_WGMMA)
              `ifdef VX_CFG_TCU_SPARSE_ENABLE
                 || (ibuf_in.op_type == INST_TCU_WGMMA_SP)
              `endif
                 ;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire wg_is_sparse = (ibuf_in.op_type == INST_TCU_WGMMA_SP);
`endif
    wire wg_a_from_smem = ibuf_in.op_args.tcu.a_from_smem;
    wire wg_needs_setup =
        `ifdef VX_CFG_TCU_FEDP2K
            !wg_a_from_smem
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            && !wg_is_sparse
        `endif
        `else
            1'b0
        `endif
            ;

    // Variable NRC based on cd_nregs: 0→8, 1→16, 2→32
    // Loop order: m (inner) → n → k (outer)  [K-outer]
    // m_steps=2; k_steps is 2 normally and 1 with FEDP2K.
    // K-outer lets independent (m,n) tiles overlap FEDP latency.
    localparam LG_M_WG = $clog2(TCU_WG_M_STEPS);  // 1
    localparam LG_K_WG = $clog2(TCU_WG_K_STEPS);   // 1
    localparam LG_N_WG_MAX = $clog2(TCU_WG_N_STEPS); // 4 (for NRC=32)

    // Pre-computed (m×n) uop counts per k-step for each cd_nregs value
    localparam WG_MN_NR8  =  8;   // nrc=8:  n_steps=4,  mn=8
    localparam WG_MN_NR16 = 16;   // nrc=16: n_steps=8,  mn=16
    localparam WG_MN_NR32 = 32;   // nrc=32: n_steps=16, mn=32

    // Pre-computed compute uop counts for each cd_nregs value (dense)
    localparam WG_UOPS_NR8  =  WG_MN_NR8  * TCU_WG_K_STEPS;
    localparam WG_UOPS_NR16 = WG_MN_NR16 * TCU_WG_K_STEPS;
    localparam WG_UOPS_NR32 = WG_MN_NR32 * TCU_WG_K_STEPS;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    localparam WG_UOPS_SP_NR8  =  WG_MN_NR8  * TCU_WG_K_STEPS_SP;
    localparam WG_UOPS_SP_NR16 = WG_MN_NR16 * TCU_WG_K_STEPS_SP;
    localparam WG_UOPS_SP_NR32 = WG_MN_NR32 * TCU_WG_K_STEPS_SP;
`endif

    // Mux uop count based on cd_nregs: 0→8, 1→16, 2→32
    reg [UOP_CTR_W-1:0] wg_uop_cnt;
    reg [UOP_CTR_W-1:0] wg_compute_uop_cnt;
    always_comb begin
        case (ibuf_in.op_args.tcu.cd_nregs)
            3'd0: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_NR8);
            3'd1: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_NR16);
            default: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_NR32);
        endcase
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        if (wg_is_sparse) begin
            case (ibuf_in.op_args.tcu.cd_nregs)
                3'd0: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_SP_NR8);
                3'd1: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_SP_NR16);
                default: wg_compute_uop_cnt = UOP_CTR_W'(WG_UOPS_SP_NR32);
            endcase
            wg_uop_cnt = wg_compute_uop_cnt;
        end else
    `endif
            wg_uop_cnt = wg_compute_uop_cnt + UOP_CTR_W'(wg_needs_setup);
    end

    // K-outer index extraction:
    //   Dense:  ctr = k * (n_steps * m_steps) + n * m_steps + m
    //   Sparse: ctr = n * m_steps + m  (k always 0)
    // Since n_steps varies by cd_nregs, k-index bit position shifts.
    // m is always bit 0 (m_steps=2). n and k extracted via mux.
    wire is_wg_setup_uop = wg_needs_setup && (ctr == '0);
    wire [`UP(CTR_W)-1:0] wg_idx_ctr = wg_needs_setup
        ? (is_wg_setup_uop ? '0 : (ctr - `UP(CTR_W)'(1)))
        : ctr;
    wire is_wg_first_compute_uop = !is_wg_setup_uop && (wg_idx_ctr == '0);
    wire [`UP(LG_M_WG)-1:0] wg_m_index = wg_idx_ctr[0 +: `UP(LG_M_WG)];
    reg [`UP(LG_K_WG)-1:0] wg_k_index;
    reg [`UP(LG_N_WG_MAX)-1:0] wg_n_index;
    always_comb begin
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        if (wg_is_sparse) begin
            // Sparse: k always 0, n right after m
            wg_k_index = '0;
            wg_n_index = `UP(LG_N_WG_MAX)'(wg_idx_ctr[LG_M_WG +: `UP(LG_N_WG_MAX)]);
        end else
    `endif
        begin
            // Dense K-outer: m (inner) → n (middle) → k (outer)
            // n width varies by cd_nregs; k bit shifts accordingly.
            case (ibuf_in.op_args.tcu.cd_nregs)
                3'd0: begin // nrc=8, n_steps=4, LG_N=2
                    wg_n_index = `UP(LG_N_WG_MAX)'(wg_idx_ctr[LG_M_WG +: 2]);
                    wg_k_index = `UP(LG_K_WG)'(wg_idx_ctr[LG_M_WG + 2 +: `UP(LG_K_WG)]);
                end
                3'd1: begin // nrc=16, n_steps=8, LG_N=3
                    wg_n_index = `UP(LG_N_WG_MAX)'(wg_idx_ctr[LG_M_WG +: 3]);
                    wg_k_index = `UP(LG_K_WG)'(wg_idx_ctr[LG_M_WG + 3 +: `UP(LG_K_WG)]);
                end
                default: begin // nrc=32, n_steps=16, LG_N=4
                    wg_n_index = `UP(LG_N_WG_MAX)'(wg_idx_ctr[LG_M_WG +: LG_N_WG_MAX]);
                    wg_k_index = `UP(LG_K_WG)'(wg_idx_ctr[LG_M_WG + LG_N_WG_MAX +: `UP(LG_K_WG)]);
                end
            endcase
        end
    end

    // Accumulator register index: n * m_steps + m  (n-major layout)
    wire [4:0] wg_rs3_off = (5'(wg_n_index) << LG_M_WG) | 5'(wg_m_index);
    // Fixed A register base for RS mode: f24..f27
    localparam [4:0] wg_ra_base = TCU_WG_RA;

    // Register offsets for from-reg mode
    // Dense A: rs1_off = m * 2 + k  (NRA=4 regs at ra_base: f24..f27)
    // Sparse A: rs1_off = m  (NRA compressed to 2 regs f24,f25; f26,f27 hold metadata)
    wire [`UP(CTR_W)-1:0] wg_rs1_reg_off_dense = (`UP(CTR_W)'(wg_m_index) << 1) | `UP(CTR_W)'(wg_k_index);
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    wire [`UP(CTR_W)-1:0] wg_rs1_reg_off = (wg_is_sparse && !wg_a_from_smem)
        ? `UP(CTR_W)'(wg_m_index)
        : wg_rs1_reg_off_dense;
`else
    wire [`UP(CTR_W)-1:0] wg_rs1_reg_off = wg_rs1_reg_off_dense;
`endif
    if (`UP(CTR_W) > 5) begin : g_unused_wg_rs1_off
        `UNUSED_VAR (wg_rs1_reg_off[`UP(CTR_W)-1 : 5])
    end
`endif

`ifdef VX_CFG_TCU_TMEM_ENABLE
    // UMMA: A/B from SMEM via descriptors (fixed regs x10/x11, same as
    // WGMMA's smem-A convention), C/D accumulator in TMEM addressed via a
    // handle carried in a fixed reg (x12)
    wire is_umma = (ibuf_in.op_type == INST_TCU_UMMA);
    wire is_first_uop_umma = (ctr == '0);

    // NRC encoding (0..4 -> 8/16/32/64/128), decoded from cd_nregs
    wire [2:0] umma_nrc_enc = ibuf_in.op_args.tcu.cd_nregs;

    localparam LG_N_UMMA_MAX = $clog2(TCU_UMMA_N_STEPS); // 6 (NRC=128)

    localparam UMMA_UOPS_NR8   =   8 * TCU_WG_K_STEPS;
    localparam UMMA_UOPS_NR16  =  16 * TCU_WG_K_STEPS;
    localparam UMMA_UOPS_NR32  =  32 * TCU_WG_K_STEPS;
    localparam UMMA_UOPS_NR64  =  64 * TCU_WG_K_STEPS;
    localparam UMMA_UOPS_NR128 = 128 * TCU_WG_K_STEPS;

    reg [UOP_CTR_W-1:0] umma_uop_cnt;
    always_comb begin
        case (umma_nrc_enc)
            3'd0: umma_uop_cnt = UOP_CTR_W'(UMMA_UOPS_NR8);
            3'd1: umma_uop_cnt = UOP_CTR_W'(UMMA_UOPS_NR16);
            3'd2: umma_uop_cnt = UOP_CTR_W'(UMMA_UOPS_NR32);
            3'd3: umma_uop_cnt = UOP_CTR_W'(UMMA_UOPS_NR64);
            default: umma_uop_cnt = UOP_CTR_W'(UMMA_UOPS_NR128);
        endcase
    end

    // K-outer index extraction, same shape as WGMMA's:
    //   ctr = k * (n_steps * m_steps) + n * m_steps + m
    // m is always bit 0 (m_steps=2); n's width varies by NRC, so k's bit
    // position shifts accordingly
    wire [`UP(LG_M_WG)-1:0] umma_m_index = ctr[0 +: `UP(LG_M_WG)];
    reg [`UP(LG_K_WG)-1:0] umma_k_index;
    reg [`UP(LG_N_UMMA_MAX)-1:0] umma_n_index;
    always_comb begin
        case (umma_nrc_enc)
            3'd0: begin // NRC=8, n_steps=4
                umma_n_index = `UP(LG_N_UMMA_MAX)'(ctr[LG_M_WG +: 2]);
                umma_k_index = `UP(LG_K_WG)'(ctr >> (LG_M_WG + 2));
            end
            3'd1: begin // NRC=16, n_steps=8
                umma_n_index = `UP(LG_N_UMMA_MAX)'(ctr[LG_M_WG +: 3]);
                umma_k_index = `UP(LG_K_WG)'(ctr >> (LG_M_WG + 3));
            end
            3'd2: begin // NRC=32, n_steps=16
                umma_n_index = `UP(LG_N_UMMA_MAX)'(ctr[LG_M_WG +: 4]);
                umma_k_index = `UP(LG_K_WG)'(ctr >> (LG_M_WG + 4));
            end
            3'd3: begin // NRC=64, n_steps=32
                umma_n_index = `UP(LG_N_UMMA_MAX)'(ctr[LG_M_WG +: 5]);
                umma_k_index = `UP(LG_K_WG)'(ctr >> (LG_M_WG + 5));
            end
            default: begin // NRC=128, n_steps=64
                umma_n_index = `UP(LG_N_UMMA_MAX)'(ctr[LG_M_WG +: 6]);
                umma_k_index = `UP(LG_K_WG)'(ctr >> (LG_M_WG + 6));
            end
        endcase
    end
`endif

`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // is_sparse covers WMMA_SP (and WGMMA_SP, since the WGMMA path branches earlier).
    wire is_sparse = (ibuf_in.op_type == INST_TCU_WMMA_SP)
              `ifdef VX_CFG_TCU_WGMMA_ENABLE
                  || (ibuf_in.op_type == INST_TCU_WGMMA_SP)
              `endif
                  ;
`endif

    // Dense FEDP2K RS adds one descriptor setup uop. Other WGMMA modes fuse
    // descriptor transport into their first compute uop.
    assign uop_count =
`ifdef VX_CFG_TCU_TMEM_ENABLE
        is_umma ? umma_uop_cnt :
`endif
`ifdef VX_CFG_TCU_WGMMA_ENABLE
        is_wgmma ? wg_uop_cnt :
`endif
`ifdef VX_CFG_TCU_SPARSE_ENABLE
        is_sparse ? (SYM_SPARSE ? UOP_CTR_W'(TCU_UOPS) : UOP_CTR_W'(TCU_UOPS / 2)) :
`endif
        UOP_CTR_W'(TCU_UOPS);

    wire [`UP(CTR_W)-1:0] eff_ctr = ctr;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    // Parametric symmetric tmask for sparse mode
    // sym_mask_lo[t] = 1 for threads where (t % tcN) < (tcN/2)
    logic [`VX_CFG_NUM_THREADS-1:0] sym_mask_lo;
    if (SYM_SPARSE) begin : g_sym_mask
        for (genvar t = 0; t < `VX_CFG_NUM_THREADS; ++t) begin : g_bit
            assign sym_mask_lo[t] = ((t % TCU_TC_N) < (TCU_TC_N / 2)) ? 1'b1 : 1'b0;
        end
    end else begin : g_sym_mask
        assign sym_mask_lo = '0;
    end
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

`ifdef VX_CFG_TCU_SPARSE_ENABLE
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
`ifdef VX_CFG_TCU_SPARSE_ENABLE
    logic [3:0] n_sp_s;
    logic [3:0] m_sp_s;
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        n_sp_s = '0;
        m_sp_s = '0;
    `endif
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        if (is_umma) begin
            ibuf_r.op_args.tcu.step_m = 2'(umma_m_index);
            ibuf_r.op_args.tcu.step_n = 7'(umma_n_index);
            ibuf_r.op_args.tcu.step_k = 2'(umma_k_index);
            // No register writeback
            ibuf_r.wb = 1'b0;
            // A descriptor (x10), B descriptor (x11), TMEM handle (x12)
            // all fixed integer regs, read once on the first uop only
            ibuf_r.rs1 = make_reg_num(REG_TYPE_I, 5'd10);
            ibuf_r.rs2 = make_reg_num(REG_TYPE_I, 5'd11);
            ibuf_r.rs3 = make_reg_num(REG_TYPE_I, 5'd12);
            ibuf_r.used_rs[0] = is_first_uop_umma;
            ibuf_r.used_rs[1] = is_first_uop_umma;
            ibuf_r.used_rs[2] = is_first_uop_umma;
        end else
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        if (is_wgmma) begin
            ibuf_r.op_args.tcu.step_m = is_wg_setup_uop ? 2'd0 : 2'(wg_m_index);
            ibuf_r.op_args.tcu.step_n = is_wg_setup_uop ? 7'd0 : 7'(wg_n_index);
            ibuf_r.op_args.tcu.step_k = is_wg_setup_uop ? 2'd0 : 2'(wg_k_index);
            ibuf_r.wb  = !is_wg_setup_uop;
            ibuf_r.rd  = make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off);
            ibuf_r.rs3 = make_reg_num(REG_TYPE_F, TCU_WG_RC + wg_rs3_off);
            ibuf_r.used_rs[2] = !is_wg_setup_uop;
            if (is_wg_setup_uop) begin
                ibuf_r.used_rs[0] = 1'b0;
            end else if (is_wg_first_compute_uop && wg_a_from_smem) begin
                ibuf_r.rs1 = make_reg_num(REG_TYPE_I, 5'd10);
                ibuf_r.used_rs[0] = 1'b1;
            end else if (wg_a_from_smem) begin
                ibuf_r.rs1 = make_reg_num(REG_TYPE_I, 5'd10);
                ibuf_r.used_rs[0] = 1'b0;
            end else begin
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, wg_ra_base + 5'(wg_rs1_reg_off));
                ibuf_r.used_rs[0] = 1'b1;
            end
            // Dense FEDP2K RS reserves rs2 for upper A and transports the B
            // descriptor in its setup uop. Other modes read x11 on first compute.
            ibuf_r.rs2 = make_reg_num(REG_TYPE_I, 5'd11);
            if (is_wg_setup_uop || (!wg_needs_setup && is_wg_first_compute_uop)) begin
                ibuf_r.used_rs[1] = 1'b1;
            end else if (wg_needs_setup) begin
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, wg_ra_base + 5'(wg_rs1_reg_off) + 5'd1);
                ibuf_r.used_rs[1] = 1'b1;
            end else begin
                ibuf_r.used_rs[1] = 1'b0;
            end
        end else
    `endif
        begin
    `ifdef VX_CFG_TCU_SPARSE_ENABLE
        if (SYM_SPARSE) begin
            ibuf_r.tmask = is_sparse
                ? (eff_ctr[0] ? ibuf_in.tmask & ~sym_mask_lo : ibuf_in.tmask &  sym_mask_lo)
                : ibuf_in.tmask;
            n_sp_s = 4'(eff_ctr[0 +: (LG_N + LG_K)]);
            m_sp_s = 4'(eff_ctr[(LG_N + LG_K) +: LG_M]);
        end

        ibuf_r.op_args.tcu.step_m = SYM_SPARSE && is_sparse ? 2'(m_sp_s) : 2'(m_index);
        ibuf_r.op_args.tcu.step_n = SYM_SPARSE && is_sparse ? 7'(n_sp_s >> LG_K) : 7'(n_index);
        ibuf_r.op_args.tcu.step_k = SYM_SPARSE && is_sparse ? 2'(0)      : 2'(k_index);
        ibuf_r.wb = 1'b1;
        ibuf_r.rd = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.rs1 = make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = make_reg_num(REG_TYPE_F, rs2);
        ibuf_r.rs3 = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = 1'b1;
    `else
        ibuf_r.op_args.tcu.step_m = 2'(m_index);
        ibuf_r.op_args.tcu.step_n = 7'(n_index);
        ibuf_r.op_args.tcu.step_k = 2'(k_index);
        ibuf_r.wb  = 1'b1;
        ibuf_r.rd  = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.rs1 = make_reg_num(REG_TYPE_F, rs1);
        ibuf_r.rs2 = make_reg_num(REG_TYPE_F, rs2);
        ibuf_r.rs3 = make_reg_num(REG_TYPE_F, rs3);
        ibuf_r.used_rs[0] = 1'b1;
        ibuf_r.used_rs[1] = 1'b1;
        ibuf_r.used_rs[2] = 1'b1;
    `endif
        end
    `ifdef VX_CFG_TCU_TMEM_ENABLE
        if (is_umma) begin
            ibuf_r.fu_lock   = (uop_idx == '0);
            ibuf_r.fu_unlock = (uop_idx == (uop_count - UOP_CTR_W'(1)));
            // No is_wg_setup_uop-equivalent exclusion needed: that only exists for
            // WGMMA's RS mode
            ibuf_r.op_args.tcu.is_first_uop = is_first_uop_umma;
            ibuf_r.op_args.tcu.is_last_uop  = (uop_idx == (uop_count - UOP_CTR_W'(1)));
        end else
    `endif
    `ifdef VX_CFG_TCU_WGMMA_ENABLE
        if (is_wgmma) begin
            ibuf_r.fu_lock   = (uop_idx == '0);
            ibuf_r.fu_unlock = (uop_idx == (uop_count - UOP_CTR_W'(1)));
            // Expose first/last-uop predicates via op_args.tcu so downstream
            // consumers (bbuf, lockstep) read a single source of truth.
            ibuf_r.op_args.tcu.is_first_uop = is_wg_first_compute_uop;
            ibuf_r.op_args.tcu.is_last_uop  = !is_wg_setup_uop && (uop_idx == (uop_count - UOP_CTR_W'(1)));
        end
    `endif
        // Other TCU uops retain 11 so they cannot enter a locked WGMMA sequence.
    end

    assign ibuf_out = ibuf_r;

endmodule
