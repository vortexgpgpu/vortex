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

`ifndef VX_TCU_PKG_VH
`define VX_TCU_PKG_VH

`include "VX_define.vh"

`IGNORE_UNUSED_BEGIN

package VX_tcu_pkg;

    import VX_gpu_pkg::*;

    // WARNING: Changing this list requires updating format utility functions below
    // Supported floating-point types (prefer setting bfloat versions to odd)
    localparam TCU_FP32_ID  = 0;
    localparam TCU_TF32_ID  = 1;
    localparam TCU_FP16_ID  = 2;
    localparam TCU_BF16_ID  = 3;
    localparam TCU_FP8_ID   = 4;
    localparam TCU_BF8_ID   = 5;
    // MX formats have 2nd-MSB set
    localparam TCU_MXFP8_ID = 8;
    localparam TCU_MXBF8_ID = 9;
    localparam TCU_MXFP4_ID = 10;
    localparam TCU_NVFP4_ID = 11;
    // Supported integer-point types (prefer setting unsigned versions to even)
    localparam TCU_I32_ID   = 16;
    localparam TCU_I8_ID    = 17;
    localparam TCU_U8_ID    = 18;
    localparam TCU_I4_ID    = 19;
    localparam TCU_U4_ID    = 20;
    localparam TCU_FMT_WIDTH= 5;

    // Set configuration parameters
    localparam TCU_NT = `VX_CFG_NUM_THREADS;

    localparam TCU_WG_NRA = 4;  // A registers per warp (fixed)
    localparam TCU_WG_NR = 32;  // max NRC (C/D registers, variable via cd_nregs)

    localparam TCU_NR = 8;
    localparam TCU_DK = 0;
    localparam TCU_DP = 0;

    // Tile dimensions
    localparam TCU_TILE_CAP = TCU_NT * TCU_NR;
    localparam TCU_LG_TILE_CAP = $clog2(TCU_TILE_CAP);
    localparam TCU_TILE_EN = TCU_LG_TILE_CAP / 2;
    localparam TCU_TILE_EM = TCU_LG_TILE_CAP - TCU_TILE_EN;

    localparam TCU_TILE_M = 1 << TCU_TILE_EM;
    localparam TCU_TILE_N = 1 << TCU_TILE_EN;
    localparam TCU_TILE_K = (TCU_DK != 0) ? TCU_DK : (TCU_DP != 0) ? TCU_DP : (TCU_TILE_CAP / ((TCU_TILE_M > TCU_TILE_N) ? TCU_TILE_M : TCU_TILE_N));

    // Block dimensions
    localparam TCU_BLOCK_CAP = TCU_NT;
    localparam TCU_LG_BLOCK_CAP = $clog2(TCU_BLOCK_CAP);
    localparam TCU_BLOCK_EN = TCU_LG_BLOCK_CAP / 2;
    localparam TCU_BLOCK_EM = TCU_LG_BLOCK_CAP - TCU_BLOCK_EN;

    localparam TCU_TC_M = 1 << TCU_BLOCK_EM;
    localparam TCU_TC_N = 1 << TCU_BLOCK_EN;
    localparam TCU_TC_K = (TCU_DP != 0) ? TCU_DP : (TCU_BLOCK_CAP / ((TCU_TC_M > TCU_TC_N) ? TCU_TC_M : TCU_TC_N));

    // Step counts
    localparam TCU_M_STEPS = TCU_TILE_M / TCU_TC_M;
    localparam TCU_N_STEPS = TCU_TILE_N / TCU_TC_N;
    localparam TCU_K_STEPS = TCU_TILE_K / TCU_TC_K;

    // A micro-tiling
    localparam TCU_A_BLOCK_SIZE = TCU_TC_M * TCU_TC_K;
    localparam TCU_A_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_A_BLOCK_SIZE;

    // B micro-tiling (dense)
    localparam TCU_B_BLOCK_SIZE = TCU_TC_K * TCU_TC_N;
    localparam TCU_B_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_B_BLOCK_SIZE;

    // WGMMA per-warp tile dimensions (NRA=4 fixed, NRC=NR variable).
    // Derived from block geometry: xtileM = 2*tcM, xtileK = 2*tcK.
    // Dense FEDP width doubles under VX_CFG_TCU_FEDP2K
    localparam TCU_WG_TILE_M = 2 * TCU_TC_M;
    localparam TCU_WG_TILE_K = 2 * TCU_TC_K;
`ifdef VX_CFG_TCU_FEDP2K
    localparam TCU_WG_FEDP_K = 2 * TCU_TC_K;
`else
    localparam TCU_WG_FEDP_K = TCU_TC_K;
`endif
    localparam TCU_WG_TILE_N = (TCU_WG_NR * TCU_NT) / TCU_WG_TILE_M;

    // WG step counts: block geometry (TC_M/TC_N/TC_K) unchanged, tile is larger
    localparam TCU_WG_M_STEPS = TCU_WG_TILE_M / TCU_TC_M;
    localparam TCU_WG_N_STEPS = TCU_WG_TILE_N / TCU_TC_N;
    localparam TCU_WG_K_STEPS = TCU_WG_TILE_K / TCU_WG_FEDP_K;

    localparam TCU_WG_UOPS = TCU_WG_M_STEPS * TCU_WG_N_STEPS * TCU_WG_K_STEPS;

    // WG A/B micro-tiling (block geometry is shared with non-WG)
    localparam TCU_WG_A_BLOCK_SIZE = TCU_TC_M * TCU_TC_K;
    localparam TCU_WG_A_DATA_SIZE  = TCU_TC_M * TCU_WG_FEDP_K;
    localparam TCU_WG_A_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_WG_A_BLOCK_SIZE;

    localparam TCU_WG_B_BLOCK_SIZE = TCU_WG_FEDP_K * TCU_TC_N;
    localparam TCU_WG_B_SUB_BLOCKS = TCU_BLOCK_CAP / TCU_WG_B_BLOCK_SIZE;

    // WGMMA shared-memory descriptor: the slab's local-memory byte offset,
    // and above it the row stride in 32-bit words.
    localparam TCU_SMEM_DESC_OFF_BITS = 18;

    // Symmetric sparse flag (NT=4, NT=16: block_em == block_en). It selects the
    // WMMA_SP layout only: WGMMA_SP always reads the full interleaved layout.
    localparam SYM_SPARSE = (TCU_BLOCK_EM == TCU_BLOCK_EN);

    // B micro-tiling (sparse 2:4)
    // NT=8/32: standard interleaved layout (tcK × tcN × 2 = NT lanes per block)
    // NT=16 (SYM_SPARSE): WMMA_SP uses column-pair layout (2 cols × tcK × 2 = NT lanes);
    //   WGMMA_SP needs the full tcK × tcN × 2 candidate lanes (may exceed TCU_BLOCK_CAP).
    localparam TCU_B_BLOCK_SIZE_SP    = SYM_SPARSE ? TCU_BLOCK_CAP : (TCU_TC_K * TCU_TC_N) * 2;
    localparam TCU_B_SUB_BLOCKS_SP    = TCU_BLOCK_CAP / TCU_B_BLOCK_SIZE_SP;
    // WGMMA_SP always needs the full candidate lane set, regardless of SYM_SPARSE.
    localparam TCU_WG_B_BLOCK_SIZE_SP = TCU_TC_K * TCU_TC_N * 2;
    localparam TCU_WG_RS2_WIDTH_DENSE =
        (TCU_WG_B_BLOCK_SIZE > TCU_BLOCK_CAP) ? TCU_WG_B_BLOCK_SIZE : TCU_BLOCK_CAP;
    localparam TCU_WG_RS2_WIDTH = `VX_CFG_TCU_SPARSE_ENABLED
        ? ((TCU_WG_B_BLOCK_SIZE_SP > TCU_WG_RS2_WIDTH_DENSE) ? TCU_WG_B_BLOCK_SIZE_SP : TCU_WG_RS2_WIDTH_DENSE)
        : TCU_WG_RS2_WIDTH_DENSE;

    localparam TCU_MIN_FMT_WIDTH = 4; //int4
    localparam TCU_MAX_ELT_RATIO = 32 / TCU_MIN_FMT_WIDTH;

    // Max metadata widths (sized for widest type: 4-bit elements, I_RATIO=8)
    localparam TCU_MAX_META_ROW_WIDTH   = TCU_TC_K * 2 * TCU_MAX_ELT_RATIO;
    localparam TCU_MAX_META_BLOCK_WIDTH = TCU_NT   * 2 * TCU_MAX_ELT_RATIO;

    // Meta-store micro-op expansion parameters (WMMA)
    localparam TCU_META_PER_WARP_DEPTH = TCU_M_STEPS * (TCU_K_STEPS / 2);
    localparam TCU_META_COLS_PER_LOAD  = (TCU_BLOCK_CAP >= TCU_META_PER_WARP_DEPTH)
        ? (TCU_BLOCK_CAP / TCU_META_PER_WARP_DEPTH) : 1;

    // Partial-bank write parameters (NT < PER_WARP_DEPTH)
    localparam TCU_BANKS_PER_STORE = (TCU_NT < TCU_META_PER_WARP_DEPTH)
        ? TCU_NT : TCU_META_PER_WARP_DEPTH;
    localparam TCU_STORES_PER_COL = (TCU_META_PER_WARP_DEPTH + TCU_NT - 1) / TCU_NT;

`ifdef VX_CFG_TCU_WGMMA_ENABLE
    // Meta-store micro-op expansion parameters (WGMMA RS sparse)
    localparam TCU_WG_META_PER_WARP_DEPTH = TCU_WG_M_STEPS * (TCU_WG_K_STEPS / 2);
    localparam TCU_WG_META_COLS_PER_LOAD  = (TCU_BLOCK_CAP >= TCU_WG_META_PER_WARP_DEPTH)
        ? (TCU_BLOCK_CAP / TCU_WG_META_PER_WARP_DEPTH) : 1;
    localparam TCU_WG_STORES_PER_COL = (TCU_WG_META_PER_WARP_DEPTH + TCU_NT - 1) / TCU_NT;
`endif

    // Register counts
    localparam TCU_NRA = (TCU_TILE_M * TCU_TILE_K) / TCU_NT;
    localparam TCU_NRB = (TCU_TILE_N * TCU_TILE_K) / TCU_NT;
    localparam TCU_NRC = (TCU_TILE_M * TCU_TILE_N) / TCU_NT;

    // Register base addresses
    localparam TCU_RC    = 0;
    localparam TCU_WG_RC = TCU_RC;  // WGMMA C accumulator starts at same base
    localparam TCU_WG_RA = 24;     // WGMMA A register base (fixed f24..f27)
    localparam TCU_RA = 10;
    localparam TCU_RB = (TCU_NRB == 4) ? 28 : 24;

    localparam TCU_UOPS = TCU_M_STEPS * TCU_N_STEPS * TCU_K_STEPS;

    localparam TCU_MAX_INPUTS = TCU_TC_K * TCU_MAX_ELT_RATIO;

    localparam TCU_LG_M    = $clog2(TCU_M_STEPS);
    localparam TCU_LG_N    = $clog2(TCU_N_STEPS);
    localparam TCU_LG_K    = $clog2(TCU_K_STEPS);
    localparam TCU_LG_B_SB = $clog2(TCU_B_SUB_BLOCKS);

    // Register offsets of the WMMA operands of step (m, n, k), permuted so the
    // three operands of a step never share a register bank.
    //   one B sub-block:   A={m[0],~m[hi],k}, B={n^k,~k}, C={m[0],~m[hi],XNOR(m[hi],n)}
    //   two B sub-blocks:  A={k[0],~m,m^k[hi]}, B={k[0],k[hi]^np,~np}, C={n[0],~m,n[hi]}
    //   asymmetric sparse: B={n[hi],~(n[0]^k),~k}, C={n[hi],m,~(m^n[0])}
    function automatic logic [2:0] tcu_ra_off(input logic [`UP(TCU_LG_M)-1:0] m,
                                              input logic [`UP(TCU_LG_K)-1:0] k);
        if (TCU_LG_B_SB == 0) begin
            return 3'({m[0], ~m[`UP(TCU_LG_M)-1], k[0]});
        end else begin
            return 3'({k[0], ~m[0], m[0] ^ k[`UP(TCU_LG_K)-1]});
        end
    endfunction

    function automatic logic [2:0] tcu_rb_off(input logic [`UP(TCU_LG_N)-1:0] n,
                                              input logic [`UP(TCU_LG_K)-1:0] k);
        if (TCU_LG_B_SB == 0) begin
            return 3'({n[0] ^ k[0], ~k[0]});
        end else begin
            return 3'({k[0], k[`UP(TCU_LG_K)-1] ^ n[`UP(TCU_LG_N)-1], ~n[`UP(TCU_LG_N)-1]});
        end
    endfunction

    function automatic logic [2:0] tcu_rc_off(input logic [`UP(TCU_LG_M)-1:0] m,
                                              input logic [`UP(TCU_LG_N)-1:0] n);
        if (TCU_LG_B_SB == 0) begin
            return 3'({m[0], ~m[`UP(TCU_LG_M)-1], ~(m[`UP(TCU_LG_M)-1] ^ n[0])});
        end else begin
            return 3'({n[0], ~m[0], n[`UP(TCU_LG_N)-1]});
        end
    endfunction

    function automatic logic [2:0] tcu_rb_off_sp(input logic [`UP(TCU_LG_N)-1:0] n,
                                                 input logic [`UP(TCU_LG_K)-1:0] k);
        return 3'({n[`UP(TCU_LG_N)-1], ~(n[0] ^ k[0]), ~k[0]});
    endfunction

    function automatic logic [2:0] tcu_rc_off_sp(input logic [`UP(TCU_LG_M)-1:0] m,
                                                 input logic [`UP(TCU_LG_N)-1:0] n);
        return 3'({n[`UP(TCU_LG_N)-1], m[0], ~(m[0] ^ n[0])});
    endfunction

    // Matrix fragment load/store (vx_ldm/vx_stm) shapes: {ctx, role[1:0], trans, sparse}.
    // Uop e moves fragment register fd + e between it and x[rs1 | grp] +
    // (imm[11:5] | col), the element the WMMA/WGMMA expansion reads from it.
    localparam TCU_LDM_A    = 5'b00000; // WMMA A, row-major
    localparam TCU_LDM_A_SP = 5'b00001; // WMMA sparse A, row-major
    localparam TCU_LDM_B    = 5'b00110; // WMMA B, col-major
    localparam TCU_LDM_B_SP = 5'b00111; // WMMA symmetric sparse B, col-major
    localparam TCU_LDM_C    = 5'b01000; // WMMA C, row-major
    localparam TCU_LDM_WG_C = 5'b11000; // WGMMA C, row-major, one 8-register chunk

    // Decode replaces the shape in imm[4:0] with its index, so the per-warp
    // expander decodes 3 bits instead of 5, and marks an 8-register fragment
    // in bytesel.
    localparam TCU_LDM_IDX_A    = 3'd0;
    localparam TCU_LDM_IDX_A_SP = 3'd1;
    localparam TCU_LDM_IDX_B    = 3'd2;
    localparam TCU_LDM_IDX_B_SP = 3'd3;
    localparam TCU_LDM_IDX_C    = 3'd4;
    localparam TCU_LDM_IDX_WG_C = 3'd5;

    // Byte strides between the words a lane holds along a fragment row
    localparam TCU_LDM_LG_CS_K = $clog2(TCU_TC_K * 4);
    localparam TCU_LDM_LG_CS_N = $clog2(TCU_TC_N * 4);
    localparam TCU_LDM_LG_K_SP      = (TCU_LG_K > 0) ? (TCU_LG_K - 1) : 0;
    localparam TCU_LDM_COL_W        = TCU_LDM_LG_CS_K + 4;

    typedef struct packed {
        logic [1:0]               grp;
        logic [TCU_LDM_COL_W-1:0] col;
    } tcu_ldm_uop_t;

    function automatic logic tcu_ldm_valid(input logic [4:0] shape);
        case (shape)
            TCU_LDM_A,
            TCU_LDM_B,
            TCU_LDM_C:    return 1'b1;
            TCU_LDM_A_SP: return 1'(`VX_CFG_TCU_SPARSE_ENABLED);
            TCU_LDM_B_SP: return 1'(`VX_CFG_TCU_SPARSE_ENABLED && SYM_SPARSE);
            TCU_LDM_WG_C: return 1'(`VX_CFG_TCU_WGMMA_ENABLED);
            default:      return 1'b0;
        endcase
    endfunction

    function automatic logic [4:0] tcu_ldm_code(input logic [4:0] shape);
        case (shape)
            TCU_LDM_A:    return 5'(TCU_LDM_IDX_A);
            TCU_LDM_A_SP: return 5'(TCU_LDM_IDX_A_SP);
            TCU_LDM_B:    return 5'(TCU_LDM_IDX_B);
            TCU_LDM_B_SP: return 5'(TCU_LDM_IDX_B_SP);
            TCU_LDM_C:    return 5'(TCU_LDM_IDX_C);
            default:      return 5'(TCU_LDM_IDX_WG_C);
        endcase
    endfunction

    function automatic int unsigned tcu_ldm_count(input logic [2:0] idx);
        case (idx)
            TCU_LDM_IDX_A_SP: return 4;
            TCU_LDM_IDX_B,
            TCU_LDM_IDX_B_SP: return TCU_NRB;
            default:          return 8;
        endcase
    endfunction

    function automatic logic tcu_ldm_is8(input logic [4:0] shape);
        return (tcu_ldm_count(3'(tcu_ldm_code(shape))) == 8);
    endfunction

    // Uop e moves register fd + e, so the counter feeds the register add
    // directly; the shape only maps e to the row group and column of the element
    // the WMMA/WGMMA expansion reads from that register, by inverting
    // tcu_ra_off/tcu_rb_off/tcu_rc_off (a few inverted and XORed bits of e).
    // VX_uop_packld checks the inverse against them at elaboration.
    function automatic tcu_ldm_uop_t tcu_ldm_uop(input logic [2:0] idx, input logic [2:0] e);
        tcu_ldm_uop_t u;
        logic [2:0] m, n, k;
        u = '0;
        m = '0;
        n = '0;
        k = '0;
        case (idx)
            TCU_LDM_IDX_A: begin
                if (TCU_LG_B_SB == 0) begin // A={m[0],~m[1],k}
                    m = {1'b0, ~e[1], e[2]};
                    k = {2'b0, e[0]};
                end else begin              // A={k[0],~m,m^k[1]}
                    m = {2'b0, ~e[1]};
                    k = {1'b0, e[0] ^ ~e[1], e[2]};
                end
                u.grp = 2'(m);
                u.col = TCU_LDM_COL_W'(k) << TCU_LDM_LG_CS_K;
            end
            // registers in element order {m, k} over half the K steps
            TCU_LDM_IDX_A_SP: begin
                u.grp = 2'(e >> TCU_LDM_LG_K_SP);
                u.col = TCU_LDM_COL_W'(e & 3'((1 << TCU_LDM_LG_K_SP) - 1)) << TCU_LDM_LG_CS_K;
            end
            // a col-major B row is one of its columns (a pair with two sub-blocks)
            TCU_LDM_IDX_B: begin
                if (TCU_LG_B_SB == 0) begin // B={n^k,~k}
                    k = {2'b0, ~e[0]};
                    n = {2'b0, e[1] ^ ~e[0]};
                end else begin              // B={k[0],k[1]^np,~np}
                    n = {2'b0, ~e[0]};
                    k = {1'b0, e[1] ^ ~e[0], e[2]};
                end
                u.grp = 2'(n);
                u.col = TCU_LDM_COL_W'(k) << TCU_LDM_LG_CS_K;
            end
            // symmetric sparse: one row per register
            TCU_LDM_IDX_B_SP: begin
                u.grp = 2'(e);
            end
            TCU_LDM_IDX_C: begin
                if (TCU_LG_B_SB == 0) begin // C={m[0],~m[1],~(m[1]^n)}
                    m = {1'b0, ~e[1], e[2]};
                    n = {2'b0, e[0] ^ e[1]};
                end else begin              // C={n[0],~m,n[1]}
                    m = {2'b0, ~e[1]};
                    n = {1'b0, e[0], e[2]};
                end
                u.grp = 2'(m);
                u.col = TCU_LDM_COL_W'(n) << TCU_LDM_LG_CS_N;
            end
            // n-major registers {n, m}, two row groups
            default: begin
                u.grp = 2'(e[0]);
                u.col = TCU_LDM_COL_W'(e >> 1) << TCU_LDM_LG_CS_N;
            end
        endcase
        return u;
    endfunction

    // The register the WMMA/WGMMA expansion reads element (grp, col) of a
    // fragment from, through the forward permutations; -1 when (grp, col) is not
    // an element of the fragment.
    function automatic int tcu_ldm_reg(input logic [2:0] idx, input tcu_ldm_uop_t u);
        int unsigned g, c, ck, cn;
        g  = int'(u.grp);
        c  = int'(u.col);
        ck = c >> TCU_LDM_LG_CS_K;
        cn = c >> TCU_LDM_LG_CS_N;
        case (idx)
            TCU_LDM_IDX_A: begin
                if ((ck << TCU_LDM_LG_CS_K) != c || g >= TCU_M_STEPS || ck >= TCU_K_STEPS) return -1;
                return int'(tcu_ra_off(`UP(TCU_LG_M)'(g), `UP(TCU_LG_K)'(ck)));
            end
            TCU_LDM_IDX_A_SP: begin
                if ((ck << TCU_LDM_LG_CS_K) != c || g >= TCU_M_STEPS || ck >= (1 << TCU_LDM_LG_K_SP)) return -1;
                return int'((g << TCU_LDM_LG_K_SP) | ck);
            end
            TCU_LDM_IDX_B: begin
                if ((ck << TCU_LDM_LG_CS_K) != c || (g << TCU_LG_B_SB) >= `MAX(TCU_N_STEPS, 1 << TCU_LG_B_SB) || ck >= TCU_K_STEPS) return -1;
                return int'(tcu_rb_off(`UP(TCU_LG_N)'(g << TCU_LG_B_SB), `UP(TCU_LG_K)'(ck)));
            end
            TCU_LDM_IDX_B_SP: begin
                if (c != 0 || g >= TCU_NRB) return -1;
                return int'(g);
            end
            TCU_LDM_IDX_C: begin
                if ((cn << TCU_LDM_LG_CS_N) != c || g >= TCU_M_STEPS || cn >= TCU_N_STEPS) return -1;
                return int'(tcu_rc_off(`UP(TCU_LG_M)'(g), `UP(TCU_LG_N)'(cn)));
            end
            default: begin
                if ((cn << TCU_LDM_LG_CS_N) != c || g >= 2 || cn >= 4) return -1;
                return int'((cn << 1) | g);
            end
        endcase
    endfunction

    function automatic int unsigned mx_fedp_sf_count(
        input int unsigned data_bits,
        input int unsigned block_elems
    );
        automatic int unsigned sparse_ratio = `VX_CFG_TCU_SPARSE_ENABLED ? 2 : 1;
        automatic int unsigned fedp_elems = TCU_WG_FEDP_K * (32 / data_bits) * sparse_ratio;
        return (fedp_elems + block_elems - 1) / block_elems;
    endfunction

    // Per-format scale-factor counts, folded into TCU_MX_MAX_SF below.
    //
    // These are localparams rather than the one zero-argument helper function
    // they replace, because sv2v miscompiles a ZERO-ARGUMENT function while
    // folding an argument-taking one correctly: mx_fedp_sf_count() above
    // disappears from the generated Verilog entirely, but a `mx_max_fedp_sf()`
    // came out as a hierarchically-qualified call carrying a spurious argument,
    //   ...tcu_core.VX_tcu_pkg_mx_max_fedp_sf(0)
    // which Yosys rejects with "syntax error, unexpected '('". It only surfaced
    // once the module sat inside generate scopes, so the standalone `tcu` DUT
    // synthesized while `tensor` -- the same RTL nested under
    // g_cores[].tcu_unit.g_blocks[].tcu_core -- failed asic_gate before Yosys
    // read a line. Expressing the value as constants avoids that folding path.
    //
    // A disabled format contributes 1, matching the old max_sf = 1 seed, so the
    // computed value is unchanged for every configuration.
`ifdef VX_CFG_TCU_FP8_ENABLE
    localparam TCU_MX_SF_FP8 = mx_fedp_sf_count(8, 32);
`else
    localparam TCU_MX_SF_FP8 = 1;
`endif
`ifdef VX_CFG_TCU_MXFP4_ENABLE
    localparam TCU_MX_SF_MXFP4 = mx_fedp_sf_count(4, 32);
`else
    localparam TCU_MX_SF_MXFP4 = 1;
`endif
`ifdef VX_CFG_TCU_NVFP4_ENABLE
    localparam TCU_MX_SF_NVFP4 = mx_fedp_sf_count(4, 16);
`else
    localparam TCU_MX_SF_NVFP4 = 1;
`endif

    localparam TCU_MX_MAX_SF = `MAX(`MAX(TCU_MX_SF_FP8, TCU_MX_SF_MXFP4),
                                    TCU_MX_SF_NVFP4);

    // MX block scales span 2^-254 .. 2^256 on top of the elements' own range.
    `ifdef VX_CFG_TCU_TF32_ENABLE
        localparam TCU_EXP_BITS = 10;
    `elsif VX_CFG_TCU_FP16_ENABLE
        localparam TCU_EXP_BITS = 10;
    `elsif VX_CFG_TCU_MX_ENABLE
        localparam TCU_EXP_BITS = 10;
    `else
        localparam TCU_EXP_BITS = 9;
    `endif

    typedef struct packed {
        logic is_zero;
        logic is_sub;
        logic is_inf;
        logic is_nan;
    } fedp_class_t;

    typedef struct packed {
        logic is_inf;
        logic is_nan;
        logic sign;
    } fedp_excep_t;

    function automatic int exp_bits(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 8;
            TCU_FP16_ID: return 5;
            TCU_BF16_ID: return 8;
            TCU_FP8_ID:  return 4;
            TCU_BF8_ID:  return 5;
            TCU_TF32_ID: return 8;
            default:     return 0;
        endcase
    endfunction

    function automatic int sig_bits(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 23;
            TCU_FP16_ID: return 10;
            TCU_BF16_ID: return 7;
            TCU_FP8_ID:  return 3;
            TCU_BF8_ID:  return 2;
            TCU_TF32_ID: return 10;
            default:     return 0;
        endcase
    endfunction

    function automatic int sign_pos(input int fmt);
        case (fmt)
            TCU_FP32_ID: return 31;
            TCU_FP16_ID: return 15;
            TCU_BF16_ID: return 15;
            TCU_FP8_ID:  return 7;
            TCU_BF8_ID:  return 7;
            TCU_TF32_ID: return 18;
            default:     return 0;
        endcase
    endfunction

    function automatic int unsigned tcu_fmt_width(input logic [TCU_FMT_WIDTH-1:0] fmt);
        case (fmt)
            TCU_FP16_ID, TCU_BF16_ID:
                return 16;
            TCU_MXFP4_ID, TCU_NVFP4_ID, TCU_I4_ID, TCU_U4_ID:
                return 4;
            TCU_FP8_ID,
            TCU_BF8_ID,
            TCU_I8_ID,
            TCU_U8_ID,
            TCU_MXFP8_ID,
            TCU_MXBF8_ID:
                return 8;
            TCU_FP32_ID,
            TCU_I32_ID,
            TCU_TF32_ID:
                return 32;
            default:
                return 0;
        endcase
    endfunction

    function automatic logic tcu_fmt_is_int(input logic [TCU_FMT_WIDTH-1:0] fmt);
        return fmt[TCU_FMT_WIDTH-1];
    endfunction

    function automatic logic tcu_fmt_is_signed_int(input logic [TCU_FMT_WIDTH-2:0] int_fmt);
        return int_fmt[0];
    endfunction

    function automatic logic tcu_fmt_is_bfloat(input logic [TCU_FMT_WIDTH-2:0] float_fmt);
        return float_fmt[0];
    endfunction

    function automatic logic tcu_fmt_is_mx(input logic [TCU_FMT_WIDTH-1:0] fmt);
        case (fmt)
            TCU_MXFP8_ID, TCU_MXBF8_ID, TCU_MXFP4_ID, TCU_NVFP4_ID:
                return 1'b1;
            default:
                return 1'b0;
        endcase
    endfunction

    function automatic int unsigned mx_scale_block_size(input logic [TCU_FMT_WIDTH-1:0] fmt);
        case (fmt)
            TCU_MXFP8_ID, TCU_MXBF8_ID, TCU_MXFP4_ID: return 32;
            TCU_NVFP4_ID:                                          return 16;
            default:                                               return 1;
        endcase
    endfunction

    function automatic int unsigned mx_scale_blocks_k_words(
        input logic [TCU_FMT_WIDTH-1:0] fmt,
        input int unsigned tile_k_words
    );
        automatic int unsigned data_bits = tcu_fmt_width(fmt);
        automatic int unsigned block_elems = mx_scale_block_size(fmt);
        automatic int unsigned tile_elems = (data_bits != 0) ? tile_k_words * (32 / data_bits) : 0;
        return (tile_elems + block_elems - 1) / block_elems;
    endfunction

    function automatic int unsigned mx_scale_blocks_k(input logic [TCU_FMT_WIDTH-1:0] fmt);
        return mx_scale_blocks_k_words(fmt, TCU_TILE_K);
    endfunction

    function automatic logic [4:0] meta_num_cols(input logic [TCU_FMT_WIDTH-1:0] fmt);
        automatic int hw = tcu_fmt_width(fmt) / 2;
        return 5'((TCU_BLOCK_CAP + hw - 1) / hw);
    endfunction

    // Words-per-bank stride in the host's packed metadata layout:
    //   ceil((tcM * tcK * 2 * elt_ratio) / 32)
    function automatic int unsigned tcu_meta_stride_words(input logic [TCU_FMT_WIDTH-1:0] fmt);
        automatic int unsigned fb   = tcu_fmt_width(fmt);
        automatic int unsigned elr  = (fb != 0) ? (32 / fb) : 1; // input ratio
        automatic int unsigned rowb = TCU_TC_K * 2 * elr;
        return (TCU_TC_M * rowb + 31) / 32;
    endfunction

    // Tracing info
`ifdef SIMULATION
    task trace_fmt(input int level, input [TCU_FMT_WIDTH-1:0] fmt);
        case (fmt)
            TCU_FP32_ID:  `TRACE(level, ("fp32"))
            TCU_FP16_ID:  `TRACE(level, ("fp16"))
            TCU_BF16_ID:  `TRACE(level, ("bf16"))
            TCU_FP8_ID:   `TRACE(level, ("fp8"))
            TCU_BF8_ID:   `TRACE(level, ("bf8"))
            TCU_TF32_ID:  `TRACE(level, ("tf32"))
            TCU_MXFP8_ID: `TRACE(level, ("mxfp8"))
            TCU_MXBF8_ID: `TRACE(level, ("mxbf8"))
            TCU_MXFP4_ID: `TRACE(level, ("mxfp4"))
            TCU_NVFP4_ID: `TRACE(level, ("nvfp4"))
            TCU_I32_ID:   `TRACE(level, ("i32"))
            TCU_I8_ID:    `TRACE(level, ("i8"))
            TCU_U8_ID:    `TRACE(level, ("u8"))
            TCU_I4_ID:    `TRACE(level, ("i4"))
            TCU_U4_ID:    `TRACE(level, ("u4"))
            default:      `TRACE(level, ("?"))
        endcase
    endtask

    task trace_ex_op(input int level,
                     input [INST_OP_BITS-1:0] op_type,
                     input op_args_t op_args
    );
        case (INST_TCU_BITS'(op_type))
            INST_TCU_WMMA: begin
                `TRACE(level, ("WMMA."));
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%0d.%0d", op_args.tcu.step_m, op_args.tcu.step_n, op_args.tcu.step_k));
            end
        `ifdef VX_CFG_TCU_WGMMA_ENABLE
            INST_TCU_WGMMA: begin
                `TRACE(level, ("WGMMA."));
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%sS.%0d.%0d",
                    (op_args.tcu.cd_nregs == 2'd0) ? 8 : (op_args.tcu.cd_nregs == 2'd1) ? 16 : 32,
                    op_args.tcu.a_from_smem ? "S" : "R",
                    op_args.tcu.step_m, op_args.tcu.step_n));
            end
        `endif
        `ifdef TCU_META_ENABLE
            INST_TCU_LD: begin
                `TRACE(level, ("TCU_LD.%s.slot=%0d.",
                    op_args.tcu.fmt_d[4] ? "MX" : "SP", op_args.tcu.fmt_d[3:0]));
                trace_fmt(level, op_args.tcu.fmt_s);
            end
        `endif
        `ifdef VX_CFG_TCU_SPARSE_ENABLE
            INST_TCU_WMMA_SP: begin
                `TRACE(level, ("WMMA.SP."));
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%0d.%0d", op_args.tcu.step_m, op_args.tcu.step_n, op_args.tcu.step_k));
            end
          `ifdef VX_CFG_TCU_WGMMA_ENABLE
            INST_TCU_WGMMA_SP: begin
                `TRACE(level, ("WGMMA.SP."));
                trace_fmt(level, op_args.tcu.fmt_s);
                `TRACE(level, ("."));
                trace_fmt(level, op_args.tcu.fmt_d);
                `TRACE(level, (".%0d.%sS.%0d.%0d",
                    (op_args.tcu.cd_nregs == 2'd0) ? 8 : (op_args.tcu.cd_nregs == 2'd1) ? 16 : 32,
                    op_args.tcu.a_from_smem ? "S" : "R",
                    op_args.tcu.step_m, op_args.tcu.step_n));
            end
          `endif
        `endif
            default: `TRACE(level, ("?"))
        endcase
    endtask
`endif

    `DECL_EXECUTE_T (tcu, `VX_CFG_NUM_TCU_LANES);

    // tcu_tbuf_req_t — per-block uop request packed for VX_tcu_tbuf.
    // `valid` is pre-gated to WGMMA and masked by `cta_conflict`.
    typedef struct packed {
        logic                       valid;
        logic [UUID_WIDTH-1:0]      uuid;   // originating WGMMA instruction uuid
        logic [NW_WIDTH-1:0]        wid;
        logic [NCTA_WIDTH-1:0]      cta_id;
        logic [3:0]                 step_m;
        logic [3:0]                 step_n;
        logic [3:0]                 step_k;
        logic [1:0]                 cd_nregs;
        logic [`VX_CFG_XLEN-1:0]    desc_a;
        logic [`VX_CFG_XLEN-1:0]    desc_b;
        logic                       a_is_smem;
        // Precomputed per-uop expansion phase; avoids re-deriving from
        // step_m/step_n/step_k at every consumer.
        logic                       is_first_uop;
        logic                       is_last_uop;
        logic                       setup_fire;
`ifdef VX_CFG_TCU_SPARSE_ENABLE
        logic                       is_sparse;
`endif
    } tcu_tbuf_req_t;

endpackage

`IGNORE_UNUSED_END

`endif // VX_TCU_PKG_VH
