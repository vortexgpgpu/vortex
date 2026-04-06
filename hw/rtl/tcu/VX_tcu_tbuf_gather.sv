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

`ifdef TCU_WGMMA_ENABLE

//
// Tile-buffer gather (combinational output mux).
//
// Takes the flat A/B/meta buffers for the current hit slot (already
// pre-selected by VX_tcu_tbuf_fetch) and produces the operand vectors
// that VX_tcu_core expects, with format-aware element gather.
//
// A operand (rs1_data):
//   Dense:  slot_a_buf[tile_row * TILE_K + tile_col]
//   Sparse: slot_a_buf[tile_row * (TILE_K/2) + tile_col]
//           step_k already counts in half-K units for WGMMA_SP.
//
// B operand (rs2_data):
//   Dense:  format-aware gather: each lane aggregates I_RATIO elements
//           from consecutive K-rows stored in the 32-bit word buffer.
//   Sparse: produces candidate pairs (b_k0, b_k1) for VX_tcu_sp_mux.
//
// Metadata (tbuf_sp_meta):
//   Extracts the meta block for (step_m, step_k) from slot_meta_buf and
//   zero-extends to TCU_MAX_META_BLOCK_WIDTH to match VX_tcu_meta output.
//   step_k is in half-K units (same as A sparse indexing convention).
//

module VX_tcu_tbuf_gather import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID = "",
    // Tile buffer sizes (in 32-bit words); must match VX_tcu_tbuf_fetch params.
    parameter A_TOTAL        = 1,
    parameter B_TOTAL        = 1
`ifdef TCU_SPARSE_ENABLE
   ,parameter META_TOTAL_MAX = 1
`endif
) (
    // Step and format inputs
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [1:0]               req_cd_nregs,

    // Hit-slot buffers (from VX_tcu_tbuf_fetch)
    input  wire [A_TOTAL-1:0][31:0] a_buf,
    input  wire [B_TOTAL-1:0][31:0] b_buf,

`ifdef TCU_SPARSE_ENABLE
    input  wire                     is_sparse,
    input  wire [META_TOTAL_MAX-1:0][31:0] meta_buf,
    input  wire [3:0]               meta_stride, // words per wg_bank slot
`endif

    // Outputs to VX_tcu_core
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data,
    output wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data
`ifdef TCU_SPARSE_ENABLE
   ,output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta
`endif
);
    `UNUSED_SPARAM (INSTANCE_ID)
`ifdef TCU_SPARSE_ENABLE
    `UNUSED_VAR (meta_stride)
`endif

    // -----------------------------------------------------------------------
    // Derived constants
    // -----------------------------------------------------------------------

    localparam TILE_M   = TCU_WG_TILE_M;
    localparam TILE_K   = TCU_WG_TILE_K;
    localparam TILE_N   = TCU_WG_TILE_N;
    `UNUSED_PARAM (TILE_M)
    `UNUSED_PARAM (TILE_N)

    // Actual per-warp N dimension from cd_nregs: 0→8, 1→16, 2→32
    // per_warp_N = NRC * NT / TILE_M (in output-type columns)
    localparam NT_DIV_TM = TCU_NT / TILE_M;
    reg [5:0] wg_nrc;
    always_comb begin
        case (req_cd_nregs)
            2'd0: wg_nrc = 6'd8;
            2'd1: wg_nrc = 6'd16;
            default: wg_nrc = 6'd32;
        endcase
    end
    wire [6:0] actual_N = 7'(wg_nrc) * NT_DIV_TM[2:0];

    localparam LG_A_BS  = $clog2(TCU_A_BLOCK_SIZE);
    localparam LG_B_BS  = $clog2(TCU_B_BLOCK_SIZE);
    localparam OFF_W    = $clog2(TCU_BLOCK_CAP);

`ifdef TCU_SPARSE_ENABLE
    localparam LG_B_BS_SP  = $clog2(TCU_B_BLOCK_SIZE_SP);
    localparam SP_I_RATIO_32B    = 1;  // tf32/fp32
    localparam SP_I_RATIO_16B    = 2;  // fp16/bf16
    localparam SP_I_RATIO_8B     = 4;  // fp8/bf8/int8/uint8
    localparam META_ROW_BITS_32B = TCU_TC_K * 2 * SP_I_RATIO_32B;
    localparam META_ROW_BITS_16B = TCU_TC_K * 2 * SP_I_RATIO_16B;
    localparam META_ROW_BITS_8B  = TCU_TC_K * 2 * SP_I_RATIO_8B;
    localparam META_STRIDE_32B   = (TCU_TC_M * META_ROW_BITS_32B + 31) / 32;
    localparam META_STRIDE_16B   = (TCU_TC_M * META_ROW_BITS_16B + 31) / 32;
    localparam META_STRIDE_8B    = (TCU_TC_M * META_ROW_BITS_8B  + 31) / 32;
    localparam WG_HALF_K         = TCU_WG_K_STEPS / 2;
    localparam WG_META_BANKS     = TCU_WG_M_STEPS * WG_HALF_K;
    localparam META_STRIDE_MAX   = META_STRIDE_8B; // worst case (int8, I_RATIO=4)
    `UNUSED_PARAM (WG_META_BANKS)
`endif

    // -----------------------------------------------------------------------
    // A operand gather (rs1_data)
    // -----------------------------------------------------------------------
    // Buffer layout: [tile_row * TILE_K + tile_col]  (row-major, fp32 words)
    // Sparse A: only TILE_K/2 K-columns stored (2:4 compressed).
    //           step_k counts in half-K units so tile_col stays in-range.

    wire [OFF_W-1:0] a_off = (OFF_W'(req_step_m) & OFF_W'(TCU_A_SUB_BLOCKS-1)) << LG_A_BS;

    logic [TCU_BLOCK_CAP-1:0][`XLEN-1:0] rs1_mux;
    always_comb begin
        rs1_mux = '0;
        for (int i = 0; i < TCU_TC_M; ++i) begin
            for (int k = 0; k < TCU_TC_K; ++k) begin
                automatic int tile_row = int'(req_step_m) * TCU_TC_M + i;
                automatic int tile_col = int'(req_step_k) * TCU_TC_K + k;
                automatic int lane     = int'(a_off) + i * TCU_TC_K + k;
            `ifdef TCU_SPARSE_ENABLE
                if (is_sparse)
                    rs1_mux[lane] = `XLEN'(a_buf[tile_row * (TILE_K/2) + tile_col]);
                else
            `endif
                    rs1_mux[lane] = `XLEN'(a_buf[tile_row * TILE_K + tile_col]);
            end
        end
    end

    assign tbuf_rs1_data = rs1_mux;

    // -----------------------------------------------------------------------
    // B operand gather (rs2_data)
    // -----------------------------------------------------------------------
    // Buffer layout (dense, row-major fp32 words): [k_row * TILE_N + n_col]
    //   fp32:  i_ratio=1, one fp32 per element  → direct word access
    //   fp16:  i_ratio=2, two fp16 per fp32 word → sub-word extraction
    //   fp8:   i_ratio=4, four fp8  per fp32 word → sub-word extraction
    //
    // Sparse B is always dense in memory; candidate pairs (b_k0, b_k1) are
    // packed into each lane for VX_tcu_sp_mux selection.

`ifdef TCU_SPARSE_ENABLE
    wire [OFF_W-1:0] b_off = is_sparse
        ? (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS_SP-1)) << LG_B_BS_SP
        : (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1))    << LG_B_BS;
`else
    wire [OFF_W-1:0] b_off = (OFF_W'(req_step_n) & OFF_W'(TCU_B_SUB_BLOCKS-1)) << LG_B_BS;
`endif

    logic [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] rs2_mux;
    always_comb begin
        rs2_mux = '0;
    `ifdef TCU_SPARSE_ENABLE
        if (is_sparse) begin
            // Sparse: pack candidate pairs (b_k0, b_k1) into each lane.
            // Use j directly (no SYM_SPARSE folding) so all TCU_TC_N columns
            // get distinct lanes in the wider TCU_WG_RS2_WIDTH output.
            // Use actual_N (runtime) instead of TILE_N (max) for B row stride.
            for (int j = 0; j < TCU_TC_N; ++j) begin
                for (int k = 0; k < TCU_TC_K; ++k) begin
                    automatic int b_k0  = int'(req_step_k) * TCU_TC_K * 2 + k * 2;
                    automatic int b_k1  = b_k0 + 1;
                    automatic int b_col = int'(req_step_n) * TCU_TC_N + j;
                    automatic int lane0 = j * TCU_TC_K * 2 + k * 2;
                    automatic int lane1 = lane0 + 1;
                    automatic int bN    = int'(actual_N);
                    // Candidate b_k0
                    if (b_k0 < TILE_K) begin
                        case (tcu_fmt_width(req_fmt_s))
                            32:
                                rs2_mux[lane0] = `XLEN'(b_buf[b_k0 * bN + b_col]);
                            16: begin
                                automatic int bN_w = bN >> 1;
                                rs2_mux[lane0][ 0+:16] = b_buf[(b_k0*2+0)*bN_w + b_col/2][(b_col%2)*16+:16];
                                rs2_mux[lane0][16+:16] = b_buf[(b_k0*2+1)*bN_w + b_col/2][(b_col%2)*16+:16];
                            end
                            default: begin // fp8/bf8/int8/uint8
                                automatic int bN_w = bN >> 2;
                                rs2_mux[lane0][ 0+:8] = b_buf[(b_k0*4+0)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][ 8+:8] = b_buf[(b_k0*4+1)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][16+:8] = b_buf[(b_k0*4+2)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane0][24+:8] = b_buf[(b_k0*4+3)*bN_w + b_col/4][(b_col%4)*8+:8];
                            end
                        endcase
                    end
                    // Candidate b_k1
                    if (b_k1 < TILE_K) begin
                        case (tcu_fmt_width(req_fmt_s))
                            32:
                                rs2_mux[lane1] = `XLEN'(b_buf[b_k1 * bN + b_col]);
                            16: begin
                                automatic int bN_w = bN >> 1;
                                rs2_mux[lane1][ 0+:16] = b_buf[(b_k1*2+0)*bN_w + b_col/2][(b_col%2)*16+:16];
                                rs2_mux[lane1][16+:16] = b_buf[(b_k1*2+1)*bN_w + b_col/2][(b_col%2)*16+:16];
                            end
                            default: begin // fp8/bf8/int8/uint8
                                automatic int bN_w = bN >> 2;
                                rs2_mux[lane1][ 0+:8] = b_buf[(b_k1*4+0)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][ 8+:8] = b_buf[(b_k1*4+1)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][16+:8] = b_buf[(b_k1*4+2)*bN_w + b_col/4][(b_col%4)*8+:8];
                                rs2_mux[lane1][24+:8] = b_buf[(b_k1*4+3)*bN_w + b_col/4][(b_col%4)*8+:8];
                            end
                        endcase
                    end
                end
            end
        end else
    `endif
        begin
            // Dense: format-aware gather.
            // Use actual_N (runtime) instead of TILE_N (max) for B row stride.
            for (int j = 0; j < TCU_TC_N; ++j) begin
                for (int k = 0; k < TCU_TC_K; ++k) begin
                    automatic int b_row = int'(req_step_k) * TCU_TC_K + k;
                    automatic int n_col = int'(req_step_n) * TCU_TC_N + j;
                    automatic int lane  = int'(b_off) + j * TCU_TC_K + k;
                    automatic int bN    = int'(actual_N);
                    case (tcu_fmt_width(req_fmt_s))
                        32: // fp32/i32/tf32: i_ratio=1
                            rs2_mux[lane] = `XLEN'(b_buf[b_row * bN + n_col]);
                        16: begin // fp16/bf16: i_ratio=2
                            automatic int bN_w = bN >> 1;
                            rs2_mux[lane][ 0+:16] = b_buf[(b_row*2+0)*bN_w + n_col/2][(n_col%2)*16+:16];
                            rs2_mux[lane][16+:16] = b_buf[(b_row*2+1)*bN_w + n_col/2][(n_col%2)*16+:16];
                        end
                        default: begin // fp8/bf8/int8/uint8: i_ratio=4
                            automatic int bN_w = bN >> 2;
                            rs2_mux[lane][ 0+:8] = b_buf[(b_row*4+0)*bN_w + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][ 8+:8] = b_buf[(b_row*4+1)*bN_w + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][16+:8] = b_buf[(b_row*4+2)*bN_w + n_col/4][(n_col%4)*8+:8];
                            rs2_mux[lane][24+:8] = b_buf[(b_row*4+3)*bN_w + n_col/4][(n_col%4)*8+:8];
                        end
                    endcase
                end
            end
        end
    end

    assign tbuf_rs2_data = rs2_mux;

    // -----------------------------------------------------------------------
    // Metadata extraction (WGMMA_SP only)
    // -----------------------------------------------------------------------
    // slot_meta_buf layout: WG_META_BANKS consecutive slots, each
    //   meta_stride 32-bit words wide.
    //   Bank index: wg_bank = step_m * WG_HALF_K + step_k
    //   (step_k is in half-K units, matching the A sparse convention.)
    // Output is zero-extended to TCU_MAX_META_BLOCK_WIDTH to be compatible
    // with the VX_tcu_meta vld_block format consumed by VX_tcu_sp_mux.

`ifdef TCU_SPARSE_ENABLE
    logic [META_STRIDE_MAX*32-1:0] extracted_meta;

    always_comb begin
        extracted_meta = '0;
        begin
            automatic int wg_bank = int'(req_step_m) * WG_HALF_K + int'(req_step_k);
            case (tcu_fmt_width(req_fmt_s))
                32: begin // tf32/fp32: i_ratio=1
                    for (int w = 0; w < META_STRIDE_32B; ++w) begin
                        automatic int idx = wg_bank * META_STRIDE_32B + w;
                        if (idx < META_TOTAL_MAX)
                            extracted_meta[w*32 +: 32] = meta_buf[idx];
                    end
                end
                16: begin // fp16/bf16: i_ratio=2
                    for (int w = 0; w < META_STRIDE_16B; ++w) begin
                        automatic int idx = wg_bank * META_STRIDE_16B + w;
                        if (idx < META_TOTAL_MAX)
                            extracted_meta[w*32 +: 32] = meta_buf[idx];
                    end
                end
                default: begin // fp8/bf8/int8/uint8: i_ratio=4
                    for (int w = 0; w < META_STRIDE_8B; ++w) begin
                        automatic int idx = wg_bank * META_STRIDE_8B + w;
                        if (idx < META_TOTAL_MAX)
                            extracted_meta[w*32 +: 32] = meta_buf[idx];
                    end
                end
            endcase
        end
    end

    // Zero-extend to the full block width that VX_tcu_sp_mux expects.
    // WGMMA supports up to int8 (I_RATIO=4); TCU_MAX_META_BLOCK_WIDTH is
    // sized for int4 (I_RATIO=8), so this is always a zero-extension.
    assign tbuf_sp_meta = TCU_MAX_META_BLOCK_WIDTH'(extracted_meta);
`endif

endmodule

`endif // TCU_WGMMA_ENABLE
