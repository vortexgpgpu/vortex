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
// WGMMA tile stream-buffer (per-warp slot cache).
//
// Thin wrapper that computes shared parameters and connects:
//   VX_tcu_tbuf_fetch  — slot management, LMEM fetch FSM, data capture
//   VX_tcu_tbuf_gather — format-aware A/B/meta operand gather
//
// Prefetches A, B, and optional sparse metadata tiles from local memory
// using a dedicated bank-parallel read port that bypasses the LSU crossbar.
// All LMEM banks are read simultaneously each cycle, yielding NUM_BANKS
// words/cycle throughput.
//
// Each active warp gets its own slot (direct-mapped by wid).  Slots are
// allocated on the first µop of a new tile (step_m==step_n==step_k==0)
// and evicted implicitly when the same slot index is re-allocated for a
// different tile or on the next outer-loop iteration.
//
// Assumptions:
//   - Tile base addresses (descriptor[15:0]) are bank-aligned.
//   - Tiles are stored row-major and packed (ldm == tile column count).
//   - For WGMMA_SP, step_k counts in half-K units (same as WMMA_SP).
//

module VX_tcu_tbuf import VX_gpu_pkg::*, VX_tcu_pkg::*; #(
    parameter `STRING INSTANCE_ID     = "",
    parameter         TCU_TBUF_SIZE   = `NUM_WARPS,
    parameter         NUM_BANKS       = 4,
    parameter         BANK_ADDR_WIDTH = 12
) (
    input  wire clk,
    input  wire reset,

`ifdef PERF_ENABLE
    output wire [PERF_CTR_BITS-1:0] tbuf_fetch_stalls,
    output wire [PERF_CTR_BITS-1:0] lmem_reads,
`endif

    // Execute-side observation
    input  wire                     req_valid,
    input  wire                     req_fire,   // execute consumed current µop
    input  wire [NW_WIDTH-1:0]      req_wid,
    input  wire                     req_is_sparse,
    input  wire [3:0]               req_step_m,
    input  wire [3:0]               req_step_n,
    input  wire [3:0]               req_step_k,
    input  wire [3:0]               req_fmt_s,
    input  wire [1:0]               req_cd_nregs,
    input  wire [`XLEN-1:0]         req_desc_a,
    input  wire [`XLEN-1:0]         req_desc_b,

    // LMEM read port
    VX_tcu_lmem_if.master           tcu_lmem_if,

    // Tile buffer outputs
    output wire [TCU_BLOCK_CAP-1:0][`XLEN-1:0] tbuf_rs1_data,
    output wire [TCU_WG_RS2_WIDTH-1:0][`XLEN-1:0] tbuf_rs2_data,
`ifdef TCU_SPARSE_ENABLE
    output wire [TCU_MAX_META_BLOCK_WIDTH-1:0] tbuf_sp_meta,
`endif
    output wire                     tbuf_ready
);

    // -----------------------------------------------------------------------
    // Derived tile-dimension and buffer-size constants
    // -----------------------------------------------------------------------

    localparam TILE_M = TCU_WG_TILE_M;
    localparam TILE_K = TCU_WG_TILE_K;
    localparam TILE_N = TCU_WG_TILE_N;

    // Buffer sizes in 32-bit words (format-agnostic; sub-word packing done in gather).
    localparam A_TOTAL = TILE_M * TILE_K;
    localparam B_TOTAL = TILE_K * TILE_N;

`ifdef TCU_SPARSE_ENABLE
    // Metadata buffer: worst-case format is int8 (I_RATIO=4).
    localparam SP_I_RATIO_8B     = 4;
    localparam META_ROW_BITS_8B  = TCU_TC_K * 2 * SP_I_RATIO_8B;
    localparam META_STRIDE_8B    = (TCU_TC_M * META_ROW_BITS_8B + 31) / 32;
    localparam WG_HALF_K         = TCU_WG_K_STEPS / 2;
    localparam WG_META_BANKS     = TCU_WG_M_STEPS * WG_HALF_K;
    localparam META_TOTAL_MAX    = WG_META_BANKS * META_STRIDE_8B;
`endif

    // -----------------------------------------------------------------------
    // Fetch engine
    // -----------------------------------------------------------------------

    wire                     tbuf_hit;
    wire [A_TOTAL-1:0][31:0] hit_a_buf;
    wire [B_TOTAL-1:0][31:0] hit_b_buf;
`ifdef TCU_SPARSE_ENABLE
    wire                     hit_is_sparse;
    wire [META_TOTAL_MAX-1:0][31:0] hit_meta_buf;
    wire [3:0]               hit_meta_stride;
`endif

    VX_tcu_tbuf_fetch #(
        .INSTANCE_ID    (`SFORMATF(("%s-fetch", INSTANCE_ID))),
        .TCU_TBUF_SIZE  (TCU_TBUF_SIZE),
        .NUM_BANKS      (NUM_BANKS),
        .BANK_ADDR_WIDTH(BANK_ADDR_WIDTH),
        .A_TOTAL        (A_TOTAL),
        .B_TOTAL        (B_TOTAL)
    `ifdef TCU_SPARSE_ENABLE
       ,.META_TOTAL_MAX (META_TOTAL_MAX)
    `endif
    ) tbuf_fetch (
        .clk            (clk),
        .reset          (reset),
    `ifdef PERF_ENABLE
        .tbuf_fetch_stalls(tbuf_fetch_stalls),
        .lmem_reads       (lmem_reads),
    `endif
        .req_valid      (req_valid),
        .req_fire       (req_fire),
        .req_wid        (req_wid),
        .req_is_sparse  (req_is_sparse),
        .req_step_m     (req_step_m),
        .req_step_n     (req_step_n),
        .req_step_k     (req_step_k),
        .req_fmt_s      (req_fmt_s),
        .req_desc_a     (req_desc_a),
        .req_desc_b     (req_desc_b),
        .tcu_lmem_if    (tcu_lmem_if),
        .tbuf_hit       (tbuf_hit),
        .tbuf_ready     (tbuf_ready),
        .hit_a_buf      (hit_a_buf),
        .hit_b_buf      (hit_b_buf)
    `ifdef TCU_SPARSE_ENABLE
       ,.hit_is_sparse  (hit_is_sparse),
        .hit_meta_buf   (hit_meta_buf),
        .hit_meta_stride(hit_meta_stride)
    `endif
    );

    // -----------------------------------------------------------------------
    // Gather (pure combinational)
    // -----------------------------------------------------------------------

    VX_tcu_tbuf_gather #(
        .INSTANCE_ID    (`SFORMATF(("%s-gather", INSTANCE_ID))),
        .A_TOTAL        (A_TOTAL),
        .B_TOTAL        (B_TOTAL)
    `ifdef TCU_SPARSE_ENABLE
       ,.META_TOTAL_MAX (META_TOTAL_MAX)
    `endif
    ) tbuf_gather (
        .req_step_m     (req_step_m),
        .req_step_n     (req_step_n),
        .req_step_k     (req_step_k),
        .req_fmt_s      (req_fmt_s),
        .req_cd_nregs   (req_cd_nregs),
        .a_buf          (hit_a_buf),
        .b_buf          (hit_b_buf),
    `ifdef TCU_SPARSE_ENABLE
        .is_sparse      (hit_is_sparse),
        .meta_buf       (hit_meta_buf),
        .meta_stride    (hit_meta_stride),
    `endif
        .tbuf_rs1_data  (tbuf_rs1_data),
        .tbuf_rs2_data  (tbuf_rs2_data)
    `ifdef TCU_SPARSE_ENABLE
       ,.tbuf_sp_meta   (tbuf_sp_meta)
    `endif
    );

    `UNUSED_VAR (tbuf_hit)  // consumed implicitly through tbuf_ready

endmodule

`endif // TCU_WGMMA_ENABLE
