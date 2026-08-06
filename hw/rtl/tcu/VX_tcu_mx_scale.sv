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

`ifdef VX_CFG_TCU_MX_ENABLE

// MX scale-factor decode: the read-side complement to the MX region of
// VX_tcu_meta. Given the warp-wide A/B scale rows and the current step, it
// addresses the per-(m/n, k) scale byte for every FEDP lane. This mirrors
// VX_tcu_sp_mux (the sparse namespace's read-side decode).

module VX_tcu_mx_scale import VX_gpu_pkg::*, VX_tcu_pkg::*; (
    input wire [TCU_BLOCK_CAP-1:0][31:0] meta_a,
    input wire [TCU_BLOCK_CAP-1:0][31:0] meta_b,

    input wire [3:0]  step_m,
    input wire [3:0]  step_n,
    input wire [3:0]  step_k,
    input wire [4:0]  fmt_s,
    input wire        is_wmma,
    input wire        is_sparse,

    output wire [TCU_TC_M-1:0][TCU_MX_MAX_SF-1:0][7:0] sf_a,
    output wire [TCU_TC_N-1:0][TCU_MX_MAX_SF-1:0][7:0] sf_b
);
    localparam FEDP_SF = TCU_MX_MAX_SF;
    localparam MX_IDX_W = $clog2(TCU_TILE_M > TCU_TILE_N ? TCU_TILE_M : TCU_TILE_N);
    localparam MX_K_IDX_W = `LOG2UP(TCU_TILE_K * TCU_MAX_ELT_RATIO);
    localparam MX_SCALE_IDX_W = $clog2(TCU_BLOCK_CAP * 4);

    // step_m/step_n are only consumed at MX_IDX_W width; upper bits are unused.
    `UNUSED_VAR (step_m)
    `UNUSED_VAR (step_n)

    function automatic [7:0] mx_scale_at(
        input logic [TCU_BLOCK_CAP-1:0][31:0] meta,
        input logic [4:0] fmt,
        input logic [MX_IDX_W-1:0] mn_idx,
        input logic [MX_K_IDX_W-1:0] k_base_idx
    );
        logic [MX_SCALE_IDX_W-1:0] scale_k;
        logic [MX_SCALE_IDX_W-1:0] scale_idx;
        logic [`LOG2UP(TCU_BLOCK_CAP)-1:0] word_idx;
        logic [1:0] byte_idx;
        begin
            scale_k = MX_SCALE_IDX_W'(k_base_idx / mx_scale_block_size(fmt));
            scale_idx = MX_SCALE_IDX_W'(mn_idx) * MX_SCALE_IDX_W'(mx_scale_blocks_k(fmt))
                      + MX_SCALE_IDX_W'(scale_k);
            word_idx = `LOG2UP(TCU_BLOCK_CAP)'(scale_idx >> 2);
            byte_idx = scale_idx[1:0];
            mx_scale_at = meta[word_idx][byte_idx * 8 +: 8];
        end
    endfunction

    wire [3:0] mx_elems_per_word = 4'(32 / tcu_fmt_width(fmt_s));
    wire [MX_K_IDX_W:0] mx_fedp_elems = (MX_K_IDX_W+1)'(
        (MX_K_IDX_W+1)'(TCU_TC_K) * (MX_K_IDX_W+1)'(mx_elems_per_word)
        * (MX_K_IDX_W+1)'(is_sparse ? 2 : 1));
    wire [MX_K_IDX_W-1:0] mx_k_base_idx = MX_K_IDX_W'(step_k * mx_fedp_elems);

    for (genvar i = 0; i < TCU_TC_M; ++i) begin : g_sf_a
        wire [MX_IDX_W-1:0] mx_a_idx = MX_IDX_W'(step_m) * MX_IDX_W'(TCU_TC_M) + MX_IDX_W'(i);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_elems) / FEDP_SF);
            assign sf_a[i][s] = is_wmma ? mx_scale_at(meta_a, fmt_s, mx_a_idx, mx_k_idx) : '0;
        end
    end

    for (genvar j = 0; j < TCU_TC_N; ++j) begin : g_sf_b
        wire [MX_IDX_W-1:0] mx_b_idx = MX_IDX_W'(step_n) * MX_IDX_W'(TCU_TC_N) + MX_IDX_W'(j);
        for (genvar s = 0; s < FEDP_SF; ++s) begin : g_s
            wire [MX_K_IDX_W-1:0] mx_k_idx = mx_k_base_idx + MX_K_IDX_W'((s * mx_fedp_elems) / FEDP_SF);
            assign sf_b[j][s] = is_wmma ? mx_scale_at(meta_b, fmt_s, mx_b_idx, mx_k_idx) : '0;
        end
    end

endmodule

`endif // VX_CFG_TCU_MX_ENABLE
