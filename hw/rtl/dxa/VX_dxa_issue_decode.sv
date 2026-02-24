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

/* verilator lint_off UNUSEDSIGNAL */
module VX_dxa_issue_decode import VX_gpu_pkg::*, VX_dxa_pkg::*; #(
    parameter GMEM_BYTES = `L2_LINE_SIZE,
    parameter SMEM_BYTES = LSU_WORD_SIZE
) (
    input wire [`XLEN-1:0] issue_flags,
    input wire [31:0] issue_desc_meta,
    input wire [31:0] issue_desc_tile01,
    input wire [31:0] issue_size0_raw,
    input wire [31:0] issue_size1_raw,
    input wire [31:0] issue_stride0_raw,

    output wire dxa_issue_dec_t issue_dec
);
    wire [`VX_DXA_DESC_META_ELEMSZ_BITS-1:0] issue_desc_esize_enc =
        issue_desc_meta[`VX_DXA_DESC_META_ELEMSZ_LSB +: `VX_DXA_DESC_META_ELEMSZ_BITS];
    wire [`VX_DXA_DESC_META_DIM_BITS-1:0] issue_rank_raw =
        issue_desc_meta[`VX_DXA_DESC_META_DIM_LSB +: `VX_DXA_DESC_META_DIM_BITS];

    wire [31:0] issue_rank_w = (issue_rank_raw == 0) ? 32'd1
                             : (issue_rank_raw > 5) ? 32'd5
                             : 32'(issue_rank_raw);
    wire [31:0] issue_elem_bytes_w = 32'(1) << issue_desc_esize_enc;

    wire [31:0] issue_tile0_w = (issue_desc_tile01[15:0] == 0) ? 32'd1 : 32'(issue_desc_tile01[15:0]);
    wire [31:0] issue_tile1_w = (issue_rank_w >= 2)
                           ? ((issue_desc_tile01[31:16] == 0) ? 32'd1 : 32'(issue_desc_tile01[31:16]))
                           : 32'd1;

    wire [31:0] issue_size0_w = (issue_size0_raw == 0) ? 32'd1 : issue_size0_raw;
    wire [31:0] issue_size1_w = (issue_rank_w >= 2)
                           ? ((issue_size1_raw == 0) ? 32'd1 : issue_size1_raw)
                           : 32'd1;
    wire [31:0] issue_stride0_w = (issue_rank_w >= 2) ? issue_stride0_raw : 32'd0;
    wire [31:0] issue_total_w = issue_tile0_w * issue_tile1_w;
    wire issue_is_s2g_w = issue_flags[0];
    wire issue_supported_w = (issue_rank_w <= 2)
                          && (issue_elem_bytes_w != 0)
                          && (issue_elem_bytes_w <= GMEM_BYTES)
                          && (issue_elem_bytes_w <= SMEM_BYTES);

    assign issue_dec.rank = issue_rank_w;
    assign issue_dec.elem_bytes = issue_elem_bytes_w;
    assign issue_dec.tile0 = issue_tile0_w;
    assign issue_dec.tile1 = issue_tile1_w;
    assign issue_dec.size0 = issue_size0_w;
    assign issue_dec.size1 = issue_size1_w;
    assign issue_dec.stride0 = issue_stride0_w;
    assign issue_dec.total = issue_total_w;
    assign issue_dec.is_s2g = issue_is_s2g_w;
    assign issue_dec.supported = issue_supported_w;
endmodule
/* verilator lint_on UNUSEDSIGNAL */
