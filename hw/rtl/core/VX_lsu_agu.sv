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

// VX_lsu_agu: per-lane LSU address generator. Owns all LSU address forms
// so VX_lsu_slice contains no address arithmetic:
//   - plain load/store/fence: addr = base + sext(offset)
//   - pack-load uop:          addr = base + uop_idx * stride
//     (uop_idx = offset[1:0], stride = rs2; a 2-bit shift-and-add
//      collapsed with a 3:2 compressor — no multiplier)
// Both forms are computed in parallel and selected by `pack`.

module VX_lsu_agu import VX_gpu_pkg::*; (
    input  wire [`VX_CFG_XLEN-1:0] base,    // rs1
    input  wire [`VX_CFG_XLEN-1:0] stride,  // rs2 (pack stride)
    input  wire [11:0]             offset,  // immediate offset; [1:0] = pack uop_idx
    input  wire [1:0]              pack,    // 0 = plain, non-zero = pack-load
    output wire [`VX_CFG_XLEN-1:0] addr
);
    wire is_pack = (pack != 2'b00);

    // pack: base + uop_idx * stride  (uop_idx in offset[1:0])
    wire [1:0] uop_idx = offset[1:0];
    wire [`VX_CFG_XLEN-1:0] t0 = {`VX_CFG_XLEN{uop_idx[0]}} & stride;
    wire [`VX_CFG_XLEN-1:0] t1 = {`VX_CFG_XLEN{uop_idx[1]}} & (stride << 1);
    wire [`VX_CFG_XLEN+1:0] csa_sum, csa_carry;
    VX_csa_32 #(
        .N (`VX_CFG_XLEN)
    ) pack_csa (
        .a     (base),
        .b     (t0),
        .c     (t1),
        .sum   (csa_sum),
        .carry (csa_carry)
    );
    wire [`VX_CFG_XLEN-1:0] pack_addr = csa_sum[`VX_CFG_XLEN-1:0] + csa_carry[`VX_CFG_XLEN-1:0];
    `UNUSED_VAR ({csa_sum[`VX_CFG_XLEN+1:`VX_CFG_XLEN], csa_carry[`VX_CFG_XLEN+1:`VX_CFG_XLEN]})

    // plain: base + sext(offset)
    wire [`VX_CFG_XLEN-1:0] offset_addr = base + `SEXT(`VX_CFG_XLEN, offset);

    assign addr = is_pack ? pack_addr : offset_addr;

endmodule
