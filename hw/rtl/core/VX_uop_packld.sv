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

// VX_uop_packld: Micro-op sequencer for pack-load instructions.
//
// Expands one vx_packlb_f / vx_packlh_f instruction into N sequential LSU uops:
//   pack==1 (PACKLB) → 4 uops of INST_LSU_LBU (unsigned byte loads)
//   pack==2 (PACKLH) → 2 uops of INST_LSU_LHU (unsigned halfword loads)
//
// Each uop encodes its index in op_args.lsu.offset[1:0].
// The dispatcher reads pack (non-zero) to identify pack-load uops and computes:
//   eff_rs1[lane] = rs1[lane] + rs2[lane] * uop_idx   (shift-and-add)
// The pack field is cleared in op_args forwarded to the LSU slice so it sees a
// plain LBU or LHU.  The NaN-boxing of the FLW result is handled automatically
// because is_float==1 and rd is a float register (triggers nan-box in lsu_slice).

module VX_uop_packld import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    `UNUSED_VAR ({clk, reset, start, advance, uop_idx})

    // pack==1 → PACKLB: 4 byte-load uops;
    // pack==2 → PACKLH: 2 halfword-load uops
    wire is_packlh = (ibuf_in.op_args.lsu.pack == 2'b10);
    assign uop_count = is_packlh ? UOP_CTR_W'(2) : UOP_CTR_W'(4);

    wire [1:0] idx2 = uop_idx[1:0];
    wire [XLENB_W-1:0] byte_size = XLENB_W'(is_packlh ? 1 : 0);
    wire [XLENB_W-1:0] byte_off  = XLENB_W'(is_packlh ? {idx2[0], 1'b0} : idx2);

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        ibuf_r.op_args.lsu.offset = {10'd0, idx2};
        ibuf_r.bytesel = {byte_size, byte_off};
    end
    assign ibuf_out = ibuf_r;

endmodule
