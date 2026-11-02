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

// VX_uop_packld: Micro-op sequencer for pack-load and wide load/store instructions.
//
// Expands one vx_packlb_f / vx_packlh_f instruction into N sequential LSU uops:
//   pack==1 (PACKLB) → 4 uops of INST_LSU_LBU (unsigned byte loads)
//   pack==2 (PACKLH) → 2 uops of INST_LSU_LHU (unsigned halfword loads)
//   pack==3 (vx_flw.d/.q, vx_fsw.d/.q) → 2 or 4 plain flw/fsw uops
//
// A pack-load uop encodes its index in op_args.lsu.offset[1:0]; the LSU AGU
// reads pack (non-zero) to identify it and computes
//   eff_rs1[lane] = rs1[lane] + rs2[lane] * uop_idx   (shift-and-add)
// A wide uop carries word k in offset[3:2] and register fd+k / fs2+k, and
// leaves with pack cleared, so the LSU sees a plain flw/fsw. The NaN-boxing of
// a loaded word is handled automatically because is_float==1 and rd is a float
// register (triggers nan-box in lsu_block).

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
    // pack==2 → PACKLH: 2 halfword-load uops;
    // pack==3 → wide: 4 uops if the bytesel offset marks a quad, else 2
    wire is_packlb = (ibuf_in.op_args.lsu.pack == 2'b01);
    wire is_packlh = (ibuf_in.op_args.lsu.pack == 2'b10);
    wire is_wide   = (ibuf_in.op_args.lsu.pack == 2'b11);
    wire is_pair   = is_wide && ~ibuf_in.bytesel[0];
    assign uop_count = (is_packlh || is_pair) ? UOP_CTR_W'(2) : UOP_CTR_W'(4);

    wire [1:0] idx2 = uop_idx[1:0];
    wire [XLENB_W-1:0] byte_size = ibuf_in.op_args.lsu.pack[1] ? '1 : '0;
    wire [XLENB_W-1:0] byte_off  = is_packlb ? XLENB_W'(idx2) : (is_packlh ? XLENB_W'({idx2[0], 1'b0}) : '0);

    // Decode leaves a packed load's offset zero, and a wide access's offset and
    // register have their low group bits clear, so each uop ORs its index in:
    // into offset[1:0] for a packed load; into offset[3:2] (word k) and the
    // register (fd+k, fs2+k) for a wide access, which becomes a plain flw/fsw.
    wire [1:0] wide_idx = is_wide ? idx2 : 2'b00;
    wire [1:0] pack_idx = is_wide ? 2'b00 : idx2;

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        ibuf_r.op_args.lsu.pack   = is_wide ? 2'b00 : ibuf_in.op_args.lsu.pack;
        ibuf_r.op_args.lsu.offset = ibuf_in.op_args.lsu.offset | {8'd0, wide_idx, pack_idx};
        ibuf_r.rd  = ibuf_in.rd  | NUM_REGS_BITS'(wide_idx);
        ibuf_r.rs2 = ibuf_in.rs2 | NUM_REGS_BITS'(wide_idx);
        ibuf_r.bytesel = {byte_size, byte_off};
    end
    assign ibuf_out = ibuf_r;

endmodule
