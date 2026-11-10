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

// VX_uop_packld: Micro-op sequencer for pack-load, wide and matrix load/store instructions.
//
// Expands one vx_packlb_f / vx_packlh_f instruction into N sequential LSU uops:
//   pack==1 (PACKLB) → 4 uops of INST_LSU_LBU (unsigned byte loads)
//   pack==2 (PACKLH) → 2 uops of INST_LSU_LHU (unsigned halfword loads)
//   pack==3 (vx_flw.d/.q, vx_fsw.d/.q) → 2 or 4 plain flw/fsw uops
//   pack==3 with bytesel offset 2/3 (vx_ldm, vx_stm) → 4/8 flw/fsw, one per
//   register of a matrix fragment
//
// A pack-load uop encodes its index in op_args.lsu.offset[1:0]; the LSU AGU
// reads pack (non-zero) to identify it and computes
//   eff_rs1[lane] = rs1[lane] + rs2[lane] * uop_idx   (shift-and-add)
// A wide uop carries word k in offset[3:2] and register fd+k / fs2+k, and
// leaves with pack cleared, so the LSU sees a plain flw/fsw. The NaN-boxing of
// a loaded word is handled automatically because is_float==1 and rd is a float
// register (triggers nan-box in lsu_block).
// A fragment uop e moves register fd/fs2 + e: it reads its base from
// rs1 | grp(e) and ORs its column into the immediate (whose low 5 bits decode
// set to the shape's code), for the element the fragment layout puts in that
// register.

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

`ifdef EXT_LDST_ANY_ENABLE
    // pack==1 → PACKLB: 4 byte-load uops;
    // pack==2 → PACKLH: 2 halfword-load uops;
    // pack==3 → bytesel offset 0/1: a wide pair/quad, 2 or 4 uops;
    //           bytesel offset 2/3: a 4/8-register fragment
    wire is_packlb = (ibuf_in.op_args.lsu.pack == 2'b01);
    wire is_packlh = (ibuf_in.op_args.lsu.pack == 2'b10);
    wire is_wide   = (ibuf_in.op_args.lsu.pack == 2'b11);
`ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    wire is_matrix = is_wide && ibuf_in.bytesel[1];
`else
    wire is_matrix = is_wide;
`endif
    mtx_ldm_uop_t ldm_uop;
    assign ldm_uop = mtx_ldm_uop(ibuf_in.op_args.lsu.offset[2:0], uop_idx[2:0]);

    // uop e of every shape moves the register its element occupies in the
    // fragment register layout
    for (genvar s = 0; s < 6; ++s) begin : g_ldm_check
        if (s != MTX_LDM_IDX_B_SP || MTX_SYM_SPARSE) begin : g_valid
            for (genvar e = 0; e < mtx_ldm_count(3'(s)); ++e) begin : g_uops
                `STATIC_ASSERT (mtx_ldm_reg(3'(s), mtx_ldm_uop(3'(s), 3'(e))) == e,
                    ("vx_ldm shape %0d uop %0d does not invert the fragment register permutation", s, e))
            end
        end
    end
`endif
    // Decode sets the bytesel offset to the uop count: 0 → 2 (PACKLH, a wide
    // pair), 1 → 4 (PACKLB, a wide quad), 2 → 4 and 3 → 8 (fragments)
`ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
    wire cnt8 = ibuf_in.bytesel[1] && ibuf_in.bytesel[0];
    wire cnt4 = ibuf_in.bytesel[1] ^ ibuf_in.bytesel[0];
`else
    wire cnt8 = 1'b0;
    wire cnt4 = ibuf_in.bytesel[0];
`endif
    assign uop_count = UOP_CTR_W'({cnt8, cnt4, ~(cnt8 || cnt4), 1'b0});

    wire [1:0] idx2 = uop_idx[1:0];
    wire [XLENB_W-1:0] byte_size = ibuf_in.op_args.lsu.pack[1] ? '1 : '0;
    wire [XLENB_W-1:0] byte_off  = is_packlb ? XLENB_W'(idx2) : (is_packlh ? XLENB_W'({idx2[0], 1'b0}) : '0);

    // Decode leaves a packed load's offset zero, and a wide access's offset and
    // register have their low group bits clear, so each uop ORs its index in:
    // into offset[1:0] for a packed load; into offset[3:2] (word k) and the
    // register (fd+k, fs2+k) for a wide access, which becomes a plain flw/fsw.
    // A fragment uop replaces offset[4:0] (the shape code) with its column and
    // adds its register offset, through the same adder the wide uops use.
`ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    wire [1:0] wide_idx = is_wide ? idx2 : 2'b00;
`ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
    wire is_wide_word = is_wide && ~ibuf_in.bytesel[1];
`else
    wire is_wide_word = is_wide;
`endif
`else
    wire [1:0] wide_idx = 2'b00;
`endif
    wire [1:0] pack_idx = is_wide ? 2'b00 : idx2;

`ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
    wire [11:0] off_base = is_matrix ? {ibuf_in.op_args.lsu.offset[11:5], 5'd0} : ibuf_in.op_args.lsu.offset;
    wire [11:0] off_add  = is_matrix ? 12'(ldm_uop.col) : {8'd0, wide_idx, pack_idx};
    wire [4:0]  reg_add  = is_matrix ? 5'(uop_idx[2:0]) : 5'(wide_idx);
    wire [1:0]  grp      = is_matrix ? ldm_uop.grp : 2'b00;
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        ibuf_r.op_args.lsu.pack = is_wide ? 2'b00 : ibuf_in.op_args.lsu.pack;
    `ifdef VX_CFG_EXT_LDST_MATRIX_ENABLE
        ibuf_r.op_args.lsu.offset = off_base | off_add;
        ibuf_r.rs1 = ibuf_in.rs1 | NUM_REGS_BITS'(grp);
        ibuf_r.rd  = {ibuf_in.rd[NUM_REGS_BITS-1:5], ibuf_in.rd[4:0] + reg_add};
        ibuf_r.rs2 = {ibuf_in.rs2[NUM_REGS_BITS-1:5], ibuf_in.rs2[4:0] + reg_add};
    `else
        ibuf_r.op_args.lsu.offset = ibuf_in.op_args.lsu.offset | {8'd0, wide_idx, pack_idx};
        ibuf_r.rd  = ibuf_in.rd  | NUM_REGS_BITS'(wide_idx);
        ibuf_r.rs2 = ibuf_in.rs2 | NUM_REGS_BITS'(wide_idx);
    `endif
        ibuf_r.bytesel = {byte_size, byte_off};
    `ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
        // every uop of a wide access but its last may yield the next issue
        // cycle, so several warps' accesses interleave as their scalar loads do
        ibuf_r.issue_yield = is_wide_word && (~idx2[0] || (ibuf_in.bytesel[0] && ~idx2[1]));
    `endif
    end
    assign ibuf_out = ibuf_r;
`else
    wire is_packlh = (ibuf_in.op_args.lsu.pack == 2'b10);
    assign uop_count = is_packlh ? UOP_CTR_W'(2) : UOP_CTR_W'(4);

    wire [1:0] idx2 = uop_idx[1:0];
    wire [XLENB_W-1:0] byte_size = is_packlh ? '1 : '0;
    wire [XLENB_W-1:0] byte_off  = is_packlh ? XLENB_W'({idx2[0], 1'b0}) : XLENB_W'(idx2);

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        ibuf_r.op_args.lsu.offset = {10'd0, idx2};
        ibuf_r.bytesel = {byte_size, byte_off};
    end
    assign ibuf_out = ibuf_r;
`endif

endmodule
