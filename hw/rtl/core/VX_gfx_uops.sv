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
// The graphics uop expander. Mirrors VX_tcu_uops: rewrites a fetched macro-op
// into a per-cycle stream of ordinary micro-ops, each naming its own source and
// destination registers so the standard operand collector reads and writes the
// right file.
//
// It lives beside the sequencer that instantiates it, not with any one graphics
// unit, because it belongs to neither: its two macro-ops are owned by different
// subsystems (the window ops by the RTU, the export by the OM) and it is the
// sequencer that needs them rewritten. A sequencer slot is expensive -- each one
// is a full ibuffer_t input on the output mux (uop_out_data[UOP_MAX]) plus a bit
// of its priority encoder -- and the two rewrites are combinational over the same
// ibuf_in with mutually exclusive selects (EX_SFU + INST_SFU_GFXW for the window
// ops, EX_LSU + a non-zero export_mask for the fragment export), so they share one
// slot instead of owning two.
//
//   GETWF/GETW (count uops):
//     i        : rd = window base + i (FP/GP); slot = start + i (window slot)
//   TRACE (4 uops, RTU consumer). The burst hands the RTU a ray, and the RTU
//   holds exactly one, so it carries fu_lock/fu_unlock: two warps' rays must not
//   interleave on the way in. TRACE is not issued at all unless the RTU is free
//   (VX_scoreboard's trace gate), so the locked burst can never stall on it:
//     0 CFG    : rs1 = lane-packed config (GP); rd = handle (GP); arms the walk
//     1 ORIGIN : rs1/rs2/rs3 = f0/f1/f2 -> ray beats 0..2
//     2 DIR    : rs1/rs2/rs3 = f3/f4/f5 -> ray beats 3..5
//     3 ARM    : rs1/rs2     = f6/f7    -> ray beats 6..7 (t_min, t_max)
//
//   vx_om_export -> the stores of an aperture record. A shader may emit colour
//   only, depth only, or both, so the record is one or two words and export_mask
//   says which (bit 0 = colour, bit 1 = depth):
//     colour only (early-Z owns the depth test AND write -- the common case)
//       uop 0 : sw rs2 (colour), [rs1 + 0]
//     depth only (z-prepass / shadow map -- no colour target exists)
//       uop 0 : sw rs3 (depth),  [rs1 + 4]
//     both (gl_FragDepth)
//       uop 0 : sw rs2 (colour), [rs1 + 0]
//       uop 1 : sw rs3 (depth),  [rs1 + 4]
//
//   Every export uop is an ORDINARY store. The LSU learns nothing about OM: a
//   depth uop simply renames its source register (rs2 := rs3) and bumps the
//   immediate, so the address falls out of the existing AGU (addr = base +
//   sext(offset) at pack == 0 -- which is why VX_lsu_agu's header can still claim
//   it owns every address form).
//
//   The two-word case is ATOMIC against other warps, and that is why it belongs
//   here and not in the LSU. A uop burst carries fu_lock on its first uop and
//   fu_unlock on its last; VX_scoreboard latches fu_locked to the granted warp and
//   gates every other warp's arb_valid_in until the release. So no warp can issue
//   between the colour and depth stores, the two beats stay adjacent on the source
//   port all the way to the cluster trunk, and the OM ingress needs exactly ONE
//   hold register per source port. Without the lock the ingress would have to hold
//   a half-open record per (port x warp), which does not scale. The single-word
//   cases need no pairing: count == 1 yields fu_lock = fu_unlock = 1, the
//   scoreboard's single-uop default, so nothing is locked.
//
module VX_gfx_uops import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
`ifdef VX_CFG_EXT_RTU_ENABLE
    import VX_rtu_pkg::*;
`endif
    `UNUSED_VAR ({clk, reset, start, advance})
`ifndef VX_CFG_EXT_OM_ENABLE
`ifndef VX_CFG_EXT_RTU_ENABLE
    // Nothing expands into multiple uops without OM or the RTU: pass-through.
    `UNUSED_VAR (uop_idx)
`endif
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    wire [GFXW_OP_BITS-1:0] op = ibuf_in.op_args.gfxw.op;
    wire is_trace = (op == GFXW_OP_TRACE);
`endif

`ifdef VX_CFG_EXT_OM_ENABLE
    // The fragment export is an LSU op; every window op is an SFU op. op_args is a
    // union, so export_mask is only meaningful once ex_type says LSU.
    wire is_export  = (ibuf_in.ex_type == EX_LSU);
    wire has_colour = ibuf_in.op_args.lsu.export_mask[0];
    wire has_depth  = ibuf_in.op_args.lsu.export_mask[1];
    // one uop per word the shader actually writes
    wire [UOP_CTR_W-1:0] export_count = UOP_CTR_W'(has_colour) + UOP_CTR_W'(has_depth);
    // The depth word is uop 1 when colour is also written, uop 0 when it is not.
    wire is_depth_uop = has_depth && (~has_colour || (uop_idx != '0));
`else
    wire                 is_export    = 1'b0;
    wire [UOP_CTR_W-1:0] export_count = '0;
    `UNUSED_VAR ({is_export, export_count})
`endif

    // Export: one uop per written word. TRACE: exactly 4. GETWF/GETW: `count`.
`ifdef VX_CFG_EXT_RTU_ENABLE
    assign uop_count = is_export ? export_count
                     : is_trace  ? UOP_CTR_W'(4)
                                 : UOP_CTR_W'(ibuf_in.op_args.gfxw.count);
`else
    assign uop_count = export_count;
`endif

`ifdef VX_CFG_EXT_RTU_ENABLE
    // Destination window base for GETWF/GETW (type bit + index from decode).
    // Only the RTU's windowed reads fan a destination out across consecutive
    // registers, so this is scoped to that build.
    wire [REG_TYPE_BITS-1:0] rd_type = get_reg_type(ibuf_in.rd);
    wire [RV_REGS_BITS-1:0]  rd_base = get_reg_idx(ibuf_in.rd);
`endif

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
`ifdef VX_CFG_EXT_OM_ENABLE
        if (is_export) begin
            // colour at record+0, depth at record+4
            ibuf_r.op_args.lsu.offset = is_depth_uop ? 12'd4 : 12'd0;
            // a depth uop stores rs3 -- rename it into the store-data slot so the
            // LSU sees a plain `sw rs2`
            if (is_depth_uop) begin
                ibuf_r.rs2 = ibuf_in.rs3;
            end
            // export_mask is a marker for THIS expander only; the LSU must see a
            // plain store, so it never leaves here set.
            ibuf_r.op_args.lsu.export_mask = 2'b00;
            // Hold the issue lock across a two-word record (see the header).
            ibuf_r.fu_lock   = (uop_idx == '0);
            ibuf_r.fu_unlock = (uop_idx == (export_count - UOP_CTR_W'(1)));
        end else
`endif
`ifdef VX_CFG_EXT_RTU_ENABLE
        if (is_trace) begin
            case (uop_idx[1:0])
            2'd0: begin // CFG: read rs1 config, arm the walk, write handle
                ibuf_r.op_args.gfxw.uop = GFXW_UOP_CFG;
                // rd = handle (GP), rs1 = lane-packed config (GP): keep as decoded.
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b0;
                ibuf_r.used_rs[2] = 1'b0;
                // wb/rd left as decoded (handle writeback).
            end
            2'd1: begin // ORIGIN: f0,f1,f2 -> ray beats 0..2
                ibuf_r.op_args.gfxw.uop  = GFXW_UOP_ORIGIN;
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(0));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(1));
                ibuf_r.rs3 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(2));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b1;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            2'd2: begin // DIR: f3,f4,f5 -> ray beats 3..5
                ibuf_r.op_args.gfxw.uop  = GFXW_UOP_DIR;
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(3));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(4));
                ibuf_r.rs3 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(5));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b1;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            default: begin // ARM (uop 3): f6,f7 -> ray beats 6..7
                ibuf_r.op_args.gfxw.uop  = GFXW_UOP_ARM;
                ibuf_r.rs1 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(6));
                ibuf_r.rs2 = make_reg_num(REG_TYPE_F, RV_REGS_BITS'(7));
                ibuf_r.used_rs[0] = 1'b1;
                ibuf_r.used_rs[1] = 1'b1;
                ibuf_r.used_rs[2] = 1'b0;
                ibuf_r.wb = 1'b0;
                ibuf_r.rd = '0;
            end
            endcase
            // Hold the issue lock across the burst: the RTU has room for one ray,
            // and a second warp's beats interleaving into it would corrupt both.
            ibuf_r.fu_lock   = (uop_idx == '0);
            ibuf_r.fu_unlock = (uop_idx == UOP_CTR_W'(3));
        end else begin // GETWF / GETW: one uop per window element
            ibuf_r.op_args.gfxw.uop  = uop_idx[2:0];
            ibuf_r.op_args.gfxw.slot = ibuf_in.op_args.gfxw.slot + 5'(uop_idx[4:0]);
            ibuf_r.rd = make_reg_num(rd_type, rd_base + RV_REGS_BITS'(uop_idx[4:0]));
            ibuf_r.wb = 1'b1;
            // rs1 = status word: only the first uop chains the scoreboard dep
            // (subsequent uops follow it in-order through the sequencer).
            ibuf_r.used_rs[0] = (uop_idx == '0);
            ibuf_r.used_rs[1] = 1'b0;
            ibuf_r.used_rs[2] = 1'b0;
        end
`else
        begin
            // no window ops without the RTU: the export is the only rewrite
        end
`endif
    end

    assign ibuf_out = ibuf_r;

endmodule
