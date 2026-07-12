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

// VX_om_uops — expands one vx_om_export into the stores of an aperture record.
//
// A fragment shader may emit colour only, depth only, or both, so the record is
// one or two words and export_mask says which (bit 0 = colour, bit 1 = depth):
//
//   colour only  (early-Z owns the depth test AND write — the common case)
//       uop 0:  sw rs2 (colour), [rs1 + 0]
//   depth only   (z-prepass / shadow map — no colour target exists)
//       uop 0:  sw rs3 (depth),  [rs1 + 4]
//   both         (gl_FragDepth)
//       uop 0:  sw rs2 (colour), [rs1 + 0]
//       uop 1:  sw rs3 (depth),  [rs1 + 4]
//
// Every uop is an ORDINARY store. The LSU learns nothing about OM: a depth uop
// simply renames its source register (rs2 := rs3) and bumps the immediate, so the
// address falls out of the existing AGU (`addr = base + sext(offset)` at pack==0
// — which is why VX_lsu_agu's header can still claim it owns every address form).
//
// The two-word case is ATOMIC against other warps, and that is why this belongs
// here and not in the LSU. A uop burst carries fu_lock on its first uop and
// fu_unlock on its last; VX_scoreboard latches fu_locked to the granted warp and
// gates every other warp's arb_valid_in until the release. So no warp can issue
// between the colour and depth stores, the two beats stay adjacent on the source
// port all the way to the cluster trunk, and the OM ingress needs exactly ONE
// hold register per source port. Without the lock the ingress would have to hold
// a half-open record per (port x warp), which does not scale.
// The single-word cases need no pairing at all: count==1 yields fu_lock=fu_unlock
// =1, the scoreboard's single-uop default, so nothing is locked.

module VX_om_uops import VX_gpu_pkg::*; (
    input clk,
    input reset,

    input  ibuffer_t ibuf_in,
    output ibuffer_t ibuf_out,

    input wire start,
    input wire advance,
    input wire [UOP_CTR_W-1:0] uop_idx,
    output wire [UOP_CTR_W-1:0] uop_count
);
    `UNUSED_VAR ({clk, reset, start, advance})

    wire has_colour = ibuf_in.op_args.lsu.export_mask[0];
    wire has_depth  = ibuf_in.op_args.lsu.export_mask[1];

    // one uop per word the shader actually writes
    assign uop_count = UOP_CTR_W'(has_colour) + UOP_CTR_W'(has_depth);

    // The depth word is uop 1 when colour is also written, uop 0 when it is not.
    wire is_depth_uop = has_depth && (~has_colour || (uop_idx != '0));

    ibuffer_t ibuf_r;
    always_comb begin
        ibuf_r = ibuf_in;
        // colour at record+0, depth at record+4
        ibuf_r.op_args.lsu.offset = is_depth_uop ? 12'd4 : 12'd0;
        // a depth uop stores rs3 — rename it into the store-data slot so the LSU
        // sees a plain `sw rs2`
        if (is_depth_uop) begin
            ibuf_r.rs2 = ibuf_in.rs3;
        end
        // export_mask is a marker for THIS expander only; the LSU must see a plain
        // store, so it never leaves here set.
        ibuf_r.op_args.lsu.export_mask = 2'b00;
        // Hold the issue lock across a two-word record (see the header). With
        // count==1 both bits are 1, which is the scoreboard's single-uop default
        // (acquire+release in one issue) — nothing is locked.
        ibuf_r.fu_lock   = (uop_idx == '0);
        ibuf_r.fu_unlock = (uop_idx == (uop_count - UOP_CTR_W'(1)));
    end
    assign ibuf_out = ibuf_r;

endmodule
