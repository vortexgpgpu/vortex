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

// Shared per-core graphics window: the per-(warp, lane) slot register file and
// the generic SETW / GETW / GETWF macro-ops that load and read it. The window is
// a building block reused by the FF graphics units — TEX (vx_tex4 (u,v)/result
// windows) and OM (vx_om4 payload window) — and is consumed by the RTU traversal
// engine (TRACE/WAIT/CB_RET ride the same op-selector and PE). It is therefore
// present whenever any graphics extension is enabled (EXT_GFX_ANY), not gated on
// the RTU. The RT-specific op semantics live in VX_gfx_window.sv behind
// VX_CFG_EXT_RTU_ENABLE; the traversal datapath stays in VX_rtu_pkg.
package VX_gfx_window_pkg;

`IGNORE_UNUSED_BEGIN
    // The RTU-consumer op selectors (WAIT/CB_RET) are unused in a non-RTU
    // graphics build; keep the op namespace whole rather than fragment it.

    // Window op selector stored in op_args.gfxw.op. The generic ops (SETW/GETWF/
    // GETW) are always live; the RTU consumer ops (TRACE/WAIT/CB_RET) share the
    // namespace and are inert in a non-RTU build. The (funct3,funct2) -> op
    // mapping is done in decode. Values are one disjoint 4-bit namespace.
    localparam GFXW_OP_BITS   = 4;
    localparam GFXW_OP_SETW   = 4'd0;  // funct3=6 sub1   — slot <- rs1 (window write)
    localparam GFXW_OP_TRACE = 4'd4;  // funct3=7 sub0/2 — RTU window trace macro-op
    localparam GFXW_OP_WAIT  = 4'd5;  // funct3=7 sub1   — RTU terminal block
    localparam GFXW_OP_GETWF  = 4'd6;  // funct3=6 sub2   — FP windowed read macro-op
    localparam GFXW_OP_GETW   = 4'd7;  // funct3=6 sub3   — GP windowed read macro-op
    localparam GFXW_OP_CB_RET = 4'd8;  // funct3=6 sub0   — RTU callback return
    localparam GFXW_OP_GETWS  = 4'd9;  // funct3=4        — GP windowed read, warp index from rs1 (slot)

    // Per-lane window register file, one 32-bit word per slot. Sized by the RTU
    // ray/hit state today (32 slots); TEX/OM payload windows fit within it.
    localparam GFXW_SLOT_COUNT = `VX_RT_SLOT_COUNT;
    localparam GFXW_SLOT_BITS  = `CLOG2(`VX_RT_SLOT_COUNT);

    // RASTER dispatch (FWD) payload window: the raster distributor stages the
    // per-lane record {pos_mask, pid} (2 words) into slots
    // [GFXW_FRAG_SLOT_BASE .. +GFXW_FRAG_WORDS-1]; the FS reads them back with the
    // slot-indexed GETWS and recomputes per-corner edge values from the primitive
    // edges. The base sits outside the RTU object-ray range so a fragment shader
    // can hold this record and an in-flight RTU query at once. VX_GFX_FRAG_SLOT_BASE
    // is the single source of truth, shared with the kernel ABI.
    localparam GFXW_FRAG_WORDS     = `VX_GFX_FRAG_WORDS;
    localparam GFXW_FRAG_SLOT_BASE = `VX_GFX_FRAG_SLOT_BASE;

    // Macro-op uop roles (op_args.gfxw.uop), assigned by VX_gfx_uops. The RTU
    // TRACE fill roles; for GETWF/GETW the uop field instead carries the window
    // element index.
    localparam GFXW_UOP_CFG    = 3'd0;  // uop0: unpack rs1 config, alloc, rd<-handle
    localparam GFXW_UOP_ORIGIN = 3'd1;  // uop1: f0..f2 -> origin slots
    localparam GFXW_UOP_DIR    = 3'd2;  // uop2: f3..f5 -> direction slots
    localparam GFXW_UOP_ARM    = 3'd3;  // uop3: f6,f7 -> tmin/tmax; arm the walk
`IGNORE_UNUSED_END

endpackage
