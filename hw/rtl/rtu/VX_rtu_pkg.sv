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

package VX_rtu_pkg;

`IGNORE_UNUSED_BEGIN

    // The window op selector (op_args.gfxw.op: SETW/TRACE/WAIT/GETWF/GETW/
    // CB_RET), the per-lane slot register-file dimensions, and the TRACE uop
    // roles live in VX_gfx_window_pkg — the window is shared by the FF graphics
    // units, the RTU is one consumer. This package keeps only the RTU traversal
    // datapath (bus packets + walker/PE/node configuration).

    // Kind tag on the window->RTU `req` channel (see VX_rtu_bus_if): the warp's
    // per-lane CONTINUE actions, or a window slot read return.
    localparam RTU_REQ_CONT  = 1'b0;
    localparam RTU_REQ_RDATA = 1'b1;

    // Window addressing carried on every RTU bus beat. Both are fixed by the
    // window's slot RAM geometry, not by the RTU.
    localparam RTU_SLOT_BITS = `CLOG2(`VX_RT_SLOT_COUNT);
    localparam RTU_TB_BITS   = `CLOG2(`VX_CFG_NUM_THREADS);

    // Callback metadata field widths carried on the RTU bus.
    //   action : VX_RT_CB_{IGNORE,ACCEPT,TERMINATE,DONE}  (0..3)
    //   type   : VX_RT_CB_TYPE_{ANYHIT,PROC,CHS,MISS}     (1..4)
    //   sbt    : SBT index (1 byte)
    localparam RTU_CB_ACTION_BITS = 2;
    localparam RTU_CB_TYPE_BITS   = 3;
    localparam RTU_CB_SBT_BITS    = 8;

    // ─────────────────────────────────────────────────────────────────
    // Traversal / PE configuration (from VX_CFG_RTU_*)
    // ─────────────────────────────────────────────────────────────────
    // node fan-out: 0 = flat triangle-list walker (no BVH), 4 = CW-BVH4,
    // 6 = CW-BVH6. The BVH walker/box-PE/node-decode are only instantiated when
    // RTU_BVH_WIDTH > 0; RTU_NODE_W clamps the BVH-node array dimensions to >=1
    // so the (unused) BVH node type still elaborates in a flat (WIDTH=0) build.
    localparam RTU_BVH_WIDTH   = `VX_CFG_RTU_BVH_WIDTH;
    localparam RTU_NODE_W      = (RTU_BVH_WIDTH == 0) ? 1 : RTU_BVH_WIDTH;
    // F32 FMA latency for the geometry PEs — tracks the ISA FPU through the shared
    // VX_CFG_FMA_LATENCY knob (16 for the Vivado vendor xil_fma, 8/12 for the soft
    // core in sim/ASIC). The PE FMAs pass USE_DSP=VX_CFG_FPU_USE_DSP + SUBNORM=0, so
    // on Vivado they select the SAME hardened xil_fma the FPU uses; that core's one-
    // cycle add+LZC accumulate was the 300 MHz limiter. The barrel scheduler hides
    // the added latency and the hardened IP costs less LUT. PE delay lines scale
    // with this value.
    localparam RTU_LATENCY_FMA = `VX_CFG_FMA_LATENCY;
    localparam RTU_STACK_DEPTH = `VX_CFG_RTU_STACK_DEPTH; // short-stack depth

    // TLAS (top-level acceleration structure / instancing): the CW-BVH walker
    // descends LEAF_INST nodes natively; the flat walker (WIDTH=0) iterates
    // instances over inline BLAS triangle lists under VX_CFG_RTU_TLAS_ENABLE.
    // Both transform the world ray into each instance's object space through
    // VX_rtu_xform.
    localparam RTU_CHILD_BITS  = `CLOG2(RTU_NODE_W + 1);
    localparam RTU_STACK_BITS  = `CLOG2(RTU_STACK_DEPTH + 1);

    // The reciprocals (1/det, inv_d) reuse VX_fdiv_unit through the same USE_DSP
    // knob as the FMAs. Its latency is a FIXED property of the active backend, so
    // it uses the BACKEND-keyed VX_CFG_RTU_FDIV_LATENCY (vendor xil_fdiv 28 / altera
    // 15 / soft 17) — mirroring VX_CFG_FMA_LATENCY, NOT the FPU-type-keyed
    // VX_CFG_FDIV_LATENCY (which would mis-size the RTL divider to the DPI model's
    // 15). PE delay lines + the scheduler setup window scale with this.
    localparam RTU_FDIV_LAT    = `VX_CFG_RTU_FDIV_LATENCY;

    // ─────────────────────────────────────────────────────────────────
    // CW-BVH node-kind tag (low byte of word0) and count field (bits 8..15)
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_KIND_INTERNAL = 8'd0;
    localparam RTU_KIND_LEAF_TRI = 8'd1;
    localparam RTU_KIND_LEAF_INST= 8'd2;
    localparam RTU_KIND_LEAF_PROC= 8'd3;

    // Child-offset word: bit 31 = leaf flag, bits 0..30 = byte offset from
    // BVH root, value 0 = empty (no child).
    localparam RTU_CHILD_LEAF_BIT  = 31;
    localparam RTU_CHILD_OFF_MASK  = 32'h7fffffff;

    // ─────────────────────────────────────────────────────────────────
    // Byte offsets within a CW-BVH4 internal node (64 B = one cache line).
    // A node arrives as one aligned line; fields are sliced out by these offsets.
    //   word0 kind        @ 0   (bits 0..7 kind, 8..15 num_children)
    //   float origin[3]   @ 4
    //   int8  exp[3]      @ 16
    //   u32   child_off[] @ 20  (RTU_BVH_WIDTH entries)
    //   u8    qaabb_min[] @ 20 + 4*WIDTH        (WIDTH*3 bytes)
    //   u8    qaabb_max[] @ 20 + 4*WIDTH + 3*WIDTH
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_NODE_OFF_KIND   = 0;
    localparam RTU_NODE_OFF_ORIGIN = 4;
    localparam RTU_NODE_OFF_EXP    = 16;
    localparam RTU_NODE_OFF_CHILD  = 20;
    localparam RTU_NODE_OFF_QMIN   = RTU_NODE_OFF_CHILD + 4 * RTU_BVH_WIDTH;
    localparam RTU_NODE_OFF_QMAX   = RTU_NODE_OFF_QMIN  + 3 * RTU_BVH_WIDTH;

    // Decoded-field span of a node/leaf: the highest byte the decoders read
    // (qaabb_max for a node, the third vertex for a leaf triangle). Nodes and
    // leaves are packed contiguously in the scene at arbitrary byte offsets, so
    // a structure may straddle several aligned cache lines; the scheduler
    // fetches LINES lines and byte-aligns the assembled image before decode.
    localparam RTU_NODE_DEC_BYTES  = RTU_NODE_OFF_QMAX + 3 * RTU_BVH_WIDTH;
    localparam RTU_NODE_IMG_BITS   = RTU_NODE_DEC_BYTES * 8;
    localparam RTU_LINE_BYTES      = `VX_CFG_MEM_BLOCK_SIZE;
    // worst-case lines a span of N bytes can touch = ((63 + N - 1) / 64) + 1
    localparam RTU_NODE_LINES      = ((RTU_LINE_BYTES - 1 + RTU_NODE_DEC_BYTES - 1) / RTU_LINE_BYTES) + 1;
    localparam RTU_LINE_SEL_BITS   = `CLOG2(RTU_LINE_BYTES);   // byte-in-line index width
    localparam RTU_LINES_BITS      = `CLOG2(RTU_NODE_LINES + 1);

    // 16 B scene header: root_node_offset @0, scene_kind @4.
    localparam RTU_SCENE_OFF_ROOT  = 0;
    localparam RTU_SCENE_OFF_KIND  = 4;
    localparam RTU_SCENE_KIND_TRI_LIST = 32'd0;
    localparam RTU_SCENE_KIND_TLAS = 32'd1;
    localparam RTU_SCENE_KIND_BVH4 = 32'd2;
    localparam RTU_SCENE_KIND_BVH6 = 32'd3;

    // 16 B leaf header: kind @0 (+count bits 8..15), geometry_index @4,
    // flags @8, prim_base @12. 40 B triangle: v0 @0, v1 @12, v2 @24, flags @36.
    localparam RTU_LEAF_HDR_BYTES  = 16;
    localparam RTU_LEAF_OFF_GEOM   = 4;
    localparam RTU_LEAF_OFF_FLAGS  = 8;
    localparam RTU_LEAF_OFF_PRIM   = 12;
    localparam RTU_TRI_STRIDE      = 40;

    // Per-tri / per-leaf flag-word bit layout:
    //   bit  0     OPAQUE      — clear => non-opaque => AHS/IS yield
    //   bit  1     PROCEDURAL  — yield IS instead of AHS
    //   bits 8..15 SBT_IDX     — keys the kernel's switch(sbt_idx)
    localparam RTU_TRI_FLAG_OPAQUE   = 32'h1;
    localparam RTU_TRI_FLAG_PROC     = 32'h2;
    localparam RTU_TRI_SBT_IDX_SHIFT = 8;
    localparam RTU_TRI_SBT_IDX_MASK  = 32'hff;

    // Triangle vertex byte offsets within a leaf (header + triangle record).
    localparam RTU_TRI_OFF_V0      = RTU_LEAF_HDR_BYTES;       // 16
    localparam RTU_TRI_OFF_V1      = RTU_TRI_OFF_V0 + 12;      // 28
    localparam RTU_TRI_OFF_V2      = RTU_TRI_OFF_V1 + 12;      // 40
    localparam RTU_TRI_OFF_FLAGS   = RTU_TRI_OFF_V2 + 12;      // 52 (per-triangle opacity/SBT flags)
    localparam RTU_LEAF_DEC_BYTES  = RTU_TRI_OFF_FLAGS + 4;    // 56 (through flags)

    // ─────────────────────────────────────────────────────────────────
    // Flat triangle-list scene (RTU_BVH_WIDTH=0). 16 B header: word0 =
    // triangle_count. Triangles packed contiguously at stride 40 B with NO
    // per-triangle leaf header: v0 @0, v1 @12, v2 @24, flags @36 within the
    // record.
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_SCENE_HDR_BYTES = 16;
    localparam RTU_FLAT_OFF_V0     = 0;
    localparam RTU_FLAT_OFF_V1     = 12;
    localparam RTU_FLAT_OFF_V2     = 24;
    localparam RTU_FLAT_OFF_FLAGS  = 36;
    // Decode span must cover the flag word (byte 36..39) so the AHS/IS
    // classifier sees per-tri opacity even for records straddling a line.
    localparam RTU_FLAT_DEC_BYTES  = RTU_FLAT_OFF_FLAGS + 4;   // 40 (through flags)
    localparam RTU_FLAT_IMG_BITS   = RTU_FLAT_DEC_BYTES * 8;
    localparam RTU_FLAT_LINES      = ((`VX_CFG_MEM_BLOCK_SIZE - 1 + RTU_FLAT_DEC_BYTES - 1) / `VX_CFG_MEM_BLOCK_SIZE) + 1;
    localparam RTU_FLAT_LINES_BITS = `CLOG2(RTU_FLAT_LINES + 1);

    // ─────────────────────────────────────────────────────────────────
    // TLAS instance record (64 B). The 3x4 row-major affine transform
    // (object→world) occupies floats 0..11;
    // the walker applies its inverse (VX_rtu_xform) to bring the world ray into
    // object space. The two TLAS variants share xform/blas/custom but differ in
    // where instance_id and cull_mask sit:
    //   flat TLAS : blas_off@48, custom_id@52, cull_mask@56; instance_id = loop idx
    //   BVH inst  : blas_root@48, custom_id@52, instance_id@56, cull_mask@60
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_INST_STRIDE       = 64;
    localparam RTU_INST_OFF_XFORM    = 0;    // 12 fp32 (3x4 row-major)
    localparam RTU_INST_OFF_BLAS     = 48;   // blas byte offset (flat & BVH)
    localparam RTU_INST_OFF_CUSTOM   = 52;   // custom_id (VK_INSTANCE_CUSTOM_INDEX)
    localparam RTU_INST_OFF_CULL_FLAT= 56;   // flat-TLAS cull_mask
    localparam RTU_INST_OFF_ID_BVH   = 56;   // BVH instance_id (HW-assigned)
    localparam RTU_INST_OFF_CULL_BVH = 60;   // BVH instance cull_mask
    // Instance flags (VkGeometryInstanceFlagBits, low byte) pack into the
    // reserved second byte (bits 15..8) of the cull_mask word — cull_mask uses
    // only its low byte — so both TLAS variants carry them without growing the
    // 64 B record. The walker composes them with the ray-flag / per-tri opacity
    // classifier (FORCE_{,NO_}OPAQUE override opacity; TRIANGLE_FACING_CULL_
    // DISABLE / FLIP_FACING alter face culling).
    localparam RTU_INST_FLAGS_SHIFT       = 8;    // bit offset within cull word
    localparam RTU_INST_FLAG_TRI_CULL_DIS = 8'h1; // TRIANGLE_FACING_CULL_DISABLE
    localparam RTU_INST_FLAG_TRI_FLIP     = 8'h2; // TRIANGLE_FLIP_FACING
    localparam RTU_INST_FLAG_FORCE_OPAQUE = 8'h4; // FORCE_OPAQUE
    localparam RTU_INST_FLAG_FORCE_NO_OPQ = 8'h8; // FORCE_NO_OPAQUE
    // The decoders read all 64 bytes; an instance record may straddle two lines.
    localparam RTU_INST_DEC_BYTES    = RTU_INST_STRIDE;
    localparam RTU_INST_IMG_BITS     = RTU_INST_DEC_BYTES * 8;
    localparam RTU_INST_LINES        = ((RTU_LINE_BYTES - 1 + RTU_INST_DEC_BYTES - 1) / RTU_LINE_BYTES) + 1;

    // ─────────────────────────────────────────────────────────────────
    // Decoded internal node — width-generic view the box-PE array consumes.
    // origin is fp32, exp is int8, the
    // per-child quantized AABB corners are int8 (one per axis).
    // ─────────────────────────────────────────────────────────────────
    typedef struct packed {
        logic [2:0][31:0]                      origin;     // common origin (fp32)
        logic [2:0][7:0]                       exp;        // per-axis exponent (int8)
        logic [RTU_CHILD_BITS-1:0]             n_children;
        logic [RTU_NODE_W-1:0][31:0]           child_off;  // raw child-offset words
        logic [RTU_NODE_W-1:0][2:0][7:0]       qmin;       // quantized child mins
        logic [RTU_NODE_W-1:0][2:0][7:0]       qmax;       // quantized child maxs
    } rtu_node_t;

    // ─────────────────────────────────────────────────────────────────
    // Per-lane ray descriptor (snapshot of the ray-state slots at trace).
    // ─────────────────────────────────────────────────────────────────
    typedef struct packed {
        logic [2:0][31:0]                  origin;     // fp32 ray origin
        logic [2:0][31:0]                  dir;        // fp32 ray direction
        logic [31:0]                       t_min;
        logic [31:0]                       t_max;
        logic [31:0]                       flags;      // VX_RT_FLAG_*
        logic [31:0]                       cull_mask;
        logic [`VX_CFG_MEM_ADDR_WIDTH-1:0] scene_base; // device byte address of the scene buffer
    } rtu_ray_t;

    // ─────────────────────────────────────────────────────────────────
    // Window slot spans the RTU walks as bus master. The slot map is ordered so
    // every span is contiguous (see VX_types.toml [rtu_slots]), so the RTU adds
    // an index to a base instead of carrying a beat->slot table that the window
    // would have to mirror.
    //
    //   ray input  [RAY_BASE, +RAY_SLOTS)   read once, when the warp arms
    //   hit attrs  [RES_BASE, +RES_HIT)     written by every response
    //   candidate  [RES_BASE, +RES_CAND)    + object ray, cb_type, sbt, handle
    //   status      STATUS_SLOT             written LAST
    //
    // Status is last by design: writing it is what completes the warp's parked
    // WAIT, so every slot the warp may then read must already be in place.
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_RAY_BASE    = `VX_RT_RAY_ORIGIN;
    localparam RTU_RAY_SLOTS   = 10;  // origin[3], dir[3], t_min, t_max, flags, cull_mask
    localparam RTU_RES_BASE    = `VX_RT_HIT_T;
    localparam RTU_RES_HIT     = 7;   // hit_t, hit_u, hit_v, prim, instance, geometry, custom
    localparam RTU_RES_CAND    = 16;  // + object ray (6), cb_type, cb_sbt_idx, cb_handle
    localparam RTU_STATUS_SLOT = `VX_RT_STATUS;
    localparam RTU_IDX_BITS    = `CLOG2(RTU_RES_CAND + 1);

`IGNORE_UNUSED_END

endpackage
