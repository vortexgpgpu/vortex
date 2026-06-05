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

    // CUSTOM1 / funct3=5 sub-op selector (funct7[1:0]). For set/get the
    // RTU register-file slot rides in funct7[6:2].
    localparam RTU_SUBOP_BITS  = 2;
    localparam RTU_SUBOP_SET   = 2'd0;  // slot <- rs1, no writeback
    localparam RTU_SUBOP_GET   = 2'd1;  // rd <- slot
    localparam RTU_SUBOP_TRACE = 2'd2;  // rd <- handle, start the walk
    localparam RTU_SUBOP_WAIT  = 2'd3;  // rd <- terminal status

    // Per-lane ray-state / result register file, one 32-bit word per slot.
    localparam RTU_SLOT_COUNT  = `VX_RT_SLOT_COUNT;
    localparam RTU_SLOT_BITS   = `CLOG2(`VX_RT_SLOT_COUNT);

    // ─────────────────────────────────────────────────────────────────
    // Traversal / PE configuration (from VX_CFG_RTU_*)
    // ─────────────────────────────────────────────────────────────────
    localparam RTU_BVH_WIDTH   = `VX_CFG_RTU_BVH_WIDTH;   // node fan-out (4 = CW-BVH4)
    localparam RTU_BOX_PE      = `VX_CFG_RTU_BOX_PE;      // parallel ray-AABB lanes
    localparam RTU_TRI_PE      = `VX_CFG_RTU_TRI_PE;      // parallel ray-triangle lanes
    localparam RTU_STACK_DEPTH = `VX_CFG_RTU_STACK_DEPTH; // short-stack depth
    localparam RTU_NODE_LATENCY= `VX_CFG_RTU_NODE_LATENCY;// box-PE pipeline depth
    localparam RTU_TRI_LATENCY = `VX_CFG_RTU_TRI_LATENCY; // tri-PE pipeline depth
    localparam RTU_CHILD_BITS  = `CLOG2(RTU_BVH_WIDTH + 1);
    localparam RTU_STACK_BITS  = `CLOG2(RTU_STACK_DEPTH + 1);

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
    // Byte offsets within a CW-BVH4 internal node (64 B = one cache line),
    // matching the host transcode / SimX rtu_bvh.h layout. A node arrives
    // as one aligned line; fields are sliced out by these offsets.
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

    // 16 B scene header (kRtuSceneKindBvh*): root_node_offset @0, scene_kind @4.
    localparam RTU_SCENE_OFF_ROOT  = 0;
    localparam RTU_SCENE_OFF_KIND  = 4;
    localparam RTU_SCENE_KIND_BVH4 = 32'd2;
    localparam RTU_SCENE_KIND_BVH6 = 32'd3;

    // 16 B leaf header: kind @0 (+count bits 8..15), geometry_index @4,
    // flags @8, prim_base @12. 40 B triangle: v0 @0, v1 @12, v2 @24, flags @36.
    localparam RTU_LEAF_HDR_BYTES  = 16;
    localparam RTU_LEAF_OFF_GEOM   = 4;
    localparam RTU_LEAF_OFF_FLAGS  = 8;
    localparam RTU_LEAF_OFF_PRIM   = 12;
    localparam RTU_TRI_STRIDE      = 40;

    // Triangle vertex byte offsets within a leaf (header + triangle record).
    localparam RTU_TRI_OFF_V0      = RTU_LEAF_HDR_BYTES;       // 16
    localparam RTU_TRI_OFF_V1      = RTU_TRI_OFF_V0 + 12;      // 28
    localparam RTU_TRI_OFF_V2      = RTU_TRI_OFF_V1 + 12;      // 40
    localparam RTU_LEAF_DEC_BYTES  = RTU_TRI_OFF_V2 + 12;      // 52 (through v2)

    // ─────────────────────────────────────────────────────────────────
    // Decoded internal node — width-generic view the box-PE array consumes
    // (RTL analog of SimX VxBvhNodeView). origin is fp32, exp is int8, the
    // per-child quantized AABB corners are int8 (one per axis).
    // ─────────────────────────────────────────────────────────────────
    typedef struct packed {
        logic [2:0][31:0]                      origin;     // common origin (fp32)
        logic [2:0][7:0]                       exp;        // per-axis exponent (int8)
        logic [RTU_CHILD_BITS-1:0]             n_children;
        logic [RTU_BVH_WIDTH-1:0][31:0]        child_off;  // raw child-offset words
        logic [RTU_BVH_WIDTH-1:0][2:0][7:0]    qmin;       // quantized child mins
        logic [RTU_BVH_WIDTH-1:0][2:0][7:0]    qmax;       // quantized child maxs
    } rtu_node_t;

    // ─────────────────────────────────────────────────────────────────
    // Per-lane ray descriptor (snapshot of the ray-state slots at trace).
    // ─────────────────────────────────────────────────────────────────
    typedef struct packed {
        logic [2:0][31:0] origin;       // fp32 ray origin
        logic [2:0][31:0] dir;          // fp32 ray direction
        logic [31:0]      t_min;
        logic [31:0]      t_max;
        logic [31:0]      flags;        // VX_RT_FLAG_*
        logic [31:0]      cull_mask;
        logic [31:0]      scene_base;   // device byte address of the scene buffer
    } rtu_ray_t;

endpackage
