#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 on-device triangle setup — shared host/device ABI.
//
// Tests the setup front end (charter §6.1 / gfx_v2_vertex_setup_pipeline.md
// stages A–D): VS output (resident vertex_t) -> near-plane clip -> per-(sub)tri
// edge equations + attribute deltas + screen bbox, emitted as a dense
// rast_prim_t[] + bbox[] — exactly binning stage 1's input.
//
// Clipping (stage C) is against the GL near plane z_clip + w_clip >= 0
// (Sutherland-Hodgman, fan-triangulated), so a triangle crossing the near plane
// expands to 0..2 subtriangles — the count->scan->emit machinery. This closes
// the w-clip / near-plane shear hole that host Binning() leaves open (it does
// only a screen-bbox clamp). Far/guardband side planes are deferred (overdraw,
// not a correctness hole — bbox clamp bounds lateral extent).
//
// Three CP-sequenced launches (the launch-drain is the device barrier):
//   SETUP  multi-CTA, 1 thread/tri: clip + setup -> per-tri slots + keep count
//   SCAN   single-CTA: prefix-sum keep[] -> offset[], total kept P
//   EMIT   multi-CTA, 1 thread/tri: compact kept slots -> dense prim/bbox/vtx/pid
//
// Baseline = non-indexed triangle list (assembly is i={3t,3t+1,3t+2}).
// Validation: device output is bit-for-bit vs the shared host setup math, which
// is anchored against the real Binning() oracle on the no-clip subset; clipped
// subtriangles are checked with independent geometric invariants. Output carries
// a parent-tri pid (draw order) for the binning sort downstream.

#include <stdint.h>
#include "setup_types.h"   // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

#define SETUP_STAGE_SETUP 0   // multi-CTA: per-tri full setup -> slot+keep
#define SETUP_STAGE_SCAN  1   // single-CTA: prefix-sum keep -> offset, P
#define SETUP_STAGE_EMIT  2   // multi-CTA: compact kept slots -> dense out

// Clipping makes setup variable-output: each input triangle yields 0..MAX_SUB
// kept (sub)triangles, so keep[] holds a per-tri count (not just 0/1) and the
// per-tri slots are MAX_SUB-wide. The scan/emit pattern is otherwise unchanged.
typedef struct {
  uint32_t num_prims;     // input triangle count
  uint32_t stage;
  uint32_t width, height;
  uint32_t cull_mode;     // SETUP_CULL_* (0 = none / two-sided)
  uint32_t _pad;          // keep the uint64 address block 8-byte aligned
  uint64_t verts_addr;      // setup_vertex_t[3*num_prims]      (in: triangle list)
  uint64_t slot_prim_addr;  // rast_prim_t[num_prims*MAX_SUB]   (scratch)
  uint64_t slot_bbox_addr;  // setup_bbox_t[num_prims*MAX_SUB]  (scratch)
  uint64_t slot_vtx_addr;   // clip_tri_t[num_prims*MAX_SUB]    (scratch)
  uint64_t keep_addr;       // uint32[num_prims]                (scratch: 0..MAX_SUB)
  uint64_t offset_addr;     // uint32[num_prims + 1]            (scratch)
  uint64_t tsum_addr;       // uint32[T]                        (scratch: block scan)
  uint64_t prim_addr;       // rast_prim_t[num_prims*MAX_SUB]   (out: dense)
  uint64_t bbox_addr;       // setup_bbox_t[num_prims*MAX_SUB]  (out: dense)
  uint64_t vtx_addr;        // clip_tri_t[num_prims*MAX_SUB]    (out: dense clip verts)
  uint64_t pid_addr;        // uint32[num_prims*MAX_SUB]        (out: parent tri id)
  uint64_t meta_addr;       // uint32[1] = { P }                (out)
} kernel_arg_t;

#endif
