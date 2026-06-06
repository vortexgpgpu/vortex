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
//
// rt_raycast — a port of tests/regression/raycast whose per-pixel ray /
// scene intersection runs on the PRISM RTU instead of a software BVH
// traversal. The host loads an OBJ mesh, builds a CW-BVH4 acceleration
// structure (1 triangle per leaf so the RTU's gl_GeometryIndexEXT recovers
// the global triangle index), and the kernel fires one primary ray per
// pixel through vx_rt_trace / vx_rt_wait, then shades the hit with a single
// diffuse light. Shared host/device definitions live here.

#ifndef _RT_RAYCAST_COMMON_H_
#define _RT_RAYCAST_COMMON_H_

#include <stdint.h>

#ifdef __cplusplus
#include <cmath>
#endif

// ── minimal float3 (host + device) ──────────────────────────────────
typedef struct vec3 {
  float x, y, z;
} vec3;

static inline vec3 v3(float x, float y, float z) { vec3 r = {x, y, z}; return r; }
static inline vec3 v3add(vec3 a, vec3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
static inline vec3 v3sub(vec3 a, vec3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
static inline vec3 v3scale(vec3 a, float s) { return v3(a.x*s, a.y*s, a.z*s); }
static inline float v3dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline vec3 v3cross(vec3 a, vec3 b) {
  return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static inline float v3len(vec3 a) { return sqrtf(v3dot(a, a)); }
static inline vec3 v3norm(vec3 a) {
  float l = v3len(a);
  float inv = (l > 0.f) ? (1.f / l) : 0.f;
  return v3scale(a, inv);
}

// ── CW-BVH4 on-disk layout (mirror of sim/simx/rtu/rtu_bvh.h) ────────
#define VX_BVH_SCENE_KIND          2      // kRtuSceneKindBvh4
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_NODE_BYTES          64     // CW-BVH4 internal node
#define VX_BVH_WIDTH               4
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36

// CW-BVH4 internal-node field byte offsets.
#define VX_BVH_OFF_ORIGIN          4      // float[3]
#define VX_BVH_OFF_EXP             16     // int8[3]
#define VX_BVH_OFF_CHILD           20     // uint32[4]
#define VX_BVH_OFF_QMIN            36     // uint8[4][3]
#define VX_BVH_OFF_QMAX            48     // uint8[4][3]

#define VX_BVH_KIND_INTERNAL       0
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_CHILD_LEAF_FLAG     0x80000000u
#define VX_BVH_CHILD_EMPTY         0u
#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

// Per-triangle shading record (device-side array indexed by triangle id =
// the leaf's geometry_index). Vertex normals for barycentric interpolation.
typedef struct {
  vec3 n0, n1, n2;
} tri_shade_t;

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t dst_addr;       // framebuffer: uint32 RGB per pixel
  uint64_t scene_addr;     // CW-BVH4 scene buffer
  uint64_t shade_addr;     // tri_shade_t[num_tris]
  uint32_t num_tris;
  uint32_t pad0;
  vec3  camera_pos;
  vec3  camera_forward;
  vec3  camera_right;
  vec3  camera_up;
  float viewplane_x;
  float viewplane_y;
  vec3  light_pos;
  vec3  light_color;
  vec3  ambient_color;
  vec3  background_color;
} kernel_arg_t;

// Pack a [0,1] float3 colour into 0x00RRGGBB.
static inline uint32_t pack_rgb(vec3 c) {
  float r = c.x < 0.f ? 0.f : (c.x > 1.f ? 1.f : c.x);
  float g = c.y < 0.f ? 0.f : (c.y > 1.f ? 1.f : c.y);
  float b = c.z < 0.f ? 0.f : (c.z > 1.f ? 1.f : c.z);
  uint32_t ir = (uint32_t)(r * 255.f + 0.5f);
  uint32_t ig = (uint32_t)(g * 255.f + 0.5f);
  uint32_t ib = (uint32_t)(b * 255.f + 0.5f);
  return (ir << 16) | (ig << 8) | ib;
}

#endif // _RT_RAYCAST_COMMON_H_
