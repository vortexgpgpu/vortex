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
// vortex::raytrace — RTU host library (ISA v2 proposal §5.3). The host-side
// responsibilities for the RTU are two: transcoding an acceleration structure
// into the CW-BVH<W> byte layout the RtuCore walks (build_bvh_scene), and
// programming the per-dispatch VX_DCR_RTU_* block (config_t / program). Both
// live here so the runtime owns the scene format and dispatch config while the
// kernel sees only the per-ray ISA (sw/kernel/include/vx_raytrace.h).

#ifndef __VX_RAYTRACE_HOST_H__
#define __VX_RAYTRACE_HOST_H__

#include <cstdint>
#include <cstring>
#include <vector>

#include <rtu_cfg.h>     // shared host/device CW-BVH format constants
#include <VX_types.h>    // VX_DCR_RTU_*, RTU_CFG_*
#include <vortex.h>      // vx_device_h, vx_dcr_write

namespace vortex {
namespace raytrace {

namespace detail {

// Pack the VX_DCR_RTU_CONFIG word (scene_kind / bvh_width / cull defaults).
inline uint32_t pack_config(uint32_t scene_kind, uint32_t bvh_width,
                            uint32_t cull_defaults) {
  return (scene_kind    << RTU_CFG_SCENE_KIND_LSB)
       | (bvh_width     << RTU_CFG_BVH_WIDTH_LSB)
       | (cull_defaults << RTU_CFG_CULL_LSB);
}

} // namespace detail

// ── Host-side scene preparation ─────────────────────────────────────────
//
// Transcode a host acceleration structure into the CW-BVH<W> byte layout the
// RtuCore walks. W = 4 (scene_kind=BVH4) or 6 (scene_kind=BVH6). Emits the same
// bytes the SimX walker consumes (sim/simx/rtu/rtu_bvh.h), so the host builder
// and the device share one format. The current transcode emits a single-leaf
// scene (the root IS the leaf holding every triangle) — a valid, if
// unpartitioned, BVH; spatial partitioning into internal nodes is a future
// optimization. Returns false for an empty input.
template <uint32_t W>
inline bool build_bvh_scene(const host_bvh_t& src,
                            std::vector<uint8_t>& out_scene,
                            uint64_t& out_root_offset) {
  static_assert(W == 4 || W == 6, "CW-BVH width must be 4 or 6");
  if (src.tri_count == 0 || src.tris == nullptr)
    return false;

  const uint32_t scene_kind = (W == 6) ? RTU_SCENE_KIND_BVH6 : RTU_SCENE_KIND_BVH4;
  const uint32_t root_off   = RTU_BVH_SCENE_HDR_BYTES;  // leaf follows the header

  out_scene.assign(RTU_BVH_SCENE_HDR_BYTES + RTU_BVH_LEAF_HDR_BYTES
                     + (size_t)src.tri_count * RTU_BVH_TRI_STRIDE, 0);
  uint8_t* base = out_scene.data();

  // Scene header: root offset, scene_kind, node_count (0 — root is a leaf),
  // leaf_count.
  uint32_t hdr[4] = { root_off, scene_kind, 0u, 1u };
  std::memcpy(base, hdr, sizeof(hdr));

  // Leaf header: kind|count, geometry_index, flags (per-tri flags drive
  // opacity), prim_base.
  uint32_t lh[4] = {
    RTU_BVH_KIND_LEAF_TRI | (src.tri_count << RTU_BVH_COUNT_SHIFT),
    src.geometry_index, 0u, 0u
  };
  std::memcpy(base + root_off, lh, sizeof(lh));

  // Triangles (40 B each: v0, v1, v2, flags).
  uint8_t* tri = base + root_off + RTU_BVH_LEAF_HDR_BYTES;
  for (uint32_t i = 0; i < src.tri_count; ++i) {
    const host_tri_t& s = src.tris[i];
    float verts[9] = {
      s.v0[0], s.v0[1], s.v0[2],
      s.v1[0], s.v1[1], s.v1[2],
      s.v2[0], s.v2[1], s.v2[2],
    };
    uint8_t* dst = tri + (size_t)i * RTU_BVH_TRI_STRIDE;
    std::memcpy(dst, verts, sizeof(verts));
    std::memcpy(dst + 36, &s.flags, sizeof(uint32_t));
  }

  out_root_offset = root_off;
  return true;
}

// ── Per-dispatch configuration (programs VX_DCR_RTU_* once per launch) ──

struct config_t {
  uint32_t scene_kind     = RTU_SCENE_KIND_BVH4;
  uint32_t bvh_width      = 4;
  uint32_t cull_defaults  = 0;
  uint64_t callback_entry = 0;  // mtvec dispatcher PC (Phase 2)
  uint32_t reform_thresh  = 0;  // reformation threshold (Phase 3)
};

// Write the config to the VX_DCR_RTU_* block. Call before vx_start. Returns the
// first non-zero vx_dcr_write status, or 0 on success.
inline int program(vx_device_h dev, const config_t& cfg) {
  int ret;
#define VX_RTU__W(reg, val) do { ret = vx_dcr_write(dev, (reg), (val)); if (ret) return ret; } while (0)
  VX_RTU__W(VX_DCR_RTU_CONFIG,
            detail::pack_config(cfg.scene_kind, cfg.bvh_width, cfg.cull_defaults));
  VX_RTU__W(VX_DCR_RTU_CB_ENTRY_LO, (uint32_t)(cfg.callback_entry & 0xffffffffu));
  VX_RTU__W(VX_DCR_RTU_CB_ENTRY_HI, (uint32_t)(cfg.callback_entry >> 32));
  VX_RTU__W(VX_DCR_RTU_REFORM_THRESH, cfg.reform_thresh);
#undef VX_RTU__W
  return 0;
}

} // namespace raytrace
} // namespace vortex

#endif // __VX_RAYTRACE_HOST_H__
