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

#pragma once

#include <algorithm>
#include <cmath>
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

// ── CW-BVH<W> builder ───────────────────────────────────────────────────
//
// Builds a real spatial hierarchy (internal nodes) over the host triangle
// list via top-down binned-SAH partitioning, then serializes it as the
// CW-BVH<W> byte layout the SimX walker consumes (rtu_bvh.h). The valuable,
// HW-faithful part is the internal-node tree + quantized child AABBs: the
// walker box-tests each child, descends nearest-first, and prunes whole
// sub-trees — exactly the cost the RTU is built to amortize.
//
// Leaves hold ONE triangle. The leaf format reports gl_PrimitiveID as
// `prim_base + within_leaf_index`, so a multi-triangle leaf can only preserve
// Vulkan primitive IDs when its triangles are a contiguous ascending run of
// source indices — which spatial SAH partitioning does not produce. One tri
// per leaf sets prim_base to the source index directly, so the reported
// primitive ID is always exact. Multi-tri leaf clustering (denser trees,
// fewer leaf fetches) needs a per-leaf primitive-ID remap field — a format
// extension — and is the documented follow-up.
class BvhBuilder {
public:
  BvhBuilder(const host_tri_t* tris, uint32_t geom, uint32_t width,
             std::vector<uint8_t>& out)
      : tris_(tris), geom_(geom), W_(width), out_(out) {}

  // Returns the byte offset of the root node, or UINT32_MAX on overflow.
  uint32_t build(uint32_t tri_count) {
    boxes_.resize(tri_count);
    std::vector<uint32_t> idx(tri_count);
    for (uint32_t i = 0; i < tri_count; ++i) {
      idx[i] = i;
      compute_box(i);
    }
    NodeRef root = build_node(idx);
    return overflow_ ? UINT32_MAX : root.offset;
  }

private:
  struct TriBox { float mn[3], mx[3], c[3]; };
  struct NodeRef { uint32_t offset; float mn[3], mx[3]; bool is_leaf; };

  void compute_box(uint32_t ti) {
    const host_tri_t& s = tris_[ti];
    const float* v[3] = { s.v0, s.v1, s.v2 };
    TriBox& b = boxes_[ti];
    for (int a = 0; a < 3; ++a) {
      b.mn[a] = std::min(v[0][a], std::min(v[1][a], v[2][a]));
      b.mx[a] = std::max(v[0][a], std::max(v[1][a], v[2][a]));
      b.c[a]  = (b.mn[a] + b.mx[a]) * 0.5f;
    }
  }

  static float half_area(const float mn[3], const float mx[3]) {
    float dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
    if (dx < 0) dx = 0;
    if (dy < 0) dy = 0;
    if (dz < 0) dz = 0;
    return dx * dy + dy * dz + dz * dx;
  }

  // Append a single-triangle leaf (16 B header + 40 B triangle).
  NodeRef emit_leaf(uint32_t ti) {
    uint32_t off = (uint32_t)out_.size();
    uint8_t buf[RTU_BVH_LEAF_HDR_BYTES + RTU_BVH_TRI_STRIDE] = {0};
    uint32_t kind = RTU_BVH_KIND_LEAF_TRI | (1u << RTU_BVH_COUNT_SHIFT);
    std::memcpy(buf + 0,  &kind, 4);
    std::memcpy(buf + 4,  &geom_, 4);
    // flags(+8) = 0; prim_base(+12) = source index → exact gl_PrimitiveID.
    std::memcpy(buf + 12, &ti, 4);
    const host_tri_t& s = tris_[ti];
    float verts[9] = { s.v0[0], s.v0[1], s.v0[2],
                       s.v1[0], s.v1[1], s.v1[2],
                       s.v2[0], s.v2[1], s.v2[2] };
    std::memcpy(buf + RTU_BVH_LEAF_HDR_BYTES, verts, sizeof(verts));
    std::memcpy(buf + RTU_BVH_LEAF_HDR_BYTES + 36, &s.flags, 4);
    out_.insert(out_.end(), buf, buf + sizeof(buf));

    NodeRef r; r.offset = off; r.is_leaf = true;
    for (int a = 0; a < 3; ++a) { r.mn[a] = boxes_[ti].mn[a]; r.mx[a] = boxes_[ti].mx[a]; }
    return r;
  }

  // Append an internal node over up to W children, quantizing each child AABB
  // about a common origin (= node min) and per-axis exponent. Quantization is
  // CONSERVATIVE: min rounds down, max rounds up, so the walker's reconstructed
  // child box always contains the true child box (never culls a real hit).
  NodeRef emit_internal(const NodeRef* ch, uint32_t n) {
    float mn[3] = { ch[0].mn[0], ch[0].mn[1], ch[0].mn[2] };
    float mx[3] = { ch[0].mx[0], ch[0].mx[1], ch[0].mx[2] };
    for (uint32_t i = 1; i < n; ++i)
      for (int a = 0; a < 3; ++a) {
        mn[a] = std::min(mn[a], ch[i].mn[a]);
        mx[a] = std::max(mx[a], ch[i].mx[a]);
      }

    int8_t ex[3];
    float  step[3];
    for (int a = 0; a < 3; ++a) {
      float ext = mx[a] - mn[a];
      int e = 0;
      if (ext > 0.f) {
        // smallest e with 2^e >= ext/255 → ceil(ext/2^e) <= 255 (fits uint8).
        std::frexp(ext / 255.0f, &e);
        if (e < -127) e = -127;
        if (e >  127) e =  127;
      }
      ex[a]   = (int8_t)e;
      step[a] = std::ldexp(1.0f, e);
    }

    uint32_t node_bytes = (W_ == 6) ? RTU_BVH_NODE6_BYTES : RTU_BVH_NODE4_BYTES;
    uint32_t off = (uint32_t)out_.size();
    std::vector<uint8_t> buf(node_bytes, 0);
    uint32_t kind = RTU_BVH_KIND_INTERNAL | (n << RTU_BVH_COUNT_SHIFT);
    std::memcpy(buf.data() + 0, &kind, 4);
    std::memcpy(buf.data() + RTU_BVH_NODE_ORIGIN_OFF, mn, 12);
    std::memcpy(buf.data() + RTU_BVH_NODE_EXP_OFF, ex, 3);

    uint32_t child_off = RTU_BVH_NODE_CHILD_OFF;
    uint32_t qmin_off  = child_off + 4 * W_;
    uint32_t qmax_off  = qmin_off  + 3 * W_;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t co = ch[i].offset
                  | (ch[i].is_leaf ? RTU_BVH_CHILD_LEAF_FLAG : 0u);
      std::memcpy(buf.data() + child_off + 4 * i, &co, 4);
      for (int a = 0; a < 3; ++a) {
        int qmn = (int)std::floor((ch[i].mn[a] - mn[a]) / step[a]);
        int qmx = (int)std::ceil ((ch[i].mx[a] - mn[a]) / step[a]);
        if (qmn < 0)     qmn = 0;
        if (qmn > 255)   qmn = 255;
        if (qmx < qmn)   qmx = qmn;
        if (qmx > 255)   qmx = 255;
        buf[qmin_off + 3 * i + a] = (uint8_t)qmn;
        buf[qmax_off + 3 * i + a] = (uint8_t)qmx;
      }
    }
    out_.insert(out_.end(), buf.begin(), buf.end());

    NodeRef r; r.offset = off; r.is_leaf = false;
    for (int a = 0; a < 3; ++a) { r.mn[a] = mn[a]; r.mx[a] = mx[a]; }
    return r;
  }

  // Binned-SAH binary partition of `in` (size >= 2) into A, B (both non-empty).
  // Falls back to an index median when centroids coincide or SAH degenerates,
  // so the recursion always makes progress toward single-triangle leaves.
  void split2(const std::vector<uint32_t>& in,
              std::vector<uint32_t>& A, std::vector<uint32_t>& B) {
    float cmn[3] = { boxes_[in[0]].c[0], boxes_[in[0]].c[1], boxes_[in[0]].c[2] };
    float cmx[3] = { cmn[0], cmn[1], cmn[2] };
    for (uint32_t ti : in)
      for (int a = 0; a < 3; ++a) {
        cmn[a] = std::min(cmn[a], boxes_[ti].c[a]);
        cmx[a] = std::max(cmx[a], boxes_[ti].c[a]);
      }
    int axis = 0;
    float ext = cmx[0] - cmn[0];
    for (int a = 1; a < 3; ++a)
      if (cmx[a] - cmn[a] > ext) { ext = cmx[a] - cmn[a]; axis = a; }

    if (ext <= 0.f) { median_split(in, A, B); return; }

    constexpr int NB = 12;
    struct Bin { int count; float mn[3], mx[3]; };
    Bin bins[NB];
    for (int b = 0; b < NB; ++b) {
      bins[b].count = 0;
      for (int a = 0; a < 3; ++a) { bins[b].mn[a] = 1e30f; bins[b].mx[a] = -1e30f; }
    }
    float k = NB * (1.f - 1e-5f) / ext;
    auto bin_of = [&](uint32_t ti) {
      int b = (int)(k * (boxes_[ti].c[axis] - cmn[axis]));
      return b < 0 ? 0 : (b >= NB ? NB - 1 : b);
    };
    for (uint32_t ti : in) {
      Bin& bn = bins[bin_of(ti)];
      ++bn.count;
      for (int a = 0; a < 3; ++a) {
        bn.mn[a] = std::min(bn.mn[a], boxes_[ti].mn[a]);
        bn.mx[a] = std::max(bn.mx[a], boxes_[ti].mx[a]);
      }
    }

    // Sweep split planes 1..NB-1; SAH cost = Al*Nl + Ar*Nr.
    float l_area[NB], r_area[NB];
    int   l_cnt[NB],  r_cnt[NB];
    {
      float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
      int cnt = 0;
      for (int b = 0; b < NB; ++b) {
        if (bins[b].count) {
          cnt += bins[b].count;
          for (int a = 0; a < 3; ++a) {
            mn[a] = std::min(mn[a], bins[b].mn[a]);
            mx[a] = std::max(mx[a], bins[b].mx[a]);
          }
        }
        l_cnt[b] = cnt; l_area[b] = cnt ? half_area(mn, mx) : 0.f;
      }
    }
    {
      float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
      int cnt = 0;
      for (int b = NB - 1; b >= 0; --b) {
        if (bins[b].count) {
          cnt += bins[b].count;
          for (int a = 0; a < 3; ++a) {
            mn[a] = std::min(mn[a], bins[b].mn[a]);
            mx[a] = std::max(mx[a], bins[b].mx[a]);
          }
        }
        r_cnt[b] = cnt; r_area[b] = cnt ? half_area(mn, mx) : 0.f;
      }
    }
    int   best_plane = -1;
    float best_cost  = 1e30f;
    for (int p = 1; p < NB; ++p) {
      if (l_cnt[p - 1] == 0 || r_cnt[p] == 0) continue;
      float cost = l_area[p - 1] * l_cnt[p - 1] + r_area[p] * r_cnt[p];
      if (cost < best_cost) { best_cost = cost; best_plane = p; }
    }
    if (best_plane < 0) { median_split(in, A, B); return; }

    for (uint32_t ti : in)
      (bin_of(ti) < best_plane ? A : B).push_back(ti);
    if (A.empty() || B.empty()) { A.clear(); B.clear(); median_split(in, A, B); }
  }

  // Split `in` at its index median (sorted ascending) — always progresses.
  void median_split(const std::vector<uint32_t>& in,
                    std::vector<uint32_t>& A, std::vector<uint32_t>& B) {
    std::vector<uint32_t> s(in);
    std::sort(s.begin(), s.end());
    size_t half = s.size() / 2;
    A.assign(s.begin(), s.begin() + half);
    B.assign(s.begin() + half, s.end());
  }

  // Recursive top-down build. A leaf for a single triangle; otherwise split
  // into up to W groups (repeated binary SAH on the largest group) and emit
  // one internal node over their roots.
  NodeRef build_node(const std::vector<uint32_t>& idx) {
    if (overflow_ || out_.size() > RTU_BVH_MAX_SCENE_BYTES) {
      overflow_ = true;
      return emit_leaf(idx[0]);  // bail safely; caller checks overflow_
    }
    if (idx.size() == 1)
      return emit_leaf(idx[0]);

    std::vector<std::vector<uint32_t>> groups;
    groups.push_back(idx);
    // Repeatedly split the largest splittable (size >= 2) group until we have W.
    while (groups.size() < W_) {
      int pick = -1; size_t best = 1;
      for (size_t g = 0; g < groups.size(); ++g)
        if (groups[g].size() > best) { best = groups[g].size(); pick = (int)g; }
      if (pick < 0) break;  // every group is a single triangle
      std::vector<uint32_t> a, b;
      split2(groups[pick], a, b);
      groups[pick] = std::move(a);
      groups.push_back(std::move(b));
    }

    NodeRef ch[6];
    uint32_t n = 0;
    for (auto& g : groups)
      if (!g.empty()) ch[n++] = build_node(g);
    return emit_internal(ch, n);
  }

  const host_tri_t*    tris_;
  uint32_t             geom_;
  uint32_t             W_;
  std::vector<uint8_t>& out_;
  std::vector<TriBox>  boxes_;
  bool                 overflow_ = false;
};

} // namespace detail

// ── Host-side scene preparation ─────────────────────────────────────────
//
// Transcode a host acceleration structure into the CW-BVH<W> byte layout the
// RtuCore walks. W = 4 (scene_kind=BVH4) or 6 (scene_kind=BVH6). Emits the same
// bytes the SimX walker consumes (sim/simx/rtu/rtu_bvh.h), so the host builder
// and the device share one format. A single triangle yields a one-leaf scene
// (root IS the leaf); two or more triangles build a real binned-SAH internal-
// node hierarchy with quantized child AABBs (detail::BvhBuilder). Returns false
// for empty input or a scene that exceeds the walker's pre-fetch budget.
template <uint32_t W>
inline bool build_bvh_scene(const host_bvh_t& src,
                            std::vector<uint8_t>& out_scene,
                            uint64_t& out_root_offset) {
  static_assert(W == 4 || W == 6, "CW-BVH width must be 4 or 6");
  if (src.tri_count == 0 || src.tris == nullptr)
    return false;

  const uint32_t scene_kind = (W == 6) ? RTU_SCENE_KIND_BVH6 : RTU_SCENE_KIND_BVH4;

  // Reserve the 16 B scene header; nodes/leaves are appended after it in
  // post-order (children before parents), so the root is the last node and
  // all child_offsets are absolute from the scene base.
  out_scene.assign(RTU_BVH_SCENE_HDR_BYTES, 0);

  detail::BvhBuilder builder(src.tris, src.geometry_index, W, out_scene);
  uint32_t root_off = builder.build(src.tri_count);
  if (root_off == UINT32_MAX) {       // exceeded RTU_BVH_MAX_SCENE_BYTES
    out_scene.clear();
    return false;
  }

  // Scene header (VxBvhSceneHeader): root offset, scene_kind, scene_bytes
  // (total serialized size — sizes the RtuCore pre-fetch in rtu_memory.cpp),
  // leaf_count (diagnostic = tri_count, since leaves are single-triangle).
  uint32_t hdr[4] = { root_off, scene_kind,
                      (uint32_t)out_scene.size(), src.tri_count };
  std::memcpy(out_scene.data(), hdr, sizeof(hdr));

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
