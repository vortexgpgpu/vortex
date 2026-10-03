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
// rt_smoke_tie — host driver: which of several equal / near-equal-t opaque
// hits the RTU commits.
//
// The scene stacks coincident and near-coincident geometry, within one BLAS
// (exact twin triangles, a copy an ulp-scale step off the plane) and across
// instances (a BLAS instanced twice in place, rotated a quarter turn onto
// itself, shifted by half a cell, lifted by 2^-20, and a second BLAS holding
// copies of two of the first one's triangles). The
// RTU walks its own CW-BVH4; which of those hits it keeps is settled by the
// source BVH's visit order, carried as the visit-order tables the Vulkan
// driver appends (vortexpipe vp_launch.c): a TLAS table inside the scene and
// compact BLAS tables in a separate buffer reached by scene offset, triangle
// leaves naming their parent/side, instance leaves naming their BLAS table,
// TLAS rank and the TLAS table. Here the "source" trees are binary
// median-split trees built differently from the CW-BVH4, so their order is
// not the RTU's own.
//
// The same scene is traced twice: with the tables, and with the instance
// leaves' table words zeroed (the static (instance, geometry, primitive) tie
// key). Expected results are the SimX reference's (golden.h, regenerated with
// -g); every field of a hit is compared bit-exactly, and the two runs must disagree on
// some rays, or the tables were never consulted.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include <vortex2.h>
#include <VX_types.h>
#include <raytrace.h>
#include "common.h"
#include "golden.h"

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

namespace {

const char* kernel_file = "kernel.vxbin";

vx_device_h device   = nullptr;
vx_queue_h  queue    = nullptr;
vx_module_h module_  = nullptr;
vx_kernel_h kernel   = nullptr;
vx_buffer_h scene_buf[2] = { nullptr, nullptr };
vx_buffer_h tab_buf  = nullptr;
vx_buffer_h rays_buf = nullptr;
vx_buffer_h res_buf  = nullptr;

void cleanup() {
  if (!device) return;
  for (auto b : scene_buf) if (b) vx_buffer_release(b);
  if (tab_buf)  vx_buffer_release(tab_buf);
  if (rays_buf) vx_buffer_release(rays_buf);
  if (res_buf)  vx_buffer_release(res_buf);
  if (kernel)   vx_kernel_release(kernel);
  if (module_)  vx_module_release(module_);
  if (queue)    vx_queue_release(queue);
  vx_device_release(device);
  device = nullptr;
}

constexpr uint32_t kRoot = 0xffffffffu;

struct Tri {
  float    v[9];
  uint32_t geom;
};

struct Instance {
  float    otw[12];   // object -> world
  float    wto[12];   // world -> object, as the source driver stores it
  uint32_t blas;
  uint32_t id;
  uint32_t custom;
};

struct Box {
  float mn[3], mx[3];
};

Box tri_box(const Tri& t) {
  Box b;
  for (int a = 0; a < 3; ++a) {
    b.mn[a] = std::fmin(t.v[a], std::fmin(t.v[3 + a], t.v[6 + a]));
    b.mx[a] = std::fmax(t.v[a], std::fmax(t.v[3 + a], t.v[6 + a]));
  }
  return b;
}

Box box_union(const Box& x, const Box& y) {
  Box b;
  for (int a = 0; a < 3; ++a) {
    b.mn[a] = std::fmin(x.mn[a], y.mn[a]);
    b.mx[a] = std::fmax(x.mx[a], y.mx[a]);
  }
  return b;
}

// The world box of a BLAS box under an affine map, widened by an ulp.
Box xform_box(const float m[12], const Box& b) {
  Box r;
  for (int a = 0; a < 3; ++a) { r.mn[a] = INFINITY; r.mx[a] = -INFINITY; }
  for (int c = 0; c < 8; ++c) {
    const double p[3] = { (c & 1) ? b.mx[0] : b.mn[0],
                          (c & 2) ? b.mx[1] : b.mn[1],
                          (c & 4) ? b.mx[2] : b.mn[2] };
    for (int a = 0; a < 3; ++a) {
      const double w = m[a * 4] * p[0] + m[a * 4 + 1] * p[1] + m[a * 4 + 2] * p[2] + m[a * 4 + 3];
      r.mn[a] = std::fmin(r.mn[a], std::nextafter(float(w), -INFINITY));
      r.mx[a] = std::fmax(r.mx[a], std::nextafter(float(w), INFINITY));
    }
  }
  return r;
}

// ── the source BVH: a binary median-split tree ──────────────────────────
struct SrcNode {
  Box  box;
  int  child[2];   // node index, or ~item for a leaf
};

struct SrcTree {
  std::vector<SrcNode> nodes;
  int root = 0;

  int build(const std::vector<Box>& boxes, std::vector<uint32_t> items, int depth) {
    if (items.size() == 1) return ~int(items[0]);
    // split on the axis that cycles with depth, so the order differs from
    // the CW-BVH4's longest-axis split
    const int axis = depth % 3;
    std::stable_sort(items.begin(), items.end(), [&](uint32_t a, uint32_t b) {
      return boxes[a].mn[axis] + boxes[a].mx[axis] < boxes[b].mn[axis] + boxes[b].mx[axis];
    });
    const size_t h = items.size() / 2;
    std::vector<uint32_t> l(items.begin(), items.begin() + h), r(items.begin() + h, items.end());
    const int me = int(nodes.size());
    nodes.push_back(SrcNode());
    const int c0 = build(boxes, l, depth + 1);
    const int c1 = build(boxes, r, depth + 1);
    nodes[me].child[0] = c0;
    nodes[me].child[1] = c1;
    auto cbox = [&](int c) { return c < 0 ? boxes[~c] : nodes[c].box; };
    nodes[me].box = box_union(cbox(c0), cbox(c1));
    return me;
  }

  void make(const std::vector<Box>& boxes) {
    nodes.clear();
    std::vector<uint32_t> items(boxes.size());
    for (uint32_t i = 0; i < items.size(); ++i) items[i] = i;
    root = build(boxes, items, 0);
  }

  Box child_box(const std::vector<Box>& boxes, int c) const {
    return c < 0 ? boxes[~c] : nodes[c].box;
  }
};

// The source driver's table order: depth first, child 0 first. Table node
// j records its parent/side and depth; each leaf its parent/side, and leaves
// are ranked in visit order.
struct SrcTable {
  struct Node { int src; uint32_t ps, depth; };
  std::vector<Node>     nodes;
  std::vector<uint32_t> leaf_ps;     // per item
  std::vector<uint32_t> leaf_rank;   // per item
  std::vector<uint32_t> rank_item;   // per rank

  void make(const SrcTree& t, size_t n_items) {
    nodes.clear();
    leaf_ps.assign(n_items, kRoot);
    leaf_rank.assign(n_items, 0);
    rank_item.clear();
    std::vector<std::pair<int, uint32_t>> st = { { t.root, kRoot } };
    while (!st.empty()) {
      auto [c, ps] = st.back();
      st.pop_back();
      if (c < 0) {
        leaf_ps[~c] = ps;
        leaf_rank[~c] = uint32_t(rank_item.size());
        rank_item.push_back(uint32_t(~c));
        continue;
      }
      const uint32_t idx = uint32_t(nodes.size());
      nodes.push_back({ c, ps, ps == kRoot ? 0u : nodes[ps >> 1].depth + 1 });
      st.push_back({ t.nodes[c].child[1], idx << 1 | 1u });
      st.push_back({ t.nodes[c].child[0], idx << 1 });
    }
  }
};

// ── byte image helpers ──────────────────────────────────────────────────
struct Image {
  std::vector<uint8_t> b;
  uint32_t alloc(uint32_t n, uint32_t align = 4) {
    const uint32_t off = uint32_t((b.size() + align - 1) & ~size_t(align - 1));
    b.resize(off + n, 0);
    return off;
  }
  void u32(uint32_t off, uint32_t v) { std::memcpy(&b[off], &v, 4); }
  void f32(uint32_t off, float v) { std::memcpy(&b[off], &v, 4); }
  void box(uint32_t off, const Box& x) {
    for (int a = 0; a < 3; ++a) { f32(off + 4 * a, x.mn[a]); f32(off + 12 + 4 * a, x.mx[a]); }
  }
};

// ── the RTU's CW-BVH4: post-order, children before parents ─────────────
struct RtuRef { uint32_t off; Box box; bool leaf; };

RtuRef emit_internal(Image& img, const std::vector<RtuRef>& ch) {
  Box nb = ch[0].box;
  for (size_t i = 1; i < ch.size(); ++i) nb = box_union(nb, ch[i].box);
  int8_t ex[3];
  float  step[3];
  for (int a = 0; a < 3; ++a) {
    const float ext = nb.mx[a] - nb.mn[a];
    int e = -20;
    // headroom: 250 steps must cover the extent, so the 255 the quantizer
    // may round up to still lies past it
    if (ext > 0.f) std::frexp(ext / 250.0f, &e);
    ex[a]   = int8_t(std::max(-100, std::min(100, e)));
    step[a] = std::ldexp(1.0f, ex[a]);
  }
  const uint32_t off = img.alloc(RTU_BVH_NODE4_BYTES);
  img.u32(off, RTU_BVH_KIND_INTERNAL | (uint32_t(ch.size()) << RTU_BVH_COUNT_SHIFT));
  for (int a = 0; a < 3; ++a) img.f32(off + RTU_BVH_NODE_ORIGIN_OFF + 4 * a, nb.mn[a]);
  std::memcpy(&img.b[off + RTU_BVH_NODE_EXP_OFF], ex, 3);
  const uint32_t qmin_off = RTU_BVH_NODE_CHILD_OFF + 4 * 4;
  const uint32_t qmax_off = qmin_off + 3 * 4;
  for (size_t i = 0; i < ch.size(); ++i) {
    img.u32(off + RTU_BVH_NODE_CHILD_OFF + 4 * uint32_t(i),
            ch[i].off | (ch[i].leaf ? RTU_BVH_CHILD_LEAF_FLAG : 0u));
    for (int a = 0; a < 3; ++a) {
      // conservative by a step either way
      int qmn = int(std::floor((ch[i].box.mn[a] - nb.mn[a]) / step[a])) - 1;
      int qmx = int(std::ceil((ch[i].box.mx[a] - nb.mn[a]) / step[a])) + 1;
      qmn = std::max(0, std::min(255, qmn));
      qmx = std::max(qmn, std::min(255, qmx));
      img.b[off + qmin_off + 3 * i + a] = uint8_t(qmn);
      img.b[off + qmax_off + 3 * i + a] = uint8_t(qmx);
    }
  }
  return { off, nb, false };
}

// Groups of up to four along the longest centroid axis, recursively.
template <typename EmitLeaf>
RtuRef emit_cwbvh(Image& img, const std::vector<Box>& boxes, std::vector<uint32_t> items,
                  EmitLeaf&& leaf) {
  if (items.size() == 1) return { leaf(items[0]), boxes[items[0]], true };
  float cmn[3] = { INFINITY, INFINITY, INFINITY }, cmx[3] = { -INFINITY, -INFINITY, -INFINITY };
  for (uint32_t i : items)
    for (int a = 0; a < 3; ++a) {
      const float c = boxes[i].mn[a] + boxes[i].mx[a];
      cmn[a] = std::min(cmn[a], c);
      cmx[a] = std::max(cmx[a], c);
    }
  int axis = 0;
  for (int a = 1; a < 3; ++a)
    if (cmx[a] - cmn[a] > cmx[axis] - cmn[axis]) axis = a;
  std::stable_sort(items.begin(), items.end(), [&](uint32_t a, uint32_t b) {
    return boxes[a].mn[axis] + boxes[a].mx[axis] > boxes[b].mn[axis] + boxes[b].mx[axis];
  });
  const size_t g = std::min<size_t>(4, items.size());
  std::vector<RtuRef> ch;
  for (size_t k = 0; k < g; ++k) {
    const size_t lo = items.size() * k / g, hi = items.size() * (k + 1) / g;
    ch.push_back(emit_cwbvh(img, boxes, std::vector<uint32_t>(items.begin() + lo, items.begin() + hi), leaf));
  }
  return emit_internal(img, ch);
}

// ── the scene ───────────────────────────────────────────────────────────
std::vector<std::vector<Tri>> blases;
std::vector<Instance> insts;

void make_geometry() {
  // BLAS 0, the "sail": a 4x4 grid of quads on z = 0 over [-1, 1]^2 (geometry
  // 0); an exact twin of it (geometry 1); a copy an ulp-scale step above the
  // plane on some vertices (geometry 2).
  std::vector<Tri> sail;
  auto quad = [&](float x0, float y0, float x1, float y1, uint32_t geom, auto zf) {
    Tri a = { { x0, y0, zf(x0, y0), x1, y0, zf(x1, y0), x1, y1, zf(x1, y1) }, geom };
    Tri b = { { x0, y0, zf(x0, y0), x1, y1, zf(x1, y1), x0, y1, zf(x0, y1) }, geom };
    sail.push_back(a);
    sail.push_back(b);
  };
  for (uint32_t g = 0; g < 3; ++g) {
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        const float x0 = -1.f + 0.5f * i, y0 = -1.f + 0.5f * j;
        auto z = [&](float x, float y) -> float {
          (void)i; (void)j;
          switch (g) {
          // every other grid vertex: about an ulp of t for these rays
          case 2:  return (int((x + 1.f) * 4.f + (y + 1.f) * 4.f) & 2) ? 0x1p-22f : 0.f;
          default: return 0.f;
          }
        };
        quad(x0, y0, x0 + 0.5f, y0 + 0.5f, g, z);
      }
    }
  }
  blases.push_back(sail);

  // BLAS 1: a small plate tilted across the sail, plus an exact copy of two
  // sail triangles (so an instance of it coincides with the sail there).
  std::vector<Tri> plate;
  for (int i = 0; i < 4; ++i) {
    const float x0 = -0.75f + 0.375f * i;
    plate.push_back({ { x0, -0.5f, -0.25f, x0 + 0.375f, -0.5f, 0.25f, x0 + 0.375f, 0.5f, 0.25f }, 0 });
    plate.push_back({ { x0, -0.5f, -0.25f, x0 + 0.375f, 0.5f, 0.25f, x0, 0.5f, -0.25f }, 0 });
  }
  plate.push_back({ sail[10].v[0], sail[10].v[1], sail[10].v[2], sail[10].v[3], sail[10].v[4],
                    sail[10].v[5], sail[10].v[6], sail[10].v[7], sail[10].v[8], 1 });
  plate.push_back({ sail[11].v[0], sail[11].v[1], sail[11].v[2], sail[11].v[3], sail[11].v[4],
                    sail[11].v[5], sail[11].v[6], sail[11].v[7], sail[11].v[8], 1 });
  blases.push_back(plate);

  auto inst = [&](uint32_t blas, uint32_t id, const float otw[12], const float wto[12]) {
    Instance in;
    std::memcpy(in.otw, otw, sizeof(in.otw));
    std::memcpy(in.wto, wto, sizeof(in.wto));
    in.blas = blas;
    in.id = id;
    in.custom = 0x100 + id;
    insts.push_back(in);
  };
  const float I[12]   = { 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0 };
  // a quarter turn about z maps the grid onto itself
  const float R[12]   = { 0, -1, 0, 0,  1, 0, 0, 0,  0, 0, 1, 0 };
  const float Ri[12]  = { 0, 1, 0, 0,  -1, 0, 0, 0,  0, 0, 1, 0 };
  // half a cell along x
  const float T[12]   = { 1, 0, 0, 0.25f,  0, 1, 0, 0,  0, 0, 1, 0 };
  const float Ti[12]  = { 1, 0, 0, -0.25f,  0, 1, 0, 0,  0, 0, 1, 0 };
  // lifted off the plane by 2^-20
  const float U[12]   = { 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0x1p-20f };
  const float Ui[12]  = { 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, -0x1p-20f };
  // the plate: a quarter turn about x, then shifted (the RTU inverts an
  // instance transform as a rotation, so instances stay orthonormal)
  const float P[12]   = { 1, 0, 0, 0.125f,  0, 0, -1, 0,  0, 1, 0, 0 };
  const float Pi[12]  = { 1, 0, 0, -0.125f,  0, 0, 1, 0,  0, -1, 0, 0 };
  inst(0, 0, I, I);
  inst(1, 1, I, I);
  inst(0, 2, R, Ri);
  inst(1, 3, I, I);
  inst(0, 4, T, Ti);
  inst(0, 5, U, Ui);
  inst(1, 6, P, Pi);
}

std::vector<tie_ray_t> make_rays() {
  std::vector<tie_ray_t> rays;
  auto ray = [&](float ox, float oy, float oz, float dx, float dy, float dz) {
    rays.push_back({ { ox, oy, oz }, { dx, dy, dz }, 0.001f, 100.f });
  };
  // straight down and straight up, on cell interiors, edges and vertices
  for (int j = 0; j < 8; ++j)
    for (int i = 0; i < 8; ++i) {
      const float x = -0.875f + 0.25f * i, y = -0.875f + 0.25f * j;
      const float ex = ((i + j) & 1) ? 0.125f : 0.f;   // half the rays sit on an edge
      if ((i + j * 8) & 1) ray(x + ex, y, 3.f, 0.f, 0.f, -1.f);
      else                 ray(x + ex, y + ex, -2.f, 0.f, 0.f, 1.f);
    }
  // oblique, aimed at grid lines and vertices from scattered origins
  uint32_t s = 12345u;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return float(s >> 8) / float(1u << 24); };
  for (int k = 0; k < 128; ++k) {
    const float tx = -1.f + 0.25f * float(int(rnd() * 9.f));
    const float ty = -1.f + 0.5f * rnd() * 4.f;
    const float ox = tx + (rnd() - 0.5f) * 3.f, oy = ty + (rnd() - 0.5f) * 3.f;
    const float oz = (k & 3) ? 2.f + rnd() : -2.f - rnd();
    ray(ox, oy, oz, tx - ox, ty - oy, -oz);
  }
  // shallow rays skimming the plate and the sail
  for (int k = 0; k < 64; ++k) {
    const float ox = -3.f, oy = -0.9f + 1.8f * rnd();
    const float tz = (rnd() - 0.5f) * 0.02f;
    ray(ox, oy, 0.3f + tz, 3.f, (rnd() - 0.5f) * 0.4f, -0.3f - tz * 0.5f);
  }
  return rays;
}

// Build the scene image for one scene base address. tab_base is the BLAS
// table buffer's address, or 0 to leave the tables out.
struct Built {
  Image    scene;
  Image    tabs;
  uint32_t root = 0;
};

Built build(uint64_t scene_addr, uint64_t tab_addr, bool with_tables) {
  Built out;
  Image& img = out.scene;
  img.alloc(RTU_BVH_SCENE_HDR_BYTES);

  // BLAS: source trees + tables, then the CW-BVH4 with each leaf naming its
  // parent/side in the source tree
  std::vector<uint32_t> blas_root(blases.size()), blas_tab(blases.size(), 0);
  std::vector<Box> blas_box(blases.size());
  for (size_t b = 0; b < blases.size(); ++b) {
    const auto& tris = blases[b];
    std::vector<Box> boxes;
    for (const auto& t : tris) boxes.push_back(tri_box(t));
    SrcTree st;
    st.make(boxes);
    SrcTable tab;
    tab.make(st, tris.size());

    // BLAS table: { n_nodes, 0, 64, 32 }, node[j] = { own box, ps, depth }
    const uint32_t n = uint32_t(tab.nodes.size());
    const uint32_t toff = out.tabs.alloc(64 + 32 * n, 64);
    out.tabs.u32(toff, n);
    out.tabs.u32(toff + 8, 64);
    out.tabs.u32(toff + 12, 32);
    for (uint32_t j = 0; j < n; ++j) {
      const auto& nd = tab.nodes[j];
      const uint32_t o = toff + 64 + 32 * j;
      if (nd.ps != kRoot) {
        const SrcNode& parent = st.nodes[tab.nodes[nd.ps >> 1].src];
        out.tabs.box(o, st.child_box(boxes, parent.child[nd.ps & 1]));
      }
      out.tabs.u32(o + 24, nd.ps);
      out.tabs.u32(o + 28, nd.depth);
    }
    blas_tab[b] = uint32_t(tab_addr + toff - scene_addr);

    std::vector<uint32_t> items(tris.size());
    for (uint32_t i = 0; i < items.size(); ++i) items[i] = i;
    RtuRef r = emit_cwbvh(img, boxes, items, [&](uint32_t i) {
      const uint32_t off = img.alloc(RTU_BVH_LEAF_HDR_BYTES + RTU_BVH_TRI_STRIDE);
      img.u32(off, RTU_BVH_KIND_LEAF_TRI | (1u << RTU_BVH_COUNT_SHIFT));
      img.u32(off + 4, tris[i].geom);
      img.u32(off + 8, tab.leaf_ps[i]);
      img.u32(off + 12, i);
      for (int k = 0; k < 9; ++k) img.f32(off + RTU_BVH_LEAF_HDR_BYTES + 4 * k, tris[i].v[k]);
      img.u32(off + RTU_BVH_LEAF_HDR_BYTES + 36, RTU_BVH_FLAG_OPAQUE);
      return off;
    });
    blas_root[b] = r.off;
    blas_box[b] = r.box;
  }

  // TLAS: source tree over the instances' world boxes; table in the scene
  std::vector<Box> wbox;
  for (const auto& in : insts) wbox.push_back(xform_box(in.otw, blas_box[in.blas]));
  SrcTree tt;
  tt.make(wbox);
  SrcTable ttab;
  ttab.make(tt, insts.size());
  const uint32_t nl = uint32_t(insts.size()), nn = uint32_t(ttab.nodes.size());
  const uint32_t noff = (16 + nl * 64 + 63) & ~63u;
  const uint32_t tlas_tab = img.alloc(noff + 64 * nn, 64);
  img.u32(tlas_tab, nl);
  img.u32(tlas_tab + 4, nn);
  img.u32(tlas_tab + 8, noff);
  img.u32(tlas_tab + 12, 64);
  for (uint32_t r = 0; r < nl; ++r) {
    const uint32_t item = ttab.rank_item[r];
    const uint32_t o = tlas_tab + 16 + 64 * r;
    img.u32(o, ttab.leaf_ps[item]);
    for (int k = 0; k < 12; ++k) img.f32(o + 16 + 4 * k, insts[item].wto[k]);
  }
  for (uint32_t j = 0; j < nn; ++j) {
    const SrcNode& nd = tt.nodes[ttab.nodes[j].src];
    const uint32_t o = tlas_tab + noff + 64 * j;
    img.box(o, tt.child_box(wbox, nd.child[0]));
    img.box(o + 24, tt.child_box(wbox, nd.child[1]));
    img.u32(o + 48, ttab.nodes[j].ps);
    img.u32(o + 52, ttab.nodes[j].depth);
  }

  std::vector<uint32_t> items(insts.size());
  for (uint32_t i = 0; i < items.size(); ++i) items[i] = i;
  RtuRef root = emit_cwbvh(img, wbox, items, [&](uint32_t i) {
    const Instance& in = insts[i];
    const uint32_t off = img.alloc(RTU_BVH_LEAF_HDR_BYTES + RTU_BVH_INSTANCE_STRIDE);
    img.u32(off, RTU_BVH_KIND_LEAF_INST | (1u << RTU_BVH_COUNT_SHIFT));
    if (with_tables) {
      img.u32(off + 4, blas_tab[in.blas]);
      img.u32(off + 8, ttab.leaf_rank[i]);
      img.u32(off + 12, tlas_tab);
    }
    const uint32_t rec = off + RTU_BVH_LEAF_HDR_BYTES;
    for (int k = 0; k < 12; ++k) img.f32(rec + 4 * k, in.wto[k]);
    img.u32(rec + RTU_BVH_INSTANCE_BLAS_OFF, blas_root[in.blas]);
    img.u32(rec + RTU_BVH_INSTANCE_CUSTOM_OFF, in.custom);
    img.u32(rec + RTU_BVH_INSTANCE_ID_OFF, in.id);
    img.u32(rec + RTU_BVH_INSTANCE_CULL_OFF, 0xffu);
    return off;
  });
  out.root = root.off;
  img.u32(0, root.off);
  img.u32(4, RTU_SCENE_KIND_BVH4);
  img.u32(8, uint32_t(img.b.size()));
  img.u32(12, uint32_t(insts.size()));
  img.alloc(0, 64);
  return out;
}

const char* field_name[8] = { "status", "t", "u", "v", "prim", "geom", "inst", "custom" };

}  // namespace

int main(int argc, char* argv[]) {
  const char* golden_out = nullptr;
  int c;
  while ((c = getopt(argc, argv, "g:h")) != -1) {
    if (c == 'g') golden_out = optarg;
    else { printf("usage: %s [-g golden.h]\n", argv[0]); return 0; }
  }

  make_geometry();
  std::vector<tie_ray_t> rays = make_rays();
  const uint32_t NT = VX_CFG_NUM_THREADS;
  while (rays.size() % NT) rays.push_back(rays.back());
  const uint32_t count = uint32_t(rays.size());

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Sizes do not depend on the addresses: size the buffers from a dry build.
  const Built dry = build(0, 0, true);
  const uint32_t scene_bytes = uint32_t(dry.scene.b.size());
  const uint32_t tab_bytes   = uint32_t(dry.tabs.b.size());
  uint64_t scene_addr[2], tab_addr = 0;
  for (int k = 0; k < 2; ++k) {
    RT_CHECK(vx_buffer_create(device, scene_bytes, VX_MEM_READ, &scene_buf[k]));
    RT_CHECK(vx_buffer_address(scene_buf[k], &scene_addr[k]));
  }
  RT_CHECK(vx_buffer_create(device, tab_bytes, VX_MEM_READ, &tab_buf));
  RT_CHECK(vx_buffer_address(tab_buf, &tab_addr));
  if (tab_addr <= scene_addr[0] || tab_addr - scene_addr[0] >= (1ull << 31)) {
    printf("Error: the BLAS table buffer is not reachable from the scene\n");
    cleanup();
    return 1;
  }

  RT_CHECK(vx_buffer_create(device, count * sizeof(tie_ray_t), VX_MEM_READ, &rays_buf));
  RT_CHECK(vx_buffer_create(device, count * sizeof(tie_result_t), VX_MEM_WRITE, &res_buf));
  uint64_t rays_addr = 0, res_addr = 0;
  RT_CHECK(vx_buffer_address(rays_buf, &rays_addr));
  RT_CHECK(vx_buffer_address(res_buf, &res_addr));
  RT_CHECK(vx_enqueue_write(queue, rays_buf, 0, rays.data(), count * sizeof(tie_ray_t),
                            0, nullptr, nullptr));

  std::vector<tie_result_t> got[2];
  for (int k = 0; k < 2; ++k) {
    const bool with_tables = (k == 0);
    const Built bi = build(scene_addr[k], tab_addr, with_tables);
    RT_CHECK(vx_enqueue_write(queue, scene_buf[k], 0, bi.scene.b.data(), scene_bytes,
                              0, nullptr, nullptr));
    if (with_tables)
      RT_CHECK(vx_enqueue_write(queue, tab_buf, 0, bi.tabs.b.data(), tab_bytes,
                                0, nullptr, nullptr));

    kernel_arg_t arg = {};
    arg.rays_addr    = rays_addr;
    arg.results_addr = res_addr;
    arg.scene        = uint32_t(scene_addr[k]);
    arg.count        = count;

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 1;
    li.grid_dim[0]  = count / NT;
    li.block_dim[0] = NT;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
    got[k].resize(count);
    RT_CHECK(vx_enqueue_read(queue, got[k].data(), res_buf, 0, count * sizeof(tie_result_t),
                             1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
  }

  uint32_t hits = 0, differ = 0;
  for (uint32_t i = 0; i < count; ++i) {
    hits += (got[0][i].status == VX_RT_STS_DONE_HIT);
    differ += (std::memcmp(&got[0][i], &got[1][i], sizeof(tie_result_t)) != 0);
  }
  printf("rays=%u hits=%u, table vs static-key verdicts differ on %u rays\n", count, hits, differ);

  if (golden_out) {
    FILE* f = fopen(golden_out, "w");
    if (!f) { printf("Error: cannot write %s\n", golden_out); cleanup(); return 1; }
    fprintf(f, "// Generated by rt_smoke_tie -g on the SimX reference. Do not edit.\n");
    fprintf(f, "#pragma once\n#include <stdint.h>\n");
    fprintf(f, "static const uint32_t kGoldenCount = %u;\n", count);
    fprintf(f, "static const uint32_t kGolden[2][%u][8] = {\n", count);
    for (int k = 0; k < 2; ++k) {
      fprintf(f, "  {\n");
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&got[k][i]);
        fprintf(f, "    { 0x%x, 0x%08x, 0x%08x, 0x%08x, %u, 0x%08x, %u, 0x%x },\n",
                w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
      }
      fprintf(f, "  },\n");
    }
    fprintf(f, "};\n");
    fclose(f);
    printf("wrote %s\n", golden_out);
    cleanup();
    return 0;
  }

  int errors = 0;
  if (count != kGoldenCount) {
    printf("Error: %u rays, golden has %u\n", count, kGoldenCount);
    ++errors;
  } else {
    for (int k = 0; k < 2; ++k) {
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&got[k][i]);
        // a miss carries no hit attributes
        const int nf = (kGolden[k][i][0] == VX_RT_STS_DONE_HIT) ? 8 : 1;
        for (int f = 0; f < nf; ++f) {
          if (w[f] != kGolden[k][i][f]) {
            if (errors < 20)
              printf("%s ray %u: %s got 0x%08x expected 0x%08x\n",
                     k ? "static-key" : "tables", i, field_name[f], w[f], kGolden[k][i][f]);
            ++errors;
          }
        }
      }
    }
  }
  if (differ == 0) {
    printf("Error: the visit-order tables changed no verdict\n");
    ++errors;
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED with %d errors\n", errors);
    return 1;
  }
  printf("PASSED!\n");
  return 0;
}
