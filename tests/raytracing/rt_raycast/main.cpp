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
// rt_raycast host driver. Loads an OBJ mesh, builds a CW-BVH4 acceleration
// structure (1 triangle per leaf, geometry_index = global triangle id),
// renders it on the RTU (kernel.cpp) and validates the image against a CPU
// reference that traces the same triangles with the same Möller-Trumbore
// intersector + shading. Writes both as PPM for inspection.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstring>
#include <algorithm>

#include <vortex2.h>
#include <VX_types.h>
#include "common.h"

#ifndef ASSETS_PATHS
#define ASSETS_PATHS "."
#endif

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

// ── geometry ────────────────────────────────────────────────────────
struct Tri {
  vec3 v0, v1, v2;
  vec3 n0, n1, n2;
  vec3 cmin, cmax, centroid;
};

static void tri_bounds(Tri& t) {
  t.cmin = v3(std::min({t.v0.x, t.v1.x, t.v2.x}),
              std::min({t.v0.y, t.v1.y, t.v2.y}),
              std::min({t.v0.z, t.v1.z, t.v2.z}));
  t.cmax = v3(std::max({t.v0.x, t.v1.x, t.v2.x}),
              std::max({t.v0.y, t.v1.y, t.v2.y}),
              std::max({t.v0.z, t.v1.z, t.v2.z}));
  t.centroid = v3scale(v3add(t.cmin, t.cmax), 0.5f);
}

// ── OBJ loader (positions + normals + triangulated faces) ───────────
static bool load_obj(const std::string& path, std::vector<Tri>& tris) {
  std::ifstream f(path);
  if (!f.is_open()) { std::cerr << "cannot open " << path << "\n"; return false; }
  std::vector<vec3> pos, nrm;
  std::string line;
  auto parse_idx = [](const std::string& tok, int& p, int& n) {
    p = n = 0;
    size_t s1 = tok.find('/');
    if (s1 == std::string::npos) { p = std::stoi(tok); return; }
    p = std::stoi(tok.substr(0, s1));
    size_t s2 = tok.find('/', s1 + 1);
    if (s2 != std::string::npos && s2 + 1 < tok.size())
      n = std::stoi(tok.substr(s2 + 1));
  };
  while (std::getline(f, line)) {
    std::istringstream iss(line);
    std::string ty; iss >> ty;
    if (ty == "v") { vec3 v; iss >> v.x >> v.y >> v.z; pos.push_back(v); }
    else if (ty == "vn") { vec3 n; iss >> n.x >> n.y >> n.z; nrm.push_back(n); }
    else if (ty == "f") {
      std::vector<std::string> verts; std::string t;
      while (iss >> t) verts.push_back(t);
      for (size_t j = 1; j + 1 < verts.size(); ++j) {
        int p[3], n[3];
        parse_idx(verts[0],   p[0], n[0]);
        parse_idx(verts[j],   p[1], n[1]);
        parse_idx(verts[j+1], p[2], n[2]);
        Tri tr;
        tr.v0 = pos[p[0]-1]; tr.v1 = pos[p[1]-1]; tr.v2 = pos[p[2]-1];
        if (!nrm.empty() && n[0] && n[1] && n[2]) {
          tr.n0 = nrm[n[0]-1]; tr.n1 = nrm[n[1]-1]; tr.n2 = nrm[n[2]-1];
        } else {
          vec3 fn = v3norm(v3cross(v3sub(tr.v1, tr.v0), v3sub(tr.v2, tr.v0)));
          tr.n0 = tr.n1 = tr.n2 = fn;
        }
        tri_bounds(tr);
        tris.push_back(tr);
      }
    }
  }
  return !tris.empty();
}

// ── CW-BVH4 builder (recursive median split, ≤4 children, 1 tri/leaf) ─
struct BNode {
  vec3 mn, mx;
  bool leaf;
  uint32_t tri;            // leaf: global triangle index
  int child[VX_BVH_WIDTH]; // internal: build-node indices
  int nchild;
};

struct Builder {
  const std::vector<Tri>& tris;
  std::vector<uint32_t>& order;   // tri index permutation
  std::vector<BNode> nodes;
  Builder(const std::vector<Tri>& t, std::vector<uint32_t>& o) : tris(t), order(o) {}

  void bounds(int start, int count, vec3& mn, vec3& mx) {
    mn = v3( 1e30f,  1e30f,  1e30f);
    mx = v3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < count; ++i) {
      const Tri& t = tris[order[start + i]];
      mn = v3(std::min(mn.x, t.cmin.x), std::min(mn.y, t.cmin.y), std::min(mn.z, t.cmin.z));
      mx = v3(std::max(mx.x, t.cmax.x), std::max(mx.y, t.cmax.y), std::max(mx.z, t.cmax.z));
    }
  }
  // Sort [start,count) by centroid on the longest axis, return split point.
  int split2(int start, int count) {
    vec3 mn, mx; bounds(start, count, mn, mx);
    vec3 ext = v3sub(mx, mn);
    int axis = (ext.x > ext.y) ? (ext.x > ext.z ? 0 : 2) : (ext.y > ext.z ? 1 : 2);
    auto key = [&](uint32_t ti) {
      const vec3& c = tris[ti].centroid;
      return axis == 0 ? c.x : (axis == 1 ? c.y : c.z);
    };
    std::sort(order.begin() + start, order.begin() + start + count,
              [&](uint32_t a, uint32_t b) { return key(a) < key(b); });
    return count / 2;
  }
  int build(int start, int count) {
    int id = (int)nodes.size();
    nodes.push_back(BNode{});
    if (count == 1) {
      BNode& n = nodes[id];
      n.leaf = true; n.tri = order[start]; n.nchild = 0;
      n.mn = tris[n.tri].cmin; n.mx = tris[n.tri].cmax;
      return id;
    }
    // Split into up to 4 ranges via two levels of median split.
    int m = split2(start, count);
    int ranges[4][2]; int nr = 0;
    auto add_half = [&](int s, int c) {
      if (c <= 1) { ranges[nr][0] = s; ranges[nr][1] = c; ++nr; }
      else {
        int mm = split2(s, c);
        ranges[nr][0] = s;     ranges[nr][1] = mm;     ++nr;
        ranges[nr][0] = s + mm; ranges[nr][1] = c - mm; ++nr;
      }
    };
    add_half(start, m);
    add_half(start + m, count - m);
    int kids[4]; int nk = 0;
    for (int i = 0; i < nr; ++i)
      if (ranges[i][1] > 0) kids[nk++] = build(ranges[i][0], ranges[i][1]);
    BNode& n = nodes[id];
    n.leaf = false; n.nchild = nk;
    for (int i = 0; i < nk; ++i) n.child[i] = kids[i];
    // Node bounds = union of child bounds.
    n.mn = v3( 1e30f,  1e30f,  1e30f);
    n.mx = v3(-1e30f, -1e30f, -1e30f);
    for (int i = 0; i < nk; ++i) {
      const BNode& c = nodes[kids[i]];
      n.mn = v3(std::min(n.mn.x, c.mn.x), std::min(n.mn.y, c.mn.y), std::min(n.mn.z, c.mn.z));
      n.mx = v3(std::max(n.mx.x, c.mx.x), std::max(n.mx.y, c.mx.y), std::max(n.mx.z, c.mx.z));
    }
    return id;
  }
};

// Quantize child AABBs against a node's (origin, exp). exp chosen so the
// node extent maps into [0,255]; returns per-axis exponent.
static void choose_exp(const vec3& mn, const vec3& mx, int exp[3]) {
  float ext[3] = { mx.x - mn.x, mx.y - mn.y, mx.z - mn.z };
  for (int a = 0; a < 3; ++a) {
    if (ext[a] <= 0.f) { exp[a] = -16; continue; }
    // step ≥ ext/255  →  2^exp ≥ ext/255  →  exp = ceil(log2(ext/255))
    int e = (int)std::ceil(std::log2(ext[a] / 255.0f));
    if (e < -16) e = -16;
    if (e > 16) e = 16;
    exp[a] = e;
  }
}
static uint8_t quant_lo(float v, float origin, int exp) {
  float q = std::floor((v - origin) / std::ldexp(1.0f, exp));
  if (q < 0.f) q = 0.f;
  if (q > 255.f) q = 255.f;
  return (uint8_t)q;
}
static uint8_t quant_hi(float v, float origin, int exp) {
  float q = std::ceil((v - origin) / std::ldexp(1.0f, exp));
  if (q < 0.f) q = 0.f;
  if (q > 255.f) q = 255.f;
  return (uint8_t)q;
}

// Serialize the build tree into the on-disk CW-BVH4 scene buffer.
static std::vector<uint8_t> serialize(const std::vector<BNode>& nodes, int root,
                                      const std::vector<Tri>& tris) {
  // Assign byte offsets in node-id order, after the 16 B scene header.
  std::vector<uint32_t> off(nodes.size());
  uint32_t cur = VX_BVH_SCENE_HDR_BYTES;
  for (size_t i = 0; i < nodes.size(); ++i) {
    off[i] = cur;
    cur += nodes[i].leaf ? (VX_BVH_LEAF_HDR_BYTES + VX_BVH_TRI_STRIDE)
                         : VX_BVH_NODE_BYTES;
  }
  std::vector<uint8_t> buf(cur, 0);
  uint32_t* sh = reinterpret_cast<uint32_t*>(buf.data());
  sh[0] = off[root];
  sh[1] = VX_BVH_SCENE_KIND;
  sh[2] = (uint32_t)buf.size();   // total scene bytes (pre-fetch size)
  sh[3] = (uint32_t)nodes.size();

  for (size_t i = 0; i < nodes.size(); ++i) {
    const BNode& n = nodes[i];
    uint8_t* p = buf.data() + off[i];
    if (n.leaf) {
      uint32_t* lh = reinterpret_cast<uint32_t*>(p);
      lh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);
      lh[1] = n.tri;   // geometry_index = global triangle id
      lh[2] = 0; lh[3] = 0;
      float* tv = reinterpret_cast<float*>(p + VX_BVH_LEAF_HDR_BYTES);
      const Tri& t = tris[n.tri];
      tv[0]=t.v0.x; tv[1]=t.v0.y; tv[2]=t.v0.z;
      tv[3]=t.v1.x; tv[4]=t.v1.y; tv[5]=t.v1.z;
      tv[6]=t.v2.x; tv[7]=t.v2.y; tv[8]=t.v2.z;
      uint32_t* tf = reinterpret_cast<uint32_t*>(p + VX_BVH_LEAF_HDR_BYTES
                                                 + VX_BVH_TRI_FLAGS_OFFSET);
      *tf = VX_BVH_TRI_FLAG_OPAQUE;
    } else {
      uint32_t* kind = reinterpret_cast<uint32_t*>(p);
      *kind = VX_BVH_KIND_INTERNAL | ((uint32_t)n.nchild << VX_BVH_COUNT_SHIFT);
      float* origin = reinterpret_cast<float*>(p + VX_BVH_OFF_ORIGIN);
      origin[0] = n.mn.x; origin[1] = n.mn.y; origin[2] = n.mn.z;
      int exp[3]; choose_exp(n.mn, n.mx, exp);
      int8_t* pe = reinterpret_cast<int8_t*>(p + VX_BVH_OFF_EXP);
      pe[0]=(int8_t)exp[0]; pe[1]=(int8_t)exp[1]; pe[2]=(int8_t)exp[2];
      uint32_t* child = reinterpret_cast<uint32_t*>(p + VX_BVH_OFF_CHILD);
      uint8_t* qmin = p + VX_BVH_OFF_QMIN;
      uint8_t* qmax = p + VX_BVH_OFF_QMAX;
      for (int c = 0; c < n.nchild; ++c) {
        const BNode& ch = nodes[n.child[c]];
        child[c] = off[n.child[c]] | (ch.leaf ? VX_BVH_CHILD_LEAF_FLAG : 0u);
        qmin[c*3+0] = quant_lo(ch.mn.x, n.mn.x, exp[0]);
        qmin[c*3+1] = quant_lo(ch.mn.y, n.mn.y, exp[1]);
        qmin[c*3+2] = quant_lo(ch.mn.z, n.mn.z, exp[2]);
        qmax[c*3+0] = quant_hi(ch.mx.x, n.mn.x, exp[0]);
        qmax[c*3+1] = quant_hi(ch.mx.y, n.mn.y, exp[1]);
        qmax[c*3+2] = quant_hi(ch.mx.z, n.mn.z, exp[2]);
      }
    }
  }
  return buf;
}

// ── Möller-Trumbore (identical to sim/simx/rtu/rtu_isect.cpp) ────────
static bool ray_tri(const vec3& ro, const vec3& rd, const Tri& t,
                    float tmin, float tmax, float& out_t, float& out_u, float& out_v) {
  vec3 e1 = v3sub(t.v1, t.v0), e2 = v3sub(t.v2, t.v0);
  vec3 P = v3cross(rd, e2);
  float det = v3dot(e1, P);
  const float EPS = 1e-6f;
  if (det > -EPS && det < EPS) return false;
  float inv = 1.0f / det;
  vec3 T = v3sub(ro, t.v0);
  float u = v3dot(T, P) * inv;
  if (u < 0.f || u > 1.f) return false;
  vec3 Q = v3cross(T, e1);
  float v = v3dot(rd, Q) * inv;
  if (v < 0.f || u + v > 1.f) return false;
  float tt = v3dot(e2, Q) * inv;
  if (tt < tmin || tt > tmax) return false;
  out_t = tt; out_u = u; out_v = v;
  return true;
}

static vec3 shade(const kernel_arg_t& arg, const std::vector<Tri>& tris,
                  vec3 ro, vec3 rd, float t, float u, float v, uint32_t id) {
  const Tri& tr = tris[id];
  float w0 = 1.f - u - v;
  vec3 N = v3norm(v3add(v3add(v3scale(tr.n0, w0), v3scale(tr.n1, u)),
                        v3scale(tr.n2, v)));
  vec3 I = v3add(ro, v3scale(rd, t));
  vec3 L = v3norm(v3sub(arg.light_pos, I));
  float nl = v3dot(N, L); if (nl < 0.f) nl = 0.f;
  return v3add(arg.ambient_color, v3scale(arg.light_color, nl));
}

// ── globals / cleanup ───────────────────────────────────────────────
vx_device_h device = nullptr;
vx_buffer_h scene_buf = nullptr, shade_buf = nullptr, fb_buf = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
void cleanup() {
  if (device) {
    if (scene_buf) vx_buffer_release(scene_buf);
    if (shade_buf) vx_buffer_release(shade_buf);
    if (fb_buf)    vx_buffer_release(fb_buf);
    if (kernel)    vx_kernel_release(kernel);
    if (module_)   vx_module_release(module_);
    if (queue)     vx_queue_release(queue);
    vx_device_release(device);
  }
}

static void write_ppm(const std::string& path, const std::vector<uint32_t>& img,
                      uint32_t w, uint32_t h) {
  std::ofstream f(path, std::ios::binary);
  f << "P6\n" << w << " " << h << "\n255\n";
  for (uint32_t i = 0; i < w * h; ++i) {
    uint32_t c = img[i];
    unsigned char rgb[3] = { (unsigned char)((c >> 16) & 0xff),
                             (unsigned char)((c >> 8) & 0xff),
                             (unsigned char)(c & 0xff) };
    f.write((const char*)rgb, 3);
  }
}

int main(int argc, char** argv) {
  std::string obj = std::string(ASSETS_PATHS) + "/sphere.obj";
  uint32_t W = 32, H = 24;
  std::string out = "output.ppm";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-m" && i+1 < argc) obj = argv[++i];
    else if (a == "-w" && i+1 < argc) W = std::stoi(argv[++i]);
    else if (a == "-h" && i+1 < argc) H = std::stoi(argv[++i]);
    else if (a == "-o" && i+1 < argc) out = argv[++i];
  }

  std::vector<Tri> tris;
  if (!load_obj(obj, tris)) return 1;
  std::cout << "loaded " << obj << ": " << tris.size() << " triangles\n";

  // Build CW-BVH4.
  std::vector<uint32_t> order(tris.size());
  for (uint32_t i = 0; i < tris.size(); ++i) order[i] = i;
  Builder b(tris, order);
  int root = b.build(0, (int)tris.size());
  std::vector<uint8_t> scene = serialize(b.nodes, root, tris);
  std::cout << "CW-BVH4: " << b.nodes.size() << " nodes, "
            << scene.size() << " bytes\n";
  if (scene.size() > 16384) {
    std::cerr << "scene exceeds 16 KB RTU pre-fetch budget; use a smaller mesh\n";
    return 1;
  }

  // Per-triangle shading data (indexed by geometry_index = tri id).
  std::vector<tri_shade_t> shade_arr(tris.size());
  for (uint32_t i = 0; i < tris.size(); ++i)
    shade_arr[i] = tri_shade_t{ tris[i].n0, tris[i].n1, tris[i].n2 };

  // Camera framing from mesh bounds.
  vec3 mn = v3(1e30f,1e30f,1e30f), mx = v3(-1e30f,-1e30f,-1e30f);
  for (auto& t : tris) {
    mn = v3(std::min(mn.x,t.cmin.x), std::min(mn.y,t.cmin.y), std::min(mn.z,t.cmin.z));
    mx = v3(std::max(mx.x,t.cmax.x), std::max(mx.y,t.cmax.y), std::max(mx.z,t.cmax.z));
  }
  vec3 center = v3scale(v3add(mn, mx), 0.5f);
  float radius = 0.5f * v3len(v3sub(mx, mn));

  kernel_arg_t arg = {};
  arg.dst_width = W; arg.dst_height = H; arg.num_tris = (uint32_t)tris.size();
  vec3 view_dir = v3norm(v3(0.4f, 0.35f, 1.0f));
  arg.camera_pos = v3add(center, v3scale(view_dir, 2.6f * radius));
  arg.camera_forward = v3norm(v3sub(center, arg.camera_pos));
  vec3 world_up = v3(0.f, 1.f, 0.f);
  arg.camera_right = v3norm(v3cross(arg.camera_forward, world_up));
  arg.camera_up = v3cross(arg.camera_right, arg.camera_forward);
  float aspect = (float)W / (float)H;
  float half_h = std::tan(0.5f * 50.0f * 3.14159265f / 180.0f);
  arg.viewplane_y = 2.0f * half_h;
  arg.viewplane_x = arg.viewplane_y * aspect;
  arg.light_pos = v3add(center, v3(3.f*radius, 4.f*radius, 2.f*radius));
  arg.light_color = v3(0.9f, 0.85f, 0.75f);
  arg.ambient_color = v3(0.12f, 0.12f, 0.15f);
  arg.background_color = v3(0.05f, 0.06f, 0.08f);

  // ── device render ─────────────────────────────────────────────────
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buf));
  RT_CHECK(vx_buffer_address(scene_buf, &arg.scene_addr));
  RT_CHECK(vx_buffer_create(device, (uint32_t)(shade_arr.size()*sizeof(tri_shade_t)),
                            VX_MEM_READ, &shade_buf));
  RT_CHECK(vx_buffer_address(shade_buf, &arg.shade_addr));
  uint32_t fb_bytes = W * H * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(device, fb_bytes, VX_MEM_WRITE, &fb_buf));
  RT_CHECK(vx_buffer_address(fb_buf, &arg.dst_addr));

  RT_CHECK(vx_enqueue_write(queue, scene_buf, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, shade_buf, 0, shade_arr.data(),
                            (uint32_t)(shade_arr.size()*sizeof(tri_shade_t)),
                            0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  uint64_t num_threads = 1;
  vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads);
  if (num_threads == 0) num_threads = 1;

  std::cout << "render " << W << "x" << H << " on RTU\n";
  vx_event_h lev = nullptr, rev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = kernel; li.args_host = &arg; li.args_size = sizeof(arg);
    li.ndim = 2;
    li.grid_dim[0] = (W + num_threads - 1) / num_threads;
    li.grid_dim[1] = H;
    li.block_dim[0] = (uint32_t)num_threads;
    li.block_dim[1] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &lev));
  }
  std::vector<uint32_t> dev_img(W * H, 0);
  RT_CHECK(vx_enqueue_read(queue, dev_img.data(), fb_buf, 0, fb_bytes, 1, &lev, &rev));
  RT_CHECK(vx_event_wait_value(rev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(rev); vx_event_release(lev);

  // ── CPU reference (brute force over the same triangles) ───────────
  std::vector<uint32_t> ref_img(W * H, 0);
  for (uint32_t y = 0; y < H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      float xn = (x + 0.5f) / W - 0.5f, yn = (y + 0.5f) / H - 0.5f;
      vec3 pc = v3add(v3add(v3scale(arg.camera_right, xn*arg.viewplane_x),
                            v3scale(arg.camera_up, yn*arg.viewplane_y)),
                      arg.camera_forward);
      vec3 ro = arg.camera_pos, rd = v3norm(pc);
      float best_t = 1e30f, bu = 0, bv = 0; uint32_t bid = 0; bool hit = false;
      for (uint32_t i = 0; i < tris.size(); ++i) {
        float tt, uu, vv;
        if (ray_tri(ro, rd, tris[i], 0.001f, best_t, tt, uu, vv)) {
          best_t = tt; bu = uu; bv = vv; bid = i; hit = true;
        }
      }
      vec3 c = hit ? shade(arg, tris, ro, rd, best_t, bu, bv, bid)
                   : arg.background_color;
      ref_img[x + y*W] = pack_rgb(c);
    }
  }

  write_ppm(out, dev_img, W, H);
  write_ppm("reference.ppm", ref_img, W, H);

  // ── compare ───────────────────────────────────────────────────────
  uint32_t mismatch = 0, maxdiff = 0;
  for (uint32_t i = 0; i < W*H; ++i) {
    uint32_t a = dev_img[i], r = ref_img[i];
    int dr = std::abs((int)((a>>16)&0xff) - (int)((r>>16)&0xff));
    int dg = std::abs((int)((a>>8)&0xff)  - (int)((r>>8)&0xff));
    int db = std::abs((int)(a&0xff)       - (int)(r&0xff));
    int d = std::max({dr, dg, db});
    if ((uint32_t)d > maxdiff) maxdiff = d;
    if (d > 16) ++mismatch;   // count silhouette/edge flips and tone breaks
  }
  float frac = (float)mismatch / (float)(W*H);
  std::cout << "compare: maxdiff=" << maxdiff << " mismatched=" << mismatch
            << "/" << (W*H) << " (" << (frac*100.f) << "%)\n";
  std::cout << "wrote " << out << " and reference.ppm\n";

  cleanup();

  // Allow a small fraction of edge pixels to differ (float rounding between
  // x86 host and RISC-V softfloat flips a handful of silhouette samples).
  if (frac > 0.03f) {
    std::cout << "FAILED (too many mismatched pixels)\n";
    return 1;
  }
  std::cout << "PASSED!\n";
  return 0;
}
