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
// PRISM RTU multi-node BVH test — validates vortex::raytrace::build_bvh_scene
// emits a REAL internal-node CW-BVH4 tree (not a single leaf) and that the
// SimX walker traverses it to the correct closest hit.
//
// Two checks:
//   (1) Host structural: parse the emitted scene; assert the root is an
//       INTERNAL node, the tree has >= 1 internal node, and every source
//       triangle appears exactly once as a single-triangle leaf with its
//       gl_PrimitiveID preserved (prim_base == source index).
//   (2) Device end-to-end: fire one ray per triangle (through its centroid)
//       and compare the device hit to a host brute-force Moller-Trumbore
//       oracle (status / primitive_id / t / u / v).

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include <vortex2.h>
#include <VX_types.h>
#include <raytrace.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
   do {                                                       \
     int _ret = _expr;                                        \
     if (0 == _ret) break;                                    \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
     cleanup();                                               \
     exit(-1);                                                \
   } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h rays_buffer  = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

void cleanup() {
  if (device) {
    if (scene_buffer) vx_buffer_release(scene_buffer);
    if (rays_buffer)  vx_buffer_release(rays_buffer);
    if (res_buffer)   vx_buffer_release(res_buffer);
    if (kernel)       vx_kernel_release(kernel);
    if (module_)      vx_module_release(module_);
    if (queue)        vx_queue_release(queue);
    vx_device_release(device);
  }
}

// ── Host brute-force oracle (matches sim/simx/rtu/rtu_isect.cpp::ray_triangle) ──
static bool host_ray_triangle(const float ro[3], const float rd[3],
                              const float v0[3], const float v1[3],
                              const float v2[3], float tmin, float tmax,
                              float& out_t, float& out_u, float& out_v) {
  float e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
  float e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
  float P[3]  = { rd[1]*e2[2]-rd[2]*e2[1],
                  rd[2]*e2[0]-rd[0]*e2[2],
                  rd[0]*e2[1]-rd[1]*e2[0] };
  float det = e1[0]*P[0] + e1[1]*P[1] + e1[2]*P[2];
  const float EPS = 1e-6f;
  if (det > -EPS && det < EPS) return false;
  float invDet = 1.0f / det;
  float T[3] = { ro[0]-v0[0], ro[1]-v0[1], ro[2]-v0[2] };
  float u = (T[0]*P[0]+T[1]*P[1]+T[2]*P[2]) * invDet;
  if (u < 0.f || u > 1.f) return false;
  float Q[3] = { T[1]*e1[2]-T[2]*e1[1],
                 T[2]*e1[0]-T[0]*e1[2],
                 T[0]*e1[1]-T[1]*e1[0] };
  float v = (rd[0]*Q[0]+rd[1]*Q[1]+rd[2]*Q[2]) * invDet;
  if (v < 0.f || u + v > 1.f) return false;
  float t = (e2[0]*Q[0]+e2[1]*Q[1]+e2[2]*Q[2]) * invDet;
  if (t < tmin || t > tmax) return false;
  out_t = t; out_u = u; out_v = v;
  return true;
}

// ── Host structural walk of the emitted CW-BVH4 byte layout ──
static uint32_t rd_u32(const std::vector<uint8_t>& s, uint32_t off) {
  uint32_t v = 0; std::memcpy(&v, s.data() + off, 4); return v;
}

static bool walk_scene(const std::vector<uint8_t>& scene, uint32_t off,
                       std::vector<int>& prim_seen, uint32_t& internal_nodes,
                       uint32_t& leaves, int depth) {
  if (depth > 64) { std::cout << "tree too deep (cycle?)\n"; return false; }
  uint32_t kind_word = rd_u32(scene, off);
  uint32_t kind  = kind_word & 0xffu;
  uint32_t count = (kind_word >> RTU_BVH_COUNT_SHIFT) & 0xffu;
  if (kind == RTU_BVH_KIND_INTERNAL) {
    ++internal_nodes;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t co = rd_u32(scene, off + RTU_BVH_NODE_CHILD_OFF + 4 * i);
      if (co == RTU_BVH_CHILD_EMPTY) continue;
      uint32_t child_off = co & RTU_BVH_CHILD_OFFSET_MASK;
      if (!walk_scene(scene, child_off, prim_seen, internal_nodes, leaves, depth + 1))
        return false;
    }
    return true;
  }
  if (kind == RTU_BVH_KIND_LEAF_TRI) {
    ++leaves;
    if (count != 1) {
      std::cout << "leaf at " << off << " has count " << count
                << " (expected single-triangle leaves)\n";
      return false;
    }
    uint32_t prim_base = rd_u32(scene, off + 12);
    if (prim_base >= prim_seen.size()) {
      std::cout << "leaf prim_base " << prim_base << " out of range\n";
      return false;
    }
    ++prim_seen[prim_base];
    return true;
  }
  std::cout << "unexpected node kind " << kind << " at " << off << "\n";
  return false;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // ── Build a spatially-distributed scene: a GRID×GRID array of small,
  // -z-facing opaque triangles at varied depth. The disjoint XY footprints
  // mean a +z ray through triangle i's centroid hits ONLY triangle i, giving
  // a deterministic per-ray oracle while still forcing the builder to split
  // space into a multi-level internal-node tree. ───────────────────────────
  const uint32_t GRID = 6;
  const uint32_t N    = GRID * GRID;        // 36 triangles
  const float    SPACING  = 3.0f;
  const float    HALFSIZE = 0.4f;

  std::vector<vortex::raytrace::host_tri_t> tris(N);
  std::vector<float> centroid(N * 3);
  for (uint32_t i = 0; i < N; ++i) {
    uint32_t gx = i % GRID, gy = i / GRID;
    float cx = (float)gx * SPACING;
    float cy = (float)gy * SPACING;
    float z  = 5.0f + (float)(i % 5) * 0.7f;   // varied depth per triangle
    vortex::raytrace::host_tri_t& t = tris[i];
    // CCW from -z so a +z ray hits the front face.
    t.v0[0] = cx - HALFSIZE; t.v0[1] = cy - HALFSIZE; t.v0[2] = z;
    t.v1[0] = cx + HALFSIZE; t.v1[1] = cy - HALFSIZE; t.v1[2] = z;
    t.v2[0] = cx;            t.v2[1] = cy + HALFSIZE; t.v2[2] = z;
    t.flags = RTU_BVH_FLAG_OPAQUE;
    centroid[i*3+0] = (t.v0[0] + t.v1[0] + t.v2[0]) / 3.0f;
    centroid[i*3+1] = (t.v0[1] + t.v1[1] + t.v2[1]) / 3.0f;
    centroid[i*3+2] = z;
  }

  vortex::raytrace::host_bvh_t src = { tris.data(), N, /*geometry_index*/ 0 };
  std::vector<uint8_t> scene;
  uint64_t root_offset = 0;
  if (!vortex::raytrace::build_bvh_scene<4>(src, scene, root_offset)) {
    std::cout << "build_bvh_scene failed" << std::endl;
    cleanup();
    return 1;
  }

  // ── Check (1): structural ──────────────────────────────────────────────
  int errors = 0;
  {
    uint32_t hdr_root  = rd_u32(scene, 0);
    uint32_t hdr_kind  = rd_u32(scene, 4);
    uint32_t hdr_bytes = rd_u32(scene, 8);
    uint32_t hdr_leaf  = rd_u32(scene, 12);
    std::cout << "scene: " << scene.size() << " B, root_off=" << hdr_root
              << " kind=" << hdr_kind << " leaf_count=" << hdr_leaf << std::endl;

    uint32_t root_kind = rd_u32(scene, hdr_root) & 0xffu;
    if (hdr_kind != RTU_SCENE_KIND_BVH4) { std::cout << "bad scene_kind\n"; ++errors; }
    if (hdr_bytes != scene.size())       { std::cout << "bad scene_bytes\n"; ++errors; }
    if (hdr_leaf != N)                   { std::cout << "bad leaf_count\n"; ++errors; }
    if (root_kind != RTU_BVH_KIND_INTERNAL) {
      std::cout << "root is not an INTERNAL node (kind=" << root_kind
                << ") — builder did not partition\n"; ++errors;
    }

    std::vector<int> prim_seen(N, 0);
    uint32_t internal_nodes = 0, leaves = 0;
    if (!walk_scene(scene, hdr_root, prim_seen, internal_nodes, leaves, 0)) ++errors;
    std::cout << "tree: " << internal_nodes << " internal node(s), "
              << leaves << " leaves" << std::endl;
    if (internal_nodes < 1) { std::cout << "no internal nodes emitted\n"; ++errors; }
    if (leaves != N)        { std::cout << "leaf count " << leaves
                                        << " != " << N << "\n"; ++errors; }
    for (uint32_t i = 0; i < N; ++i)
      if (prim_seen[i] != 1) {
        std::cout << "primitive " << i << " appears " << prim_seen[i]
                  << " times (expected 1)\n"; ++errors;
      }
  }
  if (errors) {
    std::cout << "STRUCTURAL CHECK FAILED with " << errors << " errors" << std::endl;
    cleanup();
    return 1;
  }
  std::cout << "structural check OK" << std::endl;

  // ── Build per-triangle rays (origin at centroid XY, z=0, dir +z) ───────
  std::vector<float> rays(N * 8);
  for (uint32_t i = 0; i < N; ++i) {
    rays[i*8+0] = centroid[i*3+0];   // origin
    rays[i*8+1] = centroid[i*3+1];
    rays[i*8+2] = 0.0f;
    rays[i*8+3] = 0.0f;              // dir +z
    rays[i*8+4] = 0.0f;
    rays[i*8+5] = 1.0f;
    rays[i*8+6] = 0.001f;            // tmin
    rays[i*8+7] = 1e30f;             // tmax
  }

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
  RT_CHECK(vx_buffer_create(device, (uint32_t)(rays.size()*sizeof(float)),
                            VX_MEM_READ, &rays_buffer));
  RT_CHECK(vx_buffer_address(rays_buffer, &kernel_arg.rays_addr));
  RT_CHECK(vx_buffer_create(device, (uint32_t)(N*sizeof(rtu_result_t)),
                            VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));
  kernel_arg.num_rays = N;

  vortex::raytrace::config_t cfg;
  cfg.scene_kind    = RTU_SCENE_KIND_BVH4;
  cfg.bvh_width     = 4;
  cfg.cull_defaults = 0xff;
  RT_CHECK(vortex::raytrace::program(device, cfg));

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, rays_buffer, 0, rays.data(),
                            (uint32_t)(rays.size()*sizeof(float)), 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch " << N << " rays" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = N;
    li.block_dim[0] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<rtu_result_t> results(N);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0,
                           (uint32_t)(N*sizeof(rtu_result_t)),
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // ── Check (2): device vs host brute-force oracle ───────────────────────
  for (uint32_t i = 0; i < N; ++i) {
    const float* ro = &rays[i*8+0];
    const float* rd = &rays[i*8+3];
    float best_t = 1e30f, best_u = 0.f, best_v = 0.f; int best_prim = -1;
    for (uint32_t j = 0; j < N; ++j) {
      float t, u, v;
      if (host_ray_triangle(ro, rd, tris[j].v0, tris[j].v1, tris[j].v2,
                            0.001f, 1e30f, t, u, v) && t < best_t) {
        best_t = t; best_u = u; best_v = v; best_prim = (int)j;
      }
    }
    bool exp_hit = (best_prim >= 0);
    const rtu_result_t& r = results[i];
    bool got_hit = (r.status == VX_RT_STS_DONE_HIT);
    if (got_hit != exp_hit) {
      std::cout << "ray " << i << " status mismatch: got " << r.status
                << " exp_hit=" << exp_hit << std::endl; ++errors; continue;
    }
    if (!exp_hit) continue;
    if ((int)r.primitive_id != best_prim) {
      std::cout << "ray " << i << " prim mismatch: got " << r.primitive_id
                << " expected " << best_prim << std::endl; ++errors;
    }
    if (std::fabs(r.hit_t - best_t) > 1e-3f) {
      std::cout << "ray " << i << " t mismatch: got " << r.hit_t
                << " expected " << best_t << std::endl; ++errors;
    }
    if (std::fabs(r.hit_u - best_u) > 1e-3f ||
        std::fabs(r.hit_v - best_v) > 1e-3f) {
      std::cout << "ray " << i << " uv mismatch: got (" << r.hit_u << ","
                << r.hit_v << ") expected (" << best_u << "," << best_v
                << ")" << std::endl; ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
