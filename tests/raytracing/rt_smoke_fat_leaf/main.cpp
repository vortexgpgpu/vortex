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
// PRISM RTU fat-leaf smoke — I7 host driver.
//
// The scene root is a SINGLE CW-BVH4 LEAF_TRI that packs K triangles (a "fat
// leaf"). The K triangles are stacked in depth along the ray, ordered
// FARTHEST-first inside the leaf, so triangle index 0 is the FARTHEST and the
// closest hit is the LAST triangle in the leaf. A walker that decodes only the
// first triangle per leaf would report the farthest hit (t=10, prim=prim_base);
// the correct walker iterates all `count` triangles and returns the nearest
// (t=5, prim=prim_base+(K-1)) — matching the SimX oracle (rtu_walker.cpp
// visit_leaf_tri iterates `count`). Run under both simx and rtlsim: both must
// agree on the closest hit.

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>

#include <vortex2.h>
#include <VX_types.h>
#include <raytrace.h>
#include <rtu_cfg.h>
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
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static const uint32_t kLeafGeom  = 7u;    // gl_GeometryIndexEXT of the fat leaf
static const uint32_t kPrimBase  = 100u;  // gl_PrimitiveID of the leaf's first tri
static const uint32_t kNumTris   = 6u;    // triangles packed in the fat leaf

void cleanup() {
  if (device) {
    if (scene_buffer) vx_buffer_release(scene_buffer);
    if (res_buffer)   vx_buffer_release(res_buffer);
    if (kernel)       vx_kernel_release(kernel);
    if (module_)      vx_module_release(module_);
    if (queue)        vx_queue_release(queue);
    vx_device_release(device);
  }
}

// Little-endian append helpers into the scene byte buffer.
static void put_u32(std::vector<uint8_t>& s, uint32_t v) {
  for (int i = 0; i < 4; ++i) s.push_back(uint8_t((v >> (8 * i)) & 0xff));
}
static void put_f32(std::vector<uint8_t>& s, float f) {
  uint32_t v; std::memcpy(&v, &f, 4); put_u32(s, v);
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // ── hand-build the scene: 16 B header + one fat LEAF_TRI ───────────────
  // header: root_off=16, scene_kind=BVH4, scene_bytes (patched), leaf_count=1.
  std::vector<uint8_t> scene;
  const uint32_t root_off = RTU_BVH_SCENE_HDR_BYTES;   // 16
  put_u32(scene, root_off);
  put_u32(scene, RTU_SCENE_KIND_BVH4);
  put_u32(scene, 0);                 // scene_bytes: patched below
  put_u32(scene, 1);                 // leaf_count (diagnostic)

  // leaf header: kind = LEAF_TRI | (K<<8); geometry_index; flags=0; prim_base.
  uint32_t kind = RTU_BVH_KIND_LEAF_TRI | (kNumTris << RTU_BVH_COUNT_SHIFT);
  put_u32(scene, kind);
  put_u32(scene, kLeafGeom);
  put_u32(scene, 0);                 // flags: OPAQUE via ray flag below
  put_u32(scene, kPrimBase);

  // K triangles, FARTHEST-first: triangle i sits at z = 5 + (K-1-i), so tri 0
  // is the farthest and tri K-1 (z=5) is the closest. Each covers the ray's
  // (0.25, 0.25) footprint.
  for (uint32_t i = 0; i < kNumTris; ++i) {
    float z = 5.0f + float(kNumTris - 1 - i);
    put_f32(scene, 0.f); put_f32(scene, 0.f); put_f32(scene, z);   // v0
    put_f32(scene, 1.f); put_f32(scene, 0.f); put_f32(scene, z);   // v1
    put_f32(scene, 0.f); put_f32(scene, 1.f); put_f32(scene, z);   // v2
    put_u32(scene, RTU_BVH_FLAG_OPAQUE);                            // tri flags
  }
  // patch scene_bytes so the SimX pre-fetch pulls the whole fat leaf.
  uint32_t scene_bytes = (uint32_t)scene.size();
  std::memcpy(scene.data() + 8, &scene_bytes, 4);

  std::cout << "scene: " << scene.size() << " B, fat LEAF_TRI with "
            << kNumTris << " tris (farthest-first)" << std::endl;

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = 1;
    li.block_dim[0] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  rtu_result_t result = {};
  RT_CHECK(vx_enqueue_read(queue, &result, res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t      = 5.f;                    // nearest triangle (last in leaf)
  const uint32_t exp_prim   = kPrimBase + kNumTris-1; // prim_base + within-leaf index
  std::cout << "oracle: HIT t=" << exp_t << " prim=" << exp_prim
            << " (iterate all " << kNumTris << " leaf tris)" << std::endl;

  int errors = 0;
  bool sts_ok  = (result.status == exp_status);
  bool t_ok    = std::fabs(result.hit_t - exp_t) < 1e-4f;
  bool prim_ok = (result.primitive_id == exp_prim);
  std::cout << "result: status=" << result.status
            << " hit_t=" << result.hit_t
            << " prim=" << result.primitive_id << std::endl;
  if (!sts_ok || !t_ok || !prim_ok) ++errors;

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
