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
// PRISM RTU smoke — Phase 4 chunk 2 host driver.
//
// Builds a tiny BVH4 scene whose root IS a leaf with one opaque
// triangle. Exercises the new scene_kind=2 walker end-to-end:
//   - scene header parsed → l.bvh_root_offset = 16
//   - drain_mem_rsp pre-fetches the full per-lane line budget
//   - compute_intersections_bvh4_lane() decodes the leaf and walks
//     the single triangle.
//
// On-disk layout (72 bytes):
//   +  0  VxBvhSceneHeader { root_node_offset=16, scene_kind=2,
//                            node_count=0, leaf_count=1 }
//   + 16  VxBvhLeafHeader  { kind = LEAF_TRI | (1 << 8),
//                            geometry_index=0, flags=0, reserved=0 }
//   + 32  VxBvhTri         { v0/v1/v2 = (0,0,5)/(1,0,5)/(0,1,5),
//                            flags = OPAQUE }

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <cstring>

#include <vortex2.h>
#include <VX_types.h>
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

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Build the BVH4 scene buffer.
  std::vector<uint8_t> scene(VX_BVH_SCENE_HDR_BYTES +
                             VX_BVH_LEAF_HDR_BYTES +
                             VX_BVH_TRI_STRIDE,
                             0);

  // Scene header.
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;            // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;                  // = 2
  sh[2] = 0;                                  // node_count
  sh[3] = 1;                                  // leaf_count

  // Leaf header at offset 16.
  uint8_t*  leaf  = scene.data() + VX_BVH_SCENE_HDR_BYTES;
  uint32_t* lh    = reinterpret_cast<uint32_t*>(leaf);
  lh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);  // kind+count
  lh[1] = 0;                                  // geometry_index
  lh[2] = 0;                                  // leaf flags
  lh[3] = 0;                                  // reserved

  // Triangle at offset 32.
  float* tri = reinterpret_cast<float*>(scene.data() + VX_BVH_SCENE_HDR_BYTES
                                        + VX_BVH_LEAF_HDR_BYTES);
  tri[0] = 0.f; tri[1] = 0.f; tri[2] = 5.f;
  tri[3] = 1.f; tri[4] = 0.f; tri[5] = 5.f;
  tri[6] = 0.f; tri[7] = 1.f; tri[8] = 5.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
      + VX_BVH_TRI_FLAGS_OFFSET);
  *tri_flags = VX_BVH_TRI_FLAG_OPAQUE;

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(),
                            VX_MEM_READ, &scene_buffer));
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

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr
            << std::dec << " bvh4 (1 leaf, 1 tri)" << std::endl;

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

  // Oracle: HIT at t=5, barycentrics (u=0.25, v=0.25), prim 0.
  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t      = 5.f;
  const float    exp_u      = 0.25f;
  const float    exp_v      = 0.25f;
  const uint32_t exp_prim   = 0;

  std::cout << "oracle: HIT t=" << exp_t << " u=" << exp_u << " v=" << exp_v
            << " prim=" << exp_prim << std::endl;

  int errors = 0;
  if (result.status != exp_status) {
    std::cout << "status mismatch: got " << result.status
              << " expected " << exp_status << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_t - exp_t) > 1e-4f) {
    std::cout << "hit_t mismatch: got " << result.hit_t
              << " expected " << exp_t << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_u - exp_u) > 1e-4f) {
    std::cout << "hit_u mismatch: got " << result.hit_u
              << " expected " << exp_u << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_v - exp_v) > 1e-4f) {
    std::cout << "hit_v mismatch: got " << result.hit_v
              << " expected " << exp_v << std::endl;
    ++errors;
  }
  if (result.primitive_id != exp_prim) {
    std::cout << "prim_id mismatch: got " << result.primitive_id
              << " expected " << exp_prim << std::endl;
    ++errors;
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
