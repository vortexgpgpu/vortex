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
// PRISM RTU procedural intersection-shader smoke — host driver.
//
// Builds a BVH4 scene whose root is a LeafProc with one procedural-AABB
// primitive wrapping a unit sphere centred at (0,0,5):
//   +  0  VxBvhSceneHeader { root_node_offset=16, scene_kind=2,
//                            node_count=0, leaf_count=1 }
//   + 16  VxBvhLeafHeader  { kind = LEAF_PROC | (1 << 8), geom=0, flags=0 }
//   + 32  VxBvhProcAabb    { min=(-2,-2,3), max=(2,2,7) }  (padded > sphere)
//
// The walker's LeafProc path ray-tests the AABB and yields IS; the
// kernel's IS computes the real sphere hit, writes hit_t + hitAttribute,
// and ACCEPTs.

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

  // Build the BVH4 LeafProc scene buffer.
  std::vector<uint8_t> scene(VX_BVH_SCENE_HDR_BYTES +
                             VX_BVH_LEAF_HDR_BYTES +
                             VX_BVH_PROC_AABB_BYTES,
                             0);

  // Scene header.
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;            // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;                  // = 2 (BVH4)
  sh[2] = (uint32_t)scene.size();             // total scene bytes (pre-fetch)
  sh[3] = 1;                                  // leaf_count

  // Leaf header at offset 16: one procedural-AABB primitive.
  uint32_t* lh = reinterpret_cast<uint32_t*>(scene.data() + VX_BVH_SCENE_HDR_BYTES);
  lh[0] = VX_BVH_KIND_LEAF_PROC | (1u << VX_BVH_COUNT_SHIFT);  // kind+count
  lh[1] = 0;                                  // geometry_index
  lh[2] = 0;                                  // leaf flags (sbt_idx = 0)
  lh[3] = 0;                                  // reserved

  // Procedural AABB at offset 32: padded looser than the unit sphere so
  // the AABB-entry candidate t (3.0) differs from the IS-computed sphere
  // hit t (4.0) — committing 4.0 proves the cb_hit_t path.
  float* aabb = reinterpret_cast<float*>(scene.data() + VX_BVH_SCENE_HDR_BYTES
                                         + VX_BVH_LEAF_HDR_BYTES);
  aabb[0] = -2.0f;  aabb[1] = -2.0f;  aabb[2] = 3.0f;   // min
  aabb[3] =  2.0f;  aabb[4] =  2.0f;  aabb[5] = 7.0f;   // max

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(),
                            VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  // Ray straight down +Z through the sphere centre.
  kernel_arg.ray_origin[0]    = 0.0f;
  kernel_arg.ray_origin[1]    = 0.0f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr
            << std::dec << " bvh4 (1 leaf_proc, 1 sphere AABB)" << std::endl;

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

  // CPU oracle: ray (0,0,0)+t(0,0,1) vs sphere (0,0,5) r=1 → near t = 4.
  //   - LeafProc must yield IS → status HIT (feature 3)
  //   - committed hit_t = the IS-computed 4.0 (feature 2 cb_hit_t commit;
  //     != the AABB-entry candidate t of 3.0). A correct 4.0 also proves
  //     the IS read the object-space ray (feature 1) — a zero/garbage ray
  //     would not produce 4.0.
  //   - hit_attr = the magic sentinel the IS wrote (feature 2 hitAttribute)
  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t      = RTU_SPHERE_CZ - RTU_SPHERE_R;   // = 4.0
  const uint32_t exp_attr   = RTU_IS_ATTR_MAGIC;

  std::cout << "oracle: HIT t=" << exp_t << " attr=0x" << std::hex << exp_attr
            << std::dec << std::endl;

  int errors = 0;
  if (result.status != exp_status) {
    std::cout << "status mismatch: got " << result.status
              << " expected " << exp_status << " (LeafProc IS yield)" << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_t - exp_t) > 1e-4f) {
    std::cout << "hit_t mismatch: got " << result.hit_t
              << " expected " << exp_t
              << " (object-space ray readback / IS hit_t commit)" << std::endl;
    ++errors;
  }
  if (result.hit_attr != exp_attr) {
    std::cout << "hit_attr mismatch: got 0x" << std::hex << result.hit_attr
              << " expected 0x" << exp_attr << std::dec
              << " (hitAttributeEXT round-trip)" << std::endl;
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
