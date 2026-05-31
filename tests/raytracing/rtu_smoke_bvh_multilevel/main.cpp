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
// PRISM RTU smoke — Phase 4 chunk 3 host driver.
//
// Builds a 2-level BVH4 scene:
//   - Root internal node with 2 leaf children
//   - Each leaf contains 1 opaque triangle at different z
// Verifies the walker descends through the internal node, visits
// both leaves, and commits the closer hit.
//
// Scene layout:
//   +  0  VxBvhSceneHeader  { root_offset=16, scene_kind=2, ... }
//   + 16  VxBvhInternalNode { kind=Internal|(2<<8),
//                             origin=(0,0,0), exp=(0,0,0) (scale 1),
//                             child[0].off=80,  AABB (0,0,4)-(1,1,6),
//                             child[1].off=136, AABB (0,0,9)-(1,1,11),
//                             children 2,3 empty }
//   + 80  VxBvhLeafHeader   { kind=LeafTri|(1<<8) }
//   + 96  VxBvhTri          { v0/v1/v2 at z=5, OPAQUE }
//   +136  VxBvhLeafHeader   { kind=LeafTri|(1<<8) }
//   +152  VxBvhTri          { v0/v1/v2 at z=10, OPAQUE }
// Total = 192 B (3 cache lines).
//
// Ray from (0.25,0.25,0) shooting +z hits both AABBs (t_near=4 and 9).
// Walker descends into child 0 first → hits tri at z=5 → updates
// best_t=5 → on pop, ray-AABB test against child 1 culls it (t_near=9
// > best_t=5), so the walker never opens child 1.

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

// Build a single leaf with one opaque triangle at the given z.
static void emit_leaf(uint8_t* out, float z) {
  uint32_t* lh = reinterpret_cast<uint32_t*>(out);
  lh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);
  lh[1] = 0;  // geometry_index
  lh[2] = 0;  // flags
  lh[3] = 0;  // reserved
  float* tri = reinterpret_cast<float*>(out + VX_BVH_LEAF_HDR_BYTES);
  tri[0] = 0.f; tri[1] = 0.f; tri[2] = z;
  tri[3] = 1.f; tri[4] = 0.f; tri[5] = z;
  tri[6] = 0.f; tri[7] = 1.f; tri[8] = z;
  uint32_t* tf = reinterpret_cast<uint32_t*>(out + VX_BVH_LEAF_HDR_BYTES
                                              + VX_BVH_TRI_FLAGS_OFFSET);
  *tf = VX_BVH_TRI_FLAG_OPAQUE;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  std::vector<uint8_t> scene(192, 0);

  // Scene header.
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;            // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;                  // = 2
  sh[2] = 1;                                  // node_count
  sh[3] = 2;                                  // leaf_count

  // Internal node at offset 16. Use exp = 0 → scale = 2^0 = 1, so
  // quantized coords are direct integer real-world units (0..255).
  uint8_t* n0 = scene.data() + VX_BVH_SCENE_HDR_BYTES;
  uint32_t* n0w = reinterpret_cast<uint32_t*>(n0);
  n0w[0] = VX_BVH_KIND_INTERNAL | (2u << VX_BVH_COUNT_SHIFT);
  // origin (3 floats @ offset 4)
  float* origin = reinterpret_cast<float*>(n0 + 4);
  origin[0] = 0.f; origin[1] = 0.f; origin[2] = 0.f;
  // exp (3 int8 @ offset 16)
  int8_t* exp_arr = reinterpret_cast<int8_t*>(n0 + 16);
  exp_arr[0] = 0; exp_arr[1] = 0; exp_arr[2] = 0;
  // pad0 (1 byte) at offset 19
  n0[19] = 0;
  // child_offsets (4 uint32 @ offset 20)
  uint32_t* coff = reinterpret_cast<uint32_t*>(n0 + 20);
  coff[0] = 80;   // leaf 1 starts at offset 80
  coff[1] = 136;  // leaf 2 starts at offset 136
  coff[2] = 0;    // empty
  coff[3] = 0;    // empty
  // qaabb_min[4][3] (12 uint8 @ offset 36)
  uint8_t* qmin = n0 + 36;
  // Child 0: AABB (0,0,4)-(1,1,6) covers tri at z=5
  qmin[0]=0; qmin[1]=0; qmin[2]=4;
  // Child 1: AABB (0,0,9)-(1,1,11) covers tri at z=10
  qmin[3]=0; qmin[4]=0; qmin[5]=9;
  qmin[6]=0; qmin[7]=0; qmin[8]=0;   // child 2 unused
  qmin[9]=0; qmin[10]=0; qmin[11]=0; // child 3 unused
  // qaabb_max[4][3] (12 uint8 @ offset 48)
  uint8_t* qmax = n0 + 48;
  qmax[0]=1; qmax[1]=1; qmax[2]=6;
  qmax[3]=1; qmax[4]=1; qmax[5]=11;
  qmax[6]=0; qmax[7]=0; qmax[8]=0;
  qmax[9]=0; qmax[10]=0; qmax[11]=0;
  // pad1 (4 bytes) at offset 60..63: already zero from vector init

  // Leaves.
  emit_leaf(scene.data() + 80,  5.f);
  emit_leaf(scene.data() + 136, 10.f);

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
            << std::dec << " bvh4 multilevel (1 internal, 2 leaves)" << std::endl;

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
  const float    exp_t      = 5.f;
  std::cout << "oracle: HIT t=" << exp_t << std::endl;

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

  cleanup();
  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
