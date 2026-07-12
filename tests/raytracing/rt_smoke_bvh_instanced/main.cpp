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
// PRISM RTU smoke — Phase 4 chunk 4 host driver.
//
// Builds a TLAS-over-BLAS BVH4 with two instances of a single shared
// BLAS. The BLAS holds a triangle at (object-space) z=5; the two
// instances translate it to world z=5 (untransformed) and world z=15
// (translated +10 along z). Verifies:
//   - Walker descends into LeafInst leaf
//   - Per-instance ray transform (translation) works
//   - World hit_t = closer instance's object_t (== 5) after transform
//   - hit_instance_id reflects the HW-assigned instance ID
//
// Scene layout (192 B):
//   +  0  VxBvhSceneHeader  { root_offset=16, scene_kind=2 }
//   + 16  VxBvhLeafHeader   { kind=LeafInst|(2<<8) }     ← root = inst leaf
//   + 32  VxBvhInstance     instance 0 (identity)        ← 64 B, blas_off=160
//   + 96  VxBvhInstance     instance 1 (translate z+10)  ← 64 B, blas_off=160
//   +160  VxBvhLeafHeader   { kind=LeafTri|(1<<8) }      ← shared BLAS
//   +176  VxBvhTri          { v0/v1/v2 at z=5, OPAQUE }  ← 40 B → ends at 216
//
// Total = 216 B (4 cache lines, fits in kRtuMaxLinesPerLane = 12).
//
// Ray from (0.25,0.25,0) +z:
//   - instance 0 (identity): object ray = world ray → hits tri at object t=5 → world t=5
//   - instance 1 (translate +10z): object ray origin = (0.25,0.25,-10) →
//     tri at obj-z=5 → t=15 in world units, but since object t=15 also
//     equals 15 (translation preserves t), this is farther
// Expected: HIT at t=5, instance_id = 0.

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

// Fill a VxBvhInstance: identity rotation + translation (tx,ty,tz),
// BLAS root offset, hw instance id.
static void emit_instance(uint8_t* out, float tx, float ty, float tz,
                          uint32_t blas_off, uint32_t custom_id,
                          uint32_t instance_id) {
  float* xform = reinterpret_cast<float*>(out);
  // Row-major 3x4 affine: [R t]. Identity R, translation t.
  xform[0]  = 1.f; xform[1]  = 0.f; xform[2]  = 0.f; xform[3]  = tx;
  xform[4]  = 0.f; xform[5]  = 1.f; xform[6]  = 0.f; xform[7]  = ty;
  xform[8]  = 0.f; xform[9]  = 0.f; xform[10] = 1.f; xform[11] = tz;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_BLAS_OFF)  = blas_off;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CUSTOM_ID) = custom_id;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_ID_OFFSET) = instance_id;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CULL_MASK) = 0xffu;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  std::vector<uint8_t> scene(216, 0);

  // Scene header.
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;   // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;         // = 2
  sh[2] = (uint32_t)scene.size();    // total scene bytes (pre-fetch)
  sh[3] = 2;                         // leaf_count (1 inst leaf + 1 tri leaf)

  // Root = LeafInst with 2 instances at offset 16.
  uint8_t* root_leaf = scene.data() + 16;
  uint32_t* rlh = reinterpret_cast<uint32_t*>(root_leaf);
  rlh[0] = VX_BVH_KIND_LEAF_INST | (2u << VX_BVH_COUNT_SHIFT);
  rlh[1] = 0;  // geometry_index
  rlh[2] = 0;  // flags
  rlh[3] = 0;  // reserved

  // Instance 0 at offset 32: identity transform, BLAS at offset 160.
  emit_instance(scene.data() + 32,
                /*tx=*/0.f, /*ty=*/0.f, /*tz=*/0.f,
                /*blas_off=*/160, /*custom_id=*/0xa0,
                /*instance_id=*/0);
  // Instance 1 at offset 96: translate +10 along z.
  emit_instance(scene.data() + 96,
                /*tx=*/0.f, /*ty=*/0.f, /*tz=*/10.f,
                /*blas_off=*/160, /*custom_id=*/0xa1,
                /*instance_id=*/1);

  // Shared BLAS: LeafTri at offset 160 with 1 opaque tri at z=5.
  uint8_t* blas = scene.data() + 160;
  uint32_t* blh = reinterpret_cast<uint32_t*>(blas);
  blh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);
  blh[1] = 0;
  blh[2] = 0;
  blh[3] = 0;
  float* tri = reinterpret_cast<float*>(blas + VX_BVH_LEAF_HDR_BYTES);
  tri[0] = 0.f; tri[1] = 0.f; tri[2] = 5.f;
  tri[3] = 1.f; tri[4] = 0.f; tri[5] = 5.f;
  tri[6] = 0.f; tri[7] = 1.f; tri[8] = 5.f;
  uint32_t* tf = reinterpret_cast<uint32_t*>(blas + VX_BVH_LEAF_HDR_BYTES
                                              + VX_BVH_TRI_FLAGS_OFFSET);
  *tf = VX_BVH_TRI_FLAG_OPAQUE;

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
            << std::dec << " bvh4 instanced (2 instances of 1 BLAS)" << std::endl;

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

  const uint32_t exp_status      = VX_RT_STS_DONE_HIT;
  const float    exp_t           = 5.f;     // closer instance (identity)
  const uint32_t exp_instance_id = 0;       // instance 0 wins (closer)
  std::cout << "oracle: HIT t=" << exp_t
            << " instance_id=" << exp_instance_id << std::endl;

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
  if (result.instance_id != exp_instance_id) {
    std::cout << "instance_id mismatch: got " << result.instance_id
              << " expected " << exp_instance_id << std::endl;
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
