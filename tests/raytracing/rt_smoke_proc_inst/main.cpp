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
// PRISM RTU smoke: procedural primitives inside instances.
//
// A TLAS leaf holds two instances of ONE procedural BLAS (a unit sphere at
// object-space (0,0,5)), translated to x=-3 (id 5, custom 0xa5) and x=+3
// (id 9, custom 0xa9). Two lanes of one warp each fire a +z ray through their
// own instance. Each lane's intersection shader must see the candidate's
// gl_InstanceID / gl_InstanceCustomIndexEXT, and the committed hit must carry
// them: an IS that indexes per-instance data by instance id (every sphere of a
// "Ray Tracing in One Weekend" scene does) reads the wrong sphere otherwise.
// The ids are non-zero on purpose, so a never-written register can't pass.
//
// Scene layout (200 B):
//   +  0  VxBvhSceneHeader { root_offset=16, scene_kind=2 }
//   + 16  VxBvhLeafHeader  { kind=LeafInst|(2<<8) }
//   + 32  VxBvhInstance    id 5, translate (-3,0,0)   -> blas_off=160
//   + 96  VxBvhInstance    id 9, translate (+3,0,0)   -> blas_off=160
//   +160  VxBvhLeafHeader  { kind=LeafProc|(1<<8) }
//   +176  VxBvhProcAabb    { min=(-1,-1,4), max=(1,1,6) }
//
// Expected per lane: DONE_HIT, t=4, one IS call, IS and hit ids = the lane's
// instance.

#include <iostream>
#include <vector>
#include <cmath>

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

static const uint32_t kInstId[NUM_RAYS]   = { 5, 9 };
static const uint32_t kInstCust[NUM_RAYS] = { 0xa5, 0xa9 };
static const float    kInstTx[NUM_RAYS]   = { -3.f, 3.f };

// Identity rotation + translation (tx,0,0); the record holds its inverse
// (world->object).
static void emit_instance(uint8_t* out, float tx, uint32_t blas_off,
                          uint32_t custom_id, uint32_t instance_id) {
  float* x = reinterpret_cast<float*>(out);
  x[0] = 1.f; x[1] = 0.f; x[2]  = 0.f; x[3]  = -tx;
  x[4] = 0.f; x[5] = 1.f; x[6]  = 0.f; x[7]  = 0.f;
  x[8] = 0.f; x[9] = 0.f; x[10] = 1.f; x[11] = 0.f;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_BLAS_OFF)  = blas_off;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CUSTOM_ID) = custom_id;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_ID_OFFSET) = instance_id;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CULL_MASK) = 0xffu;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const uint32_t blas_off = 32 + NUM_RAYS * VX_BVH_INSTANCE_STRIDE;   // 160
  std::vector<uint8_t> scene(blas_off + VX_BVH_LEAF_HDR_BYTES + VX_BVH_PROC_AABB_BYTES, 0);

  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;     // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;          // = 2 (BVH4)
  sh[2] = (uint32_t)scene.size();     // total scene bytes (pre-fetch)
  sh[3] = 2;                          // leaf_count (1 inst leaf + 1 proc leaf)

  uint32_t* rlh = reinterpret_cast<uint32_t*>(scene.data() + VX_BVH_SCENE_HDR_BYTES);
  rlh[0] = VX_BVH_KIND_LEAF_INST | ((uint32_t)NUM_RAYS << VX_BVH_COUNT_SHIFT);
  for (int i = 0; i < NUM_RAYS; ++i) {
    emit_instance(scene.data() + 32 + i * VX_BVH_INSTANCE_STRIDE,
                  kInstTx[i], blas_off, kInstCust[i], kInstId[i]);
  }

  uint32_t* blh = reinterpret_cast<uint32_t*>(scene.data() + blas_off);
  blh[0] = VX_BVH_KIND_LEAF_PROC | (1u << VX_BVH_COUNT_SHIFT);
  float* aabb = reinterpret_cast<float*>(scene.data() + blas_off + VX_BVH_LEAF_HDR_BYTES);
  aabb[0] = -1.f; aabb[1] = -1.f; aabb[2] = 4.f;   // min
  aabb[3] =  1.f; aabb[4] =  1.f; aabb[5] = 6.f;   // max

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  const uint32_t res_size = NUM_RAYS * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  for (int i = 0; i < NUM_RAYS; ++i) {
    kernel_arg.ray_origin[i][0] = kInstTx[i];
    kernel_arg.ray_origin[i][1] = 0.f;
    kernel_arg.ray_origin[i][2] = 0.f;
  }
  kernel_arg.ray_direction[0] = 0.f;
  kernel_arg.ray_direction[1] = 0.f;
  kernel_arg.ray_direction[2] = 1.f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " bvh4 (2 instances of 1 leaf_proc sphere)" << std::endl;

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
    li.block_dim[0] = NUM_RAYS;   // both rays in one warp
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  rtu_result_t res[NUM_RAYS] = {};
  RT_CHECK(vx_enqueue_read(queue, res, res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (int i = 0; i < NUM_RAYS; ++i) {
    const rtu_result_t& r = res[i];
    std::cout << "lane " << i << ": status=" << r.status << " t=" << r.hit_t
              << " hit_inst=" << r.hit_inst << " hit_cust=0x" << std::hex << r.hit_cust
              << std::dec << " is_calls=" << r.is_calls << " is_inst=" << r.is_inst
              << " is_cust=0x" << std::hex << r.is_cust << std::dec
              << " obj_ray=(" << r.obj_ray[0] << "," << r.obj_ray[1] << "," << r.obj_ray[2]
              << ")+(" << r.obj_ray[3] << "," << r.obj_ray[4] << "," << r.obj_ray[5] << ")"
              << std::endl;
    bool ok = (r.status == VX_RT_STS_DONE_HIT)
           && (std::fabs(r.hit_t - 4.f) <= 1e-4f)
           && (r.is_calls == 1)
           && (r.is_inst == kInstId[i])  && (r.is_cust == kInstCust[i])
           && (r.hit_inst == kInstId[i]) && (r.hit_cust == kInstCust[i]);
    if (!ok) {
      std::cout << "  expected: status=" << VX_RT_STS_DONE_HIT << " t=4 is_calls=1"
                << " inst=" << kInstId[i] << " cust=0x" << std::hex << kInstCust[i]
                << std::dec << " (IS and hit)" << std::endl;
      ++errors;
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
