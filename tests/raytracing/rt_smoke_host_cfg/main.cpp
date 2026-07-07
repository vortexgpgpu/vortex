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
// PRISM RTU host-config smoke — exercises the vortex::raytrace host library
// (ISA v2 §5.3): build_bvh_scene<4> transcodes a one-triangle host scene into
// the CW-BVH4 bytes the walker reads, and program() writes the VX_DCR_RTU_*
// per-dispatch config before launch. The kernel then traces one ray; the hit
// must match the bvh_basic oracle, proving the host-built scene is walkable.

#include <iostream>
#include <vector>
#include <cmath>

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

  // Host-side scene prep: one opaque triangle, transcoded to CW-BVH4 bytes by
  // the runtime library (instead of the hand-packed buffer in bvh_basic).
  vortex::raytrace::host_tri_t tri = {
    { 0.f, 0.f, 5.f },   // v0
    { 1.f, 0.f, 5.f },   // v1
    { 0.f, 1.f, 5.f },   // v2
    RTU_BVH_FLAG_OPAQUE,
  };
  vortex::raytrace::host_bvh_t src = { &tri, 1, /*geometry_index*/ 0 };
  std::vector<uint8_t> scene;
  uint64_t root_offset = 0;
  if (!vortex::raytrace::build_bvh_scene<4>(src, scene, root_offset)) {
    std::cout << "build_bvh_scene failed" << std::endl;
    cleanup();
    return 1;
  }

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(),
                            VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  // Per-dispatch config: program the VX_DCR_RTU_* block before launch.
  vortex::raytrace::config_t cfg;
  cfg.scene_kind    = RTU_SCENE_KIND_BVH4;
  cfg.bvh_width     = 4;
  cfg.cull_defaults = 0xff;
  RT_CHECK(vortex::raytrace::program(device, cfg));

  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " host-built bvh4 (" << scene.size() << " B, root_off="
            << root_offset << ")" << std::endl;

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

  // Oracle: HIT at t=5, barycentrics (0.25, 0.25), prim 0 (same as bvh_basic).
  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t = 5.f, exp_u = 0.25f, exp_v = 0.25f;
  const uint32_t exp_prim = 0;

  std::cout << "oracle: HIT t=" << exp_t << " u=" << exp_u << " v=" << exp_v
            << " prim=" << exp_prim << std::endl;

  int errors = 0;
  if (result.status != exp_status) {
    std::cout << "status mismatch: got " << result.status
              << " expected " << exp_status << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_t - exp_t) > 1e-4f) {
    std::cout << "hit_t mismatch: got " << result.hit_t << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_u - exp_u) > 1e-4f) {
    std::cout << "hit_u mismatch: got " << result.hit_u << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_v - exp_v) > 1e-4f) {
    std::cout << "hit_v mismatch: got " << result.hit_v << std::endl;
    ++errors;
  }
  if (result.primitive_id != exp_prim) {
    std::cout << "prim_id mismatch: got " << result.primitive_id << std::endl;
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
