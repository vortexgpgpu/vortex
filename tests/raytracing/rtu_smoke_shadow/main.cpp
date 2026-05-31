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
// PRISM RTU smoke — §8.8 shadow-ray host driver.
//
// Scene: 3 opaque tris at z=5, z=10, z=15. Ray (0.25,0.25,0)→+z.
// Walker visits tris in scene order (flat-list) and would normally
// commit the closest (z=5) after testing all three. With
// VX_RT_FLAG_TERMINATE_ON_FIRST_HIT, the walker commits the first
// hit it commits and bails — same final t (z=5 is first in the
// flat list, and TERMINATE doesn't reorder), but the rest of the
// scan is skipped.
//
// This test verifies correctness: status==HIT, hit_t==5. The
// efficiency claim (fewer tris tested) is implicit in the
// implementation; a Phase 8 perf-counter test would validate it
// directly.

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

  // 3-triangle flat-list scene, all opaque, at z = 5, 10, 15.
  std::vector<uint8_t> scene(RTU_SCENE_HDR_BYTES + 3 * RTU_TRI_STRIDE_BYTES, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene.data());
  hdr[0] = 3;   // triangle_count
  // hdr[1] = 0 (scene_kind = TRI_LIST)
  float zs[3] = { 5.f, 10.f, 15.f };
  for (int i = 0; i < 3; ++i) {
    uint8_t* tri = scene.data() + RTU_SCENE_HDR_BYTES + i * RTU_TRI_STRIDE_BYTES;
    float* v = reinterpret_cast<float*>(tri);
    v[0] = 0.f; v[1] = 0.f; v[2] = zs[i];
    v[3] = 1.f; v[4] = 0.f; v[5] = zs[i];
    v[6] = 0.f; v[7] = 1.f; v[8] = zs[i];
    *reinterpret_cast<uint32_t*>(tri + RTU_TRI_FLAGS_OFFSET) =
        RTU_TRI_FLAG_OPAQUE;
  }

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

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " 3 opaque tris @ z=5,10,15 (TERMINATE_ON_FIRST_HIT)" << std::endl;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 1;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  rtu_result_t result = {};
  RT_CHECK(vx_enqueue_read(queue, &result, res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Oracle: first-hit-in-scan-order is the z=5 tri (it's tri 0).
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
