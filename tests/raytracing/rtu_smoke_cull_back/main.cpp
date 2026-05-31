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
// PRISM RTU smoke — §8.8 CULL_BACK_FACING host driver.
//
// Triangle vertices (0,0,5)/(0,1,5)/(1,0,5) — vertex winding chosen
// so the geometric normal points in -z direction. Front face is the
// -z side (Vulkan: vertices CCW from outside = front).
// The kernel fires two rays, both with CULL_BACK_FACING set:
//
//   front-ray: from (0.25,0.25,0) shooting (0,0,1) — comes in from -z
//              side → hits front face → SURVIVES the cull → HIT @ t=5
//
//   back-ray:  from (0.25,0.25,10) shooting (0,0,-1) — comes in from +z
//              side → hits back face → CULLED → MISS
//
// Both rays use the same scene; the only difference is direction.

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

  // 1-triangle flat-list scene.
  std::vector<uint8_t> scene(RTU_SCENE_HDR_BYTES + RTU_TRI_STRIDE_BYTES, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene.data());
  hdr[0] = 1;
  // Winding chosen so the geometric normal points in -z (front face
  // is the -z side, matched by the +z-shooting "front-ray").
  float* v = reinterpret_cast<float*>(scene.data() + RTU_SCENE_HDR_BYTES);
  v[0] = 0.f; v[1] = 0.f; v[2] = 5.f;  // v0
  v[3] = 0.f; v[4] = 1.f; v[5] = 5.f;  // v1
  v[6] = 1.f; v[7] = 0.f; v[8] = 5.f;  // v2
  *reinterpret_cast<uint32_t*>(scene.data() + RTU_SCENE_HDR_BYTES
                                + RTU_TRI_FLAGS_OFFSET) = RTU_TRI_FLAG_OPAQUE;

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(),
                            VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.front_origin[0] = 0.25f; kernel_arg.front_origin[1] = 0.25f; kernel_arg.front_origin[2] = 0.f;
  kernel_arg.front_dir[0]    = 0.f;   kernel_arg.front_dir[1]    = 0.f;   kernel_arg.front_dir[2]    = 1.f;
  kernel_arg.back_origin[0]  = 0.25f; kernel_arg.back_origin[1]  = 0.25f; kernel_arg.back_origin[2]  = 10.f;
  kernel_arg.back_dir[0]     = 0.f;   kernel_arg.back_dir[1]     = 0.f;   kernel_arg.back_dir[2]     = -1.f;
  kernel_arg.tmin            = 0.001f;
  kernel_arg.tmax            = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " CULL_BACK_FACING: front-ray must HIT, back-ray must MISS" << std::endl;

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

  const uint32_t exp_front = VX_RT_STS_DONE_HIT;
  const uint32_t exp_back  = VX_RT_STS_DONE_MISS;
  const float    exp_t     = 5.f;
  std::cout << "oracle: front=HIT(t=" << exp_t << ") back=MISS(culled)" << std::endl;

  int errors = 0;
  if (result.front_status != exp_front) {
    std::cout << "front_status mismatch: got " << result.front_status
              << " expected " << exp_front << std::endl;
    ++errors;
  }
  if (std::fabs(result.front_t - exp_t) > 1e-4f) {
    std::cout << "front_t mismatch: got " << result.front_t
              << " expected " << exp_t << std::endl;
    ++errors;
  }
  if (result.back_status != exp_back) {
    std::cout << "back_status mismatch: got " << result.back_status
              << " expected " << exp_back
              << " (cull should have made this a MISS)" << std::endl;
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
