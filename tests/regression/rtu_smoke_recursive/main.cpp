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
// PRISM RTU recursive-trace smoke — Phase 12 host driver.

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

vx_device_h device           = nullptr;
vx_buffer_h scene_buffer     = nullptr;
vx_buffer_h sub_scene_buffer = nullptr;
vx_buffer_h res_buffer       = nullptr;
vx_buffer_h payload_buffer   = nullptr;
vx_queue_h  queue            = nullptr;
vx_module_h module_          = nullptr;
vx_kernel_h kernel           = nullptr;
kernel_arg_t kernel_arg      = {};

void cleanup() {
  if (device) {
    if (scene_buffer)     vx_buffer_release(scene_buffer);
    if (sub_scene_buffer) vx_buffer_release(sub_scene_buffer);
    if (res_buffer)       vx_buffer_release(res_buffer);
    if (payload_buffer)   vx_buffer_release(payload_buffer);
    if (kernel)           vx_kernel_release(kernel);
    if (module_)          vx_module_release(module_);
    if (queue)            vx_queue_release(queue);
    vx_device_release(device);
  }
}

static void build_opaque_scene(std::vector<uint8_t>& bytes, float z) {
  bytes.assign(RTU_SCENE_HDR_BYTES + RTU_TRI_STRIDE_BYTES, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(bytes.data());
  hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = z;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = z;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = z;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = RTU_TRI_FLAG_OPAQUE;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Parent (primary) scene: opaque tri at z=5.
  // Sub-scene used by the recursive CHS: opaque tri at z=10.
  std::vector<uint8_t> parent_bytes, sub_bytes;
  build_opaque_scene(parent_bytes, 5.f);
  build_opaque_scene(sub_bytes,    10.f);

  RT_CHECK(vx_buffer_create(device, (uint32_t)parent_bytes.size(),
                            VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
  RT_CHECK(vx_buffer_create(device, (uint32_t)sub_bytes.size(),
                            VX_MEM_READ, &sub_scene_buffer));
  RT_CHECK(vx_buffer_address(sub_scene_buffer, &kernel_arg.sub_scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  uint32_t payload_init = 0xdeadbeef;
  RT_CHECK(vx_buffer_create(device, sizeof(uint32_t),
                            VX_MEM_READ_WRITE, &payload_buffer));
  RT_CHECK(vx_buffer_address(payload_buffer, &kernel_arg.payload_addr));
  RT_CHECK(vx_enqueue_write(queue, payload_buffer, 0, &payload_init,
                            sizeof(payload_init), 0, nullptr, nullptr));

  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "parent_scene=0x" << std::hex << kernel_arg.scene_addr
            << ", sub_scene=0x"  << kernel_arg.sub_scene_addr
            << ", payload=0x"    << kernel_arg.payload_addr
            << std::dec << std::endl;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, parent_bytes.data(),
                            (uint32_t)parent_bytes.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, sub_scene_buffer, 0, sub_bytes.data(),
                            (uint32_t)sub_bytes.size(), 0, nullptr, nullptr));
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

  // Oracle:
  //   status     = HIT     (parent ray hit parent scene at t=5)
  //   hit_t      = 5       (parent regfile re-written by parent TERMINAL
  //                         AFTER the CHS sub-ray's TERMINAL)
  //   sub_status = HIT (0) (CHS's recursive trace hit sub scene at t=10)
  uint32_t exp_status     = VX_RT_STS_DONE_HIT;
  float    exp_hit_t      = 5.f;
  uint32_t exp_sub_status = VX_RT_STS_DONE_HIT;
  std::cout << "oracle: status=" << exp_status
            << " hit_t=" << exp_hit_t
            << " sub_status=" << exp_sub_status << std::endl;

  int errors = 0;
  bool sts_ok = (result.status == exp_status);
  bool t_ok   = std::fabs(result.hit_t - exp_hit_t) < 1e-4f;
  bool ss_ok  = (result.sub_status == exp_sub_status);
  if (!sts_ok || !t_ok || !ss_ok) {
    std::cout << "result: status=" << result.status
              << " hit_t=" << result.hit_t
              << " sub_status=" << result.sub_status << std::endl;
    if (!ss_ok && result.sub_status == 0xdeadbeef) {
      std::cout << "  (sub_status still sentinel — CHS dispatcher did not fire OR sub-trace did not complete)"
                << std::endl;
    }
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
