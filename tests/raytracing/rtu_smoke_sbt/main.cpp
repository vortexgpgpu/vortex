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
// PRISM RTU Shader Binding Table smoke — Phase 7 host driver.

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

vx_device_h device          = nullptr;
vx_buffer_h scene_buffer    = nullptr;
vx_buffer_h res_buffer      = nullptr;
vx_buffer_h payload_buffer  = nullptr;
vx_buffer_h sbt_buffer      = nullptr;
vx_queue_h  queue           = nullptr;
vx_module_h module_         = nullptr;
vx_kernel_h kernel          = nullptr;
kernel_arg_t kernel_arg     = {};

void cleanup() {
  if (device) {
    if (scene_buffer)   vx_buffer_release(scene_buffer);
    if (res_buffer)     vx_buffer_release(res_buffer);
    if (payload_buffer) vx_buffer_release(payload_buffer);
    if (sbt_buffer)     vx_buffer_release(sbt_buffer);
    if (kernel)         vx_kernel_release(kernel);
    if (module_)        vx_module_release(module_);
    if (queue)          vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Scene: 1 procedural primitive with sbt_idx = 1 in the upper byte
  // of the per-tri flags.
  std::vector<uint8_t> scene_bytes(RTU_SCENE_HDR_BYTES + RTU_TRI_STRIDE_BYTES, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(scene_bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  uint32_t sbt_idx = 1;
  *tri_flags = RTU_TRI_FLAG_PROC
             | ((sbt_idx & RTU_TRI_SBT_MASK) << RTU_TRI_SBT_SHIFT);

  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  uint32_t payload_init = 0xdeadbeef;
  RT_CHECK(vx_buffer_create(device, sizeof(uint32_t),
                            VX_MEM_READ_WRITE, &payload_buffer));
  RT_CHECK(vx_buffer_address(payload_buffer, &kernel_arg.payload_addr));
  RT_CHECK(vx_enqueue_write(queue, payload_buffer, 0, &payload_init,
                            sizeof(payload_init), 0, nullptr, nullptr));

  // SBT buffer: 2 records × 16 B each. The kernel populates it at
  // runtime with shader function pointers (host doesn't know device
  // addresses, so the kernel writes them itself).
  uint32_t sbt_size = 2 * RTU_SBT_RECORD_STRIDE;
  RT_CHECK(vx_buffer_create(device, sbt_size, VX_MEM_READ_WRITE, &sbt_buffer));
  RT_CHECK(vx_buffer_address(sbt_buffer, &kernel_arg.sbt_addr));

  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr
            << ", payload_addr=0x" << kernel_arg.payload_addr
            << ", sbt_addr=0x" << kernel_arg.sbt_addr << std::dec
            << " (sbt_idx=1, expect IS shader 1 to fire)" << std::endl;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene_bytes.data(),
                            scene_bytes_sz, 0, nullptr, nullptr));
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

  // Oracle: sbt_idx=1 → SBT lookup hits is_shader_1 → MAGIC_1 in payload.
  // MAGIC_0 would mean the lookup ran with sbt_idx=0 (wrong shader).
  uint32_t exp_status  = VX_RT_STS_DONE_HIT;
  uint32_t exp_payload = RTU_SBT_MAGIC_1;
  std::cout << "oracle: status=" << exp_status
            << " hit_t=5 sbt_payload=0x" << std::hex << exp_payload
            << std::dec << std::endl;

  int errors = 0;
  bool sts_ok = (result.status == exp_status);
  bool t_ok   = std::fabs(result.hit_t - 5.f) < 1e-4f;
  bool pl_ok  = (result.sbt_payload == exp_payload);
  if (!sts_ok || !t_ok || !pl_ok) {
    std::cout << "result: status=" << result.status
              << " hit_t=" << result.hit_t
              << " sbt_payload=0x" << std::hex << result.sbt_payload
              << std::dec << std::endl;
    if (!pl_ok && result.sbt_payload == RTU_SBT_MAGIC_0) {
      std::cout << "  (got MAGIC_0 — SBT lookup hit the wrong record)" << std::endl;
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
