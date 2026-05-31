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
// §8.8 instanceCullMask smoke — host driver. 2-instance TLAS, three
// rays differing only in cull_mask. Validates per-ray status, hit_t,
// and hit_instance_id against an oracle.

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

  // 2-instance TLAS sharing one inline BLAS.
  //   instance 0 at world z=5,  cull_mask = 0x01
  //   instance 1 at world z=10, cull_mask = 0x02
  //
  // Layout (200 B):
  //   [0..16)    TLAS header (primary_count=2, scene_kind=TLAS)
  //   [16..80)   instance 0
  //   [80..144)  instance 1
  //   [144..160) BLAS header
  //   [160..200) BLAS triangle in object space
  constexpr uint32_t kNumInstances = 2;
  constexpr uint32_t kBlasOff = RTU_SCENE_HDR_BYTES
                              + kNumInstances * RTU_INSTANCE_STRIDE;
  constexpr uint32_t kSceneSz = kBlasOff + RTU_SCENE_HDR_BYTES + RTU_TRI_STRIDE_BYTES;
  std::vector<uint8_t> scene_bytes(kSceneSz, 0);

  uint32_t* tlas_hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  tlas_hdr[0] = kNumInstances;
  tlas_hdr[1] = RTU_SCENE_KIND_TLAS;

  auto set_instance =
    [&](uint32_t idx, float tz, uint32_t cull_mask) {
      uint8_t* inst = scene_bytes.data() + RTU_SCENE_HDR_BYTES
                    + idx * RTU_INSTANCE_STRIDE;
      float* xform = reinterpret_cast<float*>(inst);
      xform[0] = 1.f; xform[1] = 0.f; xform[2]  = 0.f; xform[3]  = 0.f;
      xform[4] = 0.f; xform[5] = 1.f; xform[6]  = 0.f; xform[7]  = 0.f;
      xform[8] = 0.f; xform[9] = 0.f; xform[10] = 1.f; xform[11] = tz;
      uint32_t* inst_tail = reinterpret_cast<uint32_t*>(
          inst + RTU_INSTANCE_BLAS_OFF_OFF);
      inst_tail[0] = kBlasOff;
      inst_tail[1] = 0xC0DE0000u + idx;
      inst_tail[2] = cull_mask;
    };
  set_instance(0,  5.f, 0x01u);
  set_instance(1, 10.f, 0x02u);

  uint32_t* blas_hdr = reinterpret_cast<uint32_t*>(scene_bytes.data() + kBlasOff);
  blas_hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(
      scene_bytes.data() + kBlasOff + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 0.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 0.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 0.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + kBlasOff + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = RTU_TRI_FLAG_OPAQUE;

  RT_CHECK(vx_buffer_create(device, kSceneSz, VX_MEM_READ, &scene_buffer));
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
  kernel_arg.ray_cull_mask[0] = 0x01u;
  kernel_arg.ray_cull_mask[1] = 0x02u;
  kernel_arg.ray_cull_mask[2] = 0xffu;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << ", 2 instances cull_mask=(0x01, 0x02)" << std::endl;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene_bytes.data(),
                            kSceneSz, 0, nullptr, nullptr));
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
  //   ray A (0x01): only inst 0 (at z=5) is visible → HIT t=5, inst=0
  //   ray B (0x02): only inst 1 (at z=10) is visible → HIT t=10, inst=1
  //   ray C (0xff): both visible; inst 0 is closer  → HIT t=5,  inst=0
  struct expect_t { uint32_t status; float t; uint32_t inst; };
  expect_t expected[RTU_NUM_RAYS] = {
    { VX_RT_STS_DONE_HIT,  5.f, 0u },
    { VX_RT_STS_DONE_HIT, 10.f, 1u },
    { VX_RT_STS_DONE_HIT,  5.f, 0u },
  };
  std::cout << "oracle: rayA(mask=0x01) HIT t=5 inst=0; "
               "rayB(mask=0x02) HIT t=10 inst=1; "
               "rayC(mask=0xff) HIT t=5 inst=0" << std::endl;

  int errors = 0;
  for (uint32_t i = 0; i < RTU_NUM_RAYS; ++i) {
    bool sts_ok = (result.rays[i].status == expected[i].status);
    bool t_ok   = std::fabs(result.rays[i].hit_t - expected[i].t) < 1e-4f;
    bool id_ok  = (result.rays[i].hit_instance_id == expected[i].inst);
    if (!sts_ok || !t_ok || !id_ok) {
      std::cout << "ray " << i
                << ": status=" << result.rays[i].status
                << " hit_t=" << result.rays[i].hit_t
                << " inst=" << result.rays[i].hit_instance_id
                << " (expected status=" << expected[i].status
                << " t=" << expected[i].t
                << " inst=" << expected[i].inst << ")" << std::endl;
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
