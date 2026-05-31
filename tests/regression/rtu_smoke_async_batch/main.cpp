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
// §8.6 async batch smoke — host driver. Allocates RTU_ASYNC_NUM_BATCH
// independent single-triangle scenes (each triangle at a unique z so
// hit_t is unique per ray), launches the kernel, validates that the
// per-ray status + hit_t match the expected values for each handle.

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
uint32_t num_lanes = 1;

vx_device_h device       = nullptr;
vx_buffer_h scene_bufs[RTU_ASYNC_NUM_BATCH] = {nullptr};
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "PRISM RTU async batch smoke test." << std::endl;
  std::cout << "Usage: [-k kernel] [-n lanes] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': num_lanes = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    for (int i = 0; i < RTU_ASYNC_NUM_BATCH; ++i) {
      if (scene_bufs[i]) vx_buffer_release(scene_bufs[i]);
    }
    if (res_buffer) vx_buffer_release(res_buffer);
    if (kernel)     vx_kernel_release(kernel);
    if (module_)    vx_module_release(module_);
    if (queue)      vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Build NUM_BATCH single-triangle scenes. Triangle i at z = (i+1)
  // so hit_t = (i+1) for each ray (ray origin is at z=0 shooting
  // +z, so t = triangle_z - 0 = triangle_z).
  uint8_t scene_bytes[RTU_ASYNC_NUM_BATCH][64] = {};
  for (int i = 0; i < RTU_ASYNC_NUM_BATCH; ++i) {
    uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes[i]);
    hdr[0] = 1;  // triangle_count
    float* tris = reinterpret_cast<float*>(scene_bytes[i] + RTU_SCENE_HDR_BYTES);
    float z = float(i + 1);
    tris[0] = 0.f; tris[1] = 0.f; tris[2] = z;  // v0
    tris[3] = 1.f; tris[4] = 0.f; tris[5] = z;  // v1
    tris[6] = 0.f; tris[7] = 1.f; tris[8] = z;  // v2
    uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
        scene_bytes[i] + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
    *tri_flags = RTU_TRI_FLAG_OPAQUE;

    RT_CHECK(vx_buffer_create(device, sizeof(scene_bytes[i]),
                              VX_MEM_READ, &scene_bufs[i]));
    RT_CHECK(vx_buffer_address(scene_bufs[i], &kernel_arg.scene_addr[i]));
    RT_CHECK(vx_enqueue_write(queue, scene_bufs[i], 0, scene_bytes[i],
                              sizeof(scene_bytes[i]), 0, nullptr, nullptr));
  }

  // Result buffer.
  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.num_lanes        = num_lanes;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scenes:";
  for (int i = 0; i < RTU_ASYNC_NUM_BATCH; ++i)
    std::cout << " 0x" << std::hex << kernel_arg.scene_addr[i] << std::dec;
  std::cout << ", num_lanes=" << num_lanes << std::endl;

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
    li.grid_dim[0]  = num_lanes;
    li.block_dim[0] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<rtu_result_t> results(num_lanes);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Each ray i should hit at t = i+1.
  int errors = 0;
  for (uint32_t l = 0; l < num_lanes; ++l) {
    for (int i = 0; i < RTU_ASYNC_NUM_BATCH; ++i) {
      float expected_t = float(i + 1);
      bool sts_ok = (results[l].rays[i].status == VX_RT_STS_DONE_HIT);
      bool t_ok   = (std::fabs(results[l].rays[i].hit_t - expected_t) < 1e-4f);
      if (!sts_ok || !t_ok) {
        std::cout << "lane " << l << " ray " << i
                  << ": status=" << results[l].rays[i].status
                  << " hit_t=" << results[l].rays[i].hit_t
                  << " (expected status=" << VX_RT_STS_DONE_HIT
                  << " t=" << expected_t << ")" << std::endl;
        ++errors;
      }
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
