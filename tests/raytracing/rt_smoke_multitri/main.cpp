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
// PRISM RTU multi-triangle smoke — Phase 4 host driver.
//
// Builds a scene of N (default 4) coplanar opaque triangles at depths
// z = 5, 6, 7, ... 5+N-1. The ray at (0.25, 0.25, 0) along +Z hits all
// of them. Oracle: closest is the first triangle at z=5; prim_id=0.

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
uint32_t num_tris = 4;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU multi-triangle smoke (Phase 4 linear-scan walker)." << std::endl;
  std::cout << "Usage: [-k kernel] [-n num_tris] [-h]" << std::endl;
  std::cout << "  -n  number of opaque triangles in the scene (default 4, max "
            << RTU_MAX_TRIS << ")" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': num_tris    = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (num_tris == 0) num_tris = 1;
  if (num_tris > RTU_MAX_TRIS) num_tris = RTU_MAX_TRIS;
}

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

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Build scene: header + N triangles. Tri i at z = 5 + i, all opaque.
  uint32_t scene_bytes_sz = RTU_SCENE_HDR_BYTES + num_tris * RTU_TRI_STRIDE_BYTES;
  std::vector<uint8_t> scene_bytes(scene_bytes_sz, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = num_tris;
  for (uint32_t i = 0; i < num_tris; ++i) {
    uint8_t* tri_base = scene_bytes.data()
                      + RTU_SCENE_HDR_BYTES
                      + i * RTU_TRI_STRIDE_BYTES;
    float* tris = reinterpret_cast<float*>(tri_base);
    float z = 5.f + float(i);
    tris[0] = 0.f; tris[1] = 0.f; tris[2] = z;
    tris[3] = 1.f; tris[4] = 0.f; tris[5] = z;
    tris[6] = 0.f; tris[7] = 1.f; tris[8] = z;
    uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
        tri_base + RTU_TRI_FLAGS_OFFSET);
    *tri_flags = RTU_TRI_FLAG_OPAQUE;   // OPAQUE, no SBT
  }

  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.num_tris         = num_tris;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << ", num_tris=" << num_tris
            << ", scene_bytes=" << scene_bytes_sz
            << " (header + " << num_tris << "×40)"
            << std::endl;

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

  // Oracle: closest opaque tri is tri 0 at z=5; ray hits at t=5, (u, v) =
  // (0.25, 0.25), prim_id=0.
  uint32_t exp_status = VX_RT_STS_DONE_HIT;
  float exp_t = 5.f, exp_u = 0.25f, exp_v = 0.25f;
  uint32_t exp_prim = 0;
  std::cout << "oracle: HIT t=" << exp_t << " u=" << exp_u << " v=" << exp_v
            << " prim=" << exp_prim << std::endl;

  int errors = 0;
  bool ok = (result.status == exp_status)
            && (std::fabs(result.hit_t - exp_t) < 1e-4f)
            && (std::fabs(result.hit_u - exp_u) < 1e-4f)
            && (std::fabs(result.hit_v - exp_v) < 1e-4f)
            && (result.primitive_id == exp_prim);
  if (!ok) {
    std::cout << "result: status=" << result.status
              << " hit_t=" << result.hit_t
              << " hit_u=" << result.hit_u
              << " hit_v=" << result.hit_v
              << " prim=" << result.primitive_id << std::endl;
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
