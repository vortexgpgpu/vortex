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
// PRISM RTU reformation multi-warp smoke — Phase 3-A2 non-interference.
//
// Launches num_warps blocks × lanes_per_warp threads each, so the CTA
// dispatcher hands each block to a distinct warp. Every lane fires the
// same ray at the same non-opaque tri (sbt_idx=0). Each warp must
// receive its OWN CB_YIELD trap with cb_mask == 0xf — and never see a
// lane bit from another warp leak in. With --debug=3 the run log shows
// exactly num_warps "rtu-core reform cb_yield: warp=N, sbt=0, cb_mask=0xf"
// lines, each with a distinct N.

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
uint32_t num_warps      = 2;
uint32_t lanes_per_warp = 4;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU reformation multi-warp smoke (Phase 3-A2 non-interference)." << std::endl;
  std::cout << "Usage: [-k kernel] [-w warps] [-n lanes_per_warp] [-h]" << std::endl;
  std::cout << "  -w  number of concurrent warps  (default 2; one block per warp)" << std::endl;
  std::cout << "  -n  lanes per warp              (default NUM_THREADS = 4)" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "w:n:k:h")) != -1) {
    switch (c) {
    case 'w': num_warps      = atoi(optarg); break;
    case 'n': lanes_per_warp = atoi(optarg); break;
    case 'k': kernel_file    = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
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
  uint32_t total_lanes = num_warps * lanes_per_warp;

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  std::vector<uint8_t> scene_bytes(64, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(scene_bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = 0;

  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = total_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.total_lanes      = total_lanes;
  kernel_arg.lanes_per_warp   = lanes_per_warp;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << ", num_warps=" << num_warps
            << ", lanes_per_warp=" << lanes_per_warp
            << " -> total_lanes=" << total_lanes
            << " (expect 1 CB_YIELD per warp, no cross-warp batching)"
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
    li.grid_dim[0]  = num_warps;       // one block per warp
    li.block_dim[0] = lanes_per_warp;  // -> one warp per block
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<rtu_result_t> results(total_lanes);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  uint32_t exp_status = VX_RT_STS_DONE_HIT;
  float exp_t = 5.f, exp_u = 0.25f, exp_v = 0.25f;
  std::cout << "oracle: ALL " << total_lanes
            << " lanes HIT t=" << exp_t
            << " u=" << exp_u
            << " v=" << exp_v << std::endl;

  int errors = 0;
  for (uint32_t i = 0; i < total_lanes; ++i) {
    bool ok = (results[i].status == exp_status)
              && (std::fabs(results[i].hit_t - exp_t) < 1e-4f)
              && (std::fabs(results[i].hit_u - exp_u) < 1e-4f)
              && (std::fabs(results[i].hit_v - exp_v) < 1e-4f);
    if (!ok) {
      std::cout << "lane " << i
                << " (warp " << (i / lanes_per_warp)
                << ", tid " << (i % lanes_per_warp)
                << "): status=" << results[i].status
                << " hit_t=" << results[i].hit_t
                << " hit_u=" << results[i].hit_u
                << " hit_v=" << results[i].hit_v
                << std::endl;
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
