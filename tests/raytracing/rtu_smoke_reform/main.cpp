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
// PRISM RTU reformation smoke — Phase 3-A2 (option A) host driver.
//
// Launches num_lanes threads in ONE block (so they share a single warp).
// All lanes trace the same ray against the same non-opaque triangle; the
// RtuCore's reformation_dispatch must batch them into a single CB_YIELD
// with cb_mask covering all active lanes. With debug=3 the run log
// shows exactly one "rtu-core reform cb_yield" line per warp.

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
uint32_t num_lanes    = 4;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU reformation smoke (Phase 3-A2 option A)." << std::endl;
  std::cout << "Usage: [-k kernel] [-n lanes] [-h]" << std::endl;
  std::cout << "  -n  number of lanes in ONE warp (default 4, max NUM_THREADS)" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': num_lanes   = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
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

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Single non-opaque triangle, sbt_idx=0. Identical to rtu_smoke_ahs so
  // the oracle stays trivial (HIT at t=5).
  std::vector<uint8_t> scene_bytes(64, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(scene_bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = 0;  // OPAQUE clear (yield) + sbt_idx 0 (in bits 8..15)

  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

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

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << ", num_lanes=" << num_lanes
            << " (single warp, same sbt -> expect 1 batched CB_YIELD)"
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
    li.grid_dim[0]  = 1;          // single block
    li.block_dim[0] = num_lanes;  // -> single warp with N threads
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<rtu_result_t> results(num_lanes);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Oracle: every lane fires the same ray at the same NON-opaque tri,
  // dispatcher ACCEPTs -> every lane HIT at t=5, u=v=0.25.
  uint32_t exp_status = VX_RT_STS_DONE_HIT;
  float exp_t = 5.f;
  float exp_u = 0.25f;
  float exp_v = 0.25f;
  std::cout << "oracle: ALL " << num_lanes
            << " lanes HIT t=" << exp_t
            << " u=" << exp_u
            << " v=" << exp_v << std::endl;

  int errors = 0;
  for (uint32_t i = 0; i < num_lanes; ++i) {
    bool sts_ok = (results[i].status == exp_status);
    bool t_ok   = (std::fabs(results[i].hit_t - exp_t) < 1e-4f);
    bool u_ok   = (std::fabs(results[i].hit_u - exp_u) < 1e-4f);
    bool v_ok   = (std::fabs(results[i].hit_v - exp_v) < 1e-4f);
    if (!sts_ok || !t_ok || !u_ok || !v_ok) {
      std::cout << "lane " << i << ": status=" << results[i].status
                << " hit_t=" << results[i].hit_t
                << " hit_u=" << results[i].hit_u
                << " hit_v=" << results[i].hit_v
                << " (expected status=" << exp_status
                << " t=" << exp_t
                << " u=" << exp_u
                << " v=" << exp_v << ")" << std::endl;
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
