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
// PRISM RTU reformation divergent-SBT smoke — Phase 3-A2 option B host driver.
//
// Builds num_lanes per-lane scenes back-to-back (one cache line each).
// Each lane's tri carries sbt_idx = (tid / sbt_group_size). With the
// defaults (num_lanes=4, sbt_group_size=2) the warp splits 2 lanes →
// sbt 0 and 2 lanes → sbt 1. The dispatcher branches on sbt_idx, so
// sbt 0 lanes ACCEPT (HIT) and sbt 1 lanes IGNORE (MISS). Reformation
// must emit exactly TWO CB_YIELDs per warp — one per sbt — visible in
// the debug=3 log.

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

const char* kernel_file  = "kernel.vxbin";
uint32_t num_lanes       = 4;
uint32_t sbt_group_size  = 2;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU reformation divergent-SBT smoke (Phase 3-A2 option B)." << std::endl;
  std::cout << "Usage: [-k kernel] [-n lanes] [-g sbt_group_size] [-h]" << std::endl;
  std::cout << "  -n  lanes in the single warp (default 4)" << std::endl;
  std::cout << "  -g  lanes per sbt group; sbt = tid/g  (default 2)" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:g:k:h")) != -1) {
    switch (c) {
    case 'n': num_lanes      = atoi(optarg); break;
    case 'g': sbt_group_size = atoi(optarg); break;
    case 'k': kernel_file    = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (sbt_group_size == 0) sbt_group_size = 1;
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

  // Build num_lanes per-lane scenes; each fits in one RTU_SCENE_BYTES line.
  std::vector<uint8_t> scene_bytes(num_lanes * RTU_SCENE_BYTES, 0);
  for (uint32_t i = 0; i < num_lanes; ++i) {
    uint8_t* base = scene_bytes.data() + i * RTU_SCENE_BYTES;
    uint32_t* hdr = reinterpret_cast<uint32_t*>(base);
    hdr[0] = 1;  // triangle_count
    float* tris = reinterpret_cast<float*>(base + RTU_SCENE_HDR_BYTES);
    tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f;
    tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f;
    tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f;
    uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
        base + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
    uint32_t sbt = i / sbt_group_size;
    *tri_flags = (sbt & RTU_TRI_SBT_MASK) << RTU_TRI_SBT_SHIFT;  // OPAQUE bit clear
  }

  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_base_addr));

  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.num_lanes        = num_lanes;
  kernel_arg.sbt_group_size   = sbt_group_size;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  uint32_t num_sbts = (num_lanes + sbt_group_size - 1) / sbt_group_size;
  std::cout << "scene_base=0x" << std::hex << kernel_arg.scene_base_addr << std::dec
            << ", num_lanes=" << num_lanes
            << ", sbt_group_size=" << sbt_group_size
            << " -> " << num_sbts << " sbts"
            << " (expect 1 CB_YIELD per sbt, grouped by sbt_idx)"
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
    li.block_dim[0] = num_lanes;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::vector<rtu_result_t> results(num_lanes);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Oracle: dispatcher ACCEPTs iff sbt==0, IGNOREs otherwise.
  float exp_t_hit = 5.f;
  std::cout << "oracle: sbt 0 lanes HIT t=" << exp_t_hit
            << "; sbt!=0 lanes MISS" << std::endl;

  int errors = 0;
  for (uint32_t i = 0; i < num_lanes; ++i) {
    uint32_t sbt   = i / sbt_group_size;
    bool exp_hit   = (sbt == 0);
    uint32_t exp_s = exp_hit ? VX_RT_STS_DONE_HIT : VX_RT_STS_DONE_MISS;
    bool sts_ok = (results[i].status == exp_s);
    bool t_ok   = exp_hit
                    ? std::fabs(results[i].hit_t - exp_t_hit) < 1e-4f
                    : true;
    if (!sts_ok || !t_ok) {
      std::cout << "lane " << i << " (sbt=" << sbt << "): "
                << "status=" << results[i].status
                << " hit_t=" << results[i].hit_t
                << " (expected " << (exp_hit ? "HIT" : "MISS")
                << ", status=" << exp_s << ")" << std::endl;
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
