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
// PRISM RTU AHS-callback smoke — Phase 2 host driver.

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
uint32_t num_lanes    = 1;
uint32_t cb_decision  = RTU_AHS_DECISION_ACCEPT;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU AHS callback smoke test." << std::endl;
  std::cout << "Usage: [-k kernel] [-n lanes] [-d 0|1] [-h]" << std::endl;
  std::cout << "  -d 0  IGNORE the candidate hit -> expect MISS" << std::endl;
  std::cout << "  -d 1  ACCEPT the candidate hit -> expect HIT (default)" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:d:h")) != -1) {
    switch (c) {
    case 'n': num_lanes   = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'd': cb_decision = atoi(optarg) ? RTU_AHS_DECISION_ACCEPT
                                          : RTU_AHS_DECISION_IGNORE; break;
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

  // Single non-opaque triangle scene (flags=0). The same v0/v1/v2 as
  // rtu_smoke so the ray-tri math + oracle stay identical.
  std::vector<uint8_t> scene_bytes(64, 0);
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = 1;
  float* tris = reinterpret_cast<float*>(scene_bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = 0;  // NOT opaque -> AHS yield

  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.num_lanes        = num_lanes;
  kernel_arg.cb_decision      = cb_decision;
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
            << ", decision=" << (cb_decision == RTU_AHS_DECISION_ACCEPT ? "ACCEPT" : "IGNORE")
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

  // Oracle. The triangle is the same as rtu_smoke (intersects at t=5,
  // u=v=0.25). The callback decides the outcome:
  //   ACCEPT -> HIT with t=5
  //   IGNORE -> MISS (no other tris in the scene)
  bool exp_hit = (cb_decision == RTU_AHS_DECISION_ACCEPT);
  uint32_t exp_status = exp_hit ? VX_RT_STS_DONE_HIT : VX_RT_STS_DONE_MISS;
  float exp_t = exp_hit ? 5.f : 0.f;
  std::cout << "oracle: " << (exp_hit ? "HIT" : "MISS")
            << " t=" << exp_t << std::endl;

  int errors = 0;
  for (uint32_t i = 0; i < num_lanes; ++i) {
    bool sts_ok = (results[i].status == exp_status);
    bool t_ok   = !exp_hit ||
                  (std::fabs(results[i].hit_t - exp_t) < 1e-4f);
    if (!sts_ok || !t_ok) {
      std::cout << "lane " << i << ": status=" << results[i].status
                << " hit_t=" << results[i].hit_t
                << " (expected status=" << exp_status
                << " t=" << exp_t << ")" << std::endl;
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
