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
// PRISM RTU smoke test — Phase 1 host driver.
//
// Sets up a single-triangle scene in device memory, launches a 1-warp
// kernel where each lane fires a primary ray, reads back per-lane hit
// results, and validates against a CPU oracle.

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
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU smoke test." << std::endl;
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
    if (scene_buffer) vx_buffer_release(scene_buffer);
    if (res_buffer)   vx_buffer_release(res_buffer);
    if (kernel)       vx_kernel_release(kernel);
    if (module_)      vx_module_release(module_);
    if (queue)        vx_queue_release(queue);
    vx_device_release(device);
  }
}

// Möller-Trumbore ray-triangle intersection (CPU oracle).
static bool ray_tri_oracle(const float ro[3], const float rd[3],
                           const float v0[3], const float v1[3], const float v2[3],
                           float tmin, float tmax,
                           float& out_t, float& out_u, float& out_v) {
  float e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
  float e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };
  float P[3]  = { rd[1]*e2[2] - rd[2]*e2[1],
                  rd[2]*e2[0] - rd[0]*e2[2],
                  rd[0]*e2[1] - rd[1]*e2[0] };
  float det = e1[0]*P[0] + e1[1]*P[1] + e1[2]*P[2];
  if (std::fabs(det) < 1e-6f) return false;
  float invDet = 1.0f / det;
  float T[3] = { ro[0]-v0[0], ro[1]-v0[1], ro[2]-v0[2] };
  float u = (T[0]*P[0] + T[1]*P[1] + T[2]*P[2]) * invDet;
  if (u < 0.f || u > 1.f) return false;
  float Q[3] = { T[1]*e1[2] - T[2]*e1[1],
                 T[2]*e1[0] - T[0]*e1[2],
                 T[0]*e1[1] - T[1]*e1[0] };
  float v = (rd[0]*Q[0] + rd[1]*Q[1] + rd[2]*Q[2]) * invDet;
  if (v < 0.f || u + v > 1.f) return false;
  float t = (e2[0]*Q[0] + e2[1]*Q[1] + e2[2]*Q[2]) * invDet;
  if (t < tmin || t > tmax) return false;
  out_t = t; out_u = u; out_v = v;
  return true;
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  // Open device + create queue.
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Build the single-triangle scene in host memory.
  // Triangle vertices in world space:
  //   v0 = ( 0, 0, 5)
  //   v1 = ( 1, 0, 5)
  //   v2 = ( 0, 1, 5)
  // Ray from origin (0.25, 0.25, 0) shooting +z → hits triangle at t=5.
  // Phase 1 smoke marks the triangle OPAQUE so RtuCore commits the hit
  // immediately — no AHS callback path.
  std::vector<uint8_t> scene_bytes(64, 0);   // one cache line
  uint32_t* hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  hdr[0] = 1;                                  // triangle_count
  float* tris = reinterpret_cast<float*>(scene_bytes.data() + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 5.f; // v0
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 5.f; // v1
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 5.f; // v2
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = RTU_TRI_FLAG_OPAQUE;

  // Allocate device scene buffer and upload.
  uint32_t scene_bytes_sz = (uint32_t)scene_bytes.size();
  RT_CHECK(vx_buffer_create(device, scene_bytes_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  // Allocate result buffer.
  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  // Configure ray descriptor — all lanes shoot the same ray.
  kernel_arg.num_lanes        = num_lanes;
  kernel_arg.ray_pattern      = 0;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << ", num_lanes=" << num_lanes << std::endl;

  // Upload scene.
  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene_bytes.data(),
                            scene_bytes_sz, 0, nullptr, nullptr));

  // Load kernel.
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Launch — grid_dim.x = num_lanes, block_dim.x = 1.
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

  // Read back results.
  std::vector<rtu_result_t> results(num_lanes);
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // Compute CPU oracle: each lane shoots the same ray.
  float exp_t = 0.f, exp_u = 0.f, exp_v = 0.f;
  bool exp_hit = ray_tri_oracle(kernel_arg.ray_origin,
                                kernel_arg.ray_direction,
                                &tris[0], &tris[3], &tris[6],
                                kernel_arg.tmin, kernel_arg.tmax,
                                exp_t, exp_u, exp_v);
  uint32_t exp_status = exp_hit ? VX_RT_STS_DONE_HIT : VX_RT_STS_DONE_MISS;
  std::cout << "oracle: " << (exp_hit ? "HIT" : "MISS")
            << " t=" << exp_t << " u=" << exp_u << " v=" << exp_v << std::endl;

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
