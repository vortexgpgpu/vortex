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
// PRISM RTU smoke host driver: a lane that leaves its proceed loop holding a
// candidate (see common.h).

#include <iostream>
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
    if (scene_buffer) {
      vx_buffer_release(scene_buffer);
    }
    if (res_buffer) {
      vx_buffer_release(res_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_release(device);
  }
}

// One CW-BVH4 triangle leaf: a triangle per (z, opaque) entry, each covering
// the point (0.25, 0.25) in its z plane.
static std::vector<uint8_t> make_scene(const std::vector<std::pair<float, bool>>& tris) {
  std::vector<uint8_t> s(VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES +
                         tris.size() * VX_BVH_TRI_STRIDE, 0);
  uint32_t* sh = reinterpret_cast<uint32_t*>(s.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;
  sh[1] = VX_BVH_SCENE_KIND;
  sh[2] = (uint32_t)s.size();
  sh[3] = 1;
  uint32_t* lh = reinterpret_cast<uint32_t*>(s.data() + VX_BVH_SCENE_HDR_BYTES);
  lh[0] = VX_BVH_KIND_LEAF_TRI | ((uint32_t)tris.size() << VX_BVH_COUNT_SHIFT);
  for (size_t i = 0; i < tris.size(); ++i) {
    uint8_t* rec = s.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                 + i * VX_BVH_TRI_STRIDE;
    float* v = reinterpret_cast<float*>(rec);
    const float z = tris[i].first;
    v[0] = 0.f; v[1] = 0.f; v[2] = z;
    v[3] = 1.f; v[4] = 0.f; v[5] = z;
    v[6] = 0.f; v[7] = 1.f; v[8] = z;
    uint32_t flags = tris[i].second ? VX_BVH_TRI_FLAG_OPAQUE : 0u;
    memcpy(rec + VX_BVH_TRI_FLAGS_OFFSET, &flags, 4);
  }
  return s;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const std::vector<uint8_t> scene =
      make_scene({ {2.f, false}, {3.f, false}, {4.f, true} });
  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  const uint32_t res_size = RTU_NUM_LANES * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

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
    li.block_dim[0] = RTU_NUM_LANES;   // both lanes in one warp
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  rtu_result_t res[RTU_NUM_LANES] = {};
  RT_CHECK(vx_enqueue_read(queue, res, res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_device_dump_perf(device, stdout);

  struct expect_t {
    uint32_t num_cand;
    uint32_t exit_status;
  };
  const expect_t expect[RTU_NUM_LANES] = {
    { 2, VX_RT_STS_DONE_HIT },       // both candidates ignored, the opaque hit
    { 1, VX_RT_STS_YIELD_ANYHIT },   // left holding the first candidate
  };
  int errors = 0;
  for (uint32_t l = 0; l < RTU_NUM_LANES; ++l) {
    const rtu_result_t& r = res[l];
    const expect_t& e = expect[l];
    float live_t;
    memcpy(&live_t, &r.live_t, sizeof(live_t));
    bool ok = r.num_cand == e.num_cand
           && std::fabs(r.first_cand_t - 2.f) < 1e-5f
           && r.exit_status == e.exit_status
           && r.live_status == VX_RT_STS_DONE_HIT
           && std::fabs(live_t - 4.f) < 1e-5f;
    if (!ok) {
      std::cout << "lane " << l << ": n=" << r.num_cand << " first_t=" << r.first_cand_t
                << " exit=" << r.exit_status << " live=" << r.live_status
                << " live_t=" << live_t << std::endl;
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
