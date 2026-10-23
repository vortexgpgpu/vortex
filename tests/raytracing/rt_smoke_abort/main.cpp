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
// PRISM RTU verdict smoke host driver (see common.h).

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
vx_buffer_h scene_buffer[2] = {};
vx_buffer_h case_buffer  = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

void cleanup() {
  if (device) {
    for (auto b : scene_buffer) {
      if (b) {
        vx_buffer_release(b);
      }
    }
    if (case_buffer) {
      vx_buffer_release(case_buffer);
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

struct expect_t {
  uint32_t num_cand;
  float    first_cand_t;
  uint32_t status;
  float    hit_t;
  bool     by_shader;
};

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const std::vector<uint8_t> scenes[2] = {
    make_scene({ {2.f, false}, {3.f, false}, {4.f, true} }),
    make_scene({ {2.f, false}, {3.f, false} }),
  };
  for (int i = 0; i < 2; ++i) {
    RT_CHECK(vx_buffer_create(device, (uint32_t)scenes[i].size(), VX_MEM_READ, &scene_buffer[i]));
    RT_CHECK(vx_buffer_address(scene_buffer[i], &kernel_arg.scene_addr[i]));
    RT_CHECK(vx_enqueue_write(queue, scene_buffer[i], 0, scenes[i].data(),
                              (uint32_t)scenes[i].size(), 0, nullptr, nullptr));
  }

  const uint32_t I = VX_RT_CB_IGNORE, A = VX_RT_CB_ACCEPT,
                 T = VX_RT_CB_TERMINATE, X = VX_RT_CB_ABORT;
  const rtu_case_t cases[] = {
    { 0, { X, I, I, I } },
    { 0, { A, I, I, I } },
    { 0, { I, X, I, I } },
    { 0, { T, I, I, I } },
    { 0, { I, I, I, I } },
    { 1, { X, I, I, I } },
    { 1, { I, A, I, I } },
  };
  // Lane 0's expected outcome per case: the walk commits the opaque t = 4 in
  // scene A before offering the nearer non-opaque candidates, nearest first.
  const expect_t expect[] = {
    { 1, 2.f, VX_RT_STS_DONE_HIT,  4.f, false },  // ABORT keeps the opaque hit
    { 1, 2.f, VX_RT_STS_DONE_HIT,  2.f, true  },  // t = 3 lies behind the accept
    { 2, 2.f, VX_RT_STS_DONE_HIT,  4.f, false },
    { 1, 2.f, VX_RT_STS_DONE_HIT,  2.f, true  },  // TERMINATE accepts
    { 2, 2.f, VX_RT_STS_DONE_HIT,  4.f, false },
    { 1, 2.f, VX_RT_STS_DONE_MISS, 0.f, false },
    { 2, 2.f, VX_RT_STS_DONE_HIT,  3.f, true  },
  };
  const uint32_t num_cases = sizeof(cases) / sizeof(cases[0]);

  RT_CHECK(vx_buffer_create(device, sizeof(cases), VX_MEM_READ, &case_buffer));
  RT_CHECK(vx_buffer_address(case_buffer, &kernel_arg.cases_addr));
  RT_CHECK(vx_enqueue_write(queue, case_buffer, 0, cases, sizeof(cases), 0, nullptr, nullptr));

  const uint32_t res_size = num_cases * RTU_NUM_LANES * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.num_cases = num_cases;
  const float origins[RTU_NUM_LANES][3] = { { 0.25f, 0.25f, 0.f }, { 5.f, 5.f, 0.f } };
  memcpy(kernel_arg.ray_origin, origins, sizeof(origins));

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
    li.block_dim[0] = RTU_NUM_LANES;   // both lanes in one warp, one trace
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  std::vector<rtu_result_t> res(num_cases * RTU_NUM_LANES);
  RT_CHECK(vx_enqueue_read(queue, res.data(), res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_device_dump_perf(device, stdout);

  int errors = 0;
  for (uint32_t c = 0; c < num_cases; ++c) {
    const rtu_result_t& r0 = res[c * RTU_NUM_LANES + 0];
    const rtu_result_t& r1 = res[c * RTU_NUM_LANES + 1];
    const expect_t& e = expect[c];
    bool ok = r0.first_status == VX_RT_STS_YIELD_ANYHIT
           && r0.num_cand == e.num_cand
           && std::fabs(r0.first_cand_t - e.first_cand_t) < 1e-5f
           && r0.status == e.status;
    if (e.status == VX_RT_STS_DONE_HIT) {
      ok = ok && std::fabs(r0.hit_t - e.hit_t) < 1e-5f
              && (((r0.geom & VX_RT_HIT_COMMITTED_CB) != 0) == e.by_shader)
              && (r0.geom & (VX_RT_HIT_COMMITTED_PROC | VX_RT_HIT_OPAQUE)) == 0;
    }
    // Lane 1 has nothing to decide: its terminal status arrives with the first
    // record, which is lane 0's candidate.
    bool ok1 = r1.first_status == VX_RT_STS_DONE_MISS && r1.num_cand == 0
            && r1.status == VX_RT_STS_DONE_MISS;
    if (!ok || !ok1) {
      std::cout << "case " << c << ": lane0 first=" << r0.first_status << " n=" << r0.num_cand
                << " t0=" << r0.first_cand_t << " status=" << r0.status << " t=" << r0.hit_t
                << " geom=0x" << std::hex << r0.geom << std::dec
                << " | lane1 first=" << r1.first_status << " n=" << r1.num_cand
                << " status=" << r1.status << std::endl;
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
