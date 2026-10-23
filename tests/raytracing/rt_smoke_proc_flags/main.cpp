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
// PRISM RTU procedural-opacity and provenance smoke host driver (see common.h).

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
vx_buffer_h scene_buffer[RTU_NUM_SCENES] = {};
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

static std::vector<uint8_t> scene_header(size_t payload) {
  std::vector<uint8_t> s(VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES + payload, 0);
  uint32_t* sh = reinterpret_cast<uint32_t*>(s.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;
  sh[1] = VX_BVH_SCENE_KIND;
  sh[2] = (uint32_t)s.size();
  sh[3] = 1;
  return s;
}

// One procedural box around the ray's z in [3, 7]; entry t = 3.
static std::vector<uint8_t> box_scene(bool opaque) {
  auto s = scene_header(VX_BVH_PROC_AABB_BYTES);
  uint32_t* lh = reinterpret_cast<uint32_t*>(s.data() + VX_BVH_SCENE_HDR_BYTES);
  lh[0] = VX_BVH_KIND_LEAF_PROC | (1u << VX_BVH_COUNT_SHIFT);
  lh[2] = opaque ? VX_BVH_FLAG_OPAQUE : 0u;
  float* b = reinterpret_cast<float*>(s.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES);
  b[0] = -1.f; b[1] = -1.f; b[2] = 3.f;
  b[3] =  1.f; b[4] =  1.f; b[5] = 7.f;
  return s;
}

// One opaque triangle at z = 5.
static std::vector<uint8_t> tri_scene() {
  auto s = scene_header(VX_BVH_TRI_STRIDE);
  uint32_t* lh = reinterpret_cast<uint32_t*>(s.data() + VX_BVH_SCENE_HDR_BYTES);
  lh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);
  uint8_t* rec = s.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES;
  float* v = reinterpret_cast<float*>(rec);
  v[0] = 0.f; v[1] = 0.f; v[2] = 5.f;
  v[3] = 1.f; v[4] = 0.f; v[5] = 5.f;
  v[6] = 0.f; v[7] = 1.f; v[8] = 5.f;
  uint32_t flags = VX_BVH_FLAG_OPAQUE;
  memcpy(rec + VX_BVH_TRI_FLAGS_OFFSET, &flags, 4);
  return s;
}

struct expect_t {
  uint32_t first_status;   // DONE_MISS when nothing yields
  bool     cand_opaque;
  uint32_t status;
  float    hit_t;
  uint32_t provenance;     // COMMITTED_CB | COMMITTED_PROC bits of a hit
};

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const std::vector<uint8_t> scenes[RTU_NUM_SCENES] = {
    box_scene(true), box_scene(false), tri_scene(),
  };
  for (int i = 0; i < RTU_NUM_SCENES; ++i) {
    RT_CHECK(vx_buffer_create(device, (uint32_t)scenes[i].size(), VX_MEM_READ, &scene_buffer[i]));
    RT_CHECK(vx_buffer_address(scene_buffer[i], &kernel_arg.scene_addr[i]));
    RT_CHECK(vx_enqueue_write(queue, scene_buffer[i], 0, scenes[i].data(),
                              (uint32_t)scenes[i].size(), 0, nullptr, nullptr));
  }

  const uint32_t ACC = VX_RT_CB_ACCEPT, IGN = VX_RT_CB_IGNORE;
  const uint32_t PROC = VX_RT_STS_YIELD_PROC, AHS = VX_RT_STS_YIELD_ANYHIT;
  const uint32_t HIT = VX_RT_STS_DONE_HIT, MISS = VX_RT_STS_DONE_MISS;
  const uint32_t CB = VX_RT_HIT_COMMITTED_CB, CBP = VX_RT_HIT_COMMITTED_CB | VX_RT_HIT_COMMITTED_PROC;
  const rtu_case_t cases[] = {
    { 0, 0,                             ACC },
    { 1, 0,                             ACC },
    { 0, VX_RT_FLAG_NO_OPAQUE,          IGN },
    { 1, VX_RT_FLAG_OPAQUE,             ACC },
    { 0, VX_RT_FLAG_CULL_OPAQUE,        ACC },
    { 1, VX_RT_FLAG_CULL_NO_OPAQUE,     ACC },
    { 0, VX_RT_FLAG_CULL_NO_OPAQUE,     ACC },
    { 1, VX_RT_FLAG_CULL_OPAQUE,        ACC },
    { 0, VX_RT_FLAG_SKIP_AABBS,         ACC },
    { 2, 0,                             ACC },
    { 2, VX_RT_FLAG_NO_OPAQUE,          ACC },
    { 2, VX_RT_FLAG_NO_OPAQUE,          IGN },
  };
  const expect_t expect[] = {
    { PROC, true,  HIT,  4.f, CBP },
    { PROC, false, HIT,  4.f, CBP },
    { PROC, false, MISS, 0.f, 0 },
    { PROC, true,  HIT,  4.f, CBP },
    { MISS, false, MISS, 0.f, 0 },    // the opaque box is culled
    { MISS, false, MISS, 0.f, 0 },    // the non-opaque box is culled
    { PROC, true,  HIT,  4.f, CBP },
    { PROC, false, HIT,  4.f, CBP },
    { MISS, false, MISS, 0.f, 0 },    // boxes skipped
    { HIT,  false, HIT,  5.f, 0 },    // the RTU commits an opaque triangle itself
    { AHS,  false, HIT,  5.f, CB },
    { AHS,  false, MISS, 0.f, 0 },
  };
  const uint32_t num_cases = sizeof(cases) / sizeof(cases[0]);

  RT_CHECK(vx_buffer_create(device, sizeof(cases), VX_MEM_READ, &case_buffer));
  RT_CHECK(vx_buffer_address(case_buffer, &kernel_arg.cases_addr));
  RT_CHECK(vx_enqueue_write(queue, case_buffer, 0, cases, sizeof(cases), 0, nullptr, nullptr));
  const uint32_t res_size = num_cases * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));
  kernel_arg.num_cases = num_cases;

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
  std::vector<rtu_result_t> res(num_cases);
  RT_CHECK(vx_enqueue_read(queue, res.data(), res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_device_dump_perf(device, stdout);

  const uint32_t prov_mask = VX_RT_HIT_COMMITTED_CB | VX_RT_HIT_COMMITTED_PROC | VX_RT_HIT_OPAQUE;
  int errors = 0;
  for (uint32_t c = 0; c < num_cases; ++c) {
    const rtu_result_t& r = res[c];
    const expect_t& e = expect[c];
    const bool yields = e.first_status == PROC || e.first_status == AHS;
    bool ok = r.first_status == e.first_status
           && r.num_cand == (yields ? 1u : 0u)
           && r.status == e.status;
    if (yields) {
      ok = ok && ((r.cand_geom & VX_RT_HIT_OPAQUE) != 0) == e.cand_opaque
              && (r.cand_geom & (VX_RT_HIT_COMMITTED_CB | VX_RT_HIT_COMMITTED_PROC)) == 0;
    }
    if (e.status == HIT) {
      ok = ok && std::fabs(r.hit_t - e.hit_t) < 1e-5f && (r.geom & prov_mask) == e.provenance;
    }
    if (!ok) {
      std::cout << "case " << c << ": first=" << r.first_status << " cand_geom=0x" << std::hex
                << r.cand_geom << std::dec << " n=" << r.num_cand << " status=" << r.status
                << " t=" << r.hit_t << " geom=0x" << std::hex << r.geom << std::dec << std::endl;
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
