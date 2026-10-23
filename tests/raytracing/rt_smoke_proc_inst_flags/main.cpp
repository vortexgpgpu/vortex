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
// PRISM RTU smoke host driver: instance opacity flags on procedural geometry
// and a box entered at the committed distance (see common.h).

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

struct inst_t {
  float    tx, tz;
  bool     opaque_blas;
  uint32_t flags;
  uint32_t id;
};

// Lane 0 at x = -6, lane 1 at x = -2, lane 2 at x = +2 (two instances).
static const inst_t kInsts[] = {
  { -6.f, 0.f, false, RTU_INST_FORCE_OPAQUE,    10 },
  { -2.f, 0.f, true,  RTU_INST_FORCE_NO_OPAQUE, 11 },
  {  2.f, 0.f, false, 0u,                       12 },
  {  2.f, 1.f, false, 0u,                       13 },
};
static const float kRayX[RTU_NUM_LANES] = { -6.f, -2.f, 2.f };
constexpr uint32_t kNumInsts = sizeof(kInsts) / sizeof(kInsts[0]);

// Identity rotation and a translation; the record holds its inverse
// (world -> object).
static void emit_instance(uint8_t* out, const inst_t& in, uint32_t blas_off) {
  float* x = reinterpret_cast<float*>(out);
  x[0] = 1.f; x[1] = 0.f; x[2]  = 0.f; x[3]  = -in.tx;
  x[4] = 0.f; x[5] = 1.f; x[6]  = 0.f; x[7]  = 0.f;
  x[8] = 0.f; x[9] = 0.f; x[10] = 1.f; x[11] = -in.tz;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_BLAS_OFF)  = blas_off;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CUSTOM_ID) = 0u;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_ID_OFFSET) = in.id;
  *reinterpret_cast<uint32_t*>(out + VX_BVH_INSTANCE_CULL_MASK) =
      0xffu | (in.flags << RTU_INST_FLAGS_SHIFT);
}

// One procedural box over z in [3, 7] (entry t = 3 from the origin).
static void emit_box_blas(uint8_t* out, bool opaque) {
  uint32_t* lh = reinterpret_cast<uint32_t*>(out);
  lh[0] = VX_BVH_KIND_LEAF_PROC | (1u << VX_BVH_COUNT_SHIFT);
  lh[2] = opaque ? VX_BVH_FLAG_OPAQUE : 0u;
  float* b = reinterpret_cast<float*>(out + VX_BVH_LEAF_HDR_BYTES);
  b[0] = -1.f; b[1] = -1.f; b[2] = 3.f;
  b[3] =  1.f; b[4] =  1.f; b[5] = 7.f;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const uint32_t blas_size = VX_BVH_LEAF_HDR_BYTES + VX_BVH_PROC_AABB_BYTES;
  const uint32_t blas_n    = VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                           + kNumInsts * VX_BVH_INSTANCE_STRIDE;
  const uint32_t blas_o    = blas_n + blas_size;
  std::vector<uint8_t> scene(blas_o + blas_size, 0);

  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;     // root node offset
  sh[1] = VX_BVH_SCENE_KIND;
  sh[2] = (uint32_t)scene.size();
  sh[3] = 3;                          // the instance leaf and two box leaves

  uint32_t* rlh = reinterpret_cast<uint32_t*>(scene.data() + VX_BVH_SCENE_HDR_BYTES);
  rlh[0] = VX_BVH_KIND_LEAF_INST | (kNumInsts << VX_BVH_COUNT_SHIFT);
  for (uint32_t i = 0; i < kNumInsts; ++i) {
    emit_instance(scene.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                      + i * VX_BVH_INSTANCE_STRIDE,
                  kInsts[i], kInsts[i].opaque_blas ? blas_o : blas_n);
  }
  emit_box_blas(scene.data() + blas_n, false);
  emit_box_blas(scene.data() + blas_o, true);

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                            (uint32_t)scene.size(), 0, nullptr, nullptr));
  const uint32_t res_size = RTU_NUM_LANES * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));
  for (uint32_t l = 0; l < RTU_NUM_LANES; ++l) {
    kernel_arg.ray_x[l] = kRayX[l];
  }

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
    li.block_dim[0] = RTU_NUM_LANES;   // every lane in one warp
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  rtu_result_t res[RTU_NUM_LANES] = {};
  RT_CHECK(vx_enqueue_read(queue, res, res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  vx_device_dump_perf(device, stdout);

  struct expect_t {
    bool     cand_opaque;
    uint32_t num_cand;
    uint32_t inst;
  };
  const expect_t expect[RTU_NUM_LANES] = {
    { true,  1, 10 },   // FORCE_OPAQUE makes the non-opaque box opaque
    { false, 1, 11 },   // FORCE_NO_OPAQUE makes the opaque box non-opaque
    { false, 2, 13 },   // the box entered at the committed t is offered and wins
  };
  const uint32_t prov = VX_RT_HIT_COMMITTED_CB | VX_RT_HIT_COMMITTED_PROC;
  int errors = 0;
  for (uint32_t l = 0; l < RTU_NUM_LANES; ++l) {
    const rtu_result_t& r = res[l];
    const expect_t& e = expect[l];
    bool ok = r.first_status == VX_RT_STS_YIELD_PROC
           && ((r.cand_geom & VX_RT_HIT_OPAQUE) != 0) == e.cand_opaque
           && r.num_cand == e.num_cand
           && r.status == VX_RT_STS_DONE_HIT
           && std::fabs(r.hit_t - RTU_IS_T) < 1e-5f
           && (r.geom & (prov | VX_RT_HIT_OPAQUE)) == prov
           && r.inst == e.inst;
    if (!ok) {
      std::cout << "lane " << l << ": first=" << r.first_status << " cand_geom=0x" << std::hex
                << r.cand_geom << std::dec << " n=" << r.num_cand << " status=" << r.status
                << " t=" << r.hit_t << " geom=0x" << std::hex << r.geom << std::dec
                << " inst=" << r.inst << std::endl;
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
