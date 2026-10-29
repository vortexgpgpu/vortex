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
// RTU multi-candidate any-hit smoke -- host driver.
//
// One CW-BVH4 leaf, six triangles stacked along the ray (+z from z=0):
//   prim 0  t=7  non-opaque
//   prim 1  t=3  non-opaque
//   prim 2  t=5  non-opaque
//   prim 3  t=5  non-opaque   (coplanar with prim 2: a t tie)
//   prim 4  t=8  OPAQUE
//   prim 5  t=9  non-opaque   (behind the opaque one: never offered)
// A lane that accepts nothing is offered 1, 2, 3, 0 -- ascending t, the tie
// broken by record order -- and ends on the opaque prim 4. A lane that accepts
// prim P is offered the prefix of that sequence up to P and ends on P.

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
uint32_t num_lanes = 12;

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_buffer_h tgt_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU multi-candidate any-hit smoke test." << std::endl;
  std::cout << "Usage: [-k kernel] [-n lanes] [-h]" << std::endl;
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
    if (tgt_buffer)   vx_buffer_release(tgt_buffer);
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

  static const float    tri_t[RTU_MULTI_NUM_TRIS]  = {7.f, 3.f, 5.f, 5.f, 8.f, 9.f};
  static const uint32_t tri_fl[RTU_MULTI_NUM_TRIS] = {0, 0, 0, 0, VX_BVH_TRI_FLAG_OPAQUE, 0};

  std::vector<uint8_t> scene_bytes(VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                                   + RTU_MULTI_NUM_TRIS * VX_BVH_TRI_STRIDE, 0);
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene_bytes.data());
  sh[0] = VX_BVH_SCENE_HDR_BYTES;          // root_node_offset
  sh[1] = VX_BVH_SCENE_KIND;
  sh[2] = (uint32_t)scene_bytes.size();
  sh[3] = 1;                               // leaf_count
  uint32_t* lh = reinterpret_cast<uint32_t*>(scene_bytes.data() + VX_BVH_SCENE_HDR_BYTES);
  lh[0] = VX_BVH_KIND_LEAF_TRI | (RTU_MULTI_NUM_TRIS << VX_BVH_COUNT_SHIFT);
  for (uint32_t i = 0; i < RTU_MULTI_NUM_TRIS; ++i) {
    uint8_t* rec = scene_bytes.data() + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                 + i * VX_BVH_TRI_STRIDE;
    float v[9] = {0.f, 0.f, tri_t[i], 1.f, 0.f, tri_t[i], 0.f, 1.f, tri_t[i]};
    memcpy(rec, v, sizeof(v));
    memcpy(rec + VX_BVH_TRI_FLAGS_OFFSET, &tri_fl[i], sizeof(uint32_t));
  }

  // lane targets cycle through: none, then each primitive in turn
  static const uint8_t target_cycle[] = {RTU_MULTI_NONE, 1, 2, 3, 0, 5};
  const uint32_t ncycle = sizeof(target_cycle) / sizeof(target_cycle[0]);
  std::vector<uint8_t> targets(num_lanes);
  for (uint32_t i = 0; i < num_lanes; ++i)
    targets[i] = target_cycle[i % ncycle];

  uint32_t scene_sz = (uint32_t)scene_bytes.size();
  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, scene_sz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_READ_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));
  RT_CHECK(vx_buffer_create(device, num_lanes, VX_MEM_READ, &tgt_buffer));
  RT_CHECK(vx_buffer_address(tgt_buffer, &kernel_arg.targets_addr));

  kernel_arg.num_lanes        = num_lanes;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::vector<rtu_result_t> zero(num_lanes);
  memset(zero.data(), 0, res_size);
  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene_bytes.data(), scene_sz, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, tgt_buffer, 0, targets.data(), num_lanes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, res_buffer, 0, zero.data(), res_size, 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "bvh4: 1 leaf, " << RTU_MULTI_NUM_TRIS << " tris, lanes=" << num_lanes << std::endl;

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
  RT_CHECK(vx_enqueue_read(queue, results.data(), res_buffer, 0, res_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // oracle: the offer order when every candidate is ignored
  static const uint8_t order[] = {1, 2, 3, 0};
  const uint32_t norder = sizeof(order);

  int errors = 0;
  for (uint32_t i = 0; i < num_lanes; ++i) {
    uint32_t tgt = targets[i];
    uint32_t exp_n = norder, exp_prim = 4;
    for (uint32_t k = 0; k < norder; ++k) {
      if (order[k] == tgt) { exp_n = k + 1; exp_prim = tgt; break; }
    }
    float exp_t = tri_t[exp_prim];
    const rtu_result_t& r = results[i];
    bool ok = (r.status == VX_RT_STS_DONE_HIT)
           && (r.primitive_id == exp_prim)
           && (std::fabs(r.hit_t - exp_t) < 1e-4f)
           && (r.num_offered == exp_n);
    for (uint32_t k = 0; ok && k < exp_n; ++k)
      ok = (r.offered[k] == order[k]);
    if (!ok) {
      std::cout << "lane " << i << " (accepts " << tgt << "): status=" << r.status
                << " prim=" << r.primitive_id << " t=" << r.hit_t << " offered=[";
      for (uint32_t k = 0; k < r.num_offered && k < RTU_MULTI_MAX_OFFER; ++k)
        std::cout << (k ? "," : "") << uint32_t(r.offered[k]);
      std::cout << "] expected prim=" << exp_prim << " t=" << exp_t
                << " offered " << exp_n << std::endl;
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
