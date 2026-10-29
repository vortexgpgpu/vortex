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
// RTU partial-warp trace smoke -- host driver. One warp: every lane traces an
// empty CW-BVH4 scene (a miss), then the last lane alone traces a scene holding
// one opaque triangle at t=5 (a hit).

#include <iostream>
#include <unistd.h>
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
    if (scene_buffer) vx_buffer_release(scene_buffer);
    if (res_buffer)   vx_buffer_release(res_buffer);
    if (kernel)       vx_kernel_release(kernel);
    if (module_)      vx_module_release(module_);
    if (queue)        vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    if (c == 'k') kernel_file = optarg;
    else { std::cout << "Usage: [-k kernel] [-h]" << std::endl; return c == 'h' ? 0 : -1; }
  }

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  const uint32_t num_lanes = (uint32_t)num_threads;   // exactly one warp

  // Two scenes in one buffer, 64-B aligned: the empty one (a leaf of zero
  // triangles) at 0, the one-triangle one at 128.
  const uint32_t kTriOff = 128;
  std::vector<uint8_t> bytes(kTriOff + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES
                             + VX_BVH_TRI_STRIDE, 0);
  for (uint32_t s = 0; s < 2; ++s) {
    uint8_t* base = bytes.data() + (s ? kTriOff : 0);
    uint32_t ntri = s;   // empty, then one triangle
    uint32_t sh[4] = {VX_BVH_SCENE_HDR_BYTES, VX_BVH_SCENE_KIND,
                      VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES + ntri * VX_BVH_TRI_STRIDE, 1};
    memcpy(base, sh, sizeof(sh));
    uint32_t lh[4] = {VX_BVH_KIND_LEAF_TRI | (ntri << VX_BVH_COUNT_SHIFT), 0, 0, 0};
    memcpy(base + VX_BVH_SCENE_HDR_BYTES, lh, sizeof(lh));
    if (ntri) {
      uint8_t* rec = base + VX_BVH_SCENE_HDR_BYTES + VX_BVH_LEAF_HDR_BYTES;
      float v[9] = {0.f, 0.f, 5.f, 1.f, 0.f, 5.f, 0.f, 1.f, 5.f};
      uint32_t fl = VX_BVH_TRI_FLAG_OPAQUE;
      memcpy(rec, v, sizeof(v));
      memcpy(rec + VX_BVH_TRI_FLAGS_OFFSET, &fl, sizeof(fl));
    }
  }

  uint32_t res_size = num_lanes * sizeof(rtu_result_t);
  uint64_t scene_addr = 0;
  RT_CHECK(vx_buffer_create(device, bytes.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &scene_addr));
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_READ_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));
  kernel_arg.empty_scene_addr = scene_addr;
  kernel_arg.tri_scene_addr   = scene_addr + kTriOff;
  kernel_arg.num_lanes        = num_lanes;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, bytes.data(), bytes.size(), 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "one warp of " << num_lanes << " lanes; last lane traces alone" << std::endl;

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

  int errors = 0;
  for (uint32_t i = 0; i < num_lanes; ++i) {
    const rtu_result_t& r = results[i];
    bool last = (i == num_lanes - 1);
    bool ok = (r.first_status == VX_RT_STS_DONE_MISS)
           && (last ? (r.second_status == VX_RT_STS_DONE_HIT && std::fabs(r.second_t - 5.f) < 1e-4f)
                    : (r.second_status == 0xffffffffu));
    if (!ok) {
      std::cout << "lane " << i << ": first=" << r.first_status << " second=" << r.second_status
                << " t=" << r.second_t << std::endl;
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
