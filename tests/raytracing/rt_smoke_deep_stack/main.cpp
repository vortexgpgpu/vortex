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
// PRISM RTU short-stack-overflow smoke — host driver.
//
// Builds a CW-BVH4 over N triangles stacked in depth along the ray, so the
// tree is several levels deep — deeper than the modest short stack the Makefile
// configures (VX_CFG_RTU_STACK_DEPTH), so the walker overflows its stack, drops
// subtrees and restarts. Two scenes: every triangle hit (the closest is on the
// first path walked), and decoys whose boxes the ray enters first while only a
// far triangle is hit (reachable only through restarts).

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <cstring>

#include <vortex2.h>
#include <VX_types.h>
#include <raytrace.h>
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

int main(int /*argc*/, char* /*argv*/[]) {
  using namespace vortex::raytrace;

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  int errors = 0;

  // One trace of the +z ray at (0.25, 0.25) against a CW-BVH4 over `tris`;
  // the walk must return the closest hit (t, prim).
  auto run_case = [&](const char* name, const std::vector<host_tri_t>& tris,
                      float exp_t, uint32_t exp_prim) {
    host_bvh_t src = { tris.data(), (uint32_t)tris.size(), /*geometry_index*/ 0 };
    std::vector<uint8_t> scene;
    uint64_t root_offset = 0;
    if (!build_bvh_scene<4>(src, scene, root_offset)) {
      std::cout << name << ": build_bvh_scene failed" << std::endl;
      ++errors;
      return;
    }
    std::cout << name << ": scene " << scene.size() << " B, " << tris.size()
              << " tris (deep CW-BVH4)" << std::endl;

    RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
    RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));
    uint32_t res_size = sizeof(rtu_result_t);
    RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
    RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

    kernel_arg.ray_origin[0]    = 0.25f;
    kernel_arg.ray_origin[1]    = 0.25f;
    kernel_arg.ray_origin[2]    = 0.0f;
    kernel_arg.ray_direction[0] = 0.0f;
    kernel_arg.ray_direction[1] = 0.0f;
    kernel_arg.ray_direction[2] = 1.0f;
    kernel_arg.tmin             = 0.001f;
    kernel_arg.tmax             = 1e30f;

    RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene.data(),
                              (uint32_t)scene.size(), 0, nullptr, nullptr));
    if (!kernel) {
      RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
      RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));
    }

    vx_event_h launch_ev = nullptr, read_ev = nullptr;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = 1;
    li.block_dim[0] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

    rtu_result_t result = {};
    RT_CHECK(vx_enqueue_read(queue, &result, res_buffer, 0, res_size,
                             1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);
    vx_event_release(launch_ev);
    vx_buffer_release(scene_buffer); scene_buffer = nullptr;
    vx_buffer_release(res_buffer);   res_buffer = nullptr;

    std::cout << name << ": oracle HIT t=" << exp_t << " prim=" << exp_prim << std::endl;
    if (result.status != VX_RT_STS_DONE_HIT || std::fabs(result.hit_t - exp_t) >= 1e-4f
     || result.primitive_id != exp_prim) {
      std::cout << name << ": result status=" << result.status
                << " hit_t=" << result.hit_t
                << " prim=" << result.primitive_id << std::endl;
      ++errors;
    }
  };

  auto tri_at = [](float z, float x0) {
    host_tri_t t = {};
    t.v0[0] = x0;  t.v0[1] = 0.f; t.v0[2] = z;
    t.v1[0] = 1.f; t.v1[1] = 0.f; t.v1[2] = z;
    t.v2[0] = 0.f; t.v2[1] = 1.f; t.v2[2] = z;
    t.flags = RTU_BVH_FLAG_OPAQUE;
    return t;
  };

  // N opaque triangles all covering the ray's (x,y) footprint, stacked at
  // z = 5, 6, ... The SAH builder splits them into a deep tree. The ray hits
  // all of them; triangle 0 (z=5) is the closest, on the walk's first path.
  constexpr uint32_t N = 64;
  std::vector<host_tri_t> tris;
  for (uint32_t i = 0; i < N; ++i) tris.push_back(tri_at(5.0f + (float)i, 0.f));
  run_case("stacked", tris, 5.f, 0);

  // The same stack, but only one triangle, deep in the far half, covers the
  // ray: the others' boxes do (so they are walked first) while the triangles
  // miss it. The hit sits in subtrees the short stack drops, so the walk has
  // to restart, repeatedly, to reach it.
  constexpr uint32_t kHit = 41;
  tris.clear();
  for (uint32_t i = 0; i < N; ++i)
    tris.push_back(tri_at(5.0f + (float)i, (i == kHit) ? 0.f : 0.6f));
  run_case("decoys", tris, 5.f + (float)kHit, kHit);

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
