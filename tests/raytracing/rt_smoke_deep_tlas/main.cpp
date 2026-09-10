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
// PRISM RTU deep multi-instance TLAS smoke — host driver.
//
// A TLAS with NUM_INST instances, each referencing the SAME deep BLAS (a stack
// of triangles that builds a multi-level CW-BVH4 deeper than the short stack the
// Makefile configures). The instances are translated along +z so instance i
// lives at world z = base + (NUM_INST-1-i)*SPAN — i.e. instance 0 is the
// FARTHEST and the LAST instance is the CLOSEST. A +z ray hits every instance's
// every triangle; the true closest hit is the last instance's nearest triangle.
//
// Each BLAS descent overflows the short stack and must restart. With a
// single per-ray-global restart budget the deep FIRST instance would exhaust it,
// so the closest (LAST) instance would drop subtrees and return a FARTHER hit.
// The restart budget is reset on every instance/BLAS entry, so every
// instance gets a full budget and the walker returns the true closest hit —
// matching the SimX oracle (unbounded stack). Runs on simx AND rtlsim; both must
// agree on (t, prim, instance).

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

static const uint32_t kNumInst  = 4;    // TLAS instances
static const uint32_t kBlasTris = 24;   // triangles per BLAS (deep tree)
static const float    kSpan     = 40.f; // world-z separation between instances

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

  // One deep BLAS: kBlasTris opaque triangles stacked in object-space z = 5..5+N
  // so build_bvh_scene emits a multi-level CW-BVH4.
  std::vector<host_tri_t> tris(kBlasTris);
  for (uint32_t i = 0; i < kBlasTris; ++i) {
    float z = 5.0f + (float)i;
    tris[i].v0[0] = 0.f; tris[i].v0[1] = 0.f; tris[i].v0[2] = z;
    tris[i].v1[0] = 1.f; tris[i].v1[1] = 0.f; tris[i].v1[2] = z;
    tris[i].v2[0] = 0.f; tris[i].v2[1] = 1.f; tris[i].v2[2] = z;
    tris[i].flags = RTU_BVH_FLAG_OPAQUE;
  }
  host_bvh_t blas = { tris.data(), kBlasTris, /*geometry_index*/ 0 };

  // kNumInst instances of that BLAS, each translated in +z. Instance i sits at
  // world z += (kNumInst-1-i)*kSpan, so instance 0 is FARTHEST and the last
  // instance is CLOSEST (its nearest triangle at world z=5 is the global hit).
  std::vector<host_instance_t> insts(kNumInst);
  for (uint32_t i = 0; i < kNumInst; ++i) {
    host_instance_t& in = insts[i];
    std::memset(&in, 0, sizeof(in));
    in.xform[0]=1.f; in.xform[5]=1.f; in.xform[10]=1.f;   // identity 3x3
    in.xform[11] = (float)(kNumInst - 1 - i) * kSpan;     // z translation
    in.blas_index  = 0;
    in.custom_id   = 0;
    in.instance_id = i;
    in.cull_mask   = 0xff;
    in.flags       = 0;
  }

  host_tlas_t tlas = { &blas, 1, insts.data(), kNumInst };
  std::vector<uint8_t> scene;
  uint64_t root_offset = 0;
  if (!build_tlas_scene<4>(tlas, scene, root_offset)) {
    std::cout << "build_tlas_scene failed" << std::endl;
    cleanup();
    return 1;
  }
  std::cout << "scene: " << scene.size() << " B, " << kNumInst
            << " instances x " << kBlasTris << "-tri deep BLAS" << std::endl;

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

  rtu_result_t result = {};
  RT_CHECK(vx_enqueue_read(queue, &result, res_buffer, 0, res_size,
                           1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t      = 5.f;             // world-z of the closest triangle
  const uint32_t exp_prim   = 0;               // nearest tri of the closest BLAS
  const uint32_t exp_inst   = kNumInst - 1;    // the CLOSEST (last) instance
  std::cout << "oracle: HIT t=" << exp_t << " prim=" << exp_prim
            << " instance=" << exp_inst << std::endl;
  std::cout << "result: status=" << result.status << " hit_t=" << result.hit_t
            << " prim=" << result.primitive_id
            << " instance=" << result.instance_id << std::endl;

  int errors = 0;
  if (result.status != exp_status)              { std::cout << "FAIL: status\n";   ++errors; }
  if (std::fabs(result.hit_t - exp_t) > 1e-4f)  { std::cout << "FAIL: hit_t\n";    ++errors; }
  if (result.primitive_id != exp_prim)          { std::cout << "FAIL: prim\n";     ++errors; }
  if (result.instance_id != exp_inst)           { std::cout << "FAIL: instance\n"; ++errors; }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
