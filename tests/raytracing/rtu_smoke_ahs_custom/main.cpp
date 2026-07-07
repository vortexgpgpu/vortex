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
// PRISM RTU any-hit candidate instance-attribute smoke — I1 parity guard.
//
// Builds a single-instance CW-BVH4 TLAS (via build_tlas_scene) whose lone BLAS
// triangle is NON-opaque, so the walker yields an AHS *candidate* callback. The
// instance carries a distinctive custom index (0xABCD0007) and instance id (7).
// The kernel's AHS dispatcher reads gl_InstanceCustomIndexEXT / gl_InstanceID
// from the CANDIDATE register-window slots and captures them, then ACCEPTs.
//
// Oracle: the candidate stage must report custom=0xABCD0007, id=7 — the true
// instance attributes. Before the I1 fix the SimX CB_YIELD builder left those
// candidate slots at 0 (the RTL delivered the real values), a parity break.
// This test asserts the CANDIDATE read (not just the post-commit terminal read,
// which rtu_smoke_tlas_builder already covers).

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
vx_buffer_h cand_buffer  = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static const uint32_t kInstCustom = 0xABCD0007u;
static const uint32_t kInstId     = 7u;

void cleanup() {
  if (device) {
    if (scene_buffer) vx_buffer_release(scene_buffer);
    if (res_buffer)   vx_buffer_release(res_buffer);
    if (cand_buffer)  vx_buffer_release(cand_buffer);
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

  // One BLAS: a single NON-opaque triangle at object-space z=5 (flags=0 -> the
  // opacity classifier yields an AHS candidate callback for a flags-0 ray).
  host_tri_t tri = {
    { 0.f, 0.f, 5.f }, { 1.f, 0.f, 5.f }, { 0.f, 1.f, 5.f },
    /*flags*/ 0u
  };
  host_bvh_t blas = { &tri, 1, /*geometry_index*/ 0 };

  // One instance (identity xform) carrying a distinctive custom id + instance id.
  auto set_xform = [](float* m) {
    m[0]=1.f; m[1]=0.f; m[2]=0.f; m[3]=0.f;
    m[4]=0.f; m[5]=1.f; m[6]=0.f; m[7]=0.f;
    m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=0.f;
  };
  host_instance_t inst = {};
  set_xform(inst.xform);
  inst.blas_index = 0; inst.custom_id = kInstCustom;
  inst.instance_id = kInstId; inst.cull_mask = 0xff; inst.flags = 0;

  host_tlas_t tlas = { &blas, 1, &inst, 1 };
  std::vector<uint8_t> scene;
  uint64_t root_offset = 0;
  if (!build_tlas_scene<4>(tlas, scene, root_offset)) {
    std::cout << "build_tlas_scene failed" << std::endl;
    cleanup();
    return 1;
  }
  std::cout << "scene: " << scene.size() << " B (1 instance, 1 non-opaque tri)"
            << std::endl;

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(), VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  // Candidate-capture buffer, pre-seeded with a sentinel so a dispatcher that
  // never runs (or reads 0 candidate slots) is distinguishable from a real read.
  uint32_t cand_size = sizeof(rtu_cand_t);
  RT_CHECK(vx_buffer_create(device, cand_size, VX_MEM_READ | VX_MEM_WRITE, &cand_buffer));
  RT_CHECK(vx_buffer_address(cand_buffer, &kernel_arg.cand_addr));
  rtu_cand_t cand_seed = { 0xDEADBEEFu, 0xDEADBEEFu };
  RT_CHECK(vx_enqueue_write(queue, cand_buffer, 0, &cand_seed, cand_size, 0, nullptr, nullptr));

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
  vx_event_h launch_ev = nullptr, r0 = nullptr, r1 = nullptr;
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
  rtu_cand_t   cand   = {};
  RT_CHECK(vx_enqueue_read(queue, &result, res_buffer, 0, res_size, 1, &launch_ev, &r0));
  RT_CHECK(vx_enqueue_read(queue, &cand, cand_buffer, 0, cand_size, 1, &launch_ev, &r1));
  RT_CHECK(vx_event_wait_value(r0, 1, VX_TIMEOUT_INFINITE));
  RT_CHECK(vx_event_wait_value(r1, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(r0);
  vx_event_release(r1);
  vx_event_release(launch_ev);

  int errors = 0;
  std::cout << "candidate: custom=0x" << std::hex << cand.cand_instance_custom
            << " id=" << std::dec << cand.cand_instance_id << std::endl;
  std::cout << "terminal:  status=" << result.status << " hit_t=" << result.hit_t
            << " instance_id=" << result.instance_id
            << " custom=0x" << std::hex << result.instance_custom << std::dec
            << std::endl;
  std::cout << "oracle: candidate custom=0x" << std::hex << kInstCustom
            << " id=" << std::dec << kInstId << std::endl;

  // I1: the CANDIDATE stage must carry the true instance attributes.
  if (cand.cand_instance_custom != kInstCustom) {
    std::cout << "FAIL: candidate instance_custom mismatch (got 0x" << std::hex
              << cand.cand_instance_custom << std::dec << ")" << std::endl;
    ++errors;
  }
  if (cand.cand_instance_id != kInstId) {
    std::cout << "FAIL: candidate instance_id mismatch (got "
              << cand.cand_instance_id << ")" << std::endl;
    ++errors;
  }
  // Sanity on the accepted terminal hit (already covered by tlas_builder, but
  // keep the test self-contained).
  if (result.status != VX_RT_STS_DONE_HIT)          { std::cout << "FAIL: status\n";   ++errors; }
  if (std::fabs(result.hit_t - 5.f) > 1e-4f)        { std::cout << "FAIL: hit_t\n";    ++errors; }
  if (result.instance_custom != kInstCustom)        { std::cout << "FAIL: term custom\n"; ++errors; }
  if (result.instance_id != kInstId)                { std::cout << "FAIL: term id\n";  ++errors; }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
