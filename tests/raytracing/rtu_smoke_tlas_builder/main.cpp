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
// PRISM RTU host-TLAS-builder smoke — W9(d) host driver.
//
// Builds a 2-level CW-BVH4 TLAS with two instances of a single shared BLAS via
// vortex::raytrace::build_tlas_scene (no hand-laid instance records). The BLAS
// holds one opaque triangle at object z=5. Instance 0 (identity) places it at
// world z=5; instance 1 (translate +10z) at world z=15. A +z ray finds the
// closer instance 0 at t=5. Oracle: HIT t=5, instance_id=0, custom=0xC0DE0000.

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

  // One shared BLAS: a single opaque triangle at object-space z=5.
  host_tri_t tri = {
    { 0.f, 0.f, 5.f }, { 1.f, 0.f, 5.f }, { 0.f, 1.f, 5.f },
    RTU_BVH_FLAG_OPAQUE
  };
  host_bvh_t blas = { &tri, 1, /*geometry_index*/ 0 };

  // Two instances of that BLAS: identity and +10z translation.
  auto set_xform = [](float* m, float tz) {
    m[0]=1.f; m[1]=0.f; m[2]=0.f; m[3]=0.f;
    m[4]=0.f; m[5]=1.f; m[6]=0.f; m[7]=0.f;
    m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=tz;
  };
  host_instance_t insts[2] = {};
  set_xform(insts[0].xform, 0.f);
  insts[0].blas_index = 0; insts[0].custom_id = 0xC0DE0000u;
  insts[0].instance_id = 0; insts[0].cull_mask = 0xff; insts[0].flags = 0;
  set_xform(insts[1].xform, 10.f);
  insts[1].blas_index = 0; insts[1].custom_id = 0xC0DE0001u;
  insts[1].instance_id = 1; insts[1].cull_mask = 0xff; insts[1].flags = 0;

  host_tlas_t tlas = { &blas, 1, insts, 2 };
  std::vector<uint8_t> scene;
  uint64_t root_offset = 0;
  if (!build_tlas_scene<4>(tlas, scene, root_offset)) {
    std::cout << "build_tlas_scene failed" << std::endl;
    cleanup();
    return 1;
  }

  // Structural sanity: root is a LEAF_INST leaf with 2 instances.
  auto rd_u32 = [&](uint32_t off) {
    uint32_t v; std::memcpy(&v, scene.data() + off, 4); return v;
  };
  int errors = 0;
  uint32_t root = rd_u32(0);
  uint32_t root_word = rd_u32(root);
  std::cout << "scene: " << scene.size() << " B, root_off=" << root
            << " kind=" << rd_u32(4)
            << " leaf_count=" << rd_u32(12) << std::endl;
  if (rd_u32(4) != RTU_SCENE_KIND_BVH4)        { std::cout << "bad scene_kind\n"; ++errors; }
  if (rd_u32(8) != scene.size())               { std::cout << "bad scene_bytes\n"; ++errors; }
  if ((root_word & 0xffu) != RTU_BVH_KIND_LEAF_INST) { std::cout << "root not LEAF_INST\n"; ++errors; }
  if (((root_word >> RTU_BVH_COUNT_SHIFT) & 0xffu) != 2) { std::cout << "bad instance count\n"; ++errors; }
  if (errors) { std::cout << "STRUCTURAL CHECK FAILED\n"; cleanup(); return 1; }
  std::cout << "structural check OK" << std::endl;

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

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " bvh4 TLAS via build_tlas_scene (2 instances of 1 BLAS)" << std::endl;

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
  const float    exp_t      = 5.f;
  const uint32_t exp_inst   = 0;
  const uint32_t exp_custom = 0xC0DE0000u;
  std::cout << "oracle: HIT t=" << exp_t << " instance_id=" << exp_inst
            << " custom=0x" << std::hex << exp_custom << std::dec << std::endl;

  bool sts_ok = (result.status == exp_status);
  bool t_ok   = std::fabs(result.hit_t - exp_t) < 1e-4f;
  bool id_ok  = (result.instance_id == exp_inst);
  bool cst_ok = (result.instance_custom == exp_custom);
  if (!sts_ok || !t_ok || !id_ok || !cst_ok) {
    std::cout << "result: status=" << result.status
              << " hit_t=" << result.hit_t
              << " instance_id=" << result.instance_id
              << " custom=0x" << std::hex << result.instance_custom << std::dec
              << std::endl;
    ++errors;
  }

  cleanup();

  if (errors != 0) {
    std::cout << "FAILED with " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
