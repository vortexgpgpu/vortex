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
// PRISM RTU per-instance-flags smoke — W9(c) host driver.
//
// A flat TLAS with one instance wrapping an OPAQUE triangle. The instance sets
// FORCE_NO_OPAQUE (packed into cull_mask bits 15..8): the walker must treat the
// opaque triangle as non-opaque and yield an AHS callback. With the default
// IGNORE dispatcher the candidate is dropped, so the trace ends in MISS — a
// result the flag alone produces (without it the opaque triangle commits a HIT
// at t=5). Pass -f 0 to clear the flag (expect HIT) or -d 1 to ACCEPT.

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
uint32_t force_no_opaque = 1;   // -f: apply the FORCE_NO_OPAQUE instance flag
uint32_t cb_decision     = RTU_AHS_DECISION_IGNORE;  // -d

vx_device_h device       = nullptr;
vx_buffer_h scene_buffer = nullptr;
vx_buffer_h res_buffer   = nullptr;
vx_queue_h  queue        = nullptr;
vx_module_h module_      = nullptr;
vx_kernel_h kernel       = nullptr;
kernel_arg_t kernel_arg  = {};

static void show_usage() {
  std::cout << "RTU per-instance-flags smoke test." << std::endl;
  std::cout << "Usage: [-k kernel] [-f 0|1] [-d 0|1] [-h]" << std::endl;
  std::cout << "  -f 1  set FORCE_NO_OPAQUE (default) -> opaque tri yields AHS" << std::endl;
  std::cout << "  -f 0  no instance flags -> opaque tri commits directly" << std::endl;
  std::cout << "  -d 0  IGNORE the candidate (default); -d 1 ACCEPT" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:f:d:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'f': force_no_opaque = atoi(optarg) ? 1 : 0; break;
    case 'd': cb_decision = atoi(optarg) ? RTU_AHS_DECISION_ACCEPT
                                          : RTU_AHS_DECISION_IGNORE; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

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
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Single-instance flat TLAS sharing one inline 1-triangle BLAS.
  //   [0..16)    : TLAS header (primary_count=1, scene_kind=TLAS)
  //   [16..80)   : instance 0 (translate z=5, cull_mask=0xff, flags byte)
  //   [80..96)   : BLAS header (triangle_count=1)
  //   [96..136)  : BLAS opaque tri at object z=0
  constexpr uint32_t kNumInstances = 1;
  constexpr uint32_t kBlasOff = RTU_SCENE_HDR_BYTES
                              + kNumInstances * RTU_INSTANCE_STRIDE;
  constexpr uint32_t kSceneSz = kBlasOff + RTU_SCENE_HDR_BYTES + RTU_TRI_STRIDE_BYTES;
  std::vector<uint8_t> scene_bytes(kSceneSz, 0);

  uint32_t* tlas_hdr = reinterpret_cast<uint32_t*>(scene_bytes.data());
  tlas_hdr[0] = kNumInstances;
  tlas_hdr[1] = RTU_SCENE_KIND_TLAS;

  uint8_t* inst = scene_bytes.data() + RTU_SCENE_HDR_BYTES;
  float* xform = reinterpret_cast<float*>(inst);
  xform[0] = 1.f; xform[1] = 0.f; xform[2]  = 0.f; xform[3]  = 0.f;
  xform[4] = 0.f; xform[5] = 1.f; xform[6]  = 0.f; xform[7]  = 0.f;
  xform[8] = 0.f; xform[9] = 0.f; xform[10] = 1.f; xform[11] = 5.f;
  uint32_t* blas_off = reinterpret_cast<uint32_t*>(inst + RTU_INSTANCE_BLAS_OFF_OFF);
  *blas_off = kBlasOff;
  uint32_t* custom_id = reinterpret_cast<uint32_t*>(inst + RTU_INSTANCE_CUSTOM_ID_OFF);
  *custom_id = 0u;
  // cull_mask low byte = 0xff (match all); instance flags in bits 15..8.
  uint32_t inst_flags = force_no_opaque ? RTU_INST_FLAG_FORCE_NO_OPQ : 0u;
  uint32_t* cull_word = reinterpret_cast<uint32_t*>(inst + RTU_INSTANCE_CULL_OFF);
  *cull_word = 0xffu | (inst_flags << RTU_INST_FLAGS_SHIFT);

  uint32_t* blas_hdr = reinterpret_cast<uint32_t*>(scene_bytes.data() + kBlasOff);
  blas_hdr[0] = 1;   // triangle_count

  float* tris = reinterpret_cast<float*>(
      scene_bytes.data() + kBlasOff + RTU_SCENE_HDR_BYTES);
  tris[0] = 0.f; tris[1] = 0.f; tris[2] = 0.f;
  tris[3] = 1.f; tris[4] = 0.f; tris[5] = 0.f;
  tris[6] = 0.f; tris[7] = 1.f; tris[8] = 0.f;
  uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
      scene_bytes.data() + kBlasOff + RTU_SCENE_HDR_BYTES + RTU_TRI_FLAGS_OFFSET);
  *tri_flags = RTU_TRI_FLAG_OPAQUE;  // opaque geometry — flag flips it

  RT_CHECK(vx_buffer_create(device, kSceneSz, VX_MEM_READ, &scene_buffer));
  RT_CHECK(vx_buffer_address(scene_buffer, &kernel_arg.scene_addr));

  uint32_t res_size = sizeof(rtu_result_t);
  RT_CHECK(vx_buffer_create(device, res_size, VX_MEM_WRITE, &res_buffer));
  RT_CHECK(vx_buffer_address(res_buffer, &kernel_arg.results_addr));

  kernel_arg.cb_decision      = cb_decision;
  kernel_arg.ray_origin[0]    = 0.25f;
  kernel_arg.ray_origin[1]    = 0.25f;
  kernel_arg.ray_origin[2]    = 0.0f;
  kernel_arg.ray_direction[0] = 0.0f;
  kernel_arg.ray_direction[1] = 0.0f;
  kernel_arg.ray_direction[2] = 1.0f;
  kernel_arg.tmin             = 0.001f;
  kernel_arg.tmax             = 1e30f;

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr << std::dec
            << " flat-TLAS (1 inst, opaque tri), FORCE_NO_OPAQUE="
            << force_no_opaque
            << ", decision=" << (cb_decision == RTU_AHS_DECISION_ACCEPT ? "ACCEPT" : "IGNORE")
            << std::endl;

  RT_CHECK(vx_enqueue_write(queue, scene_buffer, 0, scene_bytes.data(),
                            kSceneSz, 0, nullptr, nullptr));
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

  // Oracle. FORCE_NO_OPAQUE reclassifies the opaque tri as non-opaque, so it
  // yields AHS: IGNORE -> MISS, ACCEPT -> HIT. Without the flag the opaque tri
  // commits directly (HIT) and the dispatcher never runs.
  bool yields = (force_no_opaque != 0);
  bool exp_hit = !yields || (cb_decision == RTU_AHS_DECISION_ACCEPT);
  uint32_t exp_status = exp_hit ? VX_RT_STS_DONE_HIT : VX_RT_STS_DONE_MISS;
  float exp_t = exp_hit ? 5.f : 0.f;
  std::cout << "oracle: " << (exp_hit ? "HIT" : "MISS")
            << " t=" << exp_t << std::endl;

  int errors = 0;
  bool sts_ok = (result.status == exp_status);
  bool t_ok   = !exp_hit || (std::fabs(result.hit_t - exp_t) < 1e-4f);
  if (!sts_ok || !t_ok) {
    std::cout << "result: status=" << result.status
              << " hit_t=" << result.hit_t
              << " (expected status=" << exp_status << " t=" << exp_t << ")"
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
