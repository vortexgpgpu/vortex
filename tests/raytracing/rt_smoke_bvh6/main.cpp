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
// PRISM RTU smoke — CW-BVH6 host driver (scene_kind=3).
//
// Builds a scene whose root is a single 96-byte 6-wide internal node
// fanning out to six opaque-triangle leaves at decreasing depth: child i
// holds a triangle at z = 10 - i, so the NEAREST triangle (z=5) lives in
// the LAST child slot (child 5). A correct hit therefore requires the
// walker to decode all six children — a 4-wide decode would stop at
// child 3 (z=7) and return the wrong t. This exercises the width-generic
// NodeView decode + the 6-wide box-test loop end-to-end.
//
// On-disk layout (448 bytes):
//   +  0  scene header { root_node_offset=16, scene_kind=3,
//                        node_count=1, leaf_count=6 }
//   + 16  CW-BVH6 internal node (96 B): origin=(0,0,0), exp=(-4,-4,-4),
//          6 children, each a leaf at offset 112 + i*56
//   +112  6 × { leaf header (16 B) + triangle (40 B) }

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

// Quantize an axis value `v` against origin 0 and exponent `exp` (step =
// 2^exp), clamped to the uint8 grid. The reconstructed value is
// q * 2^exp, so q = round(v / 2^exp).
static uint8_t quantize(float v, int exp) {
  float step = std::ldexp(1.0f, exp);
  float q = std::round(v / step);
  if (q < 0.f)   q = 0.f;
  if (q > 255.f) q = 255.f;
  return (uint8_t)q;
}

int main(int /*argc*/, char* /*argv*/[]) {
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  const uint32_t leaf_stride = VX_BVH_LEAF_HDR_BYTES + VX_BVH_TRI_STRIDE; // 56
  const uint32_t node_off    = VX_BVH_SCENE_HDR_BYTES;                    // 16
  const uint32_t leaves_off  = node_off + VX_BVH6_NODE_BYTES;             // 112
  const uint32_t scene_bytes = leaves_off + VX_BVH6_WIDTH * leaf_stride;  // 448

  std::vector<uint8_t> scene(scene_bytes, 0);

  // Scene header.
  uint32_t* sh = reinterpret_cast<uint32_t*>(scene.data());
  sh[0] = node_off;             // root_node_offset = 16
  sh[1] = VX_BVH_SCENE_KIND;    // = 3 (BVH6)
  sh[2] = (uint32_t)scene.size(); // total scene bytes (pre-fetch)
  sh[3] = VX_BVH6_WIDTH;        // leaf_count = 6

  // Internal node at offset 16.
  uint8_t* node = scene.data() + node_off;
  uint32_t* nkind = reinterpret_cast<uint32_t*>(node);
  *nkind = VX_BVH_KIND_INTERNAL | (VX_BVH6_WIDTH << VX_BVH_COUNT_SHIFT);
  float* norigin = reinterpret_cast<float*>(node + VX_BVH6_OFF_ORIGIN);
  norigin[0] = 0.f; norigin[1] = 0.f; norigin[2] = 0.f;
  const int exp = -4;  // step = 1/16 = 0.0625
  int8_t* nexp = reinterpret_cast<int8_t*>(node + VX_BVH6_OFF_EXP);
  nexp[0] = (int8_t)exp; nexp[1] = (int8_t)exp; nexp[2] = (int8_t)exp;

  uint32_t* child = reinterpret_cast<uint32_t*>(node + VX_BVH6_OFF_CHILD);
  uint8_t*  qmin  = node + VX_BVH6_OFF_QMIN;
  uint8_t*  qmax  = node + VX_BVH6_OFF_QMAX;

  for (uint32_t i = 0; i < VX_BVH6_WIDTH; ++i) {
    float z = 10.f - (float)i;            // child 5 -> z = 5 (nearest)
    uint32_t leaf_off = leaves_off + i * leaf_stride;
    child[i] = leaf_off | VX_BVH_CHILD_LEAF_FLAG;

    // Child AABB: x,y in [0, 1.0625], z in [z, z + 0.0625].
    qmin[i*3+0] = quantize(0.f, exp);   qmax[i*3+0] = quantize(1.0625f, exp);
    qmin[i*3+1] = quantize(0.f, exp);   qmax[i*3+1] = quantize(1.0625f, exp);
    qmin[i*3+2] = quantize(z, exp);     qmax[i*3+2] = (uint8_t)(quantize(z, exp) + 1);

    // Leaf header + triangle.
    uint8_t* leaf = scene.data() + leaf_off;
    uint32_t* lh = reinterpret_cast<uint32_t*>(leaf);
    lh[0] = VX_BVH_KIND_LEAF_TRI | (1u << VX_BVH_COUNT_SHIFT);
    lh[1] = i;     // geometry_index
    lh[2] = 0;     // leaf flags
    lh[3] = 0;     // reserved

    float* tri = reinterpret_cast<float*>(leaf + VX_BVH_LEAF_HDR_BYTES);
    tri[0] = 0.f; tri[1] = 0.f; tri[2] = z;
    tri[3] = 1.f; tri[4] = 0.f; tri[5] = z;
    tri[6] = 0.f; tri[7] = 1.f; tri[8] = z;
    uint32_t* tri_flags = reinterpret_cast<uint32_t*>(
        leaf + VX_BVH_LEAF_HDR_BYTES + VX_BVH_TRI_FLAGS_OFFSET);
    *tri_flags = VX_BVH_TRI_FLAG_OPAQUE;
  }

  RT_CHECK(vx_buffer_create(device, (uint32_t)scene.size(),
                            VX_MEM_READ, &scene_buffer));
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

  std::cout << "scene_addr=0x" << std::hex << kernel_arg.scene_addr
            << std::dec << " bvh6 (1 node, 6 leaves; nearest in child 5)"
            << std::endl;

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

  // Oracle: nearest triangle is in child 5 at z=5; barycentrics (0.25, 0.25).
  const uint32_t exp_status = VX_RT_STS_DONE_HIT;
  const float    exp_t      = 5.f;
  const float    exp_u      = 0.25f;
  const float    exp_v      = 0.25f;
  const uint32_t exp_geom   = 5;   // nearest triangle lives in child 5

  std::cout << "oracle: HIT t=" << exp_t << " u=" << exp_u << " v=" << exp_v
            << " geom=" << exp_geom << " (from 6th child)" << std::endl;

  int errors = 0;
  if (result.status != exp_status) {
    std::cout << "status mismatch: got " << result.status
              << " expected " << exp_status << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_t - exp_t) > 1e-4f) {
    std::cout << "hit_t mismatch: got " << result.hit_t
              << " expected " << exp_t << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_u - exp_u) > 1e-4f) {
    std::cout << "hit_u mismatch: got " << result.hit_u
              << " expected " << exp_u << std::endl;
    ++errors;
  }
  if (std::fabs(result.hit_v - exp_v) > 1e-4f) {
    std::cout << "hit_v mismatch: got " << result.hit_v
              << " expected " << exp_v << std::endl;
    ++errors;
  }
  if (result.geometry_index != exp_geom) {
    std::cout << "geometry_index mismatch: got " << result.geometry_index
              << " expected " << exp_geom << std::endl;
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
