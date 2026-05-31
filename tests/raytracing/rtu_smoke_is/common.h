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
// PRISM RTU Intersection Shader (IS) smoke — Phase 6.
//
// Scene has one primitive whose tri.flags carries PROCEDURAL (bit 1)
// instead of OPAQUE. The walker still uses the triangle vertices to
// decide whether the ray crosses the primitive's AABB-ish region
// (placeholder for a real AABB test); when it does, it yields IS
// (cb_type = VX_RT_CB_TYPE_PROC) instead of AHS, and the kernel
// dispatcher runs the shape's actual intersection test (e.g. ray-
// sphere, ray-AABB) and returns ACCEPT/IGNORE. The smoke kernel's
// dispatcher reads VX_RT_CB_TYPE, checks it equals PROC (so the host
// can verify the IS path actually fired), accepts, and writes a
// magic sentinel to the payload before mret.

#ifndef _RTU_SMOKE_IS_COMMON_H_
#define _RTU_SMOKE_IS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u
#define RTU_TRI_FLAG_PROC      0x2u

#define RTU_IS_MAGIC           0x15ec0a11u

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t is_payload;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint64_t payload_addr;
  uint32_t reserved;
  uint32_t reserved2;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_IS_COMMON_H_
