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
// PRISM RTU Closest-Hit Shader smoke — Phase 5.
//
// Single lane fires one ray at a single opaque triangle with the
// VX_RT_FLAG_ENABLE_CHS bit set. RtuCore commits the hit, then queues
// a CHS yield through the existing CB_YIELD/CB_ACTION path. The
// kernel-registered dispatcher (mtvec) reads VX_RT_HIT_T and the
// kernel-published payload pointer (VX_RT_PAYLOAD_PTR_LO), writes a
// shading result into the payload, then exits via
// vx_rt_cb_ret(VX_RT_CB_DONE) + mret. After mret the kernel observes
// (a) status=HIT from vx_rt_wait and (b) the magic value the CHS wrote
// into the payload.

#ifndef _RTU_SMOKE_CHS_COMMON_H_
#define _RTU_SMOKE_CHS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u

// CHS writes this magic XOR'd with hit_t bits to payload[0]. The host
// oracle reproduces the same expression to verify the shader ran.
#define RTU_CHS_MAGIC          0xc1054afeu

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t primitive_id;
  uint32_t chs_payload;     // written by CHS dispatcher
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint64_t payload_addr;    // CHS reads VX_RT_PAYLOAD_PTR_LO/HI
  uint32_t reserved;
  uint32_t reserved2;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_CHS_COMMON_H_
