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
// PRISM RTU Miss Shader smoke — Phase 5.
//
// Ray fires away from the lone scene triangle (direction -Z instead of
// +Z) so traversal misses. With VX_RT_FLAG_ENABLE_MISS set, RtuCore
// queues a MISS yield through the existing CB_YIELD/CB_ACTION path.
// The dispatcher reads the kernel-published payload pointer and writes
// a magic sentinel so the host can verify the MISS shader actually ran
// *before* TERMINAL retired vx_rt_wait with status=DONE_MISS.

#ifndef _RTU_SMOKE_MISS_COMMON_H_
#define _RTU_SMOKE_MISS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u

#define RTU_MISS_MAGIC         0x5113e600u

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t miss_payload;
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

#endif // _RTU_SMOKE_MISS_COMMON_H_
