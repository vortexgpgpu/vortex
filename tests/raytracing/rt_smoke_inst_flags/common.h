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
// PRISM RTU per-instance-flags smoke — W9(c).
//
// Flat TLAS, one instance (translate z=5) wrapping a single inline OPAQUE
// triangle at object z=0 (world z=5). The instance carries the
// FORCE_NO_OPAQUE VkGeometryInstanceFlag, packed into bits 15..8 of the
// instance cull_mask word. With FORCE_NO_OPAQUE the walker treats the opaque
// triangle as non-opaque and yields an any-hit callback instead of committing
// directly; the kernel's IGNORE dispatcher then drops it, turning what would be
// a HIT (t=5) into a MISS. Clearing the flag (-f 0) commits the opaque hit and
// the dispatcher never fires.

#ifndef _RTU_SMOKE_INST_FLAGS_COMMON_H_
#define _RTU_SMOKE_INST_FLAGS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES        16
#define RTU_TRI_STRIDE_BYTES       40
#define RTU_TRI_FLAGS_OFFSET       36
#define RTU_TRI_FLAG_OPAQUE        0x1u

#define RTU_SCENE_KIND_TLAS        1

#define RTU_INSTANCE_STRIDE        64
#define RTU_INSTANCE_BLAS_OFF_OFF  48
#define RTU_INSTANCE_CUSTOM_ID_OFF 52
#define RTU_INSTANCE_CULL_OFF      56

// VkGeometryInstanceFlagBits (low byte) packed into cull_mask bits 15..8.
#define RTU_INST_FLAGS_SHIFT        8
#define RTU_INST_FLAG_FORCE_NO_OPQ  0x8u

#define RTU_AHS_DECISION_ACCEPT  1
#define RTU_AHS_DECISION_IGNORE  0

typedef struct {
  uint32_t status;
  float    hit_t;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint32_t cb_decision;   // RTU_AHS_DECISION_*
  uint32_t reserved;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_INST_FLAGS_COMMON_H_
