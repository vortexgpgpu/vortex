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
// PRISM RTU TLAS-over-BLAS smoke — Phase 8.
//
// Phase 10: 2-instance TLAS sharing a single inline BLAS. Each
// instance carries its own 3x4 affine transform; the closer
// instance's hit wins and hit_instance_id reports its index.
//
// Layout:
//   [TLAS  header (16 B)]: primary_count = 2 instances, scene_kind = 1
//   [instance 0  (64 B)] : transform = translation t=(0,0,10),
//                          blas_byte_offset = 144
//   [instance 1  (64 B)] : transform = translation t=(0,0,5),
//                          blas_byte_offset = 144
//   [BLAS  header (16 B)]: triangle_count = 1
//   [BLAS  tri (40 B)]   : object-space triangle at z=0
//
// Ray fires (0.25, 0.25, 0) along +Z. Both instances are on the ray
// path; instance 0's geometry sits at world z=10, instance 1's at
// world z=5. The walker iterates both, applies each transform's
// world→object inverse, walks the BLAS in object space, and picks
// the closest world hit. Oracle: HIT t=5, hit_instance_id=1.

#ifndef _RTU_SMOKE_TLAS_COMMON_H_
#define _RTU_SMOKE_TLAS_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES        16
#define RTU_TRI_STRIDE_BYTES       40
#define RTU_TRI_FLAGS_OFFSET       36
#define RTU_TRI_FLAG_OPAQUE        0x1u

#define RTU_SCENE_KIND_TRI_LIST    0
#define RTU_SCENE_KIND_TLAS        1

#define RTU_INSTANCE_STRIDE        64
#define RTU_INSTANCE_BLAS_OFF_OFF  48
#define RTU_INSTANCE_CUSTOM_ID_OFF 52

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t hit_instance_id;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  uint32_t reserved;
  uint32_t reserved2;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_TLAS_COMMON_H_
