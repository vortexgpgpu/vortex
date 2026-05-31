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
// PRISM RTU §8.8 instanceCullMask smoke.
//
// 2-instance TLAS sharing one BLAS:
//   instance 0 at world z=5,  cull_mask=0x01
//   instance 1 at world z=10, cull_mask=0x02
//
// Three back-to-back rays (one warp, lane 0):
//   ray A — cull_mask=0x01 → hits inst 0 only            (t=5,  inst=0)
//   ray B — cull_mask=0x02 → hits inst 1 only            (t=10, inst=1)
//   ray C — cull_mask=0xff → both candidates, inst 0 closer (t=5,  inst=0)
//
// Validates both sides of the walker's gate:
//   (inst.cull_mask & ray.cull_mask) == 0 → skip the instance entirely.

#ifndef _RTU_SMOKE_CULL_MASK_COMMON_H_
#define _RTU_SMOKE_CULL_MASK_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES         16
#define RTU_TRI_STRIDE_BYTES        40
#define RTU_TRI_FLAGS_OFFSET        36
#define RTU_TRI_FLAG_OPAQUE         0x1u

#define RTU_SCENE_KIND_TRI_LIST     0
#define RTU_SCENE_KIND_TLAS         1

#define RTU_INSTANCE_STRIDE         64
#define RTU_INSTANCE_BLAS_OFF_OFF   48
#define RTU_INSTANCE_CUSTOM_ID_OFF  52
#define RTU_INSTANCE_CULL_MASK_OFF  56

#define RTU_NUM_RAYS                3

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t hit_instance_id;
  uint32_t pad;
} rtu_one_t;

typedef struct {
  rtu_one_t rays[RTU_NUM_RAYS];
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
  uint32_t ray_cull_mask[RTU_NUM_RAYS];
} kernel_arg_t;

#endif // _RTU_SMOKE_CULL_MASK_COMMON_H_
