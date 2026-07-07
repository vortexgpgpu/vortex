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

#ifndef _RTU_BVH_MULTINODE_COMMON_H_
#define _RTU_BVH_MULTINODE_COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t status;
  float    hit_t;
  float    hit_u;
  float    hit_v;
  uint32_t primitive_id;
  uint32_t pad;
} rtu_result_t;

// One ray per thread: 8 contiguous floats (origin3, dir3, tmin, tmax) — the
// same f0..f7 window vx_ray_t maps to.
typedef struct {
  uint64_t scene_addr;
  uint64_t rays_addr;     // float[num_rays * 8]
  uint64_t results_addr;  // rtu_result_t[num_rays]
  uint32_t num_rays;
  uint32_t pad;
} kernel_arg_t;

#endif // _RTU_BVH_MULTINODE_COMMON_H_
