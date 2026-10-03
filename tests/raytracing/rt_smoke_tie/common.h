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

#ifndef _RT_SMOKE_TIE_COMMON_H_
#define _RT_SMOKE_TIE_COMMON_H_

#include <stdint.h>

typedef struct {
  float origin[3];
  float dir[3];
  float tmin;
  float tmax;
} tie_ray_t;

typedef struct {
  uint32_t status;
  uint32_t t;        // float bits
  uint32_t u;
  uint32_t v;
  uint32_t prim;
  uint32_t geom;     // geometry index | VX_RT_HIT_BACK_FACING
  uint32_t inst_id;
  uint32_t inst_custom;
} tie_result_t;

typedef struct {
  uint64_t rays_addr;
  uint64_t results_addr;
  uint32_t scene;
  uint32_t count;
} kernel_arg_t;

#endif // _RT_SMOKE_TIE_COMMON_H_
