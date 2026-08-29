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
// PRISM RTU deep multi-instance TLAS smoke — I6(b) restart-budget guard.

#ifndef _RTU_SMOKE_DEEP_TLAS_COMMON_H_
#define _RTU_SMOKE_DEEP_TLAS_COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t status;
  float    hit_t;
  uint32_t primitive_id;
  uint32_t instance_id;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_DEEP_TLAS_COMMON_H_
