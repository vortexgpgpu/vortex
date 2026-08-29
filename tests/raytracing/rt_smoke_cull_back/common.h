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

#ifndef _RTU_SMOKE_CULL_BACK_COMMON_H_
#define _RTU_SMOKE_CULL_BACK_COMMON_H_

#include <stdint.h>

#define RTU_SCENE_HDR_BYTES    16
#define RTU_TRI_STRIDE_BYTES   40
#define RTU_TRI_FLAGS_OFFSET   36
#define RTU_TRI_FLAG_OPAQUE    0x1u

typedef struct {
  uint32_t front_status;   // ray from -z side: front-hit, should survive CULL_BACK
  uint32_t back_status;    // ray from +z side: back-hit, should be culled
  float    front_t;
  uint32_t pad;
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
  // Two rays, both with CULL_BACK_FACING set; one hits front, one hits back.
  float    front_origin[3];
  float    front_dir[3];
  float    back_origin[3];
  float    back_dir[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_CULL_BACK_COMMON_H_
