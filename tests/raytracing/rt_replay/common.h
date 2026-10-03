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
// rt_replay — re-traces rays recorded by the SimX RTU ray log
// (VX_RTU_RAYLOG, sim/simx/rtu/rtu_raylog.h) against the scene lines the
// recording walks read, restored at their original device addresses.

#ifndef _RT_REPLAY_COMMON_H_
#define _RT_REPLAY_COMMON_H_

#include <stdint.h>

// One ray as the kernel consumes it. Each warp's rays share scene / flags /
// cull: those ride the warp-uniform half of vx_rt_wtrace.
typedef struct {
  uint32_t scene;
  uint32_t flags;
  uint32_t cull;
  uint32_t pad;
  float    origin[3];
  float    dir[3];
  float    tmin;
  float    tmax;
} replay_ray_t;

// Status bit set when the trace yielded a candidate (the recorded ray did not).
#define REPLAY_STS_YIELDED 0x100u

typedef struct {
  uint32_t status;
  uint32_t t;        // float bits
  uint32_t u;
  uint32_t v;
  uint32_t prim;
  uint32_t geom;     // geometry index | VX_RT_HIT_BACK_FACING
  uint32_t inst_id;
  uint32_t inst_custom;
} replay_result_t;

typedef struct {
  uint64_t rays_addr;
  uint64_t results_addr;
  uint32_t count;
  uint32_t pad;
} kernel_arg_t;

#ifdef __cplusplus
// Ray-log stream layout (mirror of sim/simx/rtu/rtu_raylog.h).
#define RAYLOG_MAGIC   0x4C525856u
#define RAYLOG_VERSION 1u
#define RAYLOG_REC_LINE 1u
#define RAYLOG_REC_RAY  2u
#define RAYLOG_INFO_REPLAYABLE (1u << 8)

struct RaylogHeader {
  uint32_t magic, version, num_threads, line_bytes, bvh_width, ray_bytes;
  uint32_t reserved[2];
};

struct RaylogRay {
  uint32_t type, info, seq, epoch;
  uint32_t scene_root, ray_flags, cull_mask, warp_lane;
  uint32_t origin[3], dir[3];
  uint32_t tmin, tmax;
  uint32_t status, hit_t, hit_u, hit_v, prim, geom, inst_id, inst_custom, hit_attr;
  uint32_t reserved[3];
};
static_assert(sizeof(RaylogRay) == 112, "RaylogRay layout");
#endif

#endif // _RT_REPLAY_COMMON_H_
