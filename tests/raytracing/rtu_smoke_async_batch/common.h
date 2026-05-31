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
// §8.6 async batch smoke test. Kernel issues N back-to-back
// vx_rt_trace calls (no intervening WAIT), then drains the handles
// with N vx_rt_waits in declared order. Validates that:
//   - vx_rt_trace returns a real handle (slot index) and is
//     non-blocking (the next trace can launch with its own scene
//     even though the prior ray is still in flight).
//   - vx_rt_wait returns the per-lane TERMINAL status of the
//     specific handle (each ray is paired with its own status,
//     not the latest TERMINAL on the warp).
//   - vx_rt_get_after(slot, sts) reads the regfile after WAIT has
//     applied the matching TERMINAL response, so post-WAIT reads
//     see THAT ray's hit attrs, not a later ray's.
//
// Scene: same layout as rtu_smoke. The host stages NUM_BATCH
// independent single-triangle scenes, each at a different z, so
// each ray's hit_t differs and the kernel can detect cross-talk.

#ifndef _RTU_SMOKE_ASYNC_BATCH_COMMON_H_
#define _RTU_SMOKE_ASYNC_BATCH_COMMON_H_

#include <stdint.h>

// 4 traces in flight per kernel invocation. Smaller than the
// per-cluster pool (VX_CFG_RTU_CONTEXT_POOL=32) but big enough to
// expose any handle-mixup or slot-recycle bugs.
#define RTU_ASYNC_NUM_BATCH 4

#define RTU_SCENE_HDR_BYTES   16
#define RTU_TRI_STRIDE_BYTES  40
#define RTU_TRI_FLAGS_OFFSET  36
#define RTU_TRI_FLAG_OPAQUE   0x1u

typedef struct {
  uint32_t status;       // one per ray in the batch
  float    hit_t;        // one per ray
} rtu_one_t;

typedef struct {
  rtu_one_t rays[RTU_ASYNC_NUM_BATCH];
} rtu_result_t;

typedef struct {
  // Per-ray scene addresses (one device buffer per ray).
  uint64_t scene_addr[RTU_ASYNC_NUM_BATCH];
  uint64_t results_addr;
  uint32_t num_lanes;
  float    ray_origin[3];
  float    ray_direction[3];
  float    tmin;
  float    tmax;
} kernel_arg_t;

#endif // _RTU_SMOKE_ASYNC_BATCH_COMMON_H_
