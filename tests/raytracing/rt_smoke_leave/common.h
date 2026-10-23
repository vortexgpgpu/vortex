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
// PRISM RTU smoke: a lane that leaves its proceed loop holding a candidate.
//
// Two lanes of one warp trace the same ray into two non-opaque triangles
// (t = 2, 3) in front of an opaque one (t = 4). Lane 0 ignores every candidate
// to the end; lane 1 leaves the loop at its first candidate, so it is absent
// from every CONTINUE lane 0 issues after it. Both models take an absent lane
// to have left its loop and answer ABORT for it: its ray ends at once,
// keeping what the walk committed (the opaque t = 4), and the records that
// follow carry its terminal status, which lane 1 reads from the hit window
// once the warp has reconverged. Neither lane's trace may stall.

#ifndef _RTU_SMOKE_LEAVE_COMMON_H_
#define _RTU_SMOKE_LEAVE_COMMON_H_

#include <stdint.h>

#define VX_BVH_SCENE_KIND          2
#define VX_BVH_SCENE_HDR_BYTES     16
#define VX_BVH_LEAF_HDR_BYTES      16
#define VX_BVH_TRI_STRIDE          40
#define VX_BVH_TRI_FLAGS_OFFSET    36
#define VX_BVH_KIND_LEAF_TRI       1
#define VX_BVH_COUNT_SHIFT         8
#define VX_BVH_TRI_FLAG_OPAQUE     0x1u

#define RTU_NUM_LANES   2

typedef struct {
  uint32_t num_cand;
  float    first_cand_t;
  uint32_t exit_status;   // the status the lane last saw in its loop
  uint32_t live_status;   // its status in the hit window after reconverging
  uint32_t live_t;        // its HIT_T word there
} rtu_result_t;

typedef struct {
  uint64_t scene_addr;
  uint64_t results_addr;
} kernel_arg_t;

#endif // _RTU_SMOKE_LEAVE_COMMON_H_
