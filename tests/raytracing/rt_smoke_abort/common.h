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
// PRISM RTU verdict smoke: ABORT, and what a candidate record tells the lanes
// that have nothing to decide.
//
// Scene A stacks two non-opaque triangles (t = 2, 3) in front of an opaque one
// (t = 4) on lane 0's ray; scene B has only the two non-opaque ones. Lane 1's
// ray misses everything. Each case answers lane 0's candidates with a
// scripted verdict sequence and records, per lane, the first status, the
// candidates seen, the first candidate's distance and the terminal record:
//   - ABORT ends traversal without committing the candidate in hand, keeping
//     whatever the walk had committed (the opaque t = 4), unlike TERMINATE,
//     which accepts and stops;
//   - the terminal geometry word marks a shader-accepted hit (COMMITTED_CB);
//   - lane 1, finished with no candidate while lane 0 holds one, reads its
//     terminal status in the very first record instead of riding along.

#ifndef _RTU_SMOKE_ABORT_COMMON_H_
#define _RTU_SMOKE_ABORT_COMMON_H_

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
#define RTU_MAX_STEPS   4   // scripted verdicts per case; later ones IGNORE

typedef struct {
  uint32_t scene;                    // 0 = A, 1 = B
  uint32_t action[RTU_MAX_STEPS];    // lane 0's verdict per candidate
} rtu_case_t;

typedef struct {
  uint32_t first_status;
  uint32_t num_cand;
  float    first_cand_t;
  uint32_t status;
  float    hit_t;
  uint32_t geom;                     // raw terminal HIT_GEOMETRY_INDEX word
} rtu_result_t;

typedef struct {
  uint64_t scene_addr[2];
  uint64_t cases_addr;
  uint64_t results_addr;
  uint32_t num_cases;
  uint32_t pad;
  float    ray_origin[RTU_NUM_LANES][3];
} kernel_arg_t;

#endif // _RTU_SMOKE_ABORT_COMMON_H_
