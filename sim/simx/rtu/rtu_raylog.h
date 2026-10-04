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
// PRISM RTU — ray log (opt-in, VX_RTU_RAYLOG=<path>).
//
// Records every traced ray with its terminal result, plus every scene line a
// walk read, so a ray can be re-traced outside the workload that issued it
// (tests/raytracing/rt_replay) and compared bit-exactly against another model.
//
// Stream layout (little-endian): a RaylogHeader, then records tagged by their
// first word. A LINE record is emitted the first time a walk reads a line, and
// again, with the epoch bumped, if a later walk reads different bytes there;
// every RAY record names the epoch whose memory image it was traced against.
//
// Optional knobs:
//   VX_RTU_RAYLOG_MAX=<n>   stop after n RAY records (default 4M)
//   VX_RTU_RAYLOG_EVERY=<n> keep one terminal ray in n (default 1)
//
// The lines a SimX walk reads are not all another model may read: one that
// culls differently descends into nodes SimX never fetched. So the first trace
// against a scene also snapshots its whole image from device memory — the
// header's scene_bytes after the root, and the instance table packed below it.

#ifndef _VX_RTU_RAYLOG_H_
#define _VX_RTU_RAYLOG_H_

#include <array>
#include <cstdint>
#include <unordered_map>
#include "rtu_types.h"

namespace vortex {
class RAM;
namespace rtu { namespace raylog {

constexpr uint32_t kMagic   = 0x4C525856;   // "VXRL"
constexpr uint32_t kVersion = 1;

enum RecType : uint32_t {
  REC_LINE = 1,
  REC_RAY  = 2,
};

// RaylogRay::info bits.
constexpr uint32_t kInfoCbShift    = 0;       // bit (cb_type) set per callback yielded
constexpr uint32_t kInfoReplayable = 1u << 8; // no any-hit / intersection callback decided it

struct RaylogHeader {
  uint32_t magic;
  uint32_t version;
  uint32_t num_threads;
  uint32_t line_bytes;
  uint32_t bvh_width;
  uint32_t ray_bytes;
  uint32_t reserved[2];
};

struct RaylogLine {
  uint32_t type;      // REC_LINE
  uint32_t epoch;
  uint64_t addr;
  uint8_t  data[VX_CFG_MEM_BLOCK_SIZE];
};

struct RaylogRay {
  uint32_t type;      // REC_RAY
  uint32_t info;
  uint32_t seq;
  uint32_t epoch;
  uint32_t scene_root;
  uint32_t ray_flags;
  uint32_t cull_mask;
  uint32_t warp_lane;  // lane | warp << 8 | slot << 16
  uint32_t origin[3];  // float bits
  uint32_t dir[3];
  uint32_t tmin;
  uint32_t tmax;
  uint32_t status;     // VX_RT_STS_DONE_HIT / DONE_MISS
  uint32_t hit_t;      // float bits (0 on a miss)
  uint32_t hit_u;
  uint32_t hit_v;
  uint32_t prim;
  uint32_t geom;       // geometry index | VX_RT_HIT_BACK_FACING
  uint32_t inst_id;
  uint32_t inst_custom;
  uint32_t hit_attr;
  uint32_t reserved[3];
};
static_assert(sizeof(RaylogRay) == 112, "RaylogRay layout");

bool enabled();

// The device memory the scene snapshots read from.
void attach_ram(const RAM* ram);

// A trace landed in `slot` of the RTU at `owner`: forget the callbacks the
// slot's previous trace raised, and snapshot any scene it is the first to name.
void on_accept(const void* owner, uint32_t slot, const RtuReq& req);

// A lane of `slot` yielded a callback of `cb_type`.
void on_callback(const void* owner, uint32_t slot, uint32_t lane, uint32_t cb_type);

// A walk completed against `lines`.
void on_walk_done(const std::unordered_map<uint64_t, LineBuf>& lines);

// `slot` emitted its terminal record.
void on_terminal(const void* owner, uint32_t slot, const RtuReq& req,
                 const std::array<LaneState, VX_CFG_NUM_THREADS>& lanes);

}}}  // namespace vortex::rtu::raylog

#endif  // _VX_RTU_RAYLOG_H_
