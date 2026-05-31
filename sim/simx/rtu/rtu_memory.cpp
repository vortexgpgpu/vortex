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

#include "rtu_memory.h"

#include <cstring>

#include "rtu_types.h"   // Slot, LaneState, SlotState, PerfStats,
                         // scene-format constants, lines_for_*,
                         // tlas_bytes, kRtuMaxInstancesPerTlas,
                         // kRtuMaxTrisPerScene, kRtuLineMask
#include "rtu_bvh.h"     // CW-BVH4 scene_kind constants (kRtuSceneKindBvh4)

namespace vortex { namespace rtu {

void MemoryEngine::issue_memory() {
  if (dcache_req_.empty()) return;
  auto& port = dcache_req_.at(0);
  for (auto& s : slots_) {
    if (!s.in_use) continue;
    if (s.state != SlotState::ISSUE && s.state != SlotState::AWAIT) continue;
    // Phase 4 multi-line fetch. Each active lane issues line 0 first;
    // once the header drains (drain_mem_rsp parses triangle_count and
    // sets lines_needed), body lines 1..lines_needed-1 are issued in
    // subsequent ticks. Stay in ISSUE while any active lane still has
    // work to schedule; otherwise drop to AWAIT until rsps drain.
    bool all_issued = true;
    for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
      auto& l = s.lanes[t];
      if (!l.active) continue;
      if (l.lines_issued >= l.lines_needed) continue;
      all_issued = false;
      if (port.full()) break;
      uint32_t line_idx = l.lines_issued;
      if (line_idx == 0) {
        // Cache-line-aligned header for the smoke-test scene layout.
        uint64_t addr = uint64_t(s.req.scene_root[t]);
        uint64_t line = addr & kRtuLineMask;
        l.line_byte_off = uint32_t(addr - line);
      }
      // Subsequent lines walk sequentially from line 0.
      uint64_t base_addr  = uint64_t(s.req.scene_root[t]) & kRtuLineMask;
      uint64_t line_addr  = base_addr + uint64_t(line_idx) * VX_CFG_MEM_BLOCK_SIZE;
      uint32_t tag = next_tag_++;
      MemReq m;
      m.addr    = line_addr;
      m.op      = MemOp::LD;
      m.tag     = tag;
      m.hart_id = 0;
      m.uuid    = s.req.uuid;
      port.send(m);
      pending_[tag] = PendingFill{ uint32_t(&s - &slots_[0]),
                                    uint8_t(t),
                                    uint8_t(line_idx) };
      l.line_issued[line_idx] = true;
      ++l.lines_issued;
      ++s.pending_mem;
      ++perf_.mem_reads;
      // Recompute all_issued on remaining lanes for next loop entry.
      all_issued = true;
      for (uint32_t u = 0; u < VX_CFG_NUM_THREADS; ++u) {
        const auto& ll = s.lanes[u];
        if (ll.active && ll.lines_issued < ll.lines_needed) {
          all_issued = false; break;
        }
      }
    }
    if (all_issued) {
      s.state = (s.pending_mem == 0) ? SlotState::COMPUTE : SlotState::AWAIT;
    } else if (s.state == SlotState::AWAIT) {
      // We're back in ISSUE because lines_needed grew after a header
      // drain — issue more next tick.
      s.state = SlotState::ISSUE;
    }
  }
}

void MemoryEngine::drain_mem_rsp() {
  for (auto& ch : dcache_rsp_) {
    while (!ch.empty()) {
      auto& rsp = ch.peek();
      auto it = pending_.find(uint32_t(rsp.tag));
      if (it == pending_.end()) {
        ch.pop();
        continue;
      }
      PendingFill pf = it->second;
      pending_.erase(it);
      Slot& s = slots_[pf.slot_idx];
      LaneState& l = s.lanes[pf.lane];
      if (rsp.data) {
        std::memcpy(l.line_data[pf.line_idx].data(), rsp.data->data(),
                    VX_CFG_MEM_BLOCK_SIZE);
      }
      l.line_filled[pf.line_idx] = true;
      ++l.lines_filled;
      if (s.pending_mem > 0) --s.pending_mem;

      // Phase 4 / 8: on the header line (line 0), parse the scene
      // header. The header layout is:
      //   uint32 primary_count;  // tris (TRI_LIST) or instances (TLAS)
      //   uint32 scene_kind;     // 0 = TRI_LIST, 1 = TLAS, 2 = BVH4
      //   uint32 reserved[2];
      if (pf.line_idx == 0 && !l.header_parsed) {
        uint32_t primary_count = 0;
        uint32_t scene_kind    = 0;
        const uint8_t* hdr     = l.line_data[0].data() + l.line_byte_off;
        std::memcpy(&primary_count, hdr + 0, sizeof(uint32_t));
        std::memcpy(&scene_kind,    hdr + 4, sizeof(uint32_t));
        l.scene_kind    = scene_kind;
        l.header_parsed = true;
        uint32_t needed = 1;
        if (scene_kind == kRtuSceneKindTlas) {
          if (primary_count > kRtuMaxInstancesPerTlas) {
            primary_count = kRtuMaxInstancesPerTlas;
          }
          l.instance_count = primary_count;
          needed = lines_for_bytes(l.line_byte_off, tlas_bytes(primary_count));
        } else if (scene_kind == kRtuSceneKindBvh4) {
          // Phase 4: VxBvhSceneHeader layout (see rtu_bvh.h). Pre-fetch
          // the entire BVH up to the per-lane line budget; the walker
          // reads from line_data synchronously via read_scene_bytes.
          // Chunk-3+ work may convert this to demand-fetch as scenes
          // grow past the line budget.
          l.bvh_root_offset = primary_count;
          l.triangle_count  = 0;
          l.instance_count  = 0;
          needed = kRtuMaxLinesPerLane;
        } else {
          if (primary_count > kRtuMaxTrisPerScene) primary_count = kRtuMaxTrisPerScene;
          l.triangle_count = primary_count;
          needed = lines_for_scene(l.line_byte_off, primary_count);
        }
        if (needed > kRtuMaxLinesPerLane) needed = kRtuMaxLinesPerLane;
        if (needed > l.lines_needed) {
          l.lines_needed = needed;
          // Drop slot back to ISSUE so the body lines get scheduled.
          s.state = SlotState::ISSUE;
        }
      }

      // Transition to compute only when every active lane has all its
      // lines filled. Cross-lane lines_needed can differ if scenes are
      // per-lane (Phase 3-A2 SBT smoke).
      if (s.pending_mem == 0 && s.state == SlotState::AWAIT) {
        bool all_done = true;
        for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
          const auto& ll = s.lanes[t];
          if (ll.active && ll.lines_filled < ll.lines_needed) {
            all_done = false; break;
          }
        }
        if (all_done) s.state = SlotState::COMPUTE;
      }
      ch.pop();
    }
  }
}

}}  // namespace vortex::rtu
