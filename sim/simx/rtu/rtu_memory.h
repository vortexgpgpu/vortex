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
// PRISM RTU — cluster-side memory-fetch engine (Layer 5 of the
// rtu_implementation.md refactor, Option C / 13 files).
//
// Owns the per-line fetch FSM and the in-flight tag table. Drives
// every dcache load that the RTU issues against the cluster's dcache
// ports. Reads slot lane state to know what to fetch; writes slot
// lane state when a response lands (filling l.line_data, advancing
// l.lines_filled, parsing the scene header on line 0, and gating
// AWAIT→COMPUTE transitions).
//
// Two-method interface, called once per tick by RtuCore::Impl::tick:
//
//   drain_mem_rsp() — drain dcache responses, fill lines, parse
//                     header on line 0, gate slot to COMPUTE when
//                     all active lanes are filled.
//   issue_memory()  — emit dcache loads for slots in ISSUE or AWAIT
//                     that still have unfetched lines.
//
// In SystemC: SC_MODULE(MemoryEngine) per cluster. The future
// §8.10 BvhCache becomes a nested SC_MODULE or a peer module —
// adding it here doesn't ripple through rtu_core.

#ifndef _VX_RTU_MEMORY_H_
#define _VX_RTU_MEMORY_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <simobject.h>
#include "types.h"            // MemReq, MemRsp, SimChannel

namespace vortex { namespace rtu {

struct Slot;
struct PerfStats;

class MemoryEngine {
public:
  MemoryEngine(std::vector<Slot>& slots,
               std::vector<SimChannel<MemReq>>& dcache_req,
               std::vector<SimChannel<MemRsp>>& dcache_rsp,
               PerfStats& perf)
    : slots_(slots), dcache_req_(dcache_req), dcache_rsp_(dcache_rsp),
      perf_(perf) {}

  // Called from RtuCore::Impl::reset(). Drops in-flight tags.
  void reset() {
    pending_.clear();
    next_tag_ = 0;
  }

  void issue_memory();
  void drain_mem_rsp();

private:
  struct PendingFill { uint32_t slot_idx; uint8_t lane; uint8_t line_idx; };

  std::vector<Slot>& slots_;
  std::vector<SimChannel<MemReq>>& dcache_req_;
  std::vector<SimChannel<MemRsp>>& dcache_rsp_;
  PerfStats& perf_;

  std::unordered_map<uint32_t, PendingFill> pending_;
  uint32_t next_tag_ = 0;
};

}}  // namespace vortex::rtu

#endif  // _VX_RTU_MEMORY_H_
