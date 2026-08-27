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
// PRISM RTU — the traversal front end's memory stage.
//
// Contexts discover the node they need one step at a time and park in REQ. This
// engine turns that into cache traffic:
//
//   SELECT — pick one context (round-robin, so no context starves).
//   MSHR   — allocate an entry for the line and issue exactly one load.
//
// Duplicate fetches are not coalesced: one context, one tag, one fetch. A
// coherent warp descending a shared tree therefore re-requests each node once
// per ray, and the caches absorb it.
//
// The response side fans one line out to every context waiting on its entry.

#ifndef _VX_RTU_MEMORY_H_
#define _VX_RTU_MEMORY_H_

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "types.h"            // MemReq, MemRsp, SimChannel

namespace vortex { namespace rtu {

struct Context;
struct PerfStats;

class MemoryEngine {
public:
  MemoryEngine(std::vector<Context>& contexts,
               std::vector<SimChannel<MemReq>>& dcache_req,
               std::vector<SimChannel<MemRsp>>& dcache_rsp,
               PerfStats& perf)
    : contexts_(contexts), dcache_req_(dcache_req), dcache_rsp_(dcache_rsp),
      perf_(perf) {}

  void reset();

  // SELECT -> MSHR -> issue. At most one distinct line leaves the RTU
  // per port per tick.
  void issue_memory();

  // Drain one response per port; fan the line out to every waiting context.
  void drain_mem_rsp();

private:
  // One in-flight line fetch and the contexts riding on it — without
  // coalescing, always the single context that requested it.
  struct Mshr {
    bool     valid = false;
    uint64_t addr  = 0;
    std::vector<uint32_t> waiters;
  };

  std::vector<Context>& contexts_;
  std::vector<SimChannel<MemReq>>& dcache_req_;
  std::vector<SimChannel<MemRsp>>& dcache_rsp_;
  PerfStats& perf_;

  std::vector<Mshr>  mshr_;
  // Tag -> MSHR index. The tag is what comes back on the response; with merging
  // disabled each fetch still gets an entry, it just never gains a second
  // waiter, so one code path serves both.
  std::unordered_map<uint32_t, uint32_t> pending_;
  uint32_t next_tag_ = 0;
  uint32_t rr_ctx_   = 0;   // SELECT rotation
};

}}  // namespace vortex::rtu

#endif  // _VX_RTU_MEMORY_H_
