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

#include <cassert>
#include <cstring>

#include "rtu_types.h"   // Context, CtxState, PerfStats

namespace vortex { namespace rtu {

namespace {
// Every fetch takes an entry and never gains a second waiter. Sized to the
// context array: each context has at most one fetch outstanding, so the table
// can never fill and so can never cap memory-level parallelism.
constexpr uint32_t kMshrEntries = VX_CFG_RTU_NUM_CTX;
}  // namespace

void MemoryEngine::reset() {
  mshr_.assign(kMshrEntries, Mshr{});
  pending_.clear();
  next_tag_ = 0;
  rr_ctx_   = 0;
}

void MemoryEngine::issue_memory() {
  if (mshr_.empty()) mshr_.assign(kMshrEntries, Mshr{});
  const uint32_t nctx = uint32_t(contexts_.size());
  bool issued_this_tick = false;

  for (auto& port : dcache_req_) {
    if (port.full()) continue;

    // SELECT — round-robin so a context that keeps losing eventually wins.
    uint32_t leader = nctx;
    for (uint32_t i = 0; i < nctx; ++i) {
      uint32_t idx = (rr_ctx_ + i) % nctx;
      if (contexts_[idx].state == CtxState::REQ) { leader = idx; break; }
    }
    if (leader == nctx) break;   // nothing wants a line

    const uint64_t addr = contexts_[leader].req_addr;
    rr_ctx_ = (leader + 1) % nctx;

    // The selected context is the whole group: concurrent requests for the same
    // line are not coalesced, so a fetch never gains a second waiter.
    std::vector<uint32_t> group{leader};

    // Allocate an entry and issue exactly one load for the whole group.
    uint32_t eidx = kMshrEntries;
    for (uint32_t i = 0; i < mshr_.size(); ++i) {
      if (!mshr_[i].valid) { eidx = i; break; }
    }
    if (eidx == kMshrEntries) {
      // Table full: the group stays in REQ and retries. Back-pressure, not a
      // correctness issue — but it caps memory-level parallelism, which is the
      // whole cost of a shallow table.
      ++perf_.mshr_full_stalls;
      break;
    }

    Mshr& e = mshr_[eidx];
    e.valid = true;
    e.addr  = addr;
    e.waiters = group;

    uint32_t tag = next_tag_++;
    pending_[tag] = eidx;

    MemReq m;
    m.addr    = addr;
    m.op      = MemOp::LD;
    m.tag     = tag;
    m.hart_id = 0;
    m.uuid    = 0;
    port.send(m);

    for (uint32_t c : group) contexts_[c].state = CtxState::WAIT;
    ++perf_.mem_reads;
    issued_this_tick = true;
  }

  if (issued_this_tick) ++perf_.mem_issue_ticks;
}

void MemoryEngine::drain_mem_rsp() {
  // One response per port per cycle: the fill path is a single-line-wide drain,
  // and an unbounded drain would hide the memory pipeline's cost entirely.
  for (auto& ch : dcache_rsp_) {
    if (ch.empty()) continue;
    auto& rsp = ch.peek();
    auto it = pending_.find(uint32_t(rsp.tag));
    if (it == pending_.end()) {
      ch.pop();
      continue;
    }
    Mshr& e = mshr_.at(it->second);
    pending_.erase(it);

    for (uint32_t c : e.waiters) {
      Context& cx = contexts_.at(c);
      // A retired context waiting on a line means the group was formed from
      // something other than a live request, which selecting only a context in
      // REQ is meant to make impossible.
      assert(cx.state != CtxState::DONE && "RTU: retired context in a fill group");
      if (!cx.valid || cx.state != CtxState::WAIT) continue;
      if (rsp.data) {
        std::memcpy(cx.lines[e.addr].data(), rsp.data->data(),
                    VX_CFG_MEM_BLOCK_SIZE);
      }
      cx.state = CtxState::PROBE;
    }
    e.valid = false;
    e.waiters.clear();
    ch.pop();
  }
}

}}  // namespace vortex::rtu
