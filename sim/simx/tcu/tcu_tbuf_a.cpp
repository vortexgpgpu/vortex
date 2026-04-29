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

#include "tcu_tbuf_a.h"
#include "constants.h"
#include "debug.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>

using namespace vortex;

namespace {
constexpr uint64_t kLineMask = ~uint64_t(MEM_BLOCK_SIZE - 1);
}

class TcuTbufA::Impl {
public:
  Impl(TcuTbufA* simobject)
    : simobject_(simobject)
  {}

  void reset() {
    pending_q_.clear();
    inflight_.clear();
    resident_.clear();
    next_tag_ = 0;
    reads_ = 0;
  }

  void plan(const std::vector<uint64_t>& line_addrs) {
    // Phase B: additive plan. Requests any not-resident, not-in-flight line.
    // Eviction is explicit via `invalidate()`. Phase C will introduce
    // working-set eviction policies (k-stripe rollover, etc).
    std::unordered_set<uint64_t> inflight_set;
    for (auto& kv : inflight_) inflight_set.insert(kv.second);
    for (auto a : line_addrs) {
      uint64_t line = a & kLineMask;
      if (resident_.count(line)) continue;
      if (inflight_set.count(line)) continue;
      pending_q_.push_back(line);
      inflight_set.insert(line);
    }
  }

  bool ready() const {
    return pending_q_.empty() && inflight_.empty();
  }

  std::shared_ptr<mem_block_t> read_line(uint64_t line_addr) const {
    auto it = resident_.find(line_addr & kLineMask);
    if (it == resident_.end()) return nullptr;
    return it->second;
  }

  void invalidate() {
    resident_.clear();
  }

  uint64_t reads() const { return reads_; }

  void tick() {
    // 1) drain responses (one per tick keeps things deterministic and
    //    matches a single bank-row port; in-flight count limited by the
    //    channel capacity, not by an MSHR).
    auto& rsp = simobject_->lmem_rsp_in;
    if (!rsp.empty()) {
      auto& r = rsp.peek();
      auto it = inflight_.find(r.tag);
      if (it != inflight_.end()) {
        if (r.data) {
          resident_[it->second] = r.data;
        }
        inflight_.erase(it);
      }
      rsp.pop();
    }

    // 2) submit one pending request if the outgoing channel has room.
    auto& req = simobject_->lmem_req_out;
    if (!pending_q_.empty() && !req.full()) {
      uint64_t addr = pending_q_.front();
      uint32_t tag = next_tag_++;
      MemReq r(addr, /*write*/false, AddrType::Shared, tag, /*cid*/0, /*uuid*/0);
      req.send(r, 1);
      inflight_[tag] = addr;
      pending_q_.pop_front();
      ++reads_;
    }
  }

private:
  TcuTbufA* simobject_;
  std::deque<uint64_t> pending_q_;
  std::unordered_map<uint32_t, uint64_t> inflight_;
  std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>> resident_;
  uint32_t next_tag_ = 0;
  uint64_t reads_ = 0;
};

TcuTbufA::TcuTbufA(const SimContext& ctx, const char* name)
  : SimObject<TcuTbufA>(ctx, name)
  , lmem_req_out(this)
  , lmem_rsp_in(this)
  , impl_(new Impl(this))
{}

TcuTbufA::~TcuTbufA() { delete impl_; }

void TcuTbufA::on_reset() { impl_->reset(); }
void TcuTbufA::on_tick()  { impl_->tick(); }

void TcuTbufA::plan(const std::vector<uint64_t>& line_addrs) {
  impl_->plan(line_addrs);
}

bool TcuTbufA::ready() const { return impl_->ready(); }

std::shared_ptr<mem_block_t> TcuTbufA::read_line(uint64_t line_addr) const {
  return impl_->read_line(line_addr);
}

void TcuTbufA::invalidate() { impl_->invalidate(); }

uint64_t TcuTbufA::reads() const { return impl_->reads(); }
