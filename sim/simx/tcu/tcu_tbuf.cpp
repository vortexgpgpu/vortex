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

#include "tcu_tbuf.h"
#include "constants.h"
#include "debug.h"
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <array>

using namespace vortex;

namespace {

constexpr uint64_t kLineMask = ~uint64_t(MEM_BLOCK_SIZE - 1);

// 2Q+1 sources fan into one external port. Source IDs:
//   0 .. NUM_TCU_BLOCKS-1                      → abuf[b]
//   NUM_TCU_BLOCKS .. 2*NUM_TCU_BLOCKS-1       → mbuf[b]
//   2*NUM_TCU_BLOCKS                           → bbuf
constexpr uint32_t kNumSources = 2 * NUM_TCU_BLOCKS + 1;
constexpr uint32_t kAOffset    = 0;
constexpr uint32_t kMOffset    = NUM_TCU_BLOCKS;
constexpr uint32_t kBOffset    = 2 * NUM_TCU_BLOCKS;

// Per-source line cache. Resident, in-flight and pending state are tracked
// independently per source; the wrapper arbitrates the shared LMEM port.
struct LineBuf {
  std::deque<uint64_t> pending_q_;
  std::unordered_map<uint32_t, uint64_t> inflight_;   // per-source tag → addr
  std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>> resident_;
  uint32_t next_tag_ = 0;
  uint64_t reads_ = 0;

  void plan(const std::vector<uint64_t>& line_addrs) {
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

  std::shared_ptr<mem_block_t> read(uint64_t line_addr) const {
    auto it = resident_.find(line_addr & kLineMask);
    if (it == resident_.end()) return nullptr;
    return it->second;
  }

  void invalidate() {
    resident_.clear();
  }

  void reset() {
    pending_q_.clear();
    inflight_.clear();
    resident_.clear();
    next_tag_ = 0;
    reads_ = 0;
  }
};

// Pack/unpack the source ID alongside the per-source tag in MemReq::tag.
constexpr uint32_t kSrcShift = 16;
constexpr uint32_t kSubTagMask = (1u << kSrcShift) - 1;

inline uint32_t pack_tag(uint32_t source, uint32_t sub_tag) {
  return (source << kSrcShift) | (sub_tag & kSubTagMask);
}
inline uint32_t unpack_source(uint32_t tag)  { return tag >> kSrcShift; }
inline uint32_t unpack_sub_tag(uint32_t tag) { return tag & kSubTagMask; }

} // namespace

class TcuTbuf::Impl {
public:
  Impl(TcuTbuf* simobject) : simobject_(simobject) {}

  void reset() {
    for (auto& b : bufs_) b.reset();
    rr_next_ = 0;
  }

  void plan(uint32_t source, const std::vector<uint64_t>& line_addrs) {
    bufs_.at(source).plan(line_addrs);
  }

  bool ready(uint32_t source) const {
    return bufs_.at(source).ready();
  }

  std::shared_ptr<mem_block_t> read(uint32_t source, uint64_t line_addr) const {
    return bufs_.at(source).read(line_addr);
  }

  void invalidate(uint32_t source) {
    bufs_.at(source).invalidate();
  }

  uint64_t reads() const {
    uint64_t total = 0;
    for (auto& b : bufs_) total += b.reads_;
    return total;
  }

  void tick() {
    // 1) drain one response and route it to the source that issued it.
    auto& rsp = simobject_->lmem_rsp_in;
    if (!rsp.empty()) {
      auto& r = rsp.peek();
      uint32_t source = unpack_source(r.tag);
      uint32_t sub_tag = unpack_sub_tag(r.tag);
      if (source < kNumSources) {
        auto& buf = bufs_.at(source);
        auto it = buf.inflight_.find(sub_tag);
        if (it != buf.inflight_.end()) {
          if (r.data) buf.resident_[it->second] = r.data;
          buf.inflight_.erase(it);
        }
      }
      rsp.pop();
    }

    // 2) round-robin pick one source with pending work and submit one req.
    auto& req = simobject_->lmem_req_out;
    if (req.full()) return;
    for (uint32_t i = 0; i < kNumSources; ++i) {
      uint32_t s = (rr_next_ + i) % kNumSources;
      auto& buf = bufs_.at(s);
      if (buf.pending_q_.empty()) continue;
      uint64_t addr = buf.pending_q_.front();
      uint32_t sub_tag = buf.next_tag_++;
      uint32_t tag = pack_tag(s, sub_tag);
      MemReq m(MemOp::LD, addr, /*data*/nullptr, /*byteen*/0, tag, /*hart_id*/0, /*uuid*/0);
      m.flags.local = 1;   // TCU TBUF reads from LMEM
      req.send(m, 1);
      buf.inflight_[sub_tag] = addr;
      buf.pending_q_.pop_front();
      ++buf.reads_;
      rr_next_ = (s + 1) % kNumSources;
      break;
    }
  }

private:
  TcuTbuf* simobject_;
  std::array<LineBuf, kNumSources> bufs_;
  uint32_t rr_next_ = 0;
};

TcuTbuf::TcuTbuf(const SimContext& ctx, const char* name)
  : SimObject<TcuTbuf>(ctx, name)
  , lmem_req_out(this)
  , lmem_rsp_in(this)
  , impl_(new Impl(this))
{}

TcuTbuf::~TcuTbuf() { delete impl_; }

void TcuTbuf::on_reset() { impl_->reset(); }
void TcuTbuf::on_tick()  { impl_->tick(); }

void TcuTbuf::plan_a(uint32_t b, const std::vector<uint64_t>& line_addrs) {
  impl_->plan(kAOffset + b, line_addrs);
}
void TcuTbuf::plan_m(uint32_t b, const std::vector<uint64_t>& line_addrs) {
  impl_->plan(kMOffset + b, line_addrs);
}
void TcuTbuf::plan_b(const std::vector<uint64_t>& line_addrs) {
  impl_->plan(kBOffset, line_addrs);
}

bool TcuTbuf::ready_a(uint32_t b) const { return impl_->ready(kAOffset + b); }
bool TcuTbuf::ready_m(uint32_t b) const { return impl_->ready(kMOffset + b); }
bool TcuTbuf::ready_b() const           { return impl_->ready(kBOffset); }

std::shared_ptr<mem_block_t> TcuTbuf::read_a(uint32_t b, uint64_t line_addr) const {
  return impl_->read(kAOffset + b, line_addr);
}
std::shared_ptr<mem_block_t> TcuTbuf::read_m(uint32_t b, uint64_t line_addr) const {
  return impl_->read(kMOffset + b, line_addr);
}
std::shared_ptr<mem_block_t> TcuTbuf::read_b(uint64_t line_addr) const {
  return impl_->read(kBOffset, line_addr);
}

void TcuTbuf::invalidate_a(uint32_t b) { impl_->invalidate(kAOffset + b); }
void TcuTbuf::invalidate_m(uint32_t b) { impl_->invalidate(kMOffset + b); }
void TcuTbuf::invalidate_b()           { impl_->invalidate(kBOffset); }

uint64_t TcuTbuf::reads() const { return impl_->reads(); }
