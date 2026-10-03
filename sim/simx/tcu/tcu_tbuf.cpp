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
#include <deque>
#include <array>

using namespace vortex;

namespace {

constexpr uint64_t kLineMask = ~uint64_t(VX_CFG_MEM_BLOCK_SIZE - 1);

// Q+1 buffers share one LMEM port. Buffer IDs, in priority order:
//   0 .. VX_CFG_NUM_TCU_BLOCKS-1   → abuf[b]
//   VX_CFG_NUM_TCU_BLOCKS          → bbuf
constexpr uint32_t kNumBuffers = VX_CFG_NUM_TCU_BLOCKS + 1;
constexpr uint32_t kAOffset    = 0;
constexpr uint32_t kBOffset    = VX_CFG_NUM_TCU_BLOCKS;

struct Buffer {
  std::deque<TcuTbuf::LmemRead> pending_;
  std::unordered_map<uint32_t, uint64_t> inflight_;   // tag → line
  std::unordered_map<uint64_t, std::shared_ptr<mem_block_t>> lines_;
  uint64_t fill_cycle_ = 0;
  bool     rsp_seen_ = false;   // a response arrived this cycle
  uint32_t next_tag_ = 0;
  uint64_t reads_ = 0;

  bool filling() const {
    return !pending_.empty() || !inflight_.empty();
  }

  void fill(std::vector<TcuTbuf::LmemRead>&& reads) {
    this->invalidate();
    for (auto& r : reads) {
      pending_.push_back(std::move(r));
    }
    fill_cycle_ = SimPlatform::instance().cycles();
  }

  std::shared_ptr<mem_block_t> read(uint64_t line_addr) const {
    auto it = lines_.find(line_addr & kLineMask);
    if (it == lines_.end()) {
      return nullptr;
    }
    return it->second;
  }

  // A response still in flight for an abandoned fill finds no tag and is dropped.
  void invalidate() {
    pending_.clear();
    inflight_.clear();
    lines_.clear();
  }

  void reset() {
    this->invalidate();
    rsp_seen_ = false;
    next_tag_ = 0;
    reads_ = 0;
  }
};

// Pack the buffer ID alongside the per-buffer tag in MemReq::tag.
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
    for (auto& b : bufs_) {
      b.reset();
    }
  }

  Buffer& buf(uint32_t source) {
    return bufs_.at(source);
  }

  const Buffer& buf(uint32_t source) const {
    return bufs_.at(source);
  }

  uint64_t reads() const {
    uint64_t total = 0;
    for (auto& b : bufs_) {
      total += b.reads_;
    }
    return total;
  }

  void tick() {
    for (auto& b : bufs_) {
      b.rsp_seen_ = false;
    }

    for (auto& rsp : simobject_->lmem_rsp_in) {
      if (rsp.empty()) {
        continue;
      }
      auto& r = rsp.peek();
      uint32_t source = unpack_source(r.tag);
      if (source < kNumBuffers) {
        auto& b = bufs_.at(source);
        auto it = b.inflight_.find(unpack_sub_tag(r.tag));
        if (it != b.inflight_.end()) {
          if (r.data) {
            b.lines_[it->second] = r.data;
          }
          b.inflight_.erase(it);
          b.rsp_seen_ = true;
          if (!b.filling()) {
            DT(3, simobject_->name() << " " << this->buf_name(source) << ": READY");
          }
        }
      }
      rsp.pop();
    }

    // An abuf issues its next read in the cycle its previous one returns;
    // the bbuf issues it the cycle after.
    uint64_t cycle = SimPlatform::instance().cycles();
    for (uint32_t s = 0; s < kNumBuffers; ++s) {
      auto& b = bufs_.at(s);
      if (b.pending_.empty() || !b.inflight_.empty()) {
        continue;
      }
      if (cycle <= b.fill_cycle_) {
        continue;
      }
      if (s == kBOffset && b.rsp_seen_) {
        continue;
      }
      this->issue(s, b);
      break;
    }
  }

private:
  void issue(uint32_t source, Buffer& b) {
    auto& lines = b.pending_.front();
    uint32_t n = std::min<uint32_t>(lines.size(), LMEM_PORTS);
    for (uint32_t p = 0; p < n; ++p) {
      if (simobject_->lmem_req_out.at(p).full()) {
        return;
      }
    }
    for (uint32_t p = 0; p < n; ++p) {
      uint64_t addr = lines.at(p);
      // inflight_ is keyed by the tag as it appears on the wire, so the
      // counter must be masked here and not only inside pack_tag().
      uint32_t sub_tag = (b.next_tag_++) & kSubTagMask;
      MemReq m(MemOp::LD, addr, /*data*/nullptr, /*byteen*/0,
               pack_tag(source, sub_tag), /*hart_id*/0, /*uuid*/0);
      m.flags.local = 1;
      simobject_->lmem_req_out.at(p).send(m, 1);
      b.inflight_[sub_tag] = addr;
    }
    DT(3, simobject_->name() << " " << this->buf_name(source)
          << ": rd_req addr=0x" << std::hex << lines.front() << std::dec
          << ", lines=" << n);
    // A read wider than the port set finishes as a further read.
    if (n < lines.size()) {
      lines.erase(lines.begin(), lines.begin() + n);
    } else {
      b.pending_.pop_front();
    }
    ++b.reads_;
  }

  std::string buf_name(uint32_t source) const {
    return (source == kBOffset) ? std::string("bbuf")
                                : ("abuf" + std::to_string(source - kAOffset));
  }

  TcuTbuf* simobject_;
  std::array<Buffer, kNumBuffers> bufs_;
};

TcuTbuf::TcuTbuf(const SimContext& ctx, const char* name)
  : SimObject<TcuTbuf>(ctx, name)
  , lmem_req_out(LMEM_PORTS, this)
  , lmem_rsp_in(LMEM_PORTS, this)
  , impl_(new Impl(this))
{}

TcuTbuf::~TcuTbuf() { delete impl_; }

void TcuTbuf::on_reset() { impl_->reset(); }
void TcuTbuf::on_tick()  { impl_->tick(); }

void TcuTbuf::fill_a(uint32_t b, std::vector<LmemRead> reads) {
  impl_->buf(kAOffset + b).fill(std::move(reads));
}
void TcuTbuf::fill_b(std::vector<LmemRead> reads) {
  impl_->buf(kBOffset).fill(std::move(reads));
}

bool TcuTbuf::filling_a(uint32_t b) const { return impl_->buf(kAOffset + b).filling(); }
bool TcuTbuf::filling_b() const           { return impl_->buf(kBOffset).filling(); }

std::shared_ptr<mem_block_t> TcuTbuf::read_a(uint32_t b, uint64_t line_addr) const {
  return impl_->buf(kAOffset + b).read(line_addr);
}
std::shared_ptr<mem_block_t> TcuTbuf::read_b(uint64_t line_addr) const {
  return impl_->buf(kBOffset).read(line_addr);
}

void TcuTbuf::invalidate_a(uint32_t b) { impl_->buf(kAOffset + b).invalidate(); }
void TcuTbuf::invalidate_b()           { impl_->buf(kBOffset).invalidate(); }

uint64_t TcuTbuf::reads() const { return impl_->reads(); }
