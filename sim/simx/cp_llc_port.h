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

#pragma once

#include <simobject.h>
#include <cstring>
#include <deque>
#include <vector>
#include "types.h"
#include "constants.h"
#include "mem_block_pool.h"
#include "debug.h"

namespace vortex {

// The command processor's DMA port into the last-level cache's client slot.
// Requests in the IO aperture carry the IO attribute and bypass the cache; a
// write flagged no-allocate writes around it on a miss. A launch that follows
// DMA writes first sends an L1-only flush token behind them, which the
// last-level cache answers once its inputs have drained.
class CpLlcPort : public SimObject<CpLlcPort> {
public:
  using Ptr = std::shared_ptr<CpLlcPort>;

  SimChannel<MemReq> MemReqOut;
  SimChannel<MemRsp> MemRspIn;

  // one 4 KB DMA burst of reads in flight
  static constexpr uint32_t kReadSlots = 4096 / CP_MEM_DATA_SIZE;

  CpLlcPort(const SimContext& ctx, const char* name)
    : SimObject<CpLlcPort>(ctx, name)
    , MemReqOut(this)
    , MemRspIn(this)
    , slots_(kReadSlots)
  {}

  // Queue one LLC word of a transfer; `src` is null for a read into `dst`.
  void push(uint64_t addr, const uint8_t* src, uint8_t* dst, uint32_t offset,
            uint32_t size, bool noalloc) {
    MemReq req;
    req.addr = addr;
    req.op = src ? MemOp::ST : MemOp::LD;
    req.flags.io = (get_addr_type(addr) == AddrType::IO);
    req.flags.noalloc = src && noalloc;
    // the word's bytes sit at its offset in the memory block
    uint32_t boff = uint32_t(addr % VX_CFG_MEM_BLOCK_SIZE) + offset;
    if (src) {
      req.data = make_mem_block();
      std::memset(req.data->data(), 0, req.data->size());
      std::memcpy(req.data->data() + boff, src, size);
      req.byteen = ((size < 64) ? ((1ull << size) - 1) : ~0ull) << boff;
    } else {
      uint32_t woff = uint32_t(addr % VX_CFG_MEM_BLOCK_SIZE);
      req.byteen = ((CP_MEM_DATA_SIZE < 64) ? ((1ull << CP_MEM_DATA_SIZE) - 1) : ~0ull) << woff;
    }
    queue_.push_back({req, dst, boff, size});
    this->tick_wake();
  }

  // Every queued request has entered the port, and every read has returned.
  bool idle() const {
    return queue_.empty() && reads_ == 0 && !fencing_;
  }

  // DMA writes entered the port since the last fence.
  bool writes_pending() const {
    return wr_pending_;
  }

  // Send the launch fence behind the pending writes.
  void fence() {
    fencing_ = true;
    fence_sent_ = false;
    this->tick_wake();
  }

protected:
  void on_reset() {
    queue_.clear();
    free_slots_.clear();
    for (uint32_t i = 0; i < kReadSlots; ++i) {
      free_slots_.push_back(kReadSlots - 1 - i);
    }
    reads_ = 0;
    wr_pending_ = false;
    fencing_ = false;
    fence_sent_ = false;
  }

  void on_tick() {
    if (!MemRspIn.empty()) {
      auto& rsp = MemRspIn.peek();
      DT(3, this->name() << " rsp: " << rsp);
      if (rsp.tag == kFenceTag) {
        fencing_ = false;
        wr_pending_ = false;
      } else {
        uint32_t slot = rsp.tag >> 1;
        auto& rd = slots_.at(slot);
        std::memcpy(rd.dst, rsp.data->data() + rd.offset, rd.size);
        free_slots_.push_back(slot);
        --reads_;
      }
      MemRspIn.pop();
    }
    if (fencing_) {
      if (!fence_sent_) {
        MemReq token;
        token.flags.flush = 1;
        token.flags.flush_l1 = 1;
        token.tag = kFenceTag;
        if (MemReqOut.try_send(token, 1)) {
          DT(3, this->name() << " fence: " << token);
          fence_sent_ = true;
        }
      }
      return;
    }
    if (queue_.empty()) {
      if (reads_ == 0 && MemRspIn.empty()) {
        this->tick_sleep();
      }
      return;
    }
    auto& entry = queue_.front();
    bool is_read = !entry.req.is_write();
    if (is_read && free_slots_.empty()) {
      return;
    }
    MemReq req = entry.req;
    uint32_t slot = 0;
    if (is_read) {
      slot = free_slots_.back();
      req.tag = slot << 1;
    } else {
      req.tag = 0;
    }
    if (!MemReqOut.try_send(req, 1)) {
      return;
    }
    DT(3, this->name() << " req: " << req);
    if (is_read) {
      free_slots_.pop_back();
      slots_.at(slot) = {entry.dst, entry.offset, entry.size};
      ++reads_;
    } else {
      wr_pending_ = true;
    }
    queue_.pop_front();
  }

private:
  static constexpr uint32_t kFenceTag = 1;

  struct entry_t {
    MemReq   req;
    uint8_t* dst;
    uint32_t offset;
    uint32_t size;
  };

  struct read_slot_t {
    uint8_t* dst = nullptr;
    uint32_t offset = 0;
    uint32_t size = 0;
  };

  std::deque<entry_t>      queue_;
  std::vector<read_slot_t> slots_;
  std::vector<uint32_t>    free_slots_;
  uint32_t reads_ = 0;
  bool wr_pending_ = false;
  bool fencing_ = false;
  bool fence_sent_ = false;

  friend class SimObject<CpLlcPort>;
};

}
