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

#include "tma_engine.h"

#ifdef EXT_TMA_ENABLE

#include <algorithm>
#include <limits>
#include "core.h"

namespace vortex {

namespace {

constexpr uint32_t kDecodeCycles = 2;
constexpr uint32_t kElemIssueCycles = 1;
constexpr uint32_t kGmemReadRspCycles = 8;
constexpr uint32_t kSmemReadRspCycles = 2;
constexpr uint32_t kGmemWriteAckCycles = 6;
constexpr uint32_t kSmemWriteAckCycles = 2;
constexpr uint32_t kCompletionCycles = 1;

inline uint64_t ceil_div(uint64_t n, uint64_t d) {
  return (n + d - 1) / d;
}

} // namespace

TmaEngine::TmaEngine(Core* core)
  : core_(core)
  , has_active_(false) {}

void TmaEngine::reset() {
  queue_.clear();
  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

bool TmaEngine::issue(uint32_t desc_slot,
                      uint32_t smem_addr,
                      const uint32_t coords[5],
                      uint32_t flags,
                      uint32_t bar_id) {
  if (queue_.size() >= kQueueDepth) {
    return false;
  }

  Request req;
  req.desc_slot = desc_slot;
  req.smem_addr = smem_addr;
  req.flags = flags;
  req.bar_id = bar_id;
  for (uint32_t i = 0; i < req.coords.size(); ++i) {
    req.coords.at(i) = coords[i];
  }
  queue_.push_back(req);
  return true;
}

void TmaEngine::tick() {
  if (!has_active_) {
    if (!this->start_next_request()) {
      return;
    }
  }
  this->progress_active_request();
}

bool TmaEngine::start_next_request() {
  if (queue_.empty()) {
    return false;
  }

  auto req = queue_.front();
  queue_.pop_front();

  uint32_t total_elems = 0;
  uint32_t elem_bytes = 0;
  uint32_t total_cycles = 0;
  if (!this->decode_request(req, &total_elems, &elem_bytes, &total_cycles)) {
    core_->barrier_tx_done(req.bar_id);
    return false;
  }

  active_xfer_.req = req;
  active_xfer_.total_elems = total_elems;
  active_xfer_.elem_bytes = elem_bytes;
  active_xfer_.cycles_left = std::max<uint32_t>(1, total_cycles);
  has_active_ = true;
  return true;
}

bool TmaEngine::decode_request(const Request& req,
                               uint32_t* total_elems,
                               uint32_t* elem_bytes,
                               uint32_t* total_cycles) const {
  uint32_t desc_elems = 0;
  uint32_t desc_elem_bytes = 0;
  if (!core_->tma_estimate(req.desc_slot, req.flags, &desc_elems, &desc_elem_bytes)) {
    return false;
  }

  uint64_t bytes = uint64_t(desc_elems) * desc_elem_bytes;
  bool is_s2g = ((req.flags & 0x1u) != 0);
  uint32_t read_rsp_cycles = is_s2g ? kSmemReadRspCycles : kGmemReadRspCycles;
  uint32_t write_ack_cycles = is_s2g ? kGmemWriteAckCycles : kSmemWriteAckCycles;
  uint64_t per_elem_cycles = uint64_t(kElemIssueCycles + read_rsp_cycles + write_ack_cycles);
  uint64_t xfer_cycles = per_elem_cycles * desc_elems;

  // A coarse bus serialization term to account for crossbar/arb pressure.
  uint32_t bus_quantum = std::max<uint32_t>(1, std::min<uint32_t>(L2_LINE_SIZE, LSU_WORD_SIZE));
  uint64_t bus_cycles = ceil_div(bytes, bus_quantum);

  uint64_t total = uint64_t(kDecodeCycles) + xfer_cycles + bus_cycles + kCompletionCycles;

  if (total_elems) {
    *total_elems = desc_elems;
  }
  if (elem_bytes) {
    *elem_bytes = desc_elem_bytes;
  }
  if (total_cycles) {
    *total_cycles = uint32_t(std::min<uint64_t>(total, std::numeric_limits<uint32_t>::max()));
  }
  return true;
}

void TmaEngine::progress_active_request() {
  if (!has_active_) {
    return;
  }

  if (active_xfer_.cycles_left > 1) {
    --active_xfer_.cycles_left;
    return;
  }

  this->complete_active_request();
}

void TmaEngine::complete_active_request() {
  uint32_t bytes_copied = 0;
  (void)core_->tma_copy(active_xfer_.req.desc_slot,
                        active_xfer_.req.smem_addr,
                        active_xfer_.req.coords.data(),
                        active_xfer_.req.flags,
                        &bytes_copied);
  (void)bytes_copied;
  core_->barrier_tx_done(active_xfer_.req.bar_id);

  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

} // namespace vortex

#endif // EXT_TMA_ENABLE

