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

#include "dxa_engine.h"

#ifdef EXT_DXA_ENABLE

#include <algorithm>
#include <limits>
#include "core.h"

namespace vortex {

namespace {

// ── Timing constants ────────────────────────────────────────────────
// These are approximate cycle costs used by the SimX model.  They do
// NOT need to be cycle-exact w.r.t. RTL; they just need to produce a
// similar-enough completion time for barrier-release ordering.

constexpr uint32_t kDecodeCycles        = 2;   // FSM decode + desc lookup
constexpr uint32_t kCompletionCycles    = 1;   // final done signaling

// ── g2s pipeline constants (next-gen NB worker) ─────────────────────
// AG produces 1 address per cycle.  RRS issues gmem reads with up to
// kMaxOutstanding concurrent requests.  The single-entry line-cache
// means consecutive elements on the same L2 line are cache hits (no
// gmem read needed).
constexpr uint32_t kMaxOutstanding      = 8;   // matches RTL MAX_OUTSTANDING
constexpr uint32_t kGmemReadLatency     = 8;   // L2 cache-line read latency

// ── s2g serial constants (legacy path, not pipelined yet) ───────────
constexpr uint32_t kSmemReadCycles      = 2;
constexpr uint32_t kGmemWriteCycles     = 6;
constexpr uint32_t kElemIssueCycles     = 1;

inline uint64_t ceil_div(uint64_t n, uint64_t d) {
  return (n + d - 1) / d;
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// DxaEngine implementation
// ════════════════════════════════════════════════════════════════════

DxaEngine::DxaEngine(Core* core)
  : core_(core)
  , has_active_(false) {}

void DxaEngine::reset() {
  queue_.clear();
  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

bool DxaEngine::issue(uint32_t desc_slot,
                      uint32_t smem_addr,
                      const uint32_t coords[5],
                      uint32_t bar_id) {
  if (queue_.size() >= kQueueDepth) {
    return false;
  }

  Request req;
  req.desc_slot = desc_slot;
  req.smem_addr = smem_addr;
  req.bar_id = bar_id;
  for (uint32_t i = 0; i < req.coords.size(); ++i) {
    req.coords.at(i) = coords[i];
  }
  queue_.push_back(req);
  return true;
}

void DxaEngine::tick() {
  if (!has_active_) {
    if (!this->start_next_request()) {
      return;
    }
  }
  this->progress_active_request();
}

bool DxaEngine::start_next_request() {
  if (queue_.empty()) {
    return false;
  }

  auto req = queue_.front();
  queue_.pop_front();

  uint32_t total_elems = 0;
  uint32_t elem_bytes = 0;
  uint32_t total_cycles = 0;
  if (!this->decode_request(req, &total_elems, &elem_bytes, &total_cycles)) {
    core_->barrier_event_release(req.bar_id);
    return false;
  }

  active_xfer_.req = req;
  active_xfer_.total_elems = total_elems;
  active_xfer_.elem_bytes = elem_bytes;
  active_xfer_.cycles_left = std::max<uint32_t>(1, total_cycles);
  has_active_ = true;
  return true;
}

bool DxaEngine::decode_request(const Request& req,
                               uint32_t* total_elems,
                               uint32_t* elem_bytes,
                               uint32_t* total_cycles) const {
  uint32_t desc_elems = 0;
  uint32_t desc_elem_bytes = 0;
  if (!core_->dxa_estimate(req.desc_slot, &desc_elems, &desc_elem_bytes)) {
    return false;
  }

  bool is_s2g = false;  // Always g2s; s2g removed with flags
  uint64_t total = 0;

  if (is_s2g) {
    // s2g: serial per-element model (nextgen RTL currently does instant
    // completion for s2g, but we still model a small latency so that the
    // barrier fires a few cycles later, matching RTL's instant-done +
    // barrier propagation delay).
    uint64_t per_elem = uint64_t(kElemIssueCycles + kSmemReadCycles + kGmemWriteCycles);
    total = kDecodeCycles + per_elem * desc_elems + kCompletionCycles;
  } else {
    // g2s: pipelined model — mirrors the AG→RRS→WBC pipeline.
    //
    // AG issues 1 element/cycle.  The single-entry line-cache absorbs
    // consecutive accesses to the same L2 line, so the number of gmem
    // reads (cache misses) ≈ ceil(total_bytes / L2_LINE_SIZE).  With
    // kMaxOutstanding concurrent reads, the pipeline stalls only when
    // all slots are occupied.
    //
    // Timing breakdown:
    //   ag_cycles     = desc_elems      (1 elem / cycle)
    //   stall_cycles  = excess_misses × ceil(read_lat / max_out)
    //   drain_cycles  = kGmemReadLatency (wait for last response)
    //
    // total = decode + ag_cycles + stall_cycles + drain + completion

    uint64_t total_bytes = uint64_t(desc_elems) * desc_elem_bytes;
    uint32_t line_size = std::max<uint32_t>(1, L2_LINE_SIZE);
    uint64_t num_misses = ceil_div(total_bytes, line_size);

    uint64_t ag_cycles = desc_elems;

    // Stall happens when AG has issued more misses than kMaxOutstanding
    // before any slots free up.  Each "batch" of kMaxOutstanding reads
    // takes kGmemReadLatency cycles to drain.
    uint64_t excess = (num_misses > kMaxOutstanding) ? (num_misses - kMaxOutstanding) : 0;
    uint64_t stall_cycles = excess * ceil_div(kGmemReadLatency, kMaxOutstanding);

    // After AG finishes, the last outstanding reads take up to
    // kGmemReadLatency cycles to return.
    uint64_t drain_cycles = kGmemReadLatency;

    total = kDecodeCycles + ag_cycles + stall_cycles + drain_cycles + kCompletionCycles;
  }

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

void DxaEngine::progress_active_request() {
  if (!has_active_) {
    return;
  }

  if (active_xfer_.cycles_left > 1) {
    --active_xfer_.cycles_left;
    return;
  }

  this->complete_active_request();
}

void DxaEngine::complete_active_request() {
  uint32_t bytes_copied = 0;
  (void)core_->dxa_copy(active_xfer_.req.desc_slot,
                        active_xfer_.req.smem_addr,
                        active_xfer_.req.coords.data(),
                        &bytes_copied);
  (void)bytes_copied;
  core_->barrier_event_release(active_xfer_.req.bar_id);

  has_active_ = false;
  active_xfer_ = ActiveTransfer();
}

} // namespace vortex

#endif // EXT_DXA_ENABLE

