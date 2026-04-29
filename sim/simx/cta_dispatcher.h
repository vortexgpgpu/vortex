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

#include <cstdint>
#include <vector>
#include <simobject.h>
#include "types.h"
#include "kmu.h"

namespace vortex {

class Core; // forward declaration

struct cta_warp_record_t {
  bool       do_init;
  Word       PC;
  ThreadMask tmask;
  Word       mscratch;
  uint32_t   cta_id;
  uint32_t   cta_rank;
  uint32_t   cta_size;
  uint32_t   thread_idx[3];
  uint32_t   block_idx[3];
  uint32_t   block_dim[3];
  uint32_t   grid_dim[3];
  uint64_t   param;
  uint64_t   lmem_addr;
};

// CTA dispatcher: walks the pending kernel grid and admits one warp rank per
// step() into a free warp slot.
class CtaDispatcher : public SimObject<CtaDispatcher> {
public:
  CtaDispatcher(const SimContext& ctx, const char* name, Core* core);
  ~CtaDispatcher();

  // Try to dispatch one warp rank into a free warp slot.
  // Returns true on success and sets *wid_out / *rec_out.
  bool step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out);

  // Notify that a warp has exited; frees its lmem slot when all warps of the CTA are done.
  void warp_done(uint32_t wid);

  // True while CTAs remain to dispatch or active slots hold live warps.
  bool running() const {
    return has_cta_ || has_pending_ || kmu_->running();
  }

protected:
  void on_reset();

private:
  bool next_warp(bool is_init, cta_warp_record_t* out);

  Core*     core_;
  Kmu*      kmu_;
  uint32_t  num_threads_;
  uint32_t  num_warps_;
  uint64_t  lmem_base_;
  uint32_t  lmem_capacity_;

  // Ring-buffer allocation pointer (offset from lmem_base_)
  uint32_t  lmem_tail_;
  // Total bytes currently available for new allocations
  uint32_t  free_size_;

  // Per-slot tracking (indexed 0..num_warps_-1)
  std::vector<uint32_t> slot_rem_warps_;  // remaining active warps in this slot
  std::vector<uint32_t> slot_lmem_size_;  // lmem bytes allocated to this slot
  // Reverse lookup: wid → slot index
  std::vector<uint32_t> wid_to_slot_;
  // FIFO ring for in-order slot allocation
  uint32_t  head_slot_;   // oldest live slot
  uint32_t  tail_slot_;   // next slot to fill

  // Currently-in-flight CTA (being dispatched warp-rank by warp-rank)
  bool      has_cta_;
  uint32_t  cur_slot_;    // slot index assigned to the current in-flight CTA
  kmu_req_t cta_;
  uint32_t  cta_size_;
  uint32_t  rank_;
  uint32_t  block_size_rem_;
  uint32_t  thread_idx_[3];
  uint64_t  lmem_addr_;

  // CTA fetched from KMU but blocked on lmem admission
  bool      has_pending_;
  kmu_req_t pending_cta_;
  Word      cur_kernel_pc_;
  std::vector<bool> warp_init_mask_;

  friend class SimObject<CtaDispatcher>;
};

} // namespace vortex
