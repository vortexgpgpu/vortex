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
  Word       entry;
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
  uint32_t   cluster_size;
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

  // Usable slot count for a given per-CTA stride: floor(capacity/stride) capped
  // at NUM_WARPS (all slots when stride is 0). This is the occupancy bound that
  // replaces the byte ring's free_size — slots at index >= this never fit.
  uint32_t usable_slots(uint32_t stride) const;

  Core*     core_;
  Kmu*      kmu_;
  uint32_t  num_threads_;
  uint32_t  num_warps_;
  uint64_t  lmem_base_;
  uint32_t  lmem_capacity_;

  // Fixed-stride slot allocation. Every resident CTA gets LMEM base
  // slot × stride, where stride = align(lmem_size, MEM_BLOCK_SIZE) is uniform
  // within a kernel. The occupancy bound floor(capacity/stride) (capped at
  // NUM_WARPS) replaces the byte ring's free_size accounting, so no wrap or
  // padding arithmetic is needed. Slots are allocated round-robin via a tail
  // pointer and freed immediately when a CTA's last warp exits.
  std::vector<uint32_t> slot_rem_warps_;  // remaining active warps; slot free when 0
  // Reverse lookup: wid → slot index
  std::vector<uint32_t> wid_to_slot_;
  uint32_t  tail_slot_;   // next slot to allocate (round-robin over usable slots)

  // Currently-in-flight CTA (being dispatched warp-rank by warp-rank)
  bool      has_cta_;
  uint32_t  cur_slot_;    // slot index assigned to the current in-flight CTA
  kmu_req_t cta_;
  uint32_t  cta_size_;
  uint32_t  rank_;
  uint32_t  block_size_rem_;
  uint32_t  thread_idx_[3];
  uint64_t  lmem_addr_;

  // CTA fetched from KMU but blocked on slot admission
  bool      has_pending_;
  kmu_req_t pending_cta_;
  Word      cur_kernel_pc_;
  std::vector<bool> warp_init_mask_;

  // Cluster co-residency: the first CTA of a cluster reserves K consecutive
  // usable slots (pre-wrapping the tail to 0 if the window would overrun the
  // usable range), so members occupy consecutive slots — member r at LMEM base
  // issuer_base + r × stride, exactly what multicast resolves. The following
  // K-1 members (KMU clears is_first_of_cluster) take tail slots in order.

  friend class SimObject<CtaDispatcher>;
};

} // namespace vortex
