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

#include <VX_types.h>
#include "cta_dispatcher.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include <VX_config.h>
#include <cassert>

using namespace vortex;

CtaDispatcher::CtaDispatcher(const SimContext& ctx, const char* name, Core* core)
  : SimObject<CtaDispatcher>(ctx, name)
  , core_(core)
  , kmu_(&core->socket()->cluster()->processor()->kmu())
  , num_threads_(VX_CFG_NUM_THREADS)
  , num_warps_(VX_CFG_NUM_WARPS)
  , lmem_base_(VX_MEM_LMEM_BASE_ADDR)
  , lmem_capacity_(1u << VX_CFG_LMEM_LOG_SIZE)
  , slot_rem_warps_(num_warps_, 0)
  , wid_to_slot_(num_warps_, num_warps_)  // num_warps_ = invalid/unassigned
  , tail_slot_(0)
  , has_cta_(false)
  , cur_slot_(0)
  , cta_size_(0)
  , rank_(0)
  , block_size_rem_(0)
  , lmem_addr_(0)
  , has_pending_(false)
  , cur_kernel_pc_(0)
  , warp_init_mask_(num_warps_, false)
{
  thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;
}

CtaDispatcher::~CtaDispatcher() {}

void CtaDispatcher::on_reset() {
  has_cta_    = false;
  has_pending_= false;
  tail_slot_  = 0;
  for (uint32_t i = 0; i < num_warps_; ++i) {
    slot_rem_warps_[i] = 0;
    wid_to_slot_[i]    = num_warps_;  // invalid
    warp_init_mask_[i] = false;
  }
  cur_kernel_pc_ = 0;
}

uint32_t CtaDispatcher::usable_slots(uint32_t stride) const {
  if (stride == 0)
    return num_warps_;
  uint32_t fit = uint32_t(lmem_capacity_ / stride);
  if (fit > num_warps_) fit = num_warps_;
  if (fit == 0) fit = 1;  // a single CTA that barely exceeds capacity still gets one slot
  return fit;
}

bool CtaDispatcher::step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out) {
  if (!has_cta_) {
    // Load next CTA: use stashed pending CTA if available, else request from KMU.
    if (!has_pending_) {
      if (!kmu_->step(&pending_cta_)) return false;
      has_pending_ = true;
    }

    // Fixed-stride admission. Round the per-CTA LMEM size up to a
    // MEM_BLOCK_SIZE multiple — this uniform stride is the slot pitch.
    // Multicast resolves receiver destinations as `issuer_base + r * stride`
    // and the LMEM model is block-addressed (byteen-masked), so a non-aligned
    // stride would truncate the destination; the descriptor handler rounds
    // its stride the same way to stay consistent.
    uint32_t stride = (pending_cta_.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u)
                      & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
    uint32_t max_slots = usable_slots(stride);

    // Round-robin slot allocation over the usable range. Normalize the tail in
    // case a kernel transition shrank the usable count.
    if (tail_slot_ >= max_slots)
      tail_slot_ = 0;
    uint32_t base = tail_slot_;

    // Cluster start: reserve K consecutive usable slots. Pre-wrap to 0 if the
    // window would overrun the usable range so the cluster stays contiguous
    // (members must occupy consecutive slots for multicast). All K must be free
    // up front so the following members never stall mid-cluster.
    uint32_t k = 1;
    if (pending_cta_.is_first_of_cluster) {
      k = pending_cta_.cluster_dim[0]
        * pending_cta_.cluster_dim[1]
        * pending_cta_.cluster_dim[2];
      if (k == 0) k = 1;
      if (base + k > max_slots)
        base = 0;
      if (k > max_slots)
        k = max_slots;  // cluster larger than co-residency: clamp (degenerate)
      for (uint32_t i = 0; i < k; ++i) {
        if (slot_rem_warps_[base + i] != 0)
          return false;  // window not free yet — wait
      }
    } else {
      // Standalone CTA, or a following cluster member (its slot was reserved by
      // the first-of-cluster window above and is guaranteed free).
      if (slot_rem_warps_[base] != 0)
        return false;
    }

    // Reset Warp initialization states on kernel transitions
    if (pending_cta_.PC != cur_kernel_pc_) {
      cur_kernel_pc_ = pending_cta_.PC;
      for (uint32_t i = 0; i < num_warps_; ++i) {
        warp_init_mask_[i] = false;
      }
    }

    // Accept the pending CTA into its slot.
    cta_ = pending_cta_;
    has_pending_ = false;

    cta_size_       = (cta_.block_size + num_threads_ - 1) / num_threads_;
    rank_           = 0;
    block_size_rem_ = cta_.block_size;
    thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;

    cur_slot_  = base;
    lmem_addr_ = lmem_base_ + uint64_t(base) * stride;
    slot_rem_warps_[base] = cta_size_;
    tail_slot_ = (base + 1) % max_slots;

    has_cta_ = true;
  }

  // Find lowest-index free warp slot.
  int free_wid = -1;
  for (int wid = 0; wid < (int)num_warps_; ++wid) {
    if (!active_warps.test(wid)) {
      free_wid = wid;
      break;
    }
  }
  if (free_wid < 0) return false;

  if (!next_warp(!warp_init_mask_[free_wid], rec_out))
    return false;

  // Mark warp as initialized
  warp_init_mask_[free_wid] = true;

  wid_to_slot_[free_wid] = cur_slot_;
  *wid_out = uint32_t(free_wid);
  return true;
}

void CtaDispatcher::warp_done(uint32_t wid) {
  uint32_t slot = wid_to_slot_[wid];
  if (slot >= num_warps_) return;  // not a CTA-dispatcher warp
  wid_to_slot_[wid] = num_warps_;  // clear assignment
  assert(slot_rem_warps_[slot] > 0);
  // When the last warp of a CTA exits, its slot (and the LMEM region at
  // slot × stride) frees immediately for reuse — no in-order reclaim.
  --slot_rem_warps_[slot];
}

bool CtaDispatcher::next_warp(bool do_init, cta_warp_record_t* out) {
  if (!has_cta_) return false;

  out->do_init  = do_init;
  out->PC       = Word(cta_.PC);
  out->entry    = Word(cta_.entry);
  out->mscratch = Word(cta_.param);
  out->param    = cta_.param;
  out->cta_id   = cur_slot_;
  out->cta_rank = rank_;
  out->cta_size = cta_size_;

  out->thread_idx[0] = thread_idx_[0];
  out->thread_idx[1] = thread_idx_[1];
  out->thread_idx[2] = thread_idx_[2];

  out->block_idx[0] = cta_.block_idx[0];
  out->block_idx[1] = cta_.block_idx[1];
  out->block_idx[2] = cta_.block_idx[2];

  out->block_dim[0] = cta_.block_dim[0];
  out->block_dim[1] = cta_.block_dim[1];
  out->block_dim[2] = cta_.block_dim[2];

  out->grid_dim[0] = cta_.grid_dim[0];
  out->grid_dim[1] = cta_.grid_dim[1];
  out->grid_dim[2] = cta_.grid_dim[2];

  out->lmem_addr = lmem_addr_;
  out->cluster_size = cta_.cluster_dim[0] * cta_.cluster_dim[1] * cta_.cluster_dim[2];

  // Thread mask: full warp or partial last warp.
  out->tmask.resize(num_threads_);
  uint32_t active = (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;
  for (uint32_t t = 0; t < active; ++t) {
    out->tmask.set(t);
  }

  // advance thread_idx with 3-D carry propagation (additive delta + carry)
  uint32_t next_x = thread_idx_[0] + cta_.warp_step[0];
  bool wrap_x = (next_x >= cta_.block_dim[0]);
  thread_idx_[0] = wrap_x ? (next_x - cta_.block_dim[0]) : next_x;

  uint32_t next_y = thread_idx_[1] + cta_.warp_step[1] + (wrap_x ? 1 : 0);
  bool wrap_y = (next_y >= cta_.block_dim[1]);
  thread_idx_[1] = wrap_y ? (next_y - cta_.block_dim[1]) : next_y;

  thread_idx_[2] = thread_idx_[2] + cta_.warp_step[2] + (wrap_y ? 1 : 0);

  block_size_rem_ -= (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;

  ++rank_;
  if (rank_ >= cta_size_) {
    has_cta_ = false;
  }

  return true;
}
