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

#include "cta_dispatcher.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include <VX_config.h>
#include <cassert>

using namespace vortex;

CtaDispatcher::CtaDispatcher(Core* core)
  : core_(core)
  , kmu_(&core->socket()->cluster()->processor()->kmu())
  , num_threads_(core->arch().num_threads())
  , num_warps_(core->arch().num_warps())
  , lmem_base_(core->arch().local_mem_base())
  , lmem_capacity_(1u << LMEM_LOG_SIZE)
  , lmem_tail_(0)
  , free_size_(1u << LMEM_LOG_SIZE)
  , slot_rem_warps_(num_warps_, 0)
  , slot_lmem_size_(num_warps_, 0)
  , wid_to_slot_(num_warps_, num_warps_)  // num_warps_ = invalid/unassigned
  , head_slot_(0)
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

void CtaDispatcher::reset() {
  lmem_tail_ = 0;
  free_size_  = lmem_capacity_;
  has_cta_    = false;
  has_pending_= false;
  head_slot_  = 0;
  tail_slot_  = 0;
  for (uint32_t i = 0; i < num_warps_; ++i) {
    slot_rem_warps_[i] = 0;
    slot_lmem_size_[i] = 0;
    wid_to_slot_[i]    = num_warps_;  // invalid
  }
  cur_kernel_pc_ = 0;
  for (uint32_t i = 0; i < num_warps_; ++i) {
    warp_init_mask_[i] = false;
  }
}

bool CtaDispatcher::step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out) {
  if (!has_cta_) {
    // Load next CTA: use stashed pending CTA if available, else request from KMU.
    if (!has_pending_) {
      if (!kmu_->step(&pending_cta_)) return false;
      has_pending_ = true;
    }

    // Admission control: wait until enough lmem is free.
    if (pending_cta_.lmem_size > 0 && free_size_ < pending_cta_.lmem_size)
      return false;

    if (pending_cta_.PC != cur_kernel_pc_) {
      cur_kernel_pc_ = pending_cta_.PC;
      for (uint32_t i = 0; i < num_warps_; ++i) {
        warp_init_mask_[i] = false;
      }
    }

    // Accept the pending CTA.
    cta_ = pending_cta_;
    has_pending_ = false;

    cta_size_       = (cta_.block_size + num_threads_ - 1) / num_threads_;
    rank_           = 0;
    block_size_rem_ = cta_.block_size;
    thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;

    // Allocate lmem slot.
    lmem_addr_ = 0;
    if (cta_.lmem_size > 0) {
      lmem_addr_  = lmem_base_ + lmem_tail_;
      lmem_tail_  = (lmem_tail_ + cta_.lmem_size) & (lmem_capacity_ - 1);
      free_size_  -= cta_.lmem_size;
    }

    // Claim the next FIFO slot.
    cur_slot_ = tail_slot_;
    tail_slot_ = (tail_slot_ + 1) % num_warps_;
    slot_lmem_size_[cur_slot_] = cta_.lmem_size;
    slot_rem_warps_[cur_slot_] = cta_size_;

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

  if (!next_warp(rec_out)) return false;

  // If kernel already initialized on this warp, skip the prologue
  if (warp_init_mask_[free_wid]) {
    rec_out->PC -= 8;
  } else {
    warp_init_mask_[free_wid] = true;
  }

  wid_to_slot_[free_wid] = cur_slot_;
  *wid_out = uint32_t(free_wid);
  return true;
}

void CtaDispatcher::warp_done(uint32_t wid) {
  uint32_t slot = wid_to_slot_[wid];
  if (slot >= num_warps_) return;  // not a CTA-dispatcher warp
  wid_to_slot_[wid] = num_warps_;  // clear assignment
  assert(slot_rem_warps_[slot] > 0);
  if (--slot_rem_warps_[slot] == 0) {
    // All warps of this CTA are done; reclaim lmem.
    free_size_ += slot_lmem_size_[slot];
    slot_lmem_size_[slot] = 0;
    // Advance FIFO head past any completed slots.
    if (slot == head_slot_) {
      while (head_slot_ != tail_slot_ && slot_rem_warps_[head_slot_] == 0) {
        head_slot_ = (head_slot_ + 1) % num_warps_;
      }
    }
  }
}

bool CtaDispatcher::next_warp(cta_warp_record_t* out) {
  if (!has_cta_) return false;

  out->PC       = Word(cta_.PC);
  out->mscratch = Word(cta_.param);
  out->param    = cta_.param;
  out->cta_id   = cta_.cta_id;
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

  // Thread mask: full warp or partial last warp.
  out->tmask.resize(num_threads_);
  out->tmask.reset();
  uint32_t active = (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;
  for (uint32_t t = 0; t < active; ++t) out->tmask.set(t);

  // Advance thread_idx with 3-D carry propagation.
  // warp_step[d] == 0 means the warp fully covers dimension d, so always carry.
  uint32_t next_x = thread_idx_[0] + cta_.warp_step[0];
  bool wrap_x = (cta_.warp_step[0] == 0) || ((cta_.block_dim[0] > 0) && (next_x >= cta_.block_dim[0]));
  thread_idx_[0] = (cta_.block_dim[0] > 0) ? (next_x % cta_.block_dim[0]) : next_x;

  uint32_t next_y = thread_idx_[1] + (wrap_x ? cta_.warp_step[1] : 0);
  bool wrap_y = (cta_.warp_step[1] == 0 && wrap_x) || ((cta_.block_dim[1] > 0) && (next_y >= cta_.block_dim[1]));
  thread_idx_[1] = (cta_.block_dim[1] > 0) ? (next_y % cta_.block_dim[1]) : next_y;

  thread_idx_[2] += (wrap_y ? cta_.warp_step[2] : 0);

  block_size_rem_ -= (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;

  ++rank_;
  if (rank_ >= cta_size_) {
    has_cta_ = false;
  }

  return true;
}
