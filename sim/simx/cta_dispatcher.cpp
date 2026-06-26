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
#include <cstdlib>
#include <iostream>

using namespace vortex;

namespace {
bool dxa_trace_enabled() {
  static bool enabled = (nullptr != std::getenv("VX_DXA_TRACE"));
  return enabled;
}
}

CtaDispatcher::CtaDispatcher(const SimContext& ctx, const char* name, Core* core)
  : SimObject<CtaDispatcher>(ctx, name)
  , core_(core)
  , kmu_(&core->socket()->cluster()->processor()->kmu())
  , num_threads_(VX_CFG_NUM_THREADS)
  , num_warps_(VX_CFG_NUM_WARPS)
  , lmem_base_(VX_MEM_LMEM_BASE_ADDR)
  , lmem_capacity_(1u << VX_CFG_LMEM_LOG_SIZE)
  , lmem_tail_(0)
  , free_size_(1u << VX_CFG_LMEM_LOG_SIZE)
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

CtaDispatcher::~CtaDispatcher() {}

void CtaDispatcher::on_reset() {
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
      if (dxa_trace_enabled()) {
        std::cout << "DXA_TRACE cta_pending"
                  << " core=" << core_->id()
                  << " pc=0x" << std::hex << pending_cta_.PC
                  << " param=0x" << pending_cta_.param << std::dec
                  << " block_size=" << pending_cta_.block_size
                  << " lmem_size=" << pending_cta_.lmem_size
                  << " block=(" << pending_cta_.block_idx[0]
                  << "," << pending_cta_.block_idx[1]
                  << "," << pending_cta_.block_idx[2] << ")"
                  << " cluster=(" << pending_cta_.cluster_dim[0]
                  << "," << pending_cta_.cluster_dim[1]
                  << "," << pending_cta_.cluster_dim[2] << ")"
                  << std::endl;
      }
    }

    // Admission control: wait until the next FIFO slot is free and enough lmem is available.
    if (slot_rem_warps_[tail_slot_] != 0)
      return false;
    // Account for both block alignment of per-CTA allocations (see comment
    // below) and the padding that occurs when the allocation would straddle
    // the LMEM boundary. For the FIRST CTA of a cluster, additionally
    // pad so the ENTIRE cluster's K CTAs fit contiguously past the
    // current lmem_tail_ — DXA Path A multicast assumes contiguous strides.
    // First-of-cluster is signaled via `pending_cta_.is_first_of_cluster`.
    uint32_t pending_aligned = (pending_cta_.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u)
                               & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);
    uint32_t lmem_needed = pending_aligned;
    if (pending_aligned > 0) {
      uint32_t span_needed = pending_aligned;
      if (pending_cta_.is_first_of_cluster) {
        // First CTA of a new cluster: reserve the full group span for
        // the straddle check so the group stays contiguous in LMEM.
        uint32_t k = pending_cta_.cluster_dim[0]
                   * pending_cta_.cluster_dim[1]
                   * pending_cta_.cluster_dim[2];
        if (k == 0) k = 1;
        span_needed = pending_aligned * k;
      }
      if (lmem_tail_ + span_needed > lmem_capacity_) {
        lmem_needed = pending_aligned + (lmem_capacity_ - lmem_tail_);
      }
    }
    if (free_size_ < lmem_needed)
      return false;

    // Reset Warp initialization states on kernel transitions
    if (pending_cta_.PC != cur_kernel_pc_) {
      cur_kernel_pc_ = pending_cta_.PC;
      for (uint32_t i = 0; i < num_warps_; ++i) {
        warp_init_mask_[i] = false;
      }
    }

    // Accept the pending CTA.
    cta_ = pending_cta_;
    has_pending_ = false;
    if (dxa_trace_enabled()) {
      std::cout << "DXA_TRACE cta_accept"
                << " core=" << core_->id()
                << " slot=" << tail_slot_
                << " block_size=" << cta_.block_size
                << " lmem_free=" << free_size_
                << " lmem_needed=" << lmem_needed
                << std::endl;
    }

    // Round per-CTA LMEM allocation up to a MEM_BLOCK_SIZE multiple so
    // adjacent CTAs are stride-aligned. DXA Path A multicast resolves
    // receiver destinations as `issuer_addr + r * smem_stride`; if the
    // per-CTA stride is not block-aligned, the LMEM model truncates the
    // address (it's block-addressed with a byteen mask) and writes land
    // in the wrong block. The DXA descriptor handler applies the same
    // rounding to `smem_stride`, keeping the two consistent.
    uint32_t aligned_lmem_size = (cta_.lmem_size + VX_CFG_MEM_BLOCK_SIZE - 1u)
                                 & ~uint32_t(VX_CFG_MEM_BLOCK_SIZE - 1u);

    cta_size_       = (cta_.block_size + num_threads_ - 1) / num_threads_;
    rank_           = 0;
    block_size_rem_ = cta_.block_size;
    thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;

    // allocate lmem slot
    lmem_addr_ = 0;
    uint32_t lmem_cost = 0;
    if (aligned_lmem_size > 0) {
      lmem_cost = aligned_lmem_size;
      // For the first CTA of a cluster, pre-wrap so the WHOLE group's
      // K CTAs (each rounded to a block multiple) live at K contiguous
      // offsets. For subsequent CTAs in an already-placed group, fall back
      // to per-CTA wrap (the group span check at the start guaranteed K
      // fit, but defensive code handles any residual edge cases).
      uint32_t span_to_check = aligned_lmem_size;
      if (cta_.is_first_of_cluster) {
        uint32_t k = cta_.cluster_dim[0]
                   * cta_.cluster_dim[1]
                   * cta_.cluster_dim[2];
        if (k == 0) k = 1;
        span_to_check = aligned_lmem_size * k;
      }
      if (lmem_tail_ + span_to_check > lmem_capacity_) {
        lmem_cost += lmem_capacity_ - lmem_tail_;
        lmem_tail_ = 0;
      }
      lmem_addr_  = lmem_base_ + lmem_tail_;
      lmem_tail_  = (lmem_tail_ + aligned_lmem_size) & (lmem_capacity_ - 1);
      free_size_  -= lmem_cost;
    }

    // Claim the next FIFO slot.
    cur_slot_ = tail_slot_;
    tail_slot_ = (tail_slot_ + 1) % num_warps_;
    slot_lmem_size_[cur_slot_] = lmem_cost;
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

  if (!next_warp(!warp_init_mask_[free_wid], rec_out))
    return false;

  // Mark warp as initialized
  warp_init_mask_[free_wid] = true;

  wid_to_slot_[free_wid] = cur_slot_;
  *wid_out = uint32_t(free_wid);
  if (dxa_trace_enabled()) {
    std::cout << "DXA_TRACE cta_warp"
              << " core=" << core_->id()
              << " wid=" << *wid_out
              << " slot=" << cur_slot_
              << " rank=" << rec_out->cta_rank
              << "/" << rec_out->cta_size
              << " do_init=" << rec_out->do_init
              << " thread_x=" << rec_out->thread_idx[0]
              << " lmem=0x" << std::hex << rec_out->lmem_addr << std::dec
              << std::endl;
  }
  return true;
}

void CtaDispatcher::warp_done(uint32_t wid) {
  uint32_t slot = wid_to_slot_[wid];
  if (slot >= num_warps_) return;  // not a CTA-dispatcher warp
  wid_to_slot_[wid] = num_warps_;  // clear assignment
  assert(slot_rem_warps_[slot] > 0);
  if (--slot_rem_warps_[slot] == 0) {
    // only advance head and reclaim memory if the oldest CTA finished
    if (slot == head_slot_) {
      do {
        // Reclaim memory strictly in-order as the head advances
        free_size_ += slot_lmem_size_[head_slot_];
        slot_lmem_size_[head_slot_] = 0;
        head_slot_ = (head_slot_ + 1) % num_warps_;
      } while (head_slot_ != tail_slot_ && slot_rem_warps_[head_slot_] == 0);
    }
  }
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
