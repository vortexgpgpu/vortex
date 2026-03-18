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

using namespace vortex;

CtaDispatcher::CtaDispatcher(Core* core)
  : core_(core)
  , kmu_(nullptr)
  , started_(false)
  , total_cores_(1)
  , core_id_(0)
  , num_threads_(1)
  , num_warps_(1)
  , lmem_capacity_(0)
  , lmem_tail_(0)
  , iter_cta_id_(0)
  , iter_running_(false)
  , has_cta_(false)
  , cta_size_(0)
  , rank_(0)
  , block_size_rem_(0)
  , lmem_addr_(0)
{
  iter_block_idx_[0] = iter_block_idx_[1] = iter_block_idx_[2] = 0;
  thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;
}

void CtaDispatcher::do_start() {
  const auto& arch = core_->arch();

  // Bind to the processor's KMU — no copy, just a pointer.
  // ProcessorImpl::run() already called kmu_.start(); we only read config here.
  kmu_ = &core_->socket()->cluster()->processor()->kmu();

  // Initialize per-core iteration state from scratch
  iter_cta_id_        = 0;
  iter_block_idx_[0]  = iter_block_idx_[1] = iter_block_idx_[2] = 0;
  iter_running_       = kmu_->active();

  total_cores_   = arch.num_cores() * arch.num_clusters();
  core_id_       = core_->id();
  num_threads_   = arch.num_threads();
  num_warps_     = arch.num_warps();
  lmem_capacity_ = 1u << LMEM_LOG_SIZE;
  lmem_tail_     = 0;
  started_       = true;

  load_next_cta();
}

bool CtaDispatcher::step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out) {
  if (!started_) {
    do_start();
  }
  if (!has_cta_) return false;

  // find lowest-index free warp slot
  int free_wid = -1;
  for (int wid = 0; wid < (int)num_warps_; ++wid) {
    if (!active_warps.test(wid)) {
      free_wid = wid;
      break;
    }
  }
  if (free_wid < 0) return false;

  if (!next_warp(rec_out)) return false;
  *wid_out = uint32_t(free_wid);
  return true;
}

bool CtaDispatcher::next_warp(cta_warp_record_t* out) {
  if (!has_cta_) return false;

  // Fill warp record for current rank
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

  // Thread mask: full warp or partial last warp
  out->tmask.resize(num_threads_);
  out->tmask.reset();
  uint32_t active = (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;
  for (uint32_t t = 0; t < active; ++t) out->tmask.set(t);

  // Advance thread_idx by warp_step with 3-D carry propagation
  // (mirrors VX_cta_dispatch.sv DISPATCH FSM thread_idx advancement)
  uint32_t next_x = thread_idx_[0] + cta_.warp_step[0];
  bool wrap_x = (cta_.block_dim[0] > 0) && (next_x >= cta_.block_dim[0]);
  thread_idx_[0] = wrap_x ? (next_x - cta_.block_dim[0]) : next_x;

  uint32_t next_y = thread_idx_[1] + (wrap_x ? cta_.warp_step[1] : 0);
  bool wrap_y = (cta_.block_dim[1] > 0) && (next_y >= cta_.block_dim[1]);
  thread_idx_[1] = wrap_y ? (next_y - cta_.block_dim[1]) : next_y;

  thread_idx_[2] += (wrap_y ? cta_.warp_step[2] : 0);

  // Decrement remaining thread count for this CTA
  block_size_rem_ -= (block_size_rem_ >= num_threads_) ? num_threads_ : block_size_rem_;

  ++rank_;
  if (rank_ >= cta_size_) {
    // CTA fully dispatched: load next CTA
    load_next_cta();
  }

  return true;
}

void CtaDispatcher::load_next_cta() {
  has_cta_ = false;
  while (iter_running_) {
    // Build CTA request from KMU config + local iteration state
    kmu_req_t req;
    req.PC         = kmu_->PC();
    req.param      = kmu_->param();
    req.cta_id     = iter_cta_id_;
    req.lmem_size  = kmu_->lmem_size();
    req.block_size = kmu_->block_size();
    for (int i = 0; i < 3; ++i) {
      req.block_idx[i] = iter_block_idx_[i];
      req.block_dim[i] = kmu_->block_dim(i);
      req.grid_dim[i]  = kmu_->grid_dim(i);
      req.warp_step[i] = kmu_->warp_step(i);
    }

    // Advance iteration state (X innermost, Z outermost — mirrors VX_kmu.sv)
    ++iter_cta_id_;
    ++iter_block_idx_[0];
    if (iter_block_idx_[0] >= kmu_->grid_dim(0)) {
      iter_block_idx_[0] = 0;
      ++iter_block_idx_[1];
      if (iter_block_idx_[1] >= kmu_->grid_dim(1)) {
        iter_block_idx_[1] = 0;
        ++iter_block_idx_[2];
        if (iter_block_idx_[2] >= kmu_->grid_dim(2)) {
          iter_running_ = false;
        }
      }
    }

    // Round-robin assignment: this core handles CTAs where cta_id % total_cores == core_id
    if (req.cta_id % total_cores_ != core_id_) continue;

    cta_            = req;
    cta_size_       = (req.block_size + num_threads_ - 1) / num_threads_;
    rank_           = 0;
    block_size_rem_ = req.block_size;
    thread_idx_[0] = thread_idx_[1] = thread_idx_[2] = 0;

    // Assign LMEM slot: ring-buffer allocation, one slot per CTA
    lmem_addr_ = 0;
    if (req.lmem_size > 0) {
      lmem_addr_ = lmem_tail_;
      lmem_tail_ = (lmem_tail_ + req.lmem_size) & (lmem_capacity_ - 1);
    }

    has_cta_ = true;
    return;
  }
  // No more CTAs assigned to this core
}
