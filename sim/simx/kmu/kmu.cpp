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

#include "kmu.h"
#include <cassert>

using namespace vortex;

Kmu::Kmu(const SimContext& ctx, const char* name, uint32_t num_cores)
  : SimObject<Kmu>(ctx, name)
  , bus_out(num_cores, this)
  , credit_in(num_cores, this)
  , PC_(0)
  , entry_(0)
  , param_(0)
  , lmem_size_(0)
  , block_size_(0)
  , running_(false)
  , cta_id_(0)
  , rr_ptr_(0)
  , ctx_id_(0)
  , cluster_locked_(false)
  , cluster_core_(0)
  , inflight_(num_cores, 0)
{
  block_dim_[0]    = block_dim_[1]    = block_dim_[2]    = 1;
  grid_dim_[0]     = grid_dim_[1]     = grid_dim_[2]     = 1;
  cluster_dim_[0]  = cluster_dim_[1]  = cluster_dim_[2]  = 1;
  warp_step_[0]    = warp_step_[1]    = warp_step_[2]    = 1;
  group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
  intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
}

void Kmu::on_reset() {
  // Reset only the per-run progression state. The kernel descriptor (PC,
  // param, dims, block_size, lmem_size, warp_step, cluster_dim) is set
  // by dcr_write() before run() and must persist across
  // SimPlatform::on_reset().
  running_         = false;
  cta_id_          = 0;
  rr_ptr_          = 0;
  ctx_id_          = 0;
  cluster_locked_  = false;
  cluster_core_    = 0;
  group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
  intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
  inflight_.assign(inflight_.size(), 0);
}

void Kmu::dcr_write(uint32_t addr, uint32_t value) {
  switch (addr) {
  case VX_DCR_KMU_STARTUP_ADDR0: PC_    = (PC_    & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_STARTUP_ADDR1: PC_    = (PC_    &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_KERNEL_ENTRY0: entry_ = (entry_ & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_KERNEL_ENTRY1: entry_ = (entry_ &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_STARTUP_ARG0:  param_ = (param_ & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_STARTUP_ARG1:  param_ = (param_ &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
  case VX_DCR_KMU_BLOCK_DIM_X:   block_dim_[0] = value; break;
  case VX_DCR_KMU_BLOCK_DIM_Y:   block_dim_[1] = value; break;
  case VX_DCR_KMU_BLOCK_DIM_Z:   block_dim_[2] = value; break;
  case VX_DCR_KMU_GRID_DIM_X:    grid_dim_[0]  = value; break;
  case VX_DCR_KMU_GRID_DIM_Y:    grid_dim_[1]  = value; break;
  case VX_DCR_KMU_GRID_DIM_Z:    grid_dim_[2]  = value; break;
  case VX_DCR_KMU_LMEM_SIZE:     lmem_size_    = value; break;
  case VX_DCR_KMU_BLOCK_SIZE:    block_size_   = value; break;
  case VX_DCR_KMU_WARP_STEP_X:   warp_step_[0] = value; break;
  case VX_DCR_KMU_WARP_STEP_Y:   warp_step_[1] = value; break;
  case VX_DCR_KMU_WARP_STEP_Z:   warp_step_[2] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_X: cluster_dim_[0] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_Y: cluster_dim_[1] = value; break;
  case VX_DCR_KMU_CLUSTER_DIM_Z: cluster_dim_[2] = value; break;
  default: break;
  }
}

void Kmu::start() {
  running_ = (block_size_ > 0)
           && (grid_dim_[0] > 0)
           && (grid_dim_[1] > 0)
           && (grid_dim_[2] > 0)
           && (cluster_dim_[0] > 0)
           && (cluster_dim_[1] > 0)
           && (cluster_dim_[2] > 0);
  ctx_id_ += 1;
  cluster_locked_ = false;
  if (running_) {
    cta_id_          = 0;
    group_origin_[0] = group_origin_[1] = group_origin_[2] = 0;
    intra_offset_[0] = intra_offset_[1] = intra_offset_[2] = 0;
    this->tick_wake();
  }
}

void Kmu::on_tick() {
  // Collect credits returned by dispatcher admissions.
  uint32_t num_cores = bus_out.size();
  for (uint32_t c = 0; c < num_cores; ++c) {
    auto& credits = credit_in.at(c);
    while (!credits.empty()) {
      credits.pop();
      assert(inflight_[c] != 0);
      --inflight_[c];
    }
  }
  if (!running_) {
    this->tick_sleep();
    return;
  }
  // Steering: a locked cluster's remaining members stay with the owning
  // core; otherwise round-robin over the ready lanes, nearest from rr_ptr_
  // winning. One CTA per cycle — the launch bus carries a single beat. A
  // lane is ready when its transport has room and the core still owes fewer
  // than LAUNCH_CREDITS admissions, so a busy core never accumulates work.
  uint32_t target = num_cores;
  if (cluster_locked_) {
    if (!bus_out.at(cluster_core_).full()
     && inflight_[cluster_core_] < LAUNCH_CREDITS) {
      target = cluster_core_;
    }
  } else {
    for (uint32_t k = 0; k < num_cores; ++k) {
      uint32_t c = (rr_ptr_ + k) % num_cores;
      if (!bus_out.at(c).full()
       && inflight_[c] < LAUNCH_CREDITS) {
        target = c;
        break;
      }
    }
  }
  if (target == num_cores)
    return; // no ready lane this cycle
  kmu_req_t req{};
  if (!this->next_cta(&req, target))
    return;
  bus_out.at(target).send(req, 1);
  ++inflight_[target];
  if (!cluster_locked_) {
    // Message complete: advance past the granted core.
    rr_ptr_ = (target + 1) % num_cores;
  }
}

bool Kmu::next_cta(kmu_req_t* req, uint32_t core_id) {
  if (!running_) return false;

  // A cluster's CTAs must all land on one core: they address each other's local
  // memory by rank and rendezvous on a per-core barrier slot. So once a core
  // takes the first member, the rest of the cluster belongs to it.
  assert(!cluster_locked_ || (cluster_core_ == core_id));

  // Is the CTA being handed out now the last of its cluster?
  bool last_of_cluster = ((intra_offset_[0] + 1) == cluster_dim_[0])
                      && ((intra_offset_[1] + 1) == cluster_dim_[1])
                      && ((intra_offset_[2] + 1) == cluster_dim_[2]);
  cluster_locked_ = !last_of_cluster;
  cluster_core_   = core_id;

  // Effective block_idx = group_origin + intra_offset.
  uint32_t block_idx[3] = {
    group_origin_[0] + intra_offset_[0],
    group_origin_[1] + intra_offset_[1],
    group_origin_[2] + intra_offset_[2],
  };

  req->PC           = PC_;
  req->entry        = entry_;
  req->param        = param_;
  req->cta_id       = cta_id_;
  req->ctx_id       = ctx_id_;
  req->block_idx[0] = block_idx[0];
  req->block_idx[1] = block_idx[1];
  req->block_idx[2] = block_idx[2];
  req->block_dim[0] = block_dim_[0];
  req->block_dim[1] = block_dim_[1];
  req->block_dim[2] = block_dim_[2];
  req->grid_dim[0]  = grid_dim_[0];
  req->grid_dim[1]  = grid_dim_[1];
  req->grid_dim[2]  = grid_dim_[2];
  req->lmem_size    = lmem_size_;
  req->block_size   = block_size_;
  req->warp_step[0] = warp_step_[0];
  req->warp_step[1] = warp_step_[1];
  req->warp_step[2] = warp_step_[2];
  req->cluster_dim[0] = cluster_dim_[0];
  req->cluster_dim[1] = cluster_dim_[1];
  req->cluster_dim[2] = cluster_dim_[2];
  req->is_first_of_cluster = (intra_offset_[0] == 0)
                          && (intra_offset_[1] == 0)
                          && (intra_offset_[2] == 0);

  // Advance the intra-cluster offset first (fills the cluster), then the
  // group_origin in (X, Y, Z) order when the inner loop wraps.
  ++cta_id_;
  bool intra_x_wrap = (intra_offset_[0] + 1 == cluster_dim_[0]);
  if (!intra_x_wrap) {
    intra_offset_[0]++;
  } else {
    intra_offset_[0] = 0;
    bool intra_y_wrap = (intra_offset_[1] + 1 == cluster_dim_[1]);
    if (!intra_y_wrap) {
      intra_offset_[1]++;
    } else {
      intra_offset_[1] = 0;
      bool intra_z_wrap = (intra_offset_[2] + 1 == cluster_dim_[2]);
      if (!intra_z_wrap) {
        intra_offset_[2]++;
      } else {
        intra_offset_[2] = 0;
        // Cluster complete — advance group_origin in (X, Y, Z) order.
        uint32_t ox = group_origin_[0] + cluster_dim_[0];
        if (ox == grid_dim_[0]) {
          group_origin_[0] = 0;
          uint32_t oy = group_origin_[1] + cluster_dim_[1];
          if (oy == grid_dim_[1]) {
            group_origin_[1] = 0;
            uint32_t oz = group_origin_[2] + cluster_dim_[2];
            if (oz == grid_dim_[2]) {
              group_origin_[2] = 0;
              running_ = false;
            } else {
              group_origin_[2] = oz;
            }
          } else {
            group_origin_[1] = oy;
          }
        } else {
          group_origin_[0] = ox;
        }
      }
    }
  }

  return true;
}
