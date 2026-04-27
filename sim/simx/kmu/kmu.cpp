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

using namespace vortex;

Kmu::Kmu(const SimContext& ctx, const char* name)
  : SimObject<Kmu>(ctx, name)
  , PC_(0)
  , param_(0)
  , lmem_size_(0)
  , block_size_(0)
  , running_(false)
  , cta_id_(0)
{
  block_dim_[0] = block_dim_[1] = block_dim_[2] = 1;
  grid_dim_[0]  = grid_dim_[1]  = grid_dim_[2]  = 1;
  warp_step_[0] = warp_step_[1] = warp_step_[2] = 1;
  block_idx_[0] = block_idx_[1] = block_idx_[2] = 0;
}

void Kmu::on_reset() {
  // Reset only the per-run progression state. The kernel descriptor (PC,
  // param, dims, block_size, lmem_size, warp_step) is set by dcr_write()
  // before run() and must persist across SimPlatform::on_reset().
  running_      = false;
  cta_id_       = 0;
  block_idx_[0] = block_idx_[1] = block_idx_[2] = 0;
}

void Kmu::dcr_write(uint32_t addr, uint32_t value) {
  switch (addr) {
  case VX_DCR_KMU_STARTUP_ADDR0: PC_    = (PC_    & ~uint64_t(0xFFFFFFFF)) | value; break;
  case VX_DCR_KMU_STARTUP_ADDR1: PC_    = (PC_    &  uint64_t(0xFFFFFFFF)) | (uint64_t(value) << 32); break;
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
  default: break;
  }
}

void Kmu::start() {
  running_ = (block_size_ > 0)
           && (grid_dim_[0] > 0)
           && (grid_dim_[1] > 0)
           && (grid_dim_[2] > 0);
  if (running_) {
    cta_id_       = 0;
    block_idx_[0] = block_idx_[1] = block_idx_[2] = 0;
  }
}

bool Kmu::step(kmu_req_t* req) {
  if (!running_) return false;

  req->PC           = PC_;
  req->param        = param_;
  req->cta_id       = cta_id_;
  req->block_idx[0] = block_idx_[0];
  req->block_idx[1] = block_idx_[1];
  req->block_idx[2] = block_idx_[2];
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

  // Advance the CTA iterator (X-innermost, Z-outermost)
  ++cta_id_;
  uint32_t bx = block_idx_[0] + 1;
  if (bx == grid_dim_[0]) {
    block_idx_[0] = 0;
    uint32_t by = block_idx_[1] + 1;
    if (by == grid_dim_[1]) {
      block_idx_[1] = 0;
      uint32_t bz = block_idx_[2] + 1;
      if (bz == grid_dim_[2]) {
        block_idx_[2] = 0;
        running_ = false;
      } else {
        block_idx_[2] = bz;
      }
    } else {
      block_idx_[1] = by;
    }
  } else {
    block_idx_[0] = bx;
  }

  return true;
}
