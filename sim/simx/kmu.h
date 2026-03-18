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
#include "VX_types.h"

namespace vortex {

// Mirrors the kmu_req_t struct from VX_gpu_pkg.sv
struct kmu_req_t {
  uint64_t PC;
  uint64_t param;
  uint32_t cta_id;
  uint32_t block_idx[3];
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint32_t lmem_size;
  uint32_t block_size;
  uint32_t warp_step[3];
};

// Software model of VX_kmu.sv: configuration store only.
// DCR writes set the kernel launch parameters; start() activates the launch.
// CTA grid iteration is the dispatcher's responsibility (one iterator per core).
class Kmu {
public:
  Kmu();

  void reset();

  void dcr_write(uint32_t addr, uint32_t value);

  // Called by ProcessorImpl::run() to activate a kernel launch.
  // Sets active() = true when config is valid (non-zero block_size and grid dims).
  void start();

  // True after a successful start() call.
  bool active() const { return active_; }

  // Config accessors (read-only after DCR writes)
  uint64_t PC()             const { return PC_;           }
  uint64_t param()          const { return param_;        }
  uint32_t block_dim(int i) const { return block_dim_[i]; }
  uint32_t grid_dim(int i)  const { return grid_dim_[i];  }
  uint32_t lmem_size()      const { return lmem_size_;    }
  uint32_t block_size()     const { return block_size_;   }
  uint32_t warp_step(int i) const { return warp_step_[i]; }

private:
  bool     active_;
  // Config registers (written via DCR)
  uint64_t PC_;
  uint64_t param_;
  uint32_t block_dim_[3];
  uint32_t grid_dim_[3];
  uint32_t lmem_size_;
  uint32_t block_size_;
  uint32_t warp_step_[3];
};

} // namespace vortex
