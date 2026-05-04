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
 

#ifndef __VX_LAUNCH_H__
#define __VX_LAUNCH_H__

#include <stdint.h>
#include <vx_intrinsics.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kernel launch descriptor consumed by the KMU through VX_CSR_KMU_LAUNCH.
// Layout is matched byte-for-byte by the simx emulator; keep packed and
// in the exact field order below.
typedef struct {
  uint64_t pc;              // kernel entry PC (same convention as VX_DCR_KMU_STARTUP_ADDR)
  uint64_t arg;             // kernel argument pointer (passed via MSCRATCH)
  uint32_t grid_dim[3];     // grid dimensions
  uint32_t block_dim[3];    // block dimensions
  uint32_t block_size;      // threads per block (product of block_dim)
  uint32_t warp_step[3];    // thread-index stride per warp (see runtime/common/utils.cpp)
  uint32_t lmem_size;       // local memory bytes per block
} vx_kmu_launch_desc_t;

// Fill `desc` from a grid/block configuration, matching the host-side
// `prepare_kernel_launch_params` logic.
static inline void vx_launch_desc_init(vx_kmu_launch_desc_t* desc,
                                       uint64_t pc,
                                       uint64_t arg,
                                       const uint32_t grid_dim[3],
                                       const uint32_t block_dim[3],
                                       uint32_t lmem_size) {
  uint32_t threads_per_warp = (uint32_t)vx_num_threads();
  uint32_t block_size = 1;
  for (int i = 0; i < 3; ++i) {
    desc->grid_dim[i]  = grid_dim[i];
    desc->block_dim[i] = block_dim[i];
    block_size *= block_dim[i];
  }
  desc->pc         = pc;
  desc->arg        = arg;
  desc->block_size = block_size;
  desc->warp_step[0] = threads_per_warp % block_dim[0];
  desc->warp_step[1] = (threads_per_warp / block_dim[0]) % block_dim[1];
  desc->warp_step[2] = (threads_per_warp / (block_dim[0] * block_dim[1])) % block_dim[2];
  desc->lmem_size  = lmem_size;
}

// Fire off a child grid. The descriptor must remain stable in memory long
// enough for the KMU to latch all fields; because set_csr reads the struct
// synchronously in simx, the lifetime ends after this call returns.
static inline void vx_kernel_launch(const vx_kmu_launch_desc_t* desc) {
  vx_fence();
  csr_write(VX_CSR_KMU_LAUNCH, (size_t)desc);
}

#ifdef __cplusplus
}
#endif

#endif // __VX_LAUNCH_H__
