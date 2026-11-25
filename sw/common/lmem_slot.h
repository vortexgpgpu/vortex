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

// The pitch of the equal per-CTA LMEM slots the CTA dispatcher partitions LMEM
// into. VX_CFG_* come from the includer: VX_config.h in SimX, the resolved -D
// flags in the runtime (VX_config.h is private to hardware and the simulators).

#pragma once

#include <stdint.h>

namespace vortex {

// Alignment of every CTA's LMEM slot, and so of its __local_mem() base. An
// LMEM DMA agent (DXA, the WGMMA tile buffers) moves whole bank rows: a
// block-major WGMMA operand must start one, and DXA multicast steps receivers
// by whole rows.
inline constexpr uint32_t lmem_slot_align() {
  constexpr uint32_t bank_row = VX_CFG_LMEM_NUM_BANKS * (VX_CFG_XLEN / 8);
  constexpr bool has_dma = VX_CFG_EXT_DXA_ENABLED || VX_CFG_TCU_WGMMA_ENABLED;
  return (has_dma && bank_row > VX_CFG_MEM_BLOCK_SIZE) ? bank_row : VX_CFG_MEM_BLOCK_SIZE;
}

static_assert((lmem_slot_align() & (lmem_slot_align() - 1)) == 0,
              "the LMEM slot alignment must be a power of two");

inline constexpr uint32_t lmem_slot_pitch(uint32_t lmem_size) {
  return (lmem_size + lmem_slot_align() - 1u) & ~(lmem_slot_align() - 1u);
}

} // namespace vortex
