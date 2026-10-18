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

#include <VX_config.h>
#include <bitmanip.h>

#ifndef RAM_PAGE_SIZE
#define RAM_PAGE_SIZE     4096
#endif

#ifndef __MAX
#define __MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef __UP
#define __UP(x) (((x) != 0) ? (x) : 1)
#endif

#define VX_DCR_BASE_STATE(addr)   ((addr) - VX_DCR_BASE_STATE_BEGIN)
#define VX_DCR_BASE_STATE_COUNT   (VX_DCR_BASE_STATE_END-VX_DCR_BASE_STATE_BEGIN)

#define VX_DCR_DXA_DESC_SLOT(addr) (((addr) - VX_DCR_DXA_DESC_BASE) / VX_DCR_DXA_DESC_STRIDE)
#define VX_DCR_DXA_DESC_WORD(addr) (((addr) - VX_DCR_DXA_DESC_BASE) % VX_DCR_DXA_DESC_STRIDE)
#define VX_DCR_DXA_STATE_COUNT     (VX_DCR_DXA_STATE_END - VX_DCR_DXA_STATE_BEGIN)

namespace vortex {

inline constexpr uint32_t XLENB           = (VX_CFG_XLEN / 8);

// A last-level cache that alone serves every memory client is the point of
// coherence between kernels. The texture, ray-tracing and DXA ports join below
// the dcache and the raster and OM caches beside it, so a dcache with any of
// them is not one.
inline constexpr bool DCACHE_SHARED_LLC = (VX_CFG_DCACHE_ENABLED != 0) && (VX_CFG_L2_ENABLED == 0) && (VX_CFG_L3_ENABLED == 0)
                                       && (VX_CFG_NUM_CORES == 1) && (VX_CFG_NUM_CLUSTERS == 1)
                                       && !VX_CFG_EXT_TEX_ENABLED && !VX_CFG_EXT_RTU_ENABLED
                                       && !VX_CFG_EXT_RASTER_ENABLED && !VX_CFG_EXT_OM_ENABLED
                                       && !VX_CFG_EXT_DXA_ENABLED;
inline constexpr bool L2_SHARED_LLC     = (VX_CFG_L2_ENABLED != 0) && (VX_CFG_L3_ENABLED == 0) && (VX_CFG_NUM_CLUSTERS == 1);
inline constexpr bool L3_SHARED_LLC     = (VX_CFG_L3_ENABLED != 0);

// A write-back LLC must be the only instance: replicated ones never make their
// dirty lines visible to each other.
static_assert(!((VX_CFG_L2_ENABLED == 0) && (VX_CFG_L3_ENABLED == 0) && VX_CFG_DCACHE_WRITEBACK)
              || ((VX_CFG_NUM_CORES == 1) && (VX_CFG_NUM_CLUSTERS == 1)),
              "a write-back dcache that is the last-level cache must be the only dcache");
static_assert(!((VX_CFG_L2_ENABLED != 0) && (VX_CFG_L3_ENABLED == 0) && VX_CFG_L2_WRITEBACK)
              || (VX_CFG_NUM_CLUSTERS == 1),
              "a write-back L2 that is the last-level cache must be the only L2");

inline constexpr uint32_t MAX_NUM_CORES   = 1024;
inline constexpr uint32_t MAX_NUM_WARPS   = 64;
inline constexpr uint32_t MAX_NUM_REGS    = 32;
inline constexpr uint32_t LOG_NUM_REGS    = 5;
inline constexpr uint32_t NUM_SRC_REGS    = 3;

inline constexpr uint32_t LSU_WORD_SIZE   = (VX_CFG_XLEN / 8);
inline constexpr uint32_t LSU_CHANNELS    = VX_CFG_NUM_LSU_LANES;
inline constexpr uint32_t LSU_NUM_REQS	  = (VX_CFG_NUM_LSU_BLOCKS * LSU_CHANNELS);
// Mem-side queue depth: derived, not a config knob — must cover the
// outstanding pool and one LSU line's worth of words.
inline constexpr uint32_t LSU_QUEUE_OUT_SIZE = (VX_CFG_LSU_PENDING_SIZE > (VX_CFG_LSU_LINE_SIZE / LSU_WORD_SIZE)) ? VX_CFG_LSU_PENDING_SIZE : (VX_CFG_LSU_LINE_SIZE / LSU_WORD_SIZE);

// The dcache uses coalesced memory blocks. WORD_SIZE (coalescer output granule)
// is decoupled from LSU_LINE_SIZE via VX_CFG_DCACHE_WORD_SIZE so channels/banks
// scale independently of the LSU pipe count (Feature A). Default = LSU_LINE_SIZE.
inline constexpr uint32_t DCACHE_WORD_SIZE= VX_CFG_DCACHE_WORD_SIZE;
inline constexpr uint32_t DCACHE_CHANNELS = __UP((VX_CFG_NUM_LSU_LANES * XLENB) / DCACHE_WORD_SIZE);
inline constexpr uint32_t VX_CFG_DCACHE_NUM_REQS	= (VX_CFG_NUM_LSU_BLOCKS * DCACHE_CHANNELS);

inline constexpr uint32_t NUM_SOCKETS     = __UP(VX_CFG_NUM_CORES / VX_CFG_SOCKET_SIZE);

// Demand clients of the cluster L2 (sockets and gfx caches), excluding the
// walker's PTE-fetch slot.
inline constexpr uint32_t VX_CFG_L2_DEMAND_REQS  = NUM_SOCKETS * VX_CFG_L1_MEM_PORTS;
#ifdef VX_CFG_VM_ENABLE
// With one cluster, an L2 and no L3, the cluster L2 is the LLC and carries
// the device walker's PTE fetches on one extra client slot; in every other
// topology the walker is one more LLC client on the last requestor slot.
inline constexpr uint32_t PTW_ON_L2              = ((VX_CFG_NUM_CLUSTERS == 1)
                                                 && (VX_CFG_L2_ENABLED != 0)
                                                 && (VX_CFG_L3_ENABLED == 0)) ? 1 : 0;
inline constexpr uint32_t VX_CFG_L2_NUM_REQS     = VX_CFG_L2_DEMAND_REQS + PTW_ON_L2;
inline constexpr uint32_t VX_CFG_L2_PTW_IDX      = VX_CFG_L2_DEMAND_REQS;
inline constexpr uint32_t VX_CFG_L3_NUM_REQS     = VX_CFG_NUM_CLUSTERS * VX_CFG_L2_MEM_PORTS
                                                 + ((PTW_ON_L2 == 0) ? 1u : 0u);
inline constexpr uint32_t VX_CFG_L3_PTW_IDX      = VX_CFG_NUM_CLUSTERS * VX_CFG_L2_MEM_PORTS;
#else
inline constexpr uint32_t VX_CFG_L2_NUM_REQS     = VX_CFG_L2_DEMAND_REQS;
inline constexpr uint32_t VX_CFG_L3_NUM_REQS     = VX_CFG_NUM_CLUSTERS * VX_CFG_L2_MEM_PORTS;
#endif

inline constexpr uint32_t PER_ISSUE_WARPS = VX_CFG_NUM_WARPS / VX_CFG_ISSUE_WIDTH;
inline constexpr uint32_t ISSUE_WIS_BITS  = log2ceil(PER_ISSUE_WARPS);

// Fetch slice f owns the warps with wid % VX_CFG_FETCH_WIDTH == f; each has its
// own warp scheduler, fetch unit, decoder and icache request port.
inline constexpr uint32_t PER_FETCH_WARPS = VX_CFG_NUM_WARPS / VX_CFG_FETCH_WIDTH;
inline constexpr uint32_t ICACHE_NUM_REQS = VX_CFG_FETCH_WIDTH;
static_assert(VX_CFG_FETCH_WIDTH != 0 && (VX_CFG_FETCH_WIDTH & (VX_CFG_FETCH_WIDTH - 1)) == 0
              && VX_CFG_FETCH_WIDTH <= VX_CFG_NUM_WARPS
              && (VX_CFG_ISSUE_WIDTH % VX_CFG_FETCH_WIDTH) == 0,
              "FETCH_WIDTH must be a power of two that divides ISSUE_WIDTH and is at most NUM_WARPS");

} // namespace vortex
