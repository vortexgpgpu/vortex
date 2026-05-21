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

#ifndef VORTEX_RUNTIME_VX_CAPS_H
#define VORTEX_RUNTIME_VX_CAPS_H

// ============================================================================
// vx_caps.h — the single device/ISA capability decoder, shared by every
// runtime backend (xrt / opae / simx / rtlsim / gem5).
//
// The two 64-bit capability words (GPU_DEV_CAPS / GPU_ISA_CAPS) are read
// from the Command Processor regfile at the CP-internal offsets below.
// They are packed by exactly one producer per representation:
//   * RTL  — hw/rtl/cp/VX_cp_axil_regfile.sv  (gpu_dev_caps/gpu_isa_caps)
//   * C++  — sim/common/cmd_processor.cpp     (gpu_dev_caps/gpu_isa_caps)
// decode_caps() below is the matching inverse, replacing the per-backend
// get_caps() bit-slicing that previously existed in five copies.
//
// CACHE_LINE_SIZE / GLOBAL_MEM_SIZE / CLOCK_RATE / PEAK_MEM_BW are NOT
// encoded in the caps words (they are platform/runtime-specific); for
// those ids decode_caps() returns false and the backend resolves them.
// ============================================================================

#include <cstdint>
#include <vortex2.h>   // VX_CAPS_* ids

namespace vortex {

// CP regfile offsets of the static caps words (CP-internal, matching
// hw/rtl/cp/VX_cp_axil_regfile.sv). Pass these to a backend's
// cp_mmio_read(), which applies any platform base itself.
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_LO = 0x018;
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_HI = 0x01C;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_LO = 0x020;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_HI = 0x024;

// Decode a VX_CAPS_* id from the two CP capability words. Returns true
// and writes *out when the id is encoded in the words; returns false
// for ids the caller must resolve itself (CACHE_LINE_SIZE,
// GLOBAL_MEM_SIZE, CLOCK_RATE, PEAK_MEM_BW).
inline bool decode_caps(uint64_t dev_caps, uint64_t isa_caps,
                        uint32_t caps_id, uint64_t* out) {
  switch (caps_id) {
  case VX_CAPS_VERSION:
    *out = (dev_caps >> 0) & 0xff;
    return true;
  case VX_CAPS_NUM_THREADS:
    *out = 1ull << ((dev_caps >> 8) & 0x7);
    return true;
  case VX_CAPS_NUM_WARPS:
    *out = 1ull << ((dev_caps >> 11) & 0x7);
    return true;
  case VX_CAPS_NUM_CORES: {
    uint32_t socket_size  = 1u << ((dev_caps >> 14) & 0x7);
    uint32_t cluster_size = 1u << ((dev_caps >> 17) & 0x7);
    uint32_t num_clusters = 1u << ((dev_caps >> 20) & 0x7);
    *out = uint64_t(num_clusters) * cluster_size * socket_size;
    return true;
  }
  case VX_CAPS_SOCKET_SIZE:
    *out = 1ull << ((dev_caps >> 14) & 0x7);
    return true;
  case VX_CAPS_NUM_CLUSTERS:
    *out = 1ull << ((dev_caps >> 20) & 0x7);
    return true;
  case VX_CAPS_ISSUE_WIDTH:
    *out = 1ull << ((dev_caps >> 23) & 0x7);
    return true;
  case VX_CAPS_LOCAL_MEM_SIZE:
    *out = 1ull << ((dev_caps >> 26) & 0xff);
    return true;
  case VX_CAPS_ISA_FLAGS:
    *out = isa_caps;
    return true;
  case VX_CAPS_NUM_MEM_BANKS:
    *out = 1ull << ((dev_caps >> 34) & 0x7);
    return true;
  case VX_CAPS_MEM_BANK_SIZE:
    *out = 1ull << (20 + ((dev_caps >> 37) & 0x1f));
    return true;
  default:
    return false;
  }
}

// Read GPU_DEV_CAPS / GPU_ISA_CAPS from the CP regfile through a
// backend's cp_mmio_read primitive. `Reader` is any callable matching
// int(uint32_t off, uint32_t* out) — returns non-zero on failure.
// Returns 0 on success.
template <typename Reader>
inline int load_caps(Reader&& cp_mmio_read,
                     uint64_t* dev_caps, uint64_t* isa_caps) {
  uint32_t lo = 0, hi = 0;
  if (cp_mmio_read(CP_REG_GPU_DEV_CAPS_LO, &lo)) return -1;
  if (cp_mmio_read(CP_REG_GPU_DEV_CAPS_HI, &hi)) return -1;
  *dev_caps = uint64_t(lo) | (uint64_t(hi) << 32);
  if (cp_mmio_read(CP_REG_GPU_ISA_CAPS_LO, &lo)) return -1;
  if (cp_mmio_read(CP_REG_GPU_ISA_CAPS_HI, &hi)) return -1;
  *isa_caps = uint64_t(lo) | (uint64_t(hi) << 32);
  return 0;
}

} // namespace vortex

#endif // VORTEX_RUNTIME_VX_CAPS_H
