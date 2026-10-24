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
// caps.h — shared device/ISA capability decoder for all runtime backends.
//
// The two 64-bit capability words (GPU_DEV_CAPS / GPU_ISA_CAPS) are read
// from the Command Processor regfile at the CP-internal offsets below.
//
// CACHE_LINE_SIZE / GLOBAL_MEM_SIZE / CLOCK_RATE / PEAK_MEM_BW are NOT
// encoded in the caps words (they are platform/runtime-specific); for
// those ids decode_caps() returns false and the backend resolves them.
// ============================================================================

#include <cstdint>
#include <vortex2.h>   // VX_CAPS_* ids

namespace vortex {

// CP regfile offsets for the static capability words.
// Pass to a backend's cp_mmio_read(), which applies any platform base offset.
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_EXT = 0x00C;
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_LO = 0x018;
static constexpr uint32_t CP_REG_GPU_DEV_CAPS_HI = 0x01C;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_LO = 0x020;
static constexpr uint32_t CP_REG_GPU_ISA_CAPS_HI = 0x024;

// Warps, sockets per cluster, clusters and the socket size carry an exact
// count, so a machine that is not a power of two reports truly; the field is
// zero on devices built before it existed, which report only the 3-bit log2
// field of GPU_DEV_CAPS.
inline uint64_t exact_or_log2(uint64_t exact, uint64_t dev_caps, uint32_t log2_lsb) {
  return exact ? exact : (1ull << ((dev_caps >> log2_lsb) & 0x7));
}

inline uint64_t exact_or_log2(uint64_t dev_caps, uint32_t exact_lsb, uint64_t exact_mask, uint32_t log2_lsb) {
  return exact_or_log2((dev_caps >> exact_lsb) & exact_mask, dev_caps, log2_lsb);
}

// Decode a VX_CAPS_* id from the two CP capability words. Returns true
// and writes *out when the id is encoded in the words; returns false
// for ids the caller must resolve itself (CACHE_LINE_SIZE,
// GLOBAL_MEM_SIZE, CLOCK_RATE, PEAK_MEM_BW).
inline bool decode_caps(uint64_t dev_caps, uint32_t dev_caps_ext, uint64_t isa_caps,
                        uint32_t caps_id, uint64_t* out) {
  // A CP without the extension word answers zero, a DECERR sentinel or all
  // ones; any reserved bit set means the word is absent.
  const uint32_t caps_ext = (dev_caps_ext >> 8) ? 0 : dev_caps_ext;
  const uint64_t socket_size = exact_or_log2(caps_ext & 0xFF, dev_caps, 14);
  switch (caps_id) {
  case VX_CAPS_VERSION:
    *out = (dev_caps >> 0) & 0xff;
    return true;
  case VX_CAPS_NUM_THREADS:
    *out = 1ull << ((dev_caps >> 8) & 0x7);
    return true;
  case VX_CAPS_NUM_WARPS:
    *out = exact_or_log2(dev_caps, 42, 0x7F, 11);
    return true;
  case VX_CAPS_NUM_CORES: {
    uint64_t cluster_size = exact_or_log2(dev_caps, 49, 0xFF, 17);
    uint64_t num_clusters = exact_or_log2(dev_caps, 57, 0x7F, 20);
    *out = num_clusters * cluster_size * socket_size;
    return true;
  }
  case VX_CAPS_SOCKET_SIZE:
    *out = socket_size;
    return true;
  case VX_CAPS_NUM_CLUSTERS:
    *out = exact_or_log2(dev_caps, 57, 0x7F, 20);
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

// Read GPU_DEV_CAPS / GPU_DEV_CAPS_EXT / GPU_ISA_CAPS from the CP regfile
// through a backend's cp_mmio_read primitive. `Reader` is any callable
// matching int(uint32_t off, uint32_t* out) — returns non-zero on failure.
// Returns 0 on success.
template <typename Reader>
inline int load_caps(Reader&& cp_mmio_read,
                     uint64_t* dev_caps, uint32_t* dev_caps_ext, uint64_t* isa_caps) {
  uint32_t lo = 0, hi = 0;
  if (cp_mmio_read(CP_REG_GPU_DEV_CAPS_EXT, dev_caps_ext)) return -1;
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
