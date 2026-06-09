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

#ifndef __VX_DXA_HOST_H__
#define __VX_DXA_HOST_H__

#include <cstdint>
#include <initializer_list>

#include <vortex.h>
#include <VX_types.h>

namespace vortex {
namespace dxa {

namespace detail {

struct reg_write {
  uint32_t off;
  uint32_t val;
};

inline int write_regs(vx_device_h dev, uint32_t dcr,
                      std::initializer_list<reg_write> writes) {
  for (const auto& w : writes) {
    if (int ret = vx_dcr_write(dev, dcr + w.off, w.val)) {
      return ret;
    }
  }
  return 0;
}

constexpr uint32_t pack_meta(uint32_t rank, uint32_t elem_size_enc) {
  return ((rank & ((1u << VX_DXA_DESC_META_DIM_BITS) - 1u)) << VX_DXA_DESC_META_DIM_LSB)
       | ((elem_size_enc & ((1u << VX_DXA_DESC_META_ELEMSZ_BITS) - 1u)) << VX_DXA_DESC_META_ELEMSZ_LSB);
}

constexpr uint32_t pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xffffu) << 16) | (lo & 0xffffu);
}

// elem_bytes must be a power of 2 (1, 2, 4, 8). Returns log2.
constexpr uint32_t elem_size_enc(uint32_t elem_bytes) {
  uint32_t enc = 0;
  for (uint32_t v = elem_bytes; v > 1; v >>= 1) {
    ++enc;
  }
  return enc;
}

} // namespace detail

// ── Destination SMEM layout ─────────────────────────────────────────
//
// ROW_MAJOR (default): a 2D tile is laid out as smem[i1 * tile0 + i0].
// K_MAJOR (NVIDIA-TMA transposing mode): smem[i0 * tile1 + i1]; the
//   writer scatters one element per beat at +tile1*elem_bytes per
//   element. Rank must be ≤ 2; the writer drains 1 element/cycle in
//   this mode (~8× slower for B-style tiles, amortized across the many
//   WGMMA uops that consume each tile from LMEM).
enum class Layout : uint32_t {
  RowMajor = 0,
  KMajor   = 1,
};

// ── Public API ─────────────────────────────────────────────────────
//
// Each function writes only the DCR registers needed for its rank.
//   1D:  7 DCR writes
//   2D: 10 DCR writes
//   3D: 14 DCR writes
//   4D: 17 DCR writes
//   5D: 21 DCR writes
//
// `base_addr` is written to VX_DCR_DXA_DESC_BASE_{LO,HI} and is
// consumed by the DXA worker's AXI master, which bypasses the
// per-core MMU. The backing buffer must be allocated with
// VX_MEM_PHYS (see vortex2.h).

// Program a 1D DXA descriptor (7 DCR writes).
//   size0: dimension size (elements)
//   tile0: tile size (elements per transfer)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_1d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0,
    uint32_t tile0,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(1, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, 0)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 2D DXA descriptor (10 DCR writes).
//   size0, size1: dimension sizes (elements)
//   stride0_bytes: byte stride between dim-1 rows
//   tile0, tile1: tile sizes (elements per transfer dimension)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_2d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1,
    uint32_t stride0_bytes,
    uint32_t tile0, uint32_t tile1,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(2, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 3D DXA descriptor (14 DCR writes).
//   size0, size1, size2: dimension sizes (elements)
//   stride0_bytes, stride1_bytes: byte strides for dim-1 and dim-2
//   tile0, tile1, tile2: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_3d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2,
    uint32_t stride0_bytes, uint32_t stride1_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(3, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, 0)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 4D DXA descriptor (17 DCR writes).
//   size0..size3: dimension sizes (elements)
//   stride0..stride2_bytes: byte strides for dim-1 through dim-3
//   tile0..tile3: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_4d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(4, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_SIZE3_OFF,     size3},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_STRIDE2_OFF,   stride2_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE3_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, tile3)},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program a 5D DXA descriptor (21 DCR writes).
//   size0..size4: dimension sizes (elements)
//   stride0..stride3_bytes: byte strides for dim-1 through dim-4
//   tile0..tile4: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
inline int program_5d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3, uint32_t size4,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes, uint32_t stride3_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3, uint32_t tile4,
    uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(5, detail::elem_size_enc(elem_bytes));
  return detail::write_regs(dev, dcr, {
    {VX_DCR_DXA_DESC_BASE_LO_OFF,   (uint32_t)(base_addr & 0xffffffffu)},
    {VX_DCR_DXA_DESC_BASE_HI_OFF,   (uint32_t)(base_addr >> 32)},
    {VX_DCR_DXA_DESC_SIZE0_OFF,     size0},
    {VX_DCR_DXA_DESC_SIZE1_OFF,     size1},
    {VX_DCR_DXA_DESC_SIZE2_OFF,     size2},
    {VX_DCR_DXA_DESC_SIZE3_OFF,     size3},
    {VX_DCR_DXA_DESC_SIZE4_OFF,     size4},
    {VX_DCR_DXA_DESC_STRIDE0_OFF,   stride0_bytes},
    {VX_DCR_DXA_DESC_STRIDE1_OFF,   stride1_bytes},
    {VX_DCR_DXA_DESC_STRIDE2_OFF,   stride2_bytes},
    {VX_DCR_DXA_DESC_STRIDE3_OFF,   stride3_bytes},
    {VX_DCR_DXA_DESC_META_OFF,      meta},
    {VX_DCR_DXA_DESC_ESTRIDE0_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE1_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE2_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE3_OFF,  1},
    {VX_DCR_DXA_DESC_ESTRIDE4_OFF,  1},
    {VX_DCR_DXA_DESC_TILESIZE01_OFF, detail::pack_2x16(tile0, tile1)},
    {VX_DCR_DXA_DESC_TILESIZE23_OFF, detail::pack_2x16(tile2, tile3)},
    {VX_DCR_DXA_DESC_TILESIZE4_OFF, tile4},
    {VX_DCR_DXA_DESC_CFILL_OFF,     0},
  });
}

// Program the multicast SMEM stride for an existing DXA descriptor.
//   smem_stride_bytes: byte offset between consecutive CTAs' SMEM bases
// Bar stride is always 1 (hardcoded in hardware).
// Multicast is active when cta_mask has more than one bit set.
inline int set_multicast(
    vx_device_h dev, uint32_t slot,
    uint32_t smem_stride_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  return vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_SMEM_STRIDE_OFF, smem_stride_bytes);
}

// Override the destination SMEM layout for an already-programmed descriptor.
// META is rewritten from scratch (rank + elem_bytes + layout); call AFTER
// program_Nd, passing the same rank and elem_bytes you used. (DCR is
// write-only on most paths — we don't read-modify-write META; we re-pack
// it locally.)
inline int set_layout(
    vx_device_h dev, uint32_t slot,
    Layout layout,
    uint32_t rank, uint32_t elem_bytes) {
  uint32_t dcr  = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = detail::pack_meta(rank, detail::elem_size_enc(elem_bytes))
                | ((static_cast<uint32_t>(layout) & ((1u << VX_DXA_DESC_META_LAYOUT_BITS) - 1u))
                   << VX_DXA_DESC_META_LAYOUT_LSB);
  return vx_dcr_write(dev, dcr + VX_DCR_DXA_DESC_META_OFF, meta);
}

} // namespace dxa
} // namespace vortex

#endif // __VX_DXA_HOST_H__
