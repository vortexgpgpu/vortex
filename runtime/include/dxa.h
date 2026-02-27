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

#include <vortex.h>
#include <VX_types.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Internal helpers (not part of public API) ──────────────────────

static inline uint32_t vx_dxa__pack_meta(uint32_t rank, uint32_t elem_size_enc) {
  return ((rank & ((1u << VX_DXA_DESC_META_DIM_BITS) - 1u)) << VX_DXA_DESC_META_DIM_LSB)
       | ((elem_size_enc & ((1u << VX_DXA_DESC_META_ELEMSZ_BITS) - 1u)) << VX_DXA_DESC_META_ELEMSZ_LSB);
}

static inline uint32_t vx_dxa__pack_2x16(uint32_t lo, uint32_t hi) {
  return ((hi & 0xffffu) << 16) | (lo & 0xffffu);
}

// elem_bytes must be a power of 2 (1, 2, 4, 8). Returns log2.
static inline uint32_t vx_dxa__elem_size_enc(uint32_t elem_bytes) {
  uint32_t enc = 0;
  uint32_t v = elem_bytes;
  while (v > 1) { v >>= 1; ++enc; }
  return enc;
}

// ── Public API ─────────────────────────────────────────────────────
//
// Each function writes only the DCR registers needed for its rank.
//   1D:  7 DCR writes
//   2D: 10 DCR writes
//   3D: 14 DCR writes
//   4D: 17 DCR writes
//   5D: 21 DCR writes

// Program a 1D DXA descriptor (7 DCR writes).
//   size0: dimension size (elements)
//   tile0: tile size (elements per transfer)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
static inline int vx_dxa_program_desc_1d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0,
    uint32_t tile0,
    uint32_t elem_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = vx_dxa__pack_meta(1, vx_dxa__elem_size_enc(elem_bytes));
  int ret;
#define VX_DXA__W(off, val) do { ret = vx_dcr_write(dev, dcr + (off), (val)); if (ret) return ret; } while(0)
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  VX_DXA__W(VX_DCR_DXA_DESC_META_OFF, meta);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE01_OFF, vx_dxa__pack_2x16(tile0, 0));
  VX_DXA__W(VX_DCR_DXA_DESC_CFILL_OFF, 0);
#undef VX_DXA__W
  return 0;
}

// Program a 2D DXA descriptor (10 DCR writes).
//   size0, size1: dimension sizes (elements)
//   stride0_bytes: byte stride between dim-1 rows
//   tile0, tile1: tile sizes (elements per transfer dimension)
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
static inline int vx_dxa_program_desc_2d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1,
    uint32_t stride0_bytes,
    uint32_t tile0, uint32_t tile1,
    uint32_t elem_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = vx_dxa__pack_meta(2, vx_dxa__elem_size_enc(elem_bytes));
  int ret;
#define VX_DXA__W(off, val) do { ret = vx_dcr_write(dev, dcr + (off), (val)); if (ret) return ret; } while(0)
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE1_OFF, size1);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_META_OFF, meta);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE01_OFF, vx_dxa__pack_2x16(tile0, tile1));
  VX_DXA__W(VX_DCR_DXA_DESC_CFILL_OFF, 0);
#undef VX_DXA__W
  return 0;
}

// Program a 3D DXA descriptor (14 DCR writes).
//   size0, size1, size2: dimension sizes (elements)
//   stride0_bytes, stride1_bytes: byte strides for dim-1 and dim-2
//   tile0, tile1, tile2: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
static inline int vx_dxa_program_desc_3d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2,
    uint32_t stride0_bytes, uint32_t stride1_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2,
    uint32_t elem_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = vx_dxa__pack_meta(3, vx_dxa__elem_size_enc(elem_bytes));
  int ret;
#define VX_DXA__W(off, val) do { ret = vx_dcr_write(dev, dcr + (off), (val)); if (ret) return ret; } while(0)
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE1_OFF, size1);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE2_OFF, size2);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE1_OFF, stride1_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_META_OFF, meta);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE2_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE01_OFF, vx_dxa__pack_2x16(tile0, tile1));
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE23_OFF, vx_dxa__pack_2x16(tile2, 0));
  VX_DXA__W(VX_DCR_DXA_DESC_CFILL_OFF, 0);
#undef VX_DXA__W
  return 0;
}

// Program a 4D DXA descriptor (17 DCR writes).
//   size0..size3: dimension sizes (elements)
//   stride0..stride2_bytes: byte strides for dim-1 through dim-3
//   tile0..tile3: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
static inline int vx_dxa_program_desc_4d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3,
    uint32_t elem_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = vx_dxa__pack_meta(4, vx_dxa__elem_size_enc(elem_bytes));
  int ret;
#define VX_DXA__W(off, val) do { ret = vx_dcr_write(dev, dcr + (off), (val)); if (ret) return ret; } while(0)
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE1_OFF, size1);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE2_OFF, size2);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE3_OFF, size3);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE1_OFF, stride1_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE2_OFF, stride2_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_META_OFF, meta);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE2_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE3_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE01_OFF, vx_dxa__pack_2x16(tile0, tile1));
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE23_OFF, vx_dxa__pack_2x16(tile2, tile3));
  VX_DXA__W(VX_DCR_DXA_DESC_CFILL_OFF, 0);
#undef VX_DXA__W
  return 0;
}

// Program a 5D DXA descriptor (21 DCR writes).
//   size0..size4: dimension sizes (elements)
//   stride0..stride3_bytes: byte strides for dim-1 through dim-4
//   tile0..tile4: tile sizes
//   elem_bytes: element size in bytes (must be power of 2: 1, 2, 4, or 8)
static inline int vx_dxa_program_desc_5d(
    vx_device_h dev, uint32_t slot,
    uint64_t base_addr,
    uint32_t size0, uint32_t size1, uint32_t size2, uint32_t size3, uint32_t size4,
    uint32_t stride0_bytes, uint32_t stride1_bytes, uint32_t stride2_bytes, uint32_t stride3_bytes,
    uint32_t tile0, uint32_t tile1, uint32_t tile2, uint32_t tile3, uint32_t tile4,
    uint32_t elem_bytes) {
  uint32_t dcr = VX_DCR_DXA_DESC_BASE + slot * VX_DCR_DXA_DESC_STRIDE;
  uint32_t meta = vx_dxa__pack_meta(5, vx_dxa__elem_size_enc(elem_bytes));
  int ret;
#define VX_DXA__W(off, val) do { ret = vx_dcr_write(dev, dcr + (off), (val)); if (ret) return ret; } while(0)
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_LO_OFF, (uint32_t)(base_addr & 0xffffffffu));
  VX_DXA__W(VX_DCR_DXA_DESC_BASE_HI_OFF, (uint32_t)(base_addr >> 32));
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE0_OFF, size0);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE1_OFF, size1);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE2_OFF, size2);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE3_OFF, size3);
  VX_DXA__W(VX_DCR_DXA_DESC_SIZE4_OFF, size4);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE0_OFF, stride0_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE1_OFF, stride1_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE2_OFF, stride2_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_STRIDE3_OFF, stride3_bytes);
  VX_DXA__W(VX_DCR_DXA_DESC_META_OFF, meta);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE0_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE1_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE2_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE3_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_ESTRIDE4_OFF, 1);
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE01_OFF, vx_dxa__pack_2x16(tile0, tile1));
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE23_OFF, vx_dxa__pack_2x16(tile2, tile3));
  VX_DXA__W(VX_DCR_DXA_DESC_TILESIZE4_OFF, tile4);
  VX_DXA__W(VX_DCR_DXA_DESC_CFILL_OFF, 0);
#undef VX_DXA__W
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif // __VX_DXA_HOST_H__
