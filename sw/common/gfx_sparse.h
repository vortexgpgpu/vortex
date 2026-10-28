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

// Addressing of sparse (tiled) images.
//
// A sparse image stores each mip level as 64 KiB tiles of the Vulkan standard
// block shape: tile (x >> lw, y >> lh, z >> ld) is tile number
// tx + ty * TX + tz * TX * TY of the level, and a texel sits at
// (x & mw) + (y & mh) * tw + (z & md) * tw * th blocks into its tile. Array
// layers follow one another at the level's layer stride, a whole number of
// tiles. Residency is one bit per 64 KiB of the resource: bit (offset >> 16) of
// the bitmap, offset counted from the resource's start.

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GFX_SPARSE_TILE_LOG2 16

// One sparse image view: where its residency lives, where the view starts in
// the resource, and the tile shape its texels are addressed by.
//
// A view of a 3D level may start at a slice inside a tile's depth. Its origin
// is then that slice's first texel, and GFX_SPARSE_Z0 of its shape the slice's
// depth inside the tile: view slice z is tile slice z + Z0, whose texels lie
// gfx_sparse_off(.., z + Z0) - gfx_sparse_off(.., Z0) from the origin.
typedef struct {
  uint64_t bitmap;     // device address of the residency bitmap; 0 => not sparse
  uint32_t view_base;  // bytes from the resource's start to the view's origin
  uint32_t shape;      // GFX_SPARSE_SHAPE(); never 0 for a sparse view
} gfx_sparse_t;

// The tile shape: log2 of the tile's width, height and depth in blocks, and
// the bytes of one block.
#define GFX_SPARSE_SHAPE(lw, lh, ld, bpp) \
  ((uint32_t)(lw) | ((uint32_t)(lh) << 4) | ((uint32_t)(ld) << 8) | ((uint32_t)(bpp) << 16))
#define GFX_SPARSE_Z0_SHIFT 24
#define GFX_SPARSE_WITH_Z0(s, z0) ((s) | ((uint32_t)(z0) << GFX_SPARSE_Z0_SHIFT))
#define GFX_SPARSE_LW(s)  ((s) & 0xfu)
#define GFX_SPARSE_LH(s)  (((s) >> 4) & 0xfu)
#define GFX_SPARSE_LD(s)  (((s) >> 8) & 0xfu)
#define GFX_SPARSE_BPP(s) (((s) >> 16) & 0x1fu)
#define GFX_SPARSE_Z0(s)  (((s) >> GFX_SPARSE_Z0_SHIFT) & 0x1fu)

// Byte offset of block (x, y, z) from the origin of a level `w` x `h` blocks.
// z is the 3D slice; an array layer is added at the layer stride instead.
static inline uint32_t gfx_sparse_off(uint32_t shape, uint32_t w, uint32_t h,
                                      uint32_t x, uint32_t y, uint32_t z) {
  const uint32_t lw = GFX_SPARSE_LW(shape), lh = GFX_SPARSE_LH(shape),
                 ld = GFX_SPARSE_LD(shape);
  const uint32_t mw = (1u << lw) - 1u, mh = (1u << lh) - 1u, md = (1u << ld) - 1u;
  const uint32_t tx = (w + mw) >> lw, ty = (h + mh) >> lh;
  const uint32_t tile = (x >> lw) + ((y >> lh) + (z >> ld) * ty) * tx;
  const uint32_t in_tile = (x & mw) + ((y & mh) << lw) + ((z & md) << (lw + lh));
  return (tile << GFX_SPARSE_TILE_LOG2) + in_tile * GFX_SPARSE_BPP(shape);
}

// A box of one level of an image: addressed through the level's tiles when
// `shape` is set, the base then being the level's start (layer 0) and
// (x, y, z) the box's origin in it; linearly otherwise, the base then being
// the box's first texel.
typedef struct {
  uint32_t shape;    // gfx_sparse_t::shape; 0 => linear
  uint32_t w, h;     // the level's extent in blocks
  uint32_t x, y, z;  // the box's origin: blocks, and the 3D slice or array layer
} gfx_sparse_box_t;

// Byte offset of block (x, y, z) of the box from its base: `texel`, `row` and
// `slice` are the linear pitches; a tiled level's array layers keep `slice`.
static inline uint64_t gfx_sparse_box_off(const gfx_sparse_box_t* b, uint32_t texel,
                                          uint32_t row, uint64_t slice,
                                          uint32_t x, uint32_t y, uint32_t z) {
  if (!b->shape) {
    return z * slice + (uint64_t)y * row + (uint64_t)x * texel;
  }
  const uint32_t lz = b->z + z;
  if (GFX_SPARSE_LD(b->shape)) {
    return gfx_sparse_off(b->shape, b->w, b->h, b->x + x, b->y + y, lz);
  }
  return gfx_sparse_off(b->shape, b->w, b->h, b->x + x, b->y + y, 0) + lz * slice;
}

// Whether the 64 KiB tile holding byte `off` of the resource is bound.
static inline uint32_t gfx_sparse_resident(uint64_t bitmap, uint32_t off) {
  const volatile uint32_t* words = (const volatile uint32_t*)(uintptr_t)bitmap;
  return (words[off >> (GFX_SPARSE_TILE_LOG2 + 5)] >> ((off >> GFX_SPARSE_TILE_LOG2) & 31u)) & 1u;
}

#ifdef __cplusplus
}
#endif
