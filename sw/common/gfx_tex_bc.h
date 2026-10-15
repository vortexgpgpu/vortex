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

// Block-compressed texel decode (BC1-BC7), one texel at a time: a sample reads
// the 4x4 block its tap lands in and decodes only that texel, so the texture
// stays compressed in memory. Freestanding, as gfx_frag_tex.h is.

#pragma once

#include <stdint.h>

namespace gfx_tex {

// A block's 128 (or 64) bits, little-endian.
struct BcBits {
  uint64_t lo;
  uint64_t hi;
};

static inline BcBits BcLoad(const void* p, uint32_t bytes) {
  BcBits b{0, 0};
  __builtin_memcpy(&b.lo, p, 8);
  if (bytes > 8) {
    __builtin_memcpy(&b.hi, (const uint8_t*)p + 8, 8);
  }
  return b;
}

// `count` bits starting at bit `first`, low bit first.
static inline uint32_t BcGet(const BcBits& b, uint32_t first, uint32_t count) {
  if (count == 0) {
    return 0;
  }
  uint64_t v;
  if (first >= 64) {
    v = b.hi >> (first - 64);
  } else if (first + count <= 64) {
    v = b.lo >> first;
  } else {
    v = (b.lo >> first) | (b.hi << (64 - first));
  }
  return (uint32_t)(v & ((1ull << count) - 1));
}

static const uint32_t kBcPartition2[64] = {
  0x50505050, 0x40404040, 0x54545454, 0x54505040, 0x50404000, 0x55545450, 0x55545040, 0x54504000,
  0x50400000, 0x55555450, 0x55544000, 0x54400000, 0x55555440, 0x55550000, 0x55555500, 0x55000000,
  0x55150100, 0x00004054, 0x15010000, 0x00405054, 0x00004050, 0x15050100, 0x05010000, 0x40505054,
  0x00404050, 0x05010100, 0x14141414, 0x05141450, 0x01155440, 0x00555500, 0x15014054, 0x05414150,
  0x44444444, 0x55005500, 0x11441144, 0x05055050, 0x05500550, 0x11114444, 0x41144114, 0x44111144,
  0x15055054, 0x01055040, 0x05041050, 0x05455150, 0x14414114, 0x50050550, 0x41411414, 0x00141400,
  0x00041504, 0x00105410, 0x10541000, 0x04150400, 0x50410514, 0x41051450, 0x05415014, 0x14054150,
  0x41050514, 0x41505014, 0x40011554, 0x54150140, 0x50505500, 0x00555050, 0x15151010, 0x54540404,
};
static const uint32_t kBcPartition3[64] = {
  0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
  0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
  0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
  0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
  0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
  0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
  0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
  0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};
static const uint8_t kBcAnchor2[64] = {
  15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
  15,  2,  8,  2,  2,  8,  8, 15,  2,  8,  2,  2,  8,  8,  2,  2,
  15, 15,  6,  8,  2,  8, 15, 15,  2,  8,  2,  2,  2, 15, 15,  6,
   6,  2,  6,  8, 15, 15,  2,  2, 15, 15, 15, 15, 15,  2,  2, 15,
};
static const uint8_t kBcAnchor3b[64] = {
   3,  3, 15, 15,  8,  3, 15, 15,  8,  8,  6,  6,  6,  5,  3,  3,
   3,  3,  8, 15,  3,  3,  6, 10,  5,  8,  8,  6,  8,  5, 15, 15,
   8, 15,  3,  5,  6, 10,  8, 15, 15,  3, 15,  5, 15, 15, 15, 15,
   3, 15,  5,  5,  5,  8,  5, 10,  5, 10,  8, 13, 15, 12,  3,  3,
};
static const uint8_t kBcAnchor3c[64] = {
  15,  8,  8,  3, 15, 15,  3,  8, 15, 15, 15, 15, 15, 15, 15,  8,
  15,  8, 15,  3, 15,  8, 15,  8,  3, 15,  6, 10, 15, 15, 10,  8,
  15,  3, 15, 10, 10,  8,  9, 10,  6, 15,  8, 15,  3,  6,  6,  8,
  15,  3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,  3, 15, 15,  8,
};

// The 2-bit subset of texel t under a 2- or 3-subset partition.
static inline uint32_t BcSubset(uint32_t subsets, uint32_t partition, uint32_t t) {
  if (subsets == 1) {
    return 0;
  }
  const uint32_t w = (subsets == 2) ? kBcPartition2[partition] : kBcPartition3[partition];
  return (w >> (2 * t)) & 3u;
}

// The anchor texel of a subset: its index carries one bit fewer.
static inline uint32_t BcAnchor(uint32_t subsets, uint32_t partition, uint32_t subset) {
  if (subset == 0) {
    return 0;
  }
  if (subsets == 2) {
    return kBcAnchor2[partition];
  }
  return (subset == 1) ? kBcAnchor3b[partition] : kBcAnchor3c[partition];
}

// Bit offset of texel t's index within an index field of `bits` per texel, the
// anchors of `subsets` subsets one bit short.
static inline uint32_t BcIndexOffset(uint32_t subsets, uint32_t partition, uint32_t t,
                                     uint32_t bits) {
  uint32_t off = t * bits;
  for (uint32_t s = 0; s < subsets; ++s) {
    if (BcAnchor(subsets, partition, s) < t) {
      off -= 1;
    }
  }
  return off;
}

static const uint8_t kBcWeight2[4]  = { 0, 21, 43, 64 };
static const uint8_t kBcWeight3[8]  = { 0, 9, 18, 27, 37, 46, 55, 64 };
static const uint8_t kBcWeight4[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

static inline uint32_t BcWeight(uint32_t bits, uint32_t index) {
  return (bits == 2) ? kBcWeight2[index] : ((bits == 3) ? kBcWeight3[index] : kBcWeight4[index]);
}

static inline int32_t BcInterpolate(int32_t a, int32_t b, uint32_t index, uint32_t bits) {
  const int32_t w = (int32_t)BcWeight(bits, index);
  return ((64 - w) * a + w * b + 32) >> 6;
}

// RGB565 endpoint, bits replicated to eight per channel: R | G << 8 | B << 16.
static inline uint32_t Bc565(uint32_t c) {
  const uint32_t r = (c >> 11) & 0x1f, g = (c >> 5) & 0x3f, b = c & 0x1f;
  return ((r << 3) | (r >> 2)) | (((g << 2) | (g >> 4)) << 8) | (((b << 3) | (b >> 2)) << 16);
}

// a*(n-k) + b*k over n, per byte of an RGB word.
static inline uint32_t BcMix(uint32_t a, uint32_t b, uint32_t k, uint32_t n) {
  uint32_t out = 0;
  for (uint32_t c = 0; c < 3; ++c) {
    const uint32_t x = (a >> (8 * c)) & 0xff, y = (b >> (8 * c)) & 0xff;
    out |= ((x * (n - k) + y * k) / n) << (8 * c);
  }
  return out;
}

// The colour half of BC1-BC3: texel t of the 64-bit colour block at `p`, as
// R | G << 8 | B << 16 | A << 24. `punch` selects BC1's three-colour mode when
// color0 <= color1, whose fourth entry is black, transparent when
// `punch_alpha`.
static inline uint32_t BcColorBlock(const uint8_t* p, uint32_t t, bool punch,
                                    bool punch_alpha) {
  uint16_t c0, c1;
  uint32_t idx;
  __builtin_memcpy(&c0, p, 2);
  __builtin_memcpy(&c1, p + 2, 2);
  __builtin_memcpy(&idx, p + 4, 4);
  const uint32_t i = (idx >> (2 * t)) & 3u;
  const uint32_t e0 = Bc565(c0), e1 = Bc565(c1);
  if (i == 0) {
    return e0 | 0xff000000u;
  }
  if (i == 1) {
    return e1 | 0xff000000u;
  }
  if (!punch || c0 > c1) {
    return BcMix(e0, e1, i - 1, 3) | 0xff000000u;
  }
  if (i == 2) {
    return BcMix(e0, e1, 1, 2) | 0xff000000u;
  }
  return punch_alpha ? 0u : 0xff000000u;
}

// BC3's (and BC4/BC5's unsigned) 8-bit endpoint ramp, as integers: texel t of
// the 64-bit block at `p`.
static inline uint32_t BcAlphaBlock8(const uint8_t* p, uint32_t t) {
  uint64_t bits;
  __builtin_memcpy(&bits, p, 8);
  const uint32_t a0 = p[0], a1 = p[1];
  const uint32_t i = (uint32_t)(bits >> (16 + 3 * t)) & 7u;
  if (i == 0) {
    return a0;
  }
  if (i == 1) {
    return a1;
  }
  if (a0 > a1) {
    return (a0 * (8 - i) + a1 * (i - 1)) / 7;
  }
  if (i == 6) {
    return 0;
  }
  if (i == 7) {
    return 255;
  }
  return (a0 * (6 - i) + a1 * (i - 1)) / 5;
}

// BC4's ramp in float, unsigned or signed: texel t of the 64-bit block at `p`.
// A signed endpoint is snorm8 (-128 reads as -1).
static inline float BcRedBlock(const uint8_t* p, uint32_t t, bool sign) {
  uint64_t bits;
  __builtin_memcpy(&bits, p, 8);
  const uint32_t i = (uint32_t)(bits >> (16 + 3 * t)) & 7u;
  float r0, r1;
  if (sign) {
    const int32_t s0 = (int8_t)p[0], s1 = (int8_t)p[1];
    r0 = (s0 < -127) ? -1.0f : (float)s0 / 127.0f;
    r1 = (s1 < -127) ? -1.0f : (float)s1 / 127.0f;
  } else {
    r0 = (float)p[0] / 255.0f;
    r1 = (float)p[1] / 255.0f;
  }
  if (i == 0) {
    return r0;
  }
  if (i == 1) {
    return r1;
  }
  // The ramp is chosen on the decoded endpoints, so -127 and -128 (both -1)
  // take the six-step ramp, as the D3D rule and the CTS reference do.
  if (r0 > r1) {
    return (r0 * (float)(8 - i) + r1 * (float)(i - 1)) / 7.0f;
  }
  if (i == 6) {
    return sign ? -1.0f : 0.0f;
  }
  if (i == 7) {
    return 1.0f;
  }
  return (r0 * (float)(6 - i) + r1 * (float)(i - 1)) / 5.0f;
}

// BC1 (opaque or one-bit alpha), BC2 and BC3: texel t as R | G << 8 | B << 16 |
// A << 24.
static inline uint32_t BcDecodeBc1(const void* blk, uint32_t t, bool alpha) {
  return BcColorBlock((const uint8_t*)blk, t, true, alpha);
}

static inline uint32_t BcDecodeBc2(const void* blk, uint32_t t) {
  const uint8_t* p = (const uint8_t*)blk;
  const uint32_t a = (p[t >> 1] >> (4 * (t & 1))) & 0xf;
  return (BcColorBlock(p + 8, t, false, false) & 0x00ffffffu) | ((a * 17) << 24);
}

static inline uint32_t BcDecodeBc3(const void* blk, uint32_t t) {
  const uint8_t* p = (const uint8_t*)blk;
  return (BcColorBlock(p + 8, t, false, false) & 0x00ffffffu) | (BcAlphaBlock8(p, t) << 24);
}

// BC7: texel t as R | G << 8 | B << 16 | A << 24. A reserved mode (byte 0 zero)
// decodes to transparent black.
static inline uint32_t BcDecodeBc7(const void* blk, uint32_t t) {
  static const uint8_t kSubsets[8]  = { 3, 2, 3, 2, 1, 1, 1, 2 };
  static const uint8_t kPartBits[8] = { 4, 6, 6, 6, 0, 0, 0, 6 };
  static const uint8_t kColorBits[8] = { 4, 6, 5, 7, 5, 7, 7, 5 };
  static const uint8_t kAlphaBits[8] = { 0, 0, 0, 0, 6, 8, 7, 5 };
  // 0: none, 1: one per endpoint, 2: one per subset (shared by its endpoints)
  static const uint8_t kPBits[8]    = { 1, 2, 0, 1, 0, 0, 1, 1 };
  static const uint8_t kIndexBits[8] = { 3, 3, 2, 2, 2, 2, 4, 2 };
  const BcBits b = BcLoad(blk, 16);
  const uint8_t b0 = (uint8_t)b.lo;
  if (b0 == 0) {
    return 0;
  }
  const uint32_t mode = (uint32_t)__builtin_ctz(b0);
  uint32_t off = mode + 1;
  const uint32_t subsets = kSubsets[mode];
  const uint32_t partition = BcGet(b, off, kPartBits[mode]);
  off += kPartBits[mode];
  uint32_t rotation = 0, idx_mode = 0;
  if (mode == 4 || mode == 5) {
    rotation = BcGet(b, off, 2);
    off += 2;
    if (mode == 4) {
      idx_mode = BcGet(b, off, 1);
      off += 1;
    }
  }
  const uint32_t subset = BcSubset(subsets, partition, t);
  const uint32_t neps = subsets * 2;
  const uint32_t cb = kColorBits[mode], ab = kAlphaBits[mode];
  // Endpoints are stored channel-major: every endpoint's red, then green, ...
  uint32_t ep[2][4];
  for (uint32_t c = 0; c < 4; ++c) {
    const uint32_t n = (c < 3) ? cb : ab;
    for (uint32_t e = 0; e < 2; ++e) {
      ep[e][c] = BcGet(b, off + (2 * subset + e) * n, n);
    }
    off += neps * n;
  }
  uint32_t pb = 0;
  if (kPBits[mode] == 1) {
    for (uint32_t e = 0; e < 2; ++e) {
      const uint32_t p = BcGet(b, off + 2 * subset + e, 1);
      for (uint32_t c = 0; c < 4; ++c) {
        ep[e][c] = (ep[e][c] << 1) | p;
      }
    }
    off += neps;
    pb = 1;
  } else if (kPBits[mode] == 2) {
    const uint32_t p = BcGet(b, off + subset, 1);
    for (uint32_t e = 0; e < 2; ++e) {
      for (uint32_t c = 0; c < 4; ++c) {
        ep[e][c] = (ep[e][c] << 1) | p;
      }
    }
    off += subsets;
    pb = 1;
  }
  // Each channel widened to eight bits by replicating its top bits.
  for (uint32_t e = 0; e < 2; ++e) {
    for (uint32_t c = 0; c < 4; ++c) {
      const uint32_t n = ((c < 3) ? cb : ab) + pb;
      if (c == 3 && ab == 0) {
        ep[e][c] = 255;
        continue;
      }
      ep[e][c] = (ep[e][c] << (8 - n)) | (ep[e][c] >> (2 * n - 8));
    }
  }
  // The colour index field, then (modes 4 and 5) the alpha one; mode 4's index
  // mode swaps which of its two fields is the colour's.
  const uint32_t ib = kIndexBits[mode];
  const uint32_t anchor_rm = (t == BcAnchor(subsets, partition, subset)) ? 1u : 0u;
  uint32_t ci, ai, cbits, abits;
  if (mode == 4 || mode == 5) {
    const uint32_t ib2 = (mode == 4) ? 3u : 2u;   // the second field's width
    const uint32_t i0 = BcGet(b, off + (t ? t * ib - 1 : 0), ib - anchor_rm);
    const uint32_t off2 = off + 16 * ib - 1;
    const uint32_t i1 = BcGet(b, off2 + (t ? t * ib2 - 1 : 0), ib2 - anchor_rm);
    if (idx_mode) {
      ci = i1; cbits = ib2; ai = i0; abits = ib;
    } else {
      ci = i0; cbits = ib; ai = i1; abits = ib2;
    }
  } else {
    ci = BcGet(b, off + BcIndexOffset(subsets, partition, t, ib), ib - anchor_rm);
    cbits = ib;
    ai = ci;
    abits = ib;
  }
  uint32_t ch[4];
  for (uint32_t c = 0; c < 3; ++c) {
    ch[c] = (uint32_t)BcInterpolate((int32_t)ep[0][c], (int32_t)ep[1][c], ci, cbits);
  }
  ch[3] = (uint32_t)BcInterpolate((int32_t)ep[0][3], (int32_t)ep[1][3], ai, abits);
  if (rotation) {
    const uint32_t c = rotation - 1;
    const uint32_t tmp = ch[3];
    ch[3] = ch[c];
    ch[c] = tmp;
  }
  return ch[0] | (ch[1] << 8) | (ch[2] << 16) | (ch[3] << 24);
}

namespace bc6 {
// field (r0..r3, g0..g3, b0..b3 = 0..11), first bit, last bit, shift
static const uint8_t kBc6Fields[][4] = {
  { 6,  2,  2,  4},
  {10,  3,  3,  4},
  {11,  4,  4,  4},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 39,  0},
  { 7, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 49,  0},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 59,  0},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 75,  0},
  {11, 76, 76,  3},
  { 6,  2,  2,  5},
  { 7,  3,  3,  4},
  { 7,  4,  4,  5},
  { 0,  5, 11,  0},
  {11, 12, 12,  0},
  {11, 13, 13,  1},
  {10, 14, 14,  4},
  { 4, 15, 21,  0},
  {10, 22, 22,  5},
  {11, 23, 23,  2},
  { 6, 24, 24,  4},
  { 8, 25, 31,  0},
  {11, 32, 32,  3},
  {11, 33, 33,  5},
  {11, 34, 34,  4},
  { 1, 35, 40,  0},
  { 6, 41, 44,  0},
  { 5, 45, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 60,  0},
  {10, 61, 64,  0},
  { 2, 65, 70,  0},
  { 3, 71, 76,  0},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 39,  0},
  { 0, 40, 40, 10},
  { 6, 41, 44,  0},
  { 5, 45, 48,  0},
  { 4, 49, 49, 10},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 58,  0},
  { 8, 59, 59, 10},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 75,  0},
  {11, 76, 76,  3},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 38,  0},
  { 0, 39, 39, 10},
  { 7, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 49,  0},
  { 4, 50, 50, 10},
  { 7, 51, 54,  0},
  { 9, 55, 58,  0},
  { 8, 59, 59, 10},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 68,  0},
  {11, 69, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 74,  0},
  { 6, 75, 75,  4},
  {11, 76, 76,  3},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 38,  0},
  { 0, 39, 39, 10},
  {10, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 48,  0},
  { 4, 49, 49, 10},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 59,  0},
  { 8, 60, 60, 10},
  {10, 61, 64,  0},
  { 2, 65, 68,  0},
  {11, 69, 69,  1},
  {11, 70, 70,  2},
  { 3, 71, 74,  0},
  {11, 75, 75,  4},
  {11, 76, 76,  3},
  { 0,  5, 13,  0},
  {10, 14, 14,  4},
  { 4, 15, 23,  0},
  { 6, 24, 24,  4},
  { 8, 25, 33,  0},
  {11, 34, 34,  4},
  { 1, 35, 39,  0},
  { 7, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 49,  0},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 59,  0},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 75,  0},
  {11, 76, 76,  3},
  { 0,  5, 12,  0},
  { 7, 13, 13,  4},
  {10, 14, 14,  4},
  { 4, 15, 22,  0},
  {11, 23, 23,  2},
  { 6, 24, 24,  4},
  { 8, 25, 32,  0},
  {11, 33, 33,  3},
  {11, 34, 34,  4},
  { 1, 35, 40,  0},
  { 6, 41, 44,  0},
  { 5, 45, 49,  0},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 59,  0},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 70,  0},
  { 3, 71, 76,  0},
  { 0,  5, 12,  0},
  {11, 13, 13,  0},
  {10, 14, 14,  4},
  { 4, 15, 22,  0},
  { 6, 23, 23,  5},
  { 6, 24, 24,  4},
  { 8, 25, 32,  0},
  { 7, 33, 33,  5},
  {11, 34, 34,  4},
  { 1, 35, 39,  0},
  { 7, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 59,  0},
  {11, 60, 60,  1},
  {10, 61, 64,  0},
  { 2, 65, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 75,  0},
  {11, 76, 76,  3},
  { 0,  5, 12,  0},
  {11, 13, 13,  1},
  {10, 14, 14,  4},
  { 4, 15, 22,  0},
  {10, 23, 23,  5},
  { 6, 24, 24,  4},
  { 8, 25, 32,  0},
  {11, 33, 33,  5},
  {11, 34, 34,  4},
  { 1, 35, 39,  0},
  { 7, 40, 40,  4},
  { 6, 41, 44,  0},
  { 5, 45, 49,  0},
  {11, 50, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 60,  0},
  {10, 61, 64,  0},
  { 2, 65, 69,  0},
  {11, 70, 70,  2},
  { 3, 71, 75,  0},
  {11, 76, 76,  3},
  { 0,  5, 10,  0},
  { 7, 11, 11,  4},
  {11, 12, 13,  0},
  {10, 14, 14,  4},
  { 4, 15, 20,  0},
  { 6, 21, 21,  5},
  {10, 22, 22,  5},
  {11, 23, 23,  2},
  { 6, 24, 24,  4},
  { 8, 25, 30,  0},
  { 7, 31, 31,  5},
  {11, 32, 32,  3},
  {11, 33, 33,  5},
  {11, 34, 34,  4},
  { 1, 35, 40,  0},
  { 6, 41, 44,  0},
  { 5, 45, 50,  0},
  { 7, 51, 54,  0},
  { 9, 55, 60,  0},
  {10, 61, 64,  0},
  { 2, 65, 70,  0},
  { 3, 71, 76,  0},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 44,  0},
  { 5, 45, 54,  0},
  { 9, 55, 64,  0},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 43,  0},
  { 0, 44, 44, 10},
  { 5, 45, 53,  0},
  { 4, 54, 54, 10},
  { 9, 55, 63,  0},
  { 8, 64, 64, 10},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 42,  0},
  { 0, 44, 43, 10},
  { 5, 45, 52,  0},
  { 4, 54, 53, 10},
  { 9, 55, 62,  0},
  { 8, 64, 63, 10},
  { 0,  5, 14,  0},
  { 4, 15, 24,  0},
  { 8, 25, 34,  0},
  { 1, 35, 38,  0},
  { 0, 44, 39, 10},
  { 5, 45, 48,  0},
  { 4, 54, 49, 10},
  { 9, 55, 58,  0},
  { 8, 64, 59, 10},
};
// first entry, entry count, delta bits r, g, b
static const uint8_t kBc6Modes[14][5] = {
  {  0, 19,  5,  5,  5},
  { 19, 23,  6,  6,  6},
  { 42, 18,  5,  4,  4},
  { 60, 20,  4,  5,  4},
  { 80, 20,  4,  4,  5},
  {100, 19,  5,  5,  5},
  {119, 19,  6,  5,  5},
  {138, 21,  5,  6,  5},
  {159, 21,  5,  5,  6},
  {180, 22,  6,  6,  6},
  {202,  6, 10, 10, 10},
  {208,  9,  9,  9,  9},
  {217,  9,  8,  8,  8},
  {226,  9,  4,  4,  4},
};

static const uint8_t kEpBits[14] = { 10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16 };
}  // namespace bc6

static inline int32_t BcSignExtend(int32_t v, uint32_t bits) {
  const int32_t s = 32 - (int32_t)bits;
  return (int32_t)((uint32_t)v << s) >> s;
}

static inline int32_t Bc6Unquantize(int32_t x, uint32_t bits, bool sign) {
  if (sign) {
    if (bits >= 16) {
      return x;
    }
    const bool neg = x < 0;
    if (neg) {
      x = -x;
    }
    if (x != 0) {
      x = (x >= (1 << (bits - 1)) - 1) ? 0x7fff : ((x << 15) + 0x4000) >> (bits - 1);
    }
    return neg ? -x : x;
  }
  if (bits >= 15 || x == 0) {
    return x;
  }
  if (x == (1 << bits) - 1) {
    return 0xffff;
  }
  return ((x << 15) + 0x4000) >> (bits - 1);
}

// The interpolated value scaled to a half-float bit pattern.
static inline uint16_t Bc6Finish(int32_t x, bool sign) {
  if (!sign) {
    return (uint16_t)((x * 31) / 64);
  }
  if (x < 0) {
    return (uint16_t)((((-x) * 31) >> 5) | 0x8000);
  }
  return (uint16_t)((x * 31) >> 5);
}

// BC6H: texel t's red, green and blue as half-float bit patterns. A reserved
// mode decodes to zero.
static inline void BcDecodeBc6(const void* blk, uint32_t t, bool sign, uint16_t out[3]) {
  const BcBits b = BcLoad(blk, 16);
  const uint32_t m5 = (uint32_t)(b.lo & 0x1f);
  int32_t mode;
  if ((m5 & 3u) < 2u) {
    mode = (int32_t)(m5 & 3u);
  } else if (m5 == 0x13 || m5 == 0x17 || m5 == 0x1b || m5 == 0x1f) {
    mode = -1;
  } else {
    mode = (int32_t)(((m5 & 3u) == 2u ? 2u : 10u) + (m5 >> 2));
  }
  out[0] = out[1] = out[2] = 0;
  if (mode < 0) {
    return;
  }
  int32_t ep[12] = { 0 };   // r0..r3, g0..g3, b0..b3
  const uint8_t* md = bc6::kBc6Modes[mode];
  for (uint32_t k = 0; k < md[1]; ++k) {
    const uint8_t* f = bc6::kBc6Fields[md[0] + k];
    uint32_t v;
    if (f[1] <= f[2]) {
      v = BcGet(b, f[1], f[2] - f[1] + 1u);
    } else {
      // A field stored most-significant bit first.
      const uint32_t n = f[1] - f[2] + 1u;
      const uint32_t raw = BcGet(b, f[2], n);
      v = 0;
      for (uint32_t i = 0; i < n; ++i) {
        v |= ((raw >> i) & 1u) << (n - 1 - i);
      }
    }
    ep[f[0]] |= (int32_t)(v << f[3]);
  }
  const uint32_t epb = bc6::kEpBits[mode];
  const uint32_t regions = (mode > 9) ? 1u : 2u;
  const bool transformed = mode != 9 && mode != 10;
  for (uint32_t c = 0; c < 3; ++c) {
    int32_t* e = &ep[4 * c];
    if (sign) {
      e[0] = BcSignExtend(e[0], epb);
    }
    if (transformed) {
      const uint32_t db = md[2 + c];
      for (uint32_t i = 1; i < 2 * regions; ++i) {
        e[i] = (int32_t)((uint32_t)(e[0] + BcSignExtend(e[i], db)) & ((1u << epb) - 1u));
      }
    }
    if (sign) {
      for (uint32_t i = 1; i < 2 * regions; ++i) {
        e[i] = BcSignExtend(e[i], epb);
      }
    }
    for (uint32_t i = 0; i < 2 * regions; ++i) {
      e[i] = Bc6Unquantize(e[i], epb, sign);
    }
  }
  const uint32_t partition = (regions == 2) ? BcGet(b, 77, 5) : 0u;
  const uint32_t ib = (regions == 2) ? 3u : 4u;
  const uint32_t subset = BcSubset(regions, partition, t);
  const uint32_t ioff = (regions == 2 ? 82u : 65u) + BcIndexOffset(regions, partition, t, ib);
  const uint32_t anchor_rm = (t == BcAnchor(regions, partition, subset)) ? 1u : 0u;
  const uint32_t idx = BcGet(b, ioff, ib - anchor_rm);
  for (uint32_t c = 0; c < 3; ++c) {
    const int32_t* e = &ep[4 * c + 2 * subset];
    out[c] = Bc6Finish(BcInterpolate(e[0], e[1], idx, ib), sign);
  }
}

}  // namespace gfx_tex
