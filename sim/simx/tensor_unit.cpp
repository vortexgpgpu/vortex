
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

#include "tensor_unit.h"
#include "tensor_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <type_traits>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg    = vt::wmma_config_t<NUM_THREADS>;
#ifndef TCU_WG_N_MUL
#define TCU_WG_N_MUL 1
#endif
#ifndef TCU_WG_M_MUL
#define TCU_WG_M_MUL 1
#endif
using wg_cfg = vt::wgmma_config_t<NUM_THREADS, vt::fp32, vt::fp32, 8u * TCU_WG_N_MUL, TCU_WG_M_MUL>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

static inline uint8_t unpack_u8(uint32_t word, uint32_t idx) {
  return (word >> (idx * 8)) & 0xffu;
}

#ifdef TCU_MX_ENABLE
static inline uint8_t unpack_mx_scale_8(uint32_t w0,
                                        uint32_t w1,
                                        uint32_t idx,
                                        const char* axis_name) {
  // mxfp8/mxint8 ABI carries two packed words per axis.
  // For 16x16 tiles this intentionally wraps indices [8..15] to [0..7].
  if (idx >= 8) {
    idx &= 0x7;
  }
  if (idx < 4) {
    return unpack_u8(w0, idx);
  }
  if (idx < 8) {
    return unpack_u8(w1, idx - 4);
  }
  std::cout << "Error: unsupported mxfp8 scale index for " << axis_name << ": " << idx << std::endl;
  std::abort();
}

static inline uint8_t unpack_mx_scale_16(uint32_t w0,
                                         uint32_t w1,
                                         uint32_t w2,
                                         uint32_t w3,
                                         uint32_t idx,
                                         const char* axis_name) {
  if (idx >= 16) {
    idx &= 0xf;
  }
  if (idx < 4) {
    return unpack_u8(w0, idx);
  }
  if (idx < 8) {
    return unpack_u8(w1, idx - 4);
  }
  if (idx < 12) {
    return unpack_u8(w2, idx - 8);
  }
  if (idx < 16) {
    return unpack_u8(w3, idx - 12);
  }
  std::cout << "Error: unsupported nvfp4 scale index for " << axis_name << ": " << idx << std::endl;
  std::abort();
}

static inline int32_t round_shift_rne(int32_t value, uint32_t rshift) {
  if (rshift == 0) {
    return value;
  }
  if (rshift >= 31) {
    return 0;
  }
  int32_t mask = (1 << rshift) - 1;
  int32_t half = 1 << (rshift - 1);
  int32_t base = value >> rshift;
  int32_t rem = value & mask;
  if (value < 0 && rem != 0) {
    rem -= (1 << rshift);
  }
  if (rem > half) {
    return base + 1;
  }
  if (rem < -half) {
    return base - 1;
  }
  if (rem == half || rem == -half) {
    return (base & 1) ? (base + (rem > 0 ? 1 : -1)) : base;
  }
  return base;
}

static inline int32_t shl_sat_i32(int32_t value, uint32_t lshift) {
  if (lshift >= 31) {
    return (value < 0) ? std::numeric_limits<int32_t>::min()
                       : std::numeric_limits<int32_t>::max();
  }
  int64_t scaled = static_cast<int64_t>(value) << lshift;
  if (scaled > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  if (scaled < std::numeric_limits<int32_t>::min()) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(scaled);
}

static inline int32_t madd_wrap_i32(int32_t a, int32_t b, int32_t c) {
  int64_t sum = static_cast<int64_t>(a) * static_cast<int64_t>(b)
              + static_cast<int64_t>(c);
  return static_cast<int32_t>(static_cast<uint32_t>(sum));
}

static inline int32_t mxint8_scaled_i32(int8_t q, uint8_t sf) {
  int32_t value = static_cast<int32_t>(q);
  int32_t shift = static_cast<int32_t>(sf) - 133;
  if (shift >= 0) {
    return shl_sat_i32(value, static_cast<uint32_t>(shift));
  }
  return round_shift_rne(value, static_cast<uint32_t>(-shift));
}
#endif

// FMA<It, Ot>: fused multiply-add returning an Ot-typed accumulator (bit-packed in uint32).
// Widens narrow inputs/accumulator to fp32, performs mul+add, rounds once to Ot.
template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(itype a, itype b, uint32_t c) {
    otype fa = static_cast<otype>(a);
    otype fb = static_cast<otype>(b);
    otype fc = bit_cast<otype>(c);
    return bit_cast<uint32_t>(fa * fb + fc);
  }
};

// -- fp16 inputs --
template <> struct FMA<vt::fp16, vt::fp32> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_htof_s(a, 0, nullptr);
    auto fb = rv_htof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::fp16, vt::fp16> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_htof_s(a, 0, nullptr);
    auto fb = rv_htof_s(b, 0, nullptr);
    auto fc = rv_htof_s(uint16_t(c), 0, nullptr);
    return rv_ftoh_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- bf16 inputs --
template <> struct FMA<vt::bf16, vt::fp32> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_btof_s(a, 0, nullptr);
    auto fb = rv_btof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::bf16, vt::bf16> {
  static uint32_t eval(uint16_t a, uint16_t b, uint32_t c) {
    auto fa = rv_btof_s(a, 0, nullptr);
    auto fb = rv_btof_s(b, 0, nullptr);
    auto fc = rv_btof_s(uint16_t(c), 0, nullptr);
    return rv_ftob_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- fp8 inputs --
template <> struct FMA<vt::fp8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e4m3tof_s(a, 0, nullptr);
    auto fb = rv_e4m3tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::fp8, vt::fp8> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e4m3tof_s(a, 0, nullptr);
    auto fb = rv_e4m3tof_s(b, 0, nullptr);
    auto fc = rv_e4m3tof_s(uint8_t(c), 0, nullptr);
    return rv_ftoe4m3_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- bf8 inputs --
template <> struct FMA<vt::bf8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e5m2tof_s(a, 0, nullptr);
    auto fb = rv_e5m2tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::bf8, vt::bf8> {
  static uint32_t eval(uint8_t a, uint8_t b, uint32_t c) {
    auto fa = rv_e5m2tof_s(a, 0, nullptr);
    auto fb = rv_e5m2tof_s(b, 0, nullptr);
    auto fc = rv_e5m2tof_s(uint8_t(c), 0, nullptr);
    return rv_ftoe5m2_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// -- tf32 inputs --
template <> struct FMA<vt::tf32, vt::fp32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = rv_tf32tof_s(a, 0, nullptr);
    auto fb = rv_tf32tof_s(b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA<vt::tf32, vt::tf32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = rv_tf32tof_s(a, 0, nullptr);
    auto fb = rv_tf32tof_s(b, 0, nullptr);
    auto fc = rv_tf32tof_s(c, 0, nullptr);
    return rv_ftotf32_s(rv_fmadd_s(fa, fb, fc, 0, nullptr), 0, nullptr);
  }
};

// Generic FEDP: universal rule keyed on output width.
//   * Wide Ot (fp32): accumulate Σ(a_k*b_k) in fp32, add c_val last — matches
//     RTL's fp32 reduction tree with final c fold-in.
//   * Narrow Ot (fp16/bf16/fp8/bf8/…): chain FMA<It,Ot> so the accumulator is
//     rounded to Ot each step — matches host muladd_t<It,Ot> semantics.
template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
    if constexpr (std::is_same_v<Ot, vt::fp32>) {
      uint32_t acc = 0;
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
        auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
        uint32_t prod = 0;
        for (uint32_t i = 0; i < i_ratio; ++i) {
          prod = FMA<It, vt::fp32>::eval(a[i], b[i], prod);
        }
        acc = rv_fadd_s(prod, acc, 0, nullptr);
      }
      return rv_fadd_s(c_val, acc, 0, nullptr);
    } else {
      uint32_t acc = c_val;
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
        auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
        for (uint32_t i = 0; i < i_ratio; ++i) {
          acc = FMA<It, Ot>::eval(a[i], b[i], acc);
        }
      }
      return acc;
    }
  }
};

template <>
struct FEDP<vt::int4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        if (a_val & 0x8) {
          a_val |= 0xFFFFFFF0;
        }
        if (b_val & 0x8) {
          b_val |= 0xFFFFFFF0;
        }
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::uint4, vt::int32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        int32_t a_val = (a >> (i * 4)) & 0xF;
        int32_t b_val = (b >> (i * 4)) & 0xF;
        acc += a_val * b_val;
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

#ifdef TCU_MX_ENABLE
// Scaled FMA variants for MX microscaling formats (extra scale-factor args).
template <typename It, typename Ot>
struct FMA_MX;

template <> struct FMA_MX<vt::mxfp8, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint8_t sf_a, uint8_t sf_b, uint32_t c) {
    auto fa = rv_mxfp8tof_s(a, sf_a, 0, nullptr);
    auto fb = rv_mxfp8tof_s(b, sf_b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};
template <> struct FMA_MX<vt::mxint8, vt::int32> {
  static uint32_t eval(int8_t a, int8_t b, uint8_t sf_a, uint8_t sf_b, uint32_t c) {
    int32_t xa = mxint8_scaled_i32(a, sf_a);
    int32_t xb = mxint8_scaled_i32(b, sf_b);
    return bit_cast<uint32_t>(bit_cast<int32_t>(c) + xa * xb);
  }
};
template <> struct FMA_MX<vt::nvfp4, vt::fp32> {
  static uint32_t eval(uint8_t a, uint8_t b, uint8_t sf_a, uint8_t sf_b, uint32_t c) {
    auto fa = rv_nvfp4tof_s(a & 0x0f, sf_a, 0, nullptr);
    auto fb = rv_nvfp4tof_s(b & 0x0f, sf_b, 0, nullptr);
    return rv_fadd_s(rv_fmul_s(fa, fb, 0, nullptr), c, 0, nullptr);
  }
};

static uint32_t fedp_mxfp8_fp32_scaled(const reg_data_t* a_row,
                                       const reg_data_t* b_col,
                                       uint32_t c_val,
                                       uint8_t sf_a,
                                       uint8_t sf_b) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(uint8_t);
  uint32_t acc = c_val;
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const uint8_t*>(&a_row[z].u32);
    auto b = reinterpret_cast<const uint8_t*>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA_MX<vt::mxfp8, vt::fp32>::eval(a[i], b[i], sf_a, sf_b, acc);
    }
  }
  return acc;
}

static uint32_t fedp_mxint8_int32_scaled(const reg_data_t* a_row,
                                         const reg_data_t* b_col,
                                         uint32_t c_val,
                                         uint8_t sf_a,
                                         uint8_t sf_b) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(int8_t);
  uint32_t acc = c_val;
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const int8_t*>(&a_row[z].u32);
    auto b = reinterpret_cast<const int8_t*>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA_MX<vt::mxint8, vt::int32>::eval(a[i], b[i], sf_a, sf_b, acc);
    }
  }
  return acc;
}

static uint32_t fedp_nvfp4_fp32_scaled(const reg_data_t* a_row,
                                       const reg_data_t* b_col,
                                       uint32_t c_val,
                                       uint8_t sf_a,
                                       uint8_t sf_b) {
  uint32_t acc = c_val;
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    uint32_t aw = a_row[z].u32;
    uint32_t bw = b_col[z].u32;
    for (uint32_t i = 0; i < 8; ++i) {
      uint8_t a = (aw >> (i * 4)) & 0x0f;
      uint8_t b = (bw >> (i * 4)) & 0x0f;
      acc = FMA_MX<vt::nvfp4, vt::fp32>::eval(a, b, sf_a, sf_b, acc);
    }
  }
  return acc;
}
#endif

using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp32>::eval;
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::fp32>::eval;
    case vt::fp8::id:
      return FEDP<vt::fp8, vt::fp32>::eval;
    case vt::bf8::id:
      return FEDP<vt::bf8, vt::fp32>::eval;
    case vt::tf32::id:
      return FEDP<vt::tf32, vt::fp32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<vt::fp16, vt::fp16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP<vt::bf16, vt::bf16>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp8::id:
    switch (IT) {
    case vt::fp8::id:
      return FEDP<vt::fp8, vt::fp8>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf8::id:
    switch (IT) {
    case vt::bf8::id:
      return FEDP<vt::bf8, vt::bf8>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::tf32::id:
    switch (IT) {
    case vt::tf32::id:
      return FEDP<vt::tf32, vt::tf32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int8::id:
      return FEDP<vt::int8, vt::int32>::eval;
    case vt::uint8::id:
      return FEDP<vt::uint8, vt::int32>::eval;
    case vt::int4::id:
      return FEDP<vt::int4, vt::int32>::eval;
    case vt::uint4::id:
      return FEDP<vt::uint4, vt::int32>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

// Format-agnostic sparse gather: for each bword, iterate over its elem_count packed
// elements, collect those flagged by lo_mask/hi_mask respectively.
static inline uint32_t gather_sparse(uint32_t bword0, uint32_t bword1,
                                     uint32_t lo_mask, uint32_t hi_mask,
                                     uint32_t elem_bits) {
  uint32_t elem_count = 32 / elem_bits;
  uint32_t elem_mask  = (elem_bits < 32) ? ((1u << elem_bits) - 1u) : ~0u;
  assert((uint32_t)(__builtin_popcount(lo_mask) + __builtin_popcount(hi_mask)) == elem_count &&
         "gather_sparse: total selected elements must equal elem_count");
  uint32_t out = 0, k = 0;
  for (uint32_t i = 0; i < elem_count; ++i) {
    if (lo_mask & (1u << i))
      out |= ((bword0 >> (i * elem_bits)) & elem_mask) << (k++ * elem_bits);
  }
  for (uint32_t i = 0; i < elem_count; ++i) {
    if (hi_mask & (1u << i))
      out |= ((bword1 >> (i * elem_bits)) & elem_mask) << (k++ * elem_bits);
  }
  return out;
}

static inline uint32_t meta_num_cols(uint32_t fmt_s) {
  return vt::sparse_meta_num_cols(fmt_s, NUM_THREADS);
}

static inline uint32_t meta_row_width(uint32_t elem_bits) {
  // Each K-step uses (32/elem_bits) meta bits per half (lo and hi), 2 halves per row.
  return cfg::tcK * 2 * (32 / elem_bits);
}

static inline uint32_t mx_meta_words(uint32_t fmt_s) {
  if (fmt_s == vt::nvfp4::id) {
    return 8;
  }
  if (fmt_s == vt::mxfp8::id || fmt_s == vt::mxint8::id) {
    return 4;
  }
  return 0;
}

class TensorUnit::Impl {
public:

  struct lmem_desc_t {
    uint64_t base = 0;
    uint32_t ldm = 0;
    bool col_major = false;
  };

  // WGMMA v2 tile buffer state — models RTL single-slot + shared B cache.
  // B tile persists across CTA warps (fetched once per K-tile).
  // A tile is re-fetched per warp unless WGMMA_PREFETCH_A pre-loaded it.
  struct TileBufferState {
    // B buffer
    bool     b_valid = false;
    uint32_t b_desc = 0;        // cached B descriptor
    uint32_t b_fetch_wid = ~0u; // warp that last fetched B (for dirty check)
    uint32_t cur_wid = ~0u;     // warp whose A data is currently in buffer
    uint64_t ready_cycle = 0;   // cycle when B/CD buffer fill completes

    // A prefetch buffer — one slot, owned by the warp that issued PREFETCH_A
    bool     a_valid = false;           // A tile is ready for WGMMA to consume
    bool     a_prefetch_in_progress = false; // A fetch is running (not yet valid)
    uint32_t a_desc = 0;                // smem descriptor for the cached A tile
    uint32_t a_wid = ~0u;              // warp that owns the A buffer slot
    uint64_t a_ready_cycle = 0;        // cycle when the A fetch completes

    // True while a WGMMA multi-UOP sequence is in-flight in the TensorUnit
    bool tcu_active = false;

    void reset() {
      b_valid = false;
      b_desc = 0;
      b_fetch_wid = ~0u;
      cur_wid = ~0u;
      ready_cycle = 0;
      a_valid = false;
      a_prefetch_in_progress = false;
      a_desc = 0;
      a_wid = ~0u;
      a_ready_cycle = 0;
      tcu_active = false;
    }
  };

  struct APrefetchReq {
    uint32_t wid;
    uint32_t a_desc;
    uint32_t fetch_cycles;
  };

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , sparse_meta_(arch.num_warps(), std::vector<uint32_t>(kMetaBanks * kMaxMetaCols, 0))
    , mx_meta_(arch.num_warps())
    , perf_stats_()
    , tbuf_state_()
  {}

  ~Impl() {}

  void reset() {
    perf_stats_ = PerfStats();
    tbuf_state_.reset();
    for (auto& sparse_meta : sparse_meta_) {
      std::fill(sparse_meta.begin(), sparse_meta.end(), 0);
    }
    for (auto& mx_meta : mx_meta_) {
      mx_meta.fill(0);
    }
#ifdef TCU_WGMMA_ENABLE
    while (!a_prefetch_queue_.empty()) a_prefetch_queue_.pop();
#endif
  }

  void tick() {
    auto cur_cycle = SimPlatform::instance().cycles();

#ifdef TCU_WGMMA_ENABLE
    // Drain A prefetch queue once TCU is idle and no prefetch is running.
    if (!tbuf_state_.tcu_active
        && !tbuf_state_.a_prefetch_in_progress
        && !a_prefetch_queue_.empty()) {
      auto req = a_prefetch_queue_.front();
      a_prefetch_queue_.pop();
      tbuf_state_.a_prefetch_in_progress = true;
      tbuf_state_.a_ready_cycle = cur_cycle + req.fetch_cycles;
      tbuf_state_.a_desc = req.a_desc;
      tbuf_state_.a_wid  = req.wid;
      tbuf_state_.a_valid = false;
      DT(2, "WGMMA PREFETCH_A dequeued: wid=" << req.wid
         << " a_desc=0x" << std::hex << req.a_desc << std::dec
         << " ready_cycle=" << tbuf_state_.a_ready_cycle);
    }
#endif

    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
        delay = 4;
        break;
#ifdef TCU_WGMMA_ENABLE
      case TcuType::WGMMA: {
        // Track WGMMA sequence: set active on first UOP (fu_lock), clear on last (fu_unlock).
        if (trace->instr_ptr->get_fu_lock()) {
          tbuf_state_.tcu_active = true;
        }
        auto trace_data = std::dynamic_pointer_cast<ExeTraceData>(trace->data);
        // Kick off B/CD fill timer immediately so it overlaps with any A prefetch stall.
        // This means we stall for max(A_remaining, B_cycles) instead of A_remaining + B_cycles.
        int fetch_delay = trace_data ? trace_data->fetch_delay : 0;
        if (fetch_delay > 0) {
          if (tbuf_state_.ready_cycle <= cur_cycle) {
            tbuf_state_.ready_cycle = cur_cycle + fetch_delay;
          }
          trace_data->fetch_delay = 0;
        }
        // Stall if A prefetch is still in-flight.
        if (trace_data && trace_data->a_buf_pending) {
          if (cur_cycle < tbuf_state_.a_ready_cycle) {
            ++perf_stats_.abuf_prefetch_stalls;
            continue;
          }
          // A prefetch complete.
          trace_data->a_buf_pending = false;
          tbuf_state_.a_prefetch_in_progress = false;
          tbuf_state_.a_valid = true;
        }
        // Stall if B/CD fill is still in-flight.
        if (cur_cycle < tbuf_state_.ready_cycle) {
          ++perf_stats_.tbuf_stalls;
          continue;
        }
        delay = 4;
      } break;
      case TcuType::WGMMA_PREFETCH_A:
        delay = 1;
        break;
#else
      case TcuType::WGMMA: {
        auto trace_data = std::dynamic_pointer_cast<ExeTraceData>(trace->data);
        int fetch_delay = trace_data ? trace_data->fetch_delay : 0;
        if (fetch_delay > 0) {
          if (tbuf_state_.ready_cycle <= cur_cycle) {
            tbuf_state_.ready_cycle = cur_cycle + fetch_delay;
          }
          trace_data->fetch_delay = 0;
        }
        if (cur_cycle < tbuf_state_.ready_cycle) {
          ++perf_stats_.tbuf_stalls;
          continue;
        }
        delay = 4;
      } break;
#endif
      case TcuType::META_STORE:
        delay = 1;
        break;
      default:
        std::abort();
      }
      if (simobject_->Outputs.at(iw).try_send(trace, 2 + delay)) {
#ifdef TCU_WGMMA_ENABLE
        if (tcu_type == TcuType::WGMMA) {
          auto td = std::dynamic_pointer_cast<ExeTraceData>(trace->data);
          if (td && td->tbuf_cache_hit) {
            ++perf_stats_.tbuf_cache_hits;
            DT(2, "TCU tbuf_cache_hit counted, total=" << perf_stats_.tbuf_cache_hits);
          }
          if (td && td->a_buf_hit) {
            ++perf_stats_.abuf_prefetch_hits;
          }
          // Last WGMMA UOP: release TCU, invalidate A buffer (consumed), drain queue.
          if (trace->instr_ptr->get_fu_unlock()) {
            tbuf_state_.tcu_active = false;
            auto tpuArgs = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
            if (tpuArgs.is_a_smem) {
              tbuf_state_.a_valid = false; // A consumed
            }
            // Start next queued A prefetch if one is waiting.
            if (!tbuf_state_.a_prefetch_in_progress && !a_prefetch_queue_.empty()) {
              auto req = a_prefetch_queue_.front();
              a_prefetch_queue_.pop();
              tbuf_state_.a_prefetch_in_progress = true;
              tbuf_state_.a_ready_cycle = cur_cycle + req.fetch_cycles;
              tbuf_state_.a_desc = req.a_desc;
              tbuf_state_.a_wid  = req.wid;
              tbuf_state_.a_valid = false;
              DT(2, "WGMMA PREFETCH_A dequeued (post-unlock): wid=" << req.wid
                 << " a_desc=0x" << std::hex << req.a_desc << std::dec
                 << " ready_cycle=" << tbuf_state_.a_ready_cycle);
            }
          }
        }
#else
        // Count B cache hit on successful send
        if (tcu_type == TcuType::WGMMA) {
          auto td = std::dynamic_pointer_cast<ExeTraceData>(trace->data);
          if (td && td->tbuf_cache_hit) {
            ++perf_stats_.tbuf_cache_hits;
            DT(2, "TCU tbuf_cache_hit counted, total=" << perf_stats_.tbuf_cache_hits);
          }
        }
#endif
        DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
        input.pop();
      }
    }
  }

  void meta_store(uint32_t wid,
                  uint32_t fmt_s,
                  uint32_t col_idx,
                  uint32_t meta_kind,
                  const std::vector<reg_data_t>& rs1_data,
                  ExeTraceData* trace_data) {
    __unused(trace_data);

    if (meta_kind == TCU_META_KIND_MX) {
      uint32_t num_mx_words = mx_meta_words(fmt_s);
      if (col_idx >= num_mx_words || rs1_data.empty()) {
        std::cout << "Error: META_STORE MX index out of range: " << col_idx << std::endl;
        std::abort();
      }
      mx_meta_.at(wid).at(col_idx) = rs1_data.at(0).u32;
      return;
    }

    uint32_t num_cols = meta_num_cols(fmt_s);

    if (meta_kind == TCU_META_KIND_SPARSE_WG) {
      // WGMMA RS sparse: kernel scatters smem data into interleaved register
      // layout matching VX_tcu_meta's WMMA PER_WARP_DEPTH (= kMetaBanks) stride.
      // Thread mapping: src_idx = col_in_group * kMetaBanks + bank.
      // Data goes into kMetaBanks rows (same as WMMA); the reader selects
      // banks using WMMA-style {step_m, step_k_half} so m=1 lands at bank
      // (cfg::k_steps/2), not bank 1.
      constexpr uint32_t wg_cols_per_load = NUM_THREADS / kMetaBanks;
      uint32_t group = col_idx;
      uint32_t col_begin = group * wg_cols_per_load;
      uint32_t col_end = std::min(col_begin + wg_cols_per_load, num_cols);
      for (uint32_t col = col_begin; col < col_end; ++col) {
        uint32_t col_in_group = col - col_begin;
        for (uint32_t bank = 0; bank < kMetaBanks; ++bank) {
          uint32_t src_idx = col_in_group * kMetaBanks + bank;
          sparse_meta_.at(wid).at(bank * kMaxMetaCols + col) =
              rs1_data.at(src_idx).u32;
        }
      }
      return;
    }

    uint32_t total_stores = vt::sparse_meta_total_store_uops(fmt_s, cfg::stores_per_col, NUM_THREADS, cfg::meta_cols_per_load);
    if (col_idx >= total_stores) {
      // Flat per-thread store (fallback)
      for (uint32_t t = 0; t < rs1_data.size(); ++t) {
        sparse_meta_.at(wid).at(t) = rs1_data.at(t).u32;
      }
      return;
    }

    if constexpr (cfg::stores_per_col > 1) {
      // NT < per_warp_depth: col_idx enumerates (col, store_in_col) pairs.
      // Each store covers banks_per_store=NT consecutive banks of one column.
      uint32_t col = col_idx / cfg::stores_per_col;
      uint32_t store_in_col = col_idx % cfg::stores_per_col;
      if (col >= num_cols) return;
      uint32_t bank_base = store_in_col * cfg::banks_per_store;
      for (uint32_t t = 0; t < cfg::banks_per_store; ++t) {
        uint32_t bank = bank_base + t;
        if (bank >= kMetaBanks) break;
        sparse_meta_.at(wid).at(bank * kMaxMetaCols + col) = rs1_data.at(t).u32;
      }
      return;
    }

    // NT >= per_warp_depth: col_idx enumerates column groups; each group covers
    // meta_cols_per_load columns across all banks.
    uint32_t group = col_idx;
    uint32_t col_begin = group * cfg::meta_cols_per_load;
    uint32_t col_end = std::min(col_begin + cfg::meta_cols_per_load, num_cols);

    for (uint32_t col = col_begin; col < col_end; ++col) {
      uint32_t col_in_group = col - col_begin;
      uint32_t thread_offset = col_in_group * kMetaBanks;
      for (uint32_t bank = 0; bank < kMetaBanks; ++bank) {
        uint32_t src_idx = thread_offset + bank;
        sparse_meta_.at(wid).at(bank * kMaxMetaCols + col) =
            rs1_data.at(src_idx).u32;
      }
    }
  }

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            uint32_t step_k,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data,
            bool is_sparse) {
    if (is_sparse) {
      if (!vt::sparse_format_supported(fmt_s)) {
        std::cout << "Error: WMMA_SP unsupported input format: "
                  << vt::fmt_string(fmt_s) << " (id=" << fmt_s
                  << "). Supported formats: i8, u8, fp8, bf8, fp16, bf16, i4, u4." << std::endl;
        std::abort();
      }
      if ((arch_.num_threads() % cfg::b_block_size_sp) != 0) {
        std::cout << "Error: NUM_THREADS must be divisible by sparse B block size" << std::endl;
        std::abort();
      }
    }

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = is_sparse
                   ? (step_n % cfg::b_sub_blocks_sp) * cfg::b_block_size_sp
                   : (step_n % cfg::b_sub_blocks)    * cfg::b_block_size;

    // Prepare A tile [tcM][tcK]
    reg_data_t a_tile[cfg::tcM * cfg::tcK];
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t z = 0; z < cfg::tcK; ++z) {
        a_tile[i * cfg::tcK + z] = rs1_data.at(a_off + i * cfg::tcK + z);
      }
    }

    // Prepare B tile [tcM][tcN][tcK]
    reg_data_t b_tile[cfg::tcM * cfg::tcN * cfg::tcK];
    if (is_sparse) {
      constexpr uint32_t kCompression = 2;
      uint32_t ebits = elem_bits(fmt_s);
      uint32_t meta_bits = 32 / ebits;
      uint32_t bank = step_m * (cfg::k_steps / 2) + step_k;
      auto meta_bit = [&](uint32_t bit_idx) {
        return (sparse_meta_.at(wid).at(bank * kMaxMetaCols + bit_idx / 32) >> (bit_idx % 32)) & 1u;
      };
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        uint32_t row_base = i * meta_row_width(ebits);
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t j_sp = cfg::sym_sparse ? (j % (cfg::tcN / 2)) : j;
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            uint32_t b_idx = b_off + j_sp * cfg::tcK * kCompression + z * kCompression;
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < meta_bits; ++b) {
              lo |= meta_bit(row_base + meta_bits * z + b) << b;
              hi |= meta_bit(row_base + meta_bits * (cfg::tcK + z) + b) << b;
            }
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 =
                gather_sparse(rs2_data.at(b_idx).u32, rs2_data.at(b_idx + 1).u32, lo, hi, ebits);
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z] = rs2_data.at(b_off + j * cfg::tcK + z);
          }
        }
      }
    }

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data);

    trace_data->is_last_k = true;
  }

#ifdef TCU_WGMMA_ENABLE
  void wgmma_prefetch_a(uint32_t wid, uint32_t fmt_s, bool is_sparse, uint32_t a_desc) {
    auto cur_cycle = SimPlatform::instance().cycles();

    constexpr uint32_t a_total        = wg_cfg::xtileM * wg_cfg::xtileK;
    constexpr uint32_t num_banks      = LMEM_NUM_BANKS;
    constexpr uint32_t a_bank_rows    = (a_total + num_banks - 1) / num_banks;
    constexpr uint32_t a_bank_rows_sp = ((a_total / 2) + num_banks - 1) / num_banks;

    uint32_t fetch_cycles = is_sparse ? a_bank_rows_sp : a_bank_rows;
    if (is_sparse && vt::sparse_format_supported(fmt_s)) {
      uint32_t ratio = elem_ratio(fmt_s);
      uint32_t meta_row_bits = cfg::tcK * 2 * ratio;
      uint32_t meta_stride = (cfg::tcM * meta_row_bits + 31) / 32;
      uint32_t meta_total = wg_cfg::m_steps * (wg_cfg::k_steps / 2) * meta_stride;
      fetch_cycles += (meta_total + num_banks - 1) / num_banks;
    }

    if (tbuf_state_.tcu_active || tbuf_state_.a_prefetch_in_progress) {
      // TCU is busy or previous prefetch is still running — queue the request.
      a_prefetch_queue_.push({wid, a_desc, fetch_cycles});
      DT(2, "WGMMA PREFETCH_A queued: wid=" << wid
         << " a_desc=0x" << std::hex << a_desc << std::dec
         << " queue_depth=" << a_prefetch_queue_.size());
    } else {
      // TCU is idle — start prefetch immediately.
      tbuf_state_.a_prefetch_in_progress = true;
      tbuf_state_.a_ready_cycle = cur_cycle + fetch_cycles;
      tbuf_state_.a_desc  = a_desc;
      tbuf_state_.a_wid   = wid;
      tbuf_state_.a_valid = false;
      DT(2, "WGMMA PREFETCH_A started: wid=" << wid
         << " a_desc=0x" << std::hex << a_desc << std::dec
         << " fetch_cycles=" << fetch_cycles
         << " ready_cycle=" << tbuf_state_.a_ready_cycle);
    }
  }
#endif

  void wgmma(uint32_t wid,
             uint32_t fmt_s,
             uint32_t fmt_d,
             uint32_t step_m,
             uint32_t step_n,
             uint32_t step_k,
             uint32_t a_desc,
             uint32_t b_desc,
             uint32_t cd_desc,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             ExeTraceData* trace_data,
             bool is_sparse,
             uint32_t cd_nregs,
             uint32_t is_a_smem,
             uint32_t is_cd_smem) {
    uint32_t nrc = 8u << cd_nregs;
    uint32_t xtileN_actual = (nrc * NUM_THREADS) / wg_cfg::xtileM;
    if (is_sparse && !vt::sparse_format_supported(fmt_s)) {
      std::cout << "Error: WGMMA_SP unsupported input format: "
                << vt::fmt_string(fmt_s) << " (id=" << fmt_s << ")" << std::endl;
      std::abort();
    }

    uint32_t ratio    = elem_ratio(fmt_s);
    uint32_t k_words  = cfg::tcK;
    uint32_t e_bytes  = elem_bits(fmt_s) / 8;
    uint32_t e_bytes_d = elem_bits(fmt_d) / 8;

    // Decode smem descriptors: B always from smem, A optionally, C/D when is_cd_smem.
    // cd_desc is a single descriptor for the accumulator buffer used for both C reads and D writes.
    lmem_desc_t sd_a, sd_b, sd_cd;
    if (step_k == 0 && step_m == 0 && step_n == 0) {
      if (is_a_smem) {
        sd_a = {uint64_t(LMEM_BASE_ADDR) + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
        lmem_desc_[wid][0] = sd_a;
      }
      sd_b = {uint64_t(LMEM_BASE_ADDR) + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
      lmem_desc_[wid][1] = sd_b;
      if (is_cd_smem) {
        sd_cd = {uint64_t(LMEM_BASE_ADDR) + (cd_desc & 0xFFFF), (cd_desc >> 16) / e_bytes_d, false};
        lmem_desc_[wid][2] = sd_cd;
      }
    } else {
      sd_a = lmem_desc_[wid][0];
      sd_b = lmem_desc_[wid][1];
      if (is_cd_smem) {
        sd_cd = lmem_desc_[wid][2];
      }
    }

    // Prepare A tile [tcM][tcK]
    reg_data_t a_tile[cfg::tcM * cfg::tcK];
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      uint32_t a_row_idx = step_m * cfg::tcM + i;
      for (uint32_t z = 0; z < k_words; ++z) {
        if (is_a_smem) {
          uint32_t k_elem = (step_k * k_words + z) * ratio;
          a_tile[i * cfg::tcK + z].u32 = load_lmem_word(sd_a, a_row_idx, k_elem, fmt_s, false);
        } else {
          a_tile[i * cfg::tcK + z] = rs1_data.at(i * cfg::tcK + z);
        }
      }
    }

    // Prepare B tile [tcM][tcN][tcK]
    reg_data_t b_tile[cfg::tcM * cfg::tcN * cfg::tcK];
    if (is_sparse) {
      uint32_t ebits       = elem_bits(fmt_s);
      uint32_t rtl_i_ratio = 32 / ebits;
      uint32_t meta_row_w  = k_words * 2 * rtl_i_ratio;
      uint32_t meta_strd_words = (cfg::tcM * meta_row_w + 31) / 32;
      // Bank encoding mirrors VX_tcu_meta: {step_m, step_k_half} with WMMA
      // half-K width. WGMMA sparse issues step_k=0, so m=1 maps to bank
      // (cfg::k_steps/2) (matches RTL's WMMA-style SRAM layout).
      uint32_t wg_bank_rs = step_m * (cfg::k_steps / 2) + step_k;
      uint32_t wg_bank_ss = step_m * (wg_cfg::k_steps / 2) + step_k;
      uint32_t wg_bank = is_a_smem ? wg_bank_ss : wg_bank_rs;
      // For SS mode, read metadata from LMEM; for RS mode, use register data
      uint64_t meta_base = 0;
      if (is_a_smem) {
        meta_base = sd_a.base + uint64_t(wg_cfg::xtileM) * sd_a.ldm * e_bytes;
      }
      auto meta_bit_wg = [&](uint32_t bit_idx) -> uint32_t {
        uint32_t word_val = 0;
        if (is_a_smem) {
          // SS mode: read from LMEM using LMEM stride
          uint32_t word_idx = wg_bank * meta_strd_words + bit_idx / 32;
          core_->mem_read(&word_val, meta_base + uint64_t(word_idx) * 4, 4);
        } else {
          // RS mode: read from sparse_meta_ SRAM using WMMA-compatible stride
          uint32_t word_idx = wg_bank * kMaxMetaCols + bit_idx / 32;
          word_val = sparse_meta_.at(wid).at(word_idx);
        }
        return (word_val >> (bit_idx % 32)) & 1u;
      };
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        uint32_t row_base = i * meta_row_w;
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t b_col_idx = step_n * cfg::tcN + j;
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < rtl_i_ratio; ++b) {
              lo |= meta_bit_wg(row_base + rtl_i_ratio * z              + b) << b;
              hi |= meta_bit_wg(row_base + rtl_i_ratio * (k_words + z) + b) << b;
            }
            constexpr uint32_t kCompression = 2;
            uint32_t k_elem_b0 = (step_k * k_words + z) * ratio * kCompression;
            uint32_t bword0 = load_lmem_word(sd_b, k_elem_b0,         b_col_idx, fmt_s, true);
            uint32_t bword1 = load_lmem_word(sd_b, k_elem_b0 + ratio, b_col_idx, fmt_s, true);
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 =
                gather_sparse(bword0, bword1, lo, hi, ebits);
          }
        }
      }
    } else {
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t b_col_idx = step_n * cfg::tcN + j;
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t k_elem = (step_k * k_words + z) * ratio;
            b_tile[(i * cfg::tcN + j) * cfg::tcK + z].u32 =
                load_lmem_word(sd_b, k_elem, b_col_idx, fmt_s, true);
          }
        }
      }
    }

    if (is_cd_smem) {
      std::vector<reg_data_t> c_vals(cfg::tcM * cfg::tcN);
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t row = step_m * cfg::tcM + i;
          uint32_t col = step_n * cfg::tcN + j;
          uint32_t val = 0;
          core_->mem_read(&val, sd_cd.base + uint64_t(row * sd_cd.ldm + col) * e_bytes_d, e_bytes_d);
          c_vals[i * cfg::tcN + j].u32 = val;
        }
      }
      fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d, a_tile, b_tile, c_vals, rd_data);
      for (uint32_t i = 0; i < cfg::tcM; ++i) {
        for (uint32_t j = 0; j < cfg::tcN; ++j) {
          uint32_t row = step_m * cfg::tcM + i;
          uint32_t col = step_n * cfg::tcN + j;
          uint32_t val = rd_data[i * cfg::tcN + j].u32;
          core_->mem_write(&val, sd_cd.base + uint64_t(row * sd_cd.ldm + col) * e_bytes_d, e_bytes_d);
        }
      }
    } else {
      fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d, a_tile, b_tile, rs3_data, rd_data);
    }

    trace_data->is_last_k = true;

    // --- WGMMA v2 tile buffer: compute fetch cycles for timing layer ---
    // On first uop, determine how many cycles the fetch unit needs to fill
    // the tile buffer. The timing layer (tick) uses this to model stalls
    // and count perf stats.
    if (step_m == 0 && step_n == 0 && step_k == 0) {
      constexpr uint32_t a_total    = wg_cfg::xtileM * wg_cfg::xtileK;
      const     uint32_t b_total    = wg_cfg::xtileK * xtileN_actual;
      constexpr uint32_t num_banks  = LMEM_NUM_BANKS;
      constexpr uint32_t a_bank_rows    = (a_total + num_banks - 1) / num_banks;
      constexpr uint32_t a_bank_rows_sp = ((a_total / 2) + num_banks - 1) / num_banks;
      const     uint32_t b_bank_rows    = (b_total + num_banks - 1) / num_banks;

      // B dirty: same warp re-alloc with same desc (LMEM may be overwritten by DXA)
      bool b_dirty = tbuf_state_.b_valid
                  && (wid == tbuf_state_.b_fetch_wid)
                  && (b_desc == tbuf_state_.b_desc);
      bool b_cached = tbuf_state_.b_valid && !b_dirty
                   && (b_desc == tbuf_state_.b_desc);

      // A prefetch buffer check: skip A fetch cycles if the tile was pre-loaded.
      bool a_desc_match = is_a_smem
                       && (tbuf_state_.a_wid == wid)
                       && (tbuf_state_.a_desc == a_desc);
      bool a_prefetched = a_desc_match && tbuf_state_.a_valid;
      bool a_prefetch_pending = a_desc_match && tbuf_state_.a_prefetch_in_progress;

      // Signal tcu_active so concurrent PREFETCH_A requests queue instead of racing.
      tbuf_state_.tcu_active = true;

      // Fetch order: A first (entire tile), then B streamed column by column.
      // Loop order k→n→m ensures each (k,n) B column is reused across all m_steps.
      uint32_t fetch_cycles = 0;
      if (is_a_smem && !a_prefetched && !a_prefetch_pending) {
        // Normal A fetch — no prefetch available.
        fetch_cycles += is_sparse ? a_bank_rows_sp : a_bank_rows;
      }
      if (!b_cached) {
        fetch_cycles += b_bank_rows;
      }
      if (is_sparse && is_a_smem && !a_prefetched && !a_prefetch_pending) {
        uint32_t ratio = elem_ratio(fmt_s);
        uint32_t meta_row_bits = cfg::tcK * 2 * ratio;
        uint32_t meta_stride = (cfg::tcM * meta_row_bits + 31) / 32;
        uint32_t meta_total = wg_cfg::m_steps * (wg_cfg::k_steps / 2) * meta_stride;
        fetch_cycles += (meta_total + num_banks - 1) / num_banks;
      }

      // When is_cd_smem, each of TOTAL_BLOCKS blocks triggers one FETCH_C + one STORE_D.
      if (is_cd_smem) {
        constexpr uint32_t block_cap   = cfg::tcM * cfg::tcN;
        constexpr uint32_t c_bank_rows = (block_cap + num_banks - 1) / num_banks;
        const uint32_t n_steps_actual  = xtileN_actual / cfg::tcN;
        const uint32_t total_blocks    = wg_cfg::m_steps * n_steps_actual;
        fetch_cycles += total_blocks * c_bank_rows;  // FETCH_C per block
        fetch_cycles += total_blocks * c_bank_rows;  // STORE_D per block
      }

      // Count LMEM reads: B words (if not cached) + A words + metadata words + C words
      uint32_t lmem_reads = 0;
      if (!b_cached) {
        lmem_reads += b_total;
      }
      if (is_a_smem && !a_prefetched && !a_prefetch_pending) {
        lmem_reads += is_sparse ? (a_total / 2) : a_total;
      }
      if (is_sparse && is_a_smem && !a_prefetched && !a_prefetch_pending) {
        uint32_t ratio = elem_ratio(fmt_s);
        uint32_t meta_row_bits_r = cfg::tcK * 2 * ratio;
        uint32_t meta_stride_r = (cfg::tcM * meta_row_bits_r + 31) / 32;
        lmem_reads += wg_cfg::m_steps * (wg_cfg::k_steps / 2) * meta_stride_r;
      }
      if (is_cd_smem) {
        const uint32_t cd_total = wg_cfg::xtileM * xtileN_actual;
        lmem_reads += cd_total; // C reads
      }
      perf_stats_.lmem_reads += lmem_reads;

      perf_stats_.tbuf_fetch_cyc += fetch_cycles;
      if (!b_cached) {
        perf_stats_.fetch_b_cyc += b_bank_rows;
      }

      trace_data->fetch_delay     = fetch_cycles;
      trace_data->tbuf_cache_hit  = b_cached;
      trace_data->a_buf_hit       = a_prefetched;
      trace_data->a_buf_pending   = a_prefetch_pending;
      DT(2, "WGMMA tbuf: wid=" << wid
         << " b_desc=0x" << std::hex << b_desc << std::dec
         << " b_valid=" << tbuf_state_.b_valid
         << " b_dirty=" << b_dirty
         << " b_cached=" << b_cached
         << " is_a_smem=" << is_a_smem
         << " a_prefetched=" << a_prefetched
         << " a_prefetch_pending=" << a_prefetch_pending
         << " is_cd_smem=" << is_cd_smem
         << " fetch=" << fetch_cycles);

      // Update tile buffer state
      if (!b_cached) {
        ++perf_stats_.tbuf_b_misses;
        tbuf_state_.b_valid     = true;
        tbuf_state_.b_desc      = b_desc;
        tbuf_state_.b_fetch_wid = wid;
      }
      tbuf_state_.cur_wid = wid;
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  uint32_t elem_bits(uint32_t fmt_s) const {
    switch (fmt_s) {
    case vt::fp32::id:
    case vt::tf32::id:
    case vt::int32::id:
      return 32;
    case vt::fp16::id:
    case vt::bf16::id:
      return 16;
    case vt::fp8::id:
    case vt::bf8::id:
    case vt::mxfp8::id:
    case vt::mxint8::id:
    case vt::int8::id:
    case vt::uint8::id:
      return 8;
    case vt::int4::id:
    case vt::uint4::id:
    case vt::nvfp4::id:
      return 4;
    default:
      std::abort();
    }
  }

  uint32_t elem_ratio(uint32_t fmt_s) const {
    return 32 / elem_bits(fmt_s);
  }

  uint32_t load_lmem_word(const lmem_desc_t& desc, uint32_t row, uint32_t col, uint32_t fmt_s, bool pack_along_row) const {
    auto bits = elem_bits(fmt_s);
    uint32_t packed = 0;
    uint32_t elem_bytes = bits / 8;
    uint32_t elems_per_word = 4 / elem_bytes;
    uint32_t mask = (elem_bytes == 4) ? 0xffffffffu : ((1u << (8 * elem_bytes)) - 1u);

    for (uint32_t e = 0; e < elems_per_word; ++e) {
      uint32_t rr = row + (pack_along_row ? e : 0);
      uint32_t cc = col + (pack_along_row ? 0 : e);
      if (desc.col_major) {
        std::swap(rr, cc);
      }
      uint64_t addr = desc.base + uint64_t(rr * desc.ldm + cc) * elem_bytes;
      uint32_t word = 0;
      core_->mem_read(&word, addr, elem_bytes);
      packed |= (word & mask) << (8 * elem_bytes * e);
    }
    return packed;
  }

  static constexpr uint32_t kSparseKSteps = cfg::k_steps / 2;
  static constexpr uint32_t kMetaBanks = cfg::m_steps * kSparseKSteps;
  static constexpr uint32_t kMaxMetaCols = NUM_THREADS / 2;
  static constexpr uint32_t kMxMetaWords = 8;

  // FEDP tile computation for both WMMA and WGMMA.
  // a_tile: [tcM][tcK] flat array of pre-loaded A operand words.
  // b_tile: [tcM][tcN][tcK] flat array of pre-loaded B operand words
  //         (for dense, each i-slice is identical; for sparse, already gathered).
  void fedp_tile(uint32_t wid,
                 uint32_t step_m, uint32_t step_n, uint32_t step_k,
                 uint32_t fmt_s, uint32_t fmt_d,
                 const reg_data_t* a_tile,
                 const reg_data_t* b_tile,
                 const std::vector<reg_data_t>& rs3_data,
                 std::vector<reg_data_t>& rd_data) {
    __unused(wid, step_m, step_n, step_k);
  #ifdef TCU_MX_ENABLE
    bool use_mx = ((fmt_s == vt::mxfp8::id) && (fmt_d == vt::fp32::id))
               || ((fmt_s == vt::mxint8::id) && (fmt_d == vt::int32::id))
               || ((fmt_s == vt::nvfp4::id) && (fmt_d == vt::fp32::id));
  #else
    bool use_mx = false;
  #endif
    PFN_FEDP fedp = nullptr;
    if (!use_mx) {
      fedp = select_FEDP(fmt_s, fmt_d);
    }

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto t = i * cfg::tcN + j;
        auto c_val = rs3_data.at(t).u32;
        auto a_row = &a_tile[i * cfg::tcK];
        auto b_col = &b_tile[(i * cfg::tcN + j) * cfg::tcK];

        uint32_t d_val;
        if (use_mx) {
#ifdef TCU_MX_ENABLE
          const auto& mx = mx_meta_.at(wid);
          uint32_t row_idx = step_m * cfg::tcM + i;
          uint32_t col_idx = step_n * cfg::tcN + j;
          if (fmt_s == vt::nvfp4::id) {
            uint8_t sf_a = unpack_mx_scale_16(mx.at(0), mx.at(1), mx.at(2), mx.at(3), row_idx, "A");
            uint8_t sf_b = unpack_mx_scale_16(mx.at(4), mx.at(5), mx.at(6), mx.at(7), col_idx, "B");
            d_val = fedp_nvfp4_fp32_scaled(a_row, b_col, c_val, sf_a, sf_b);
          } else {
            uint8_t sf_a = unpack_mx_scale_8(mx.at(0), mx.at(1), row_idx, "A");
            uint8_t sf_b = unpack_mx_scale_8(mx.at(2), mx.at(3), col_idx, "B");
            if (fmt_s == vt::mxint8::id) {
              d_val = fedp_mxint8_int32_scaled(a_row, b_col, c_val, sf_a, sf_b);
            } else {
              d_val = fedp_mxfp8_fp32_scaled(a_row, b_col, c_val, sf_a, sf_b);
            }
          }
#else
          std::abort();
#endif
        } else {
          d_val = fedp(a_row, b_col, c_val);
        }

        rd_data.at(t).u64 = nan_box(d_val);
        DTH(3, simobject_->name() << " FEDP"
            << ": wid=" << wid << ", i=" << i << ", j=" << j
            << ", m=" << step_m << ", n=" << step_n
            << ", k=" << step_k << std::hex
            << ", a0=0x" << a_row[0].u32 << ", b0=0x" << b_col[0].u32
            << ", c=0x" << c_val << ", d=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  std::vector<std::vector<uint32_t>> sparse_meta_;
  std::vector<std::array<uint32_t, kMxMetaWords>> mx_meta_;
  std::unordered_map<uint32_t, lmem_desc_t[4]> lmem_desc_; // [0]=A, [1]=B, [2]=C, [3]=D
  PerfStats     perf_stats_;
  TileBufferState tbuf_state_;
#ifdef TCU_WGMMA_ENABLE
  std::queue<APrefetchReq> a_prefetch_queue_; // deferred PREFETCH_A requests (queued while TCU active)
#endif
};

///////////////////////////////////////////////////////////////////////////////

op_string_t TensorUnit::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::WGMMA: {
    uint32_t nrc = 8u << args.cd_nregs;
    std::string src_mode = std::string(args.is_a_smem ? "S" : "R") + "S"
                         + std::string(args.is_cd_smem ? "S" : "R"); // CD mode
    return {"WGMMA." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(nrc) + "." + src_mode
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  }
  case TcuType::META_STORE:
    return {"META_STORE." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::to_string(args.fmt_d), ""};
#ifdef TCU_WGMMA_ENABLE
  case TcuType::WGMMA_PREFETCH_A:
    return {"PREFETCH_A." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)), ""};
#endif
  default:
    std::abort();
  }
}

///////////////////////////////////////////////////////////////////////////////

uint32_t TcuUopGen::uop_count(const Instr& instr) {
  if (instr.get_fu_type() != FUType::TCU)
    return 1;

  auto tcu_type = std::get<TcuType>(instr.get_op_type());
  auto args = std::get<IntrTcuArgs>(instr.get_args());

  if (tcu_type == TcuType::WMMA) {
    using wmma = vt::wmma_config_t<NUM_THREADS>;
    bool is_sparse = args.is_sparse;
    bool is_mx = vt::mx_scale_format(args.fmt_s);
    uint32_t sparse_meta_stores = 0;
    if (is_sparse) {
      sparse_meta_stores = vt::sparse_meta_total_store_uops(args.fmt_s, wmma::stores_per_col, NUM_THREADS, wmma::meta_cols_per_load);
    }
    uint32_t mx_meta_stores = is_mx ? mx_meta_words(args.fmt_s) : 0;
    uint32_t k_count = is_sparse ? (wmma::k_steps / 2) : wmma::k_steps;
    uint32_t mma_steps = (wmma::sym_sparse && is_sparse)
                       ? (wmma::m_steps * wmma::n_steps * wmma::k_steps)
                       : (wmma::m_steps * wmma::n_steps * k_count);
    return sparse_meta_stores + mx_meta_stores + mma_steps;
  }

#ifdef TCU_WGMMA_ENABLE
  if (tcu_type == TcuType::WGMMA) {
    uint32_t nrc = 8u << args.cd_nregs;
    uint32_t k_count = args.is_sparse ? (wg_cfg::k_steps / 2) : wg_cfg::k_steps;
    uint32_t mma_uops = k_count * nrc;
    uint32_t meta_stores = 0;
    if (args.is_sparse && !args.is_a_smem) {
      // RS sparse: META_STORE uops using WGMMA-specific per_warp_depth=2
      constexpr uint32_t wg_depth = wg_cfg::m_steps * (wg_cfg::k_steps / 2);
      constexpr uint32_t wg_stores_per_col = (wg_depth + NUM_THREADS - 1) / NUM_THREADS;
      constexpr uint32_t wg_cols_per_load = (NUM_THREADS >= wg_depth) ? (NUM_THREADS / wg_depth) : 1;
      meta_stores = vt::sparse_meta_total_store_uops(args.fmt_s, wg_stores_per_col, NUM_THREADS, wg_cols_per_load);
    }
    return meta_stores + mma_uops;
  }
#endif

  return 1;
}

///////////////////////////////////////////////////////////////////////////////

Instr::Ptr TcuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  auto tcu_type = std::get<TcuType>(macro_instr.get_op_type());
  auto args = std::get<IntrTcuArgs>(macro_instr.get_args());
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint32_t total = uop_count(macro_instr);

  // Compute UUID for this micro-op
  uint32_t uuid_hi = (parent_uuid >> 32) & 0xffffffff;
  uint32_t uuid_lo = parent_uuid & 0xffffffff;
  uint32_t steps_shift = (total > 1) ? (32 - log2ceil(total)) : 0;
  uint64_t uop_uuid = (static_cast<uint64_t>(uuid_hi) << 32) | ((uop_index << steps_shift) | uuid_lo);

  auto uop_instr = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::TCU);
  uop_instr->set_parent_uuid(parent_uuid);

  if (tcu_type == TcuType::WMMA) {
    using wmma = vt::wmma_config_t<NUM_THREADS>;
    constexpr uint32_t rc_base = 0, ra_base = 10;
    constexpr uint32_t rb_base = (wmma::NRB == 4) ? 28 : 24;
    bool is_sparse = args.is_sparse;
    uint32_t fmt_s = args.fmt_s;
    uint32_t fmt_d = args.fmt_d;

    bool is_mx = vt::mx_scale_format(fmt_s);
    uint32_t sparse_meta_stores = 0;
    if (is_sparse) {
      sparse_meta_stores = vt::sparse_meta_total_store_uops(fmt_s, wmma::stores_per_col, NUM_THREADS, wmma::meta_cols_per_load);
    }
    uint32_t mx_meta_stores = is_mx ? mx_meta_words(fmt_s) : 0;
    uint32_t total_meta_stores = sparse_meta_stores + mx_meta_stores;

    if (uop_index < sparse_meta_stores) {
      // Phase 1a: sparse metadata-store uops (one per group of cols_per_load columns)
      constexpr uint32_t meta_reg0 = 14, meta_reg1 = 15;
      uint32_t group = uop_index;
      uint32_t reg_rs1 = (group > 0) ? meta_reg1 : meta_reg0;
      uop_instr->set_op_type(TcuType::META_STORE);
      uop_instr->set_args(IntrTcuArgs{false, 0, 0, fmt_s, group, 0, 0, 0, TCU_META_KIND_SPARSE, 0});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else if (uop_index < total_meta_stores) {
      // Phase 1b: MX scale metadata-store uops
      uint32_t mx_store = uop_index - sparse_meta_stores;
      uint32_t reg_rs1 = 0;
      if (fmt_s == vt::nvfp4::id) {
        constexpr uint32_t nvfp4_regs[] = {8, 9, 20, 21, 18, 19, 22, 23};
        reg_rs1 = nvfp4_regs[mx_store];
      } else {
        constexpr uint32_t mx8_regs[] = {8, 9, 18, 19};
        reg_rs1 = mx8_regs[mx_store];
      }
      uop_instr->set_op_type(TcuType::META_STORE);
      uop_instr->set_args(IntrTcuArgs{0, 0, 0, fmt_s, mx_store, 0, 0, 0, TCU_META_KIND_MX, 0});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else {
      // Phase 2: MMA uops
      uint32_t mma_idx = uop_index - total_meta_stores;
      uint32_t k_count = is_sparse ? (wmma::k_steps / 2) : wmma::k_steps;

      if (wmma::sym_sparse && is_sparse) {
        // Symmetric-sparse: flatten (m, n, k) into a single counter
        constexpr uint32_t lg_n = (wmma::n_steps > 1) ? log2ceil(wmma::n_steps) : 0;
        constexpr uint32_t lg_k = (wmma::k_steps > 1) ? log2ceil(wmma::k_steps) : 0;
        constexpr uint32_t step_bits = lg_n + lg_k;
        constexpr uint32_t step_mask = step_bits ? ((1u << step_bits) - 1) : 0;
        constexpr uint32_t sym_mask_lo = []() {
          uint32_t mask = 0;
          for (uint32_t lane = 0; lane < NUM_THREADS; ++lane)
            if ((lane % wmma::tcN) < (wmma::tcN / 2)) mask |= (1u << lane);
          return mask;
        }();
        constexpr uint32_t all_lanes = (NUM_THREADS == 32) ? 0xffffffffu : ((1u << NUM_THREADS) - 1);

        uint32_t n_sp = step_bits ? (mma_idx & step_mask) : 0;
        uint32_t m_sp = mma_idx >> step_bits;
        // n_sp encodes both actual N step (high bits) and lo/hi half (low lg_k bits).
        // Extract just the N step for accum indexing; n_sp is still used for B register selection.
        uint32_t actual_n = lg_k ? (n_sp >> lg_k) : n_sp;
        uint32_t reg_rs3 = rc_base + (mma_idx >> 1);
        uop_instr->set_op_type(TcuType::WMMA);
        uop_instr->set_args(IntrTcuArgs{true, 0, 0, fmt_s, fmt_d, m_sp, actual_n, 0, 0, 0});
        uop_instr->set_dest_reg(reg_rs3, RegType::Float);
        uop_instr->set_src_reg(0, ra_base + m_sp, RegType::Float);
        uop_instr->set_src_reg(1, rb_base + n_sp, RegType::Float);
        uop_instr->set_src_reg(2, reg_rs3, RegType::Float);
        // Symmetric sparse: no k-loop, always wb=1 and always reads C from RF
        uop_instr->set_tmask(ThreadMask(NUM_THREADS, (mma_idx & 1) ? (all_lanes & ~sym_mask_lo) : sym_mask_lo));
      } else {
        // Standard k-major triple loop (dense or non-sym sparse)
        uint32_t b_sub = is_sparse ? wmma::b_sub_blocks_sp : wmma::b_sub_blocks;
        uint32_t mn = wmma::m_steps * wmma::n_steps;
        uint32_t k = mma_idx / mn;
        uint32_t rem = mma_idx % mn;
        uint32_t m = rem / wmma::n_steps;
        uint32_t n = rem % wmma::n_steps;
        // Bank-conflict-free register offset formulas (0 stalls).
        // Permutes A, B, C offsets so all three operands always land in
        // different GPR banks for every uop.
        uint32_t reg_rs1, reg_rs2, reg_rs3;
        if (is_sparse) {
          // Sparse non-sym: A=identity, B and C permuted for 0 stalls.
          // B={n[hi], ~(n[0]^k), ~k}, C={n[hi], m, ~(m^n[0])}
          uint32_t n_hi = n >> 1;
          uint32_t n_lo = n & 1;
          uint32_t a_off = (m / wmma::a_sub_blocks) * k_count + k;
          uint32_t b_off = (n_hi << 2) | ((1 - (n_lo ^ k)) << 1) | (1 - k);
          uint32_t c_off = (n_hi << 2) | (m << 1) | (1 - (m ^ n_lo));
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        } else if (b_sub == 1) {
          // Dense Pattern A (NT=4,16,64): m_steps=4, k_steps=2, n_steps=2
          // A={m[0], ~m[hi], k}, B={n^k, ~k}, C={m[0], ~m[hi], XNOR(m[hi],n)}
          uint32_t m_hi = m >> 1;
          uint32_t m_lo = m & 1;
          uint32_t a_off = (m_lo << 2) | ((1 - m_hi) << 1) | k;
          uint32_t b_off = ((n ^ k) << 1) | (1 - k);
          uint32_t c_off = (m_lo << 2) | ((1 - m_hi) << 1) | (1 - (m_hi ^ n));
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        } else {
          // Dense Pattern B (NT=8,32): m_steps=2, k_steps=4, n_steps=4, b_sub=2
          // A={k[0], ~m, m^k[hi]}, B={k[0], k[hi]^np, ~np}, C={n[0], ~m, n[hi]}
          uint32_t k_hi = k >> 1;
          uint32_t k_lo = k & 1;
          uint32_t n_pair = n >> 1;
          uint32_t n_lo = n & 1;
          uint32_t a_off = (k_lo << 2) | ((1 - m) << 1) | (m ^ k_hi);
          uint32_t b_off = (k_lo << 2) | ((k_hi ^ n_pair) << 1) | (1 - n_pair);
          uint32_t c_off = (n_lo << 2) | ((1 - m) << 1) | n_pair;
          reg_rs1 = ra_base + a_off;
          reg_rs2 = rb_base + b_off;
          reg_rs3 = rc_base + c_off;
        }
        uop_instr->set_op_type(TcuType::WMMA);
        uop_instr->set_args(IntrTcuArgs{is_sparse, 0, 0, fmt_s, fmt_d, m, n, k, 0, 0});
        uop_instr->set_dest_reg(reg_rs3, RegType::Float);
        uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
        uop_instr->set_src_reg(1, reg_rs2, RegType::Float);
        uop_instr->set_src_reg(2, reg_rs3, RegType::Float);
      }
    }
  }
#ifdef TCU_WGMMA_ENABLE
  else if (tcu_type == TcuType::WGMMA) {
    constexpr uint32_t m_steps = wg_cfg::m_steps;
    constexpr uint32_t k_steps = wg_cfg::k_steps;
    uint32_t fmt_s = args.fmt_s;
    uint32_t fmt_d = args.fmt_d;
    bool is_sparse = args.is_sparse;
    bool is_a_smem = args.is_a_smem;
    uint32_t cd_nregs = args.cd_nregs;
    uint32_t k_count = is_sparse ? (k_steps / 2) : k_steps;
    constexpr uint32_t a0 = 10, a1 = 11;

    // RS sparse: fused meta-store uops before MMA phase
    uint32_t meta_stores = 0;
    if (is_sparse && !is_a_smem) {
      constexpr uint32_t wg_depth = wg_cfg::m_steps * (wg_cfg::k_steps / 2);
      constexpr uint32_t wg_stores_per_col = (wg_depth + NUM_THREADS - 1) / NUM_THREADS;
      constexpr uint32_t wg_cols_per_load = (NUM_THREADS >= wg_depth) ? (NUM_THREADS / wg_depth) : 1;
      meta_stores = vt::sparse_meta_total_store_uops(fmt_s, wg_stores_per_col, NUM_THREADS, wg_cols_per_load);
    }

    if (uop_index < meta_stores) {
      // Meta-store phase: preload metadata from f26/f27 into VX_tcu_meta SRAM
      constexpr uint32_t wg_meta_reg0 = 24 + wg_cfg::m_steps * (wg_cfg::k_steps / 2); // f26
      constexpr uint32_t wg_meta_reg1 = wg_meta_reg0 + 1; // f27
      uint32_t group = uop_index;
      uint32_t reg_rs1 = (group > 0) ? wg_meta_reg1 : wg_meta_reg0;
      uop_instr->set_op_type(TcuType::META_STORE);
      uop_instr->set_args(IntrTcuArgs{false, 0, 0, fmt_s, group, 0, 0, 0, TCU_META_KIND_SPARSE_WG, 0});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else {
      // MMA phase
      uint32_t mma_idx = uop_index - meta_stores;
      uint32_t ra_base = is_a_smem ? 10 : 24;

      uint32_t k, m, n;
      if (args.is_cd_smem) {
        // K-inner: k (inner) → m → n (outer), matching RTL lmem-accumulator loop order
        k = mma_idx % k_count;
        m = (mma_idx / k_count) % m_steps;
        n = mma_idx / (k_count * m_steps);
      } else {
        // A-first / B-stream: k (outer) → n (middle) → m (inner)
        // Each (k,n) B column is exhausted across all m_steps before the next is fetched.
        uint32_t nm = 8u << cd_nregs;  // = n_steps * m_steps
        k = mma_idx / nm;
        n = (mma_idx % nm) / m_steps;
        m = mma_idx % m_steps;
      }
      uint32_t r = n * m_steps + m;

      uop_instr->set_op_type(TcuType::WGMMA);
      uop_instr->set_args(IntrTcuArgs{is_sparse, is_a_smem ? 1u : 0u, cd_nregs,
                                     fmt_s, fmt_d, m, n, k, 0, args.is_cd_smem ? 1u : 0u});
      if (!args.is_cd_smem) {
        uop_instr->set_dest_reg(r, RegType::Float);
      }
      if (mma_idx == 0) {
        if (is_a_smem) {
          uop_instr->set_src_reg(0, a0, RegType::Integer);
        } else {
          uint32_t rs1_off = m * k_count + k;
          uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
        }
        uop_instr->set_src_reg(1, a1, RegType::Integer);
        // First uop carries cd_desc from x12 (a2) when C/D are smem accumulators.
        if (args.is_cd_smem) {
          uop_instr->set_src_reg(2, 12, RegType::Integer);
        }
      } else if (!is_a_smem) {
        uint32_t rs1_off = m * k_count + k;
        uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
      }
      // When is_cd_smem and mma_idx>0, C is loaded from smem; no register src needed.
      if (!args.is_cd_smem) {
        uop_instr->set_src_reg(2, r, RegType::Float);
      }
    }
    // fu_lock on first uop, fu_unlock on last uop
    uop_instr->set_fu_lock(uop_index == 0);
    uop_instr->set_fu_unlock(uop_index == (total - 1));
  }
#endif

  return uop_instr;
}

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

const TensorUnit::PerfStats &TensorUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TensorUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      uint32_t step_k,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data,
                      bool is_sparse) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, step_k,
              rs1_data, rs2_data, rs3_data,
              rd_data,
              trace_data,
              is_sparse);
}

void TensorUnit::wgmma(uint32_t wid,
                       uint32_t fmt_s,
                       uint32_t fmt_d,
                       uint32_t step_m,
                       uint32_t step_n,
                       uint32_t step_k,
                       uint32_t a_desc,
                       uint32_t b_desc,
                       uint32_t cd_desc,
                       const std::vector<reg_data_t>& rs1_data,
                       const std::vector<reg_data_t>& rs3_data,
                       std::vector<reg_data_t>& rd_data,
                       ExeTraceData* trace_data,
                       bool is_sparse,
                       uint32_t cd_nregs,
                       uint32_t is_a_smem,
                       uint32_t is_cd_smem) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, a_desc, b_desc,
               cd_desc, rs1_data, rs3_data, rd_data, trace_data,
               is_sparse, cd_nregs, is_a_smem, is_cd_smem);
}

void TensorUnit::meta_store(uint32_t wid,
                            uint32_t fmt_s,
                            uint32_t col_idx,
                            uint32_t meta_kind,
                            const std::vector<reg_data_t>& rs1_data,
                            ExeTraceData* trace_data) {
  impl_->meta_store(wid, fmt_s, col_idx, meta_kind, rs1_data, trace_data);
}

#ifdef TCU_WGMMA_ENABLE
void TensorUnit::wgmma_prefetch_a(uint32_t wid,
                                   uint32_t fmt_s,
                                   bool is_sparse,
                                   uint32_t a_desc) {
  impl_->wgmma_prefetch_a(wid, fmt_s, is_sparse, a_desc);
}
#endif

