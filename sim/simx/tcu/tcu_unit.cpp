
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

#include "tcu_unit.h"
#include "tensor_cfg.h"
#include <rvfloats.h>
#include "core.h"
#include "scheduler.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg    = vt::wmma_config_t<NUM_THREADS>;
using wg_cfg = vt::wgmma_config_t<NUM_THREADS, vt::fp32, vt::fp32>;

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
//   * Wide Ot (fp32): accumulate Σ(a_k*b_k) in fp32, add c_val last.
//   * Narrow Ot (fp16/bf16/fp8/bf8/…): chain FMA<It,Ot> so the accumulator is
//     rounded to Ot each step.
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

class TcuUnit::Impl {
public:

  struct lmem_desc_t {
    uint64_t base = 0;
    uint32_t ldm = 0;
    bool col_major = false;
  };

  Impl(TcuUnit* simobject, Core* core)
    : simobject_(simobject)
    , core_(core)
    , sparse_meta_(NUM_WARPS, std::vector<uint32_t>(kMetaBanks * kMaxMetaCols, 0))
    , mx_meta_(NUM_WARPS)
    , perf_stats_()
  {
    exec_done_.fill(false);
    wgmma_planned_.fill(false);
    in_wgmma_.fill(false);
  }

  ~Impl() {}

  void reset() {
    perf_stats_ = PerfStats();
    for (auto& sparse_meta : sparse_meta_) {
      std::fill(sparse_meta.begin(), sparse_meta.end(), 0);
    }
    for (auto& mx_meta : mx_meta_) {
      mx_meta.fill(0);
    }
    exec_done_.fill(false);
    wgmma_planned_.fill(false);
    in_wgmma_.fill(false);
    cta_owner_a_.fill(-1);
    cta_owner_b_ = -1;
    cur_block_ = 0;
  }

  void tick() {
  #ifdef TCU_WGMMA_ENABLE
    // Q-warp lock-step probe.
    // Pass 1 — identify active WGMMA blocks and prime each one's plan() on
    // first uop. WMMA / META_STORE blocks are unaffected (no Q-coupling).
    uint32_t wgmma_active = 0;
    for (uint32_t b = 0; b < NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty()) continue;
      auto trace = input.peek();
      if (std::get<TcuType>(trace->op_type) != TcuType::WGMMA) continue;
      wgmma_active |= (1u << b);

      if (wgmma_planned_.at(b)) continue;
      auto& instr = *trace->instr_ptr;
      auto tpuArgs = std::get<IntrTcuArgs>(instr.get_args());
      if (!(tpuArgs.step_m == 0 && tpuArgs.step_n == 0 && tpuArgs.step_k == 0)) {
        wgmma_planned_.at(b) = true;
        continue;
      }
      uint32_t wid = trace->wid;
      auto& rs1_data = trace->src_data[0];
      auto& rs2_data = trace->src_data[1];
      uint32_t a_desc = rs1_data.empty() ? 0 : rs1_data.at(0).u32;
      uint32_t b_desc = rs2_data.empty() ? 0 : rs2_data.at(0).u32;

      // CTA-overlap fence — defer this block's WGMMA if any other block
      // is mid-flight with a different CTA. The shared B buffer assumes
      // single-CTA occupancy across all blocks; without this fence, a
      // late-arriving CTA[i+1] WGMMA finds `any_in_wgmma==true` and the
      // invalidate_b() path is skipped, so plan_b() ADDS new lines on
      // top of CTA[i]'s resident bbuf (overlap). The deferred block
      // re-enters pass 1 next tick; once CTA[i]'s last block unlocks,
      // it plans cleanly.
      int32_t new_cta = (int32_t)core_->scheduler().warp(wid).cta_csrs.cta_id;
      bool block_other_cta_inflight = false;
      for (uint32_t k = 0; k < NUM_TCU_BLOCKS; ++k) {
        if (k == b) continue;
        if (in_wgmma_.at(k) && cta_owner_a_.at(k) != new_cta) {
          block_other_cta_inflight = true;
          break;
        }
      }
      if (block_other_cta_inflight) {
        wgmma_active &= ~(1u << b);
        wgmma_planned_.at(b) = false;
        continue;
      }

      // Drop the shared B buffer only when no other block is mid-WGMMA —
      // otherwise we'd evict their resident bytes mid-flight.
      bool any_in_wgmma = false;
      for (auto v : in_wgmma_) any_in_wgmma = any_in_wgmma || v;
      auto& tbuf = simobject_->tbuf();
      if (!any_in_wgmma) {
        tbuf->invalidate_b();
        cta_owner_b_ = -1;
      }
      tbuf->invalidate_a(b);
      tbuf->invalidate_m(b);
      this->plan_wgmma_lines(b, wid, a_desc, b_desc, tpuArgs);
      if (tbuf->ready_a(b) && tbuf->ready_m(b) && tbuf->ready_b()) {
        ++perf_stats_.tbuf_cache_hits;
      }
      in_wgmma_.at(b) = true;
      wgmma_planned_.at(b) = true;
      cta_owner_a_.at(b) = new_cta;
      if (cta_owner_b_ == -1) cta_owner_b_ = new_cta;
    }

    // Pass 2 — lock-step gate. All active WGMMA blocks must have their A/B
    // operands resident before *any* of them advances.
    if (wgmma_active != 0) {
      uint32_t ready_mask = 0;
      auto& tbuf = simobject_->tbuf();
      for (uint32_t b = 0; b < NUM_TCU_BLOCKS; ++b) {
        if (!((wgmma_active >> b) & 1u)) continue;
        auto trace = simobject_->Inputs.at(b).peek();
        auto tpuArgs = std::get<IntrTcuArgs>(trace->instr_ptr->get_args());
        bool a_ok = !tpuArgs.is_a_smem || tbuf->ready_a(b);
        bool m_ok = tbuf->ready_m(b);
        bool b_ok = tbuf->ready_b();
        if (a_ok && m_ok && b_ok) ready_mask |= (1u << b);
      }
      if (ready_mask != wgmma_active) {
        ++perf_stats_.tbuf_stalls;
        return; // hold all blocks; per-block dispatch deferred to next tick
      }
    }
  #endif

    for (uint32_t b = 0; b < NUM_TCU_BLOCKS; ++b) {
      auto& input = simobject_->Inputs.at(b);
      if (input.empty())
        continue;
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);

    #ifdef TCU_WGMMA_ENABLE
      // CTA-overlap fence deferred this block's WGMMA — skip processing
      // until pass 1 plans it (when the previous CTA's blocks all drain).
      if (tcu_type == TcuType::WGMMA && !wgmma_planned_.at(b))
        continue;
    #endif

      // Lazy execute on first peek of this trace. Side effects (LMEM
      // accumulator updates) persist across stall/backpressure retries via
      // exec_done_[b].
      if (!exec_done_.at(b)) {
        auto& instr = *trace->instr_ptr;
        auto tpuArgs = std::get<IntrTcuArgs>(instr.get_args());
        uint32_t wid = trace->wid;
        uint32_t num_threads = NUM_THREADS;
        auto& rs1_data = trace->src_data[0];
        auto& rs2_data = trace->src_data[1];
        auto& rs3_data = trace->src_data[2];
        trace->dst_data.assign(num_threads, reg_data_t{});
        auto& rd_data = trace->dst_data;

        switch (tcu_type) {
        case TcuType::WMMA:
          this->wmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                     tpuArgs.step_m, tpuArgs.step_n, tpuArgs.step_k,
                     rs1_data, rs2_data, rs3_data, rd_data,
                     tpuArgs.is_sparse);
          break;
      #ifdef TCU_WGMMA_ENABLE
        case TcuType::WGMMA: {
          uint32_t a_desc = rs1_data.empty() ? 0 : rs1_data.at(0).u32;
          uint32_t b_desc = rs2_data.empty() ? 0 : rs2_data.at(0).u32;
          cur_block_ = b;
          this->wgmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                      tpuArgs.step_m, tpuArgs.step_n, tpuArgs.step_k,
                      a_desc, b_desc, rs1_data, rs3_data, rd_data,
                      tpuArgs.is_sparse,
                      tpuArgs.cd_nregs, tpuArgs.is_a_smem);
        } break;
      #endif
        case TcuType::META_STORE:
          this->meta_store(wid, tpuArgs.fmt_s, tpuArgs.fmt_d,
                           tpuArgs.meta_kind, rs1_data);
          break;
        default:
          std::abort();
        }
        exec_done_.at(b) = true;
      }

      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
      case TcuType::WGMMA:
        delay = 4;
        break;
      case TcuType::META_STORE:
        delay = 1;
        break;
      default:
        std::abort();
      }
      if (simobject_->Outputs.at(b).try_send(trace, 2 + delay)) {
        exec_done_.at(b) = false;
      #ifdef TCU_WGMMA_ENABLE
        // WGMMA wind-down: clear plan/in-flight flags on the last uop so
        // the next WGMMA at this block re-decodes its descriptors.
        if (tcu_type == TcuType::WGMMA && trace->instr_ptr->get_fu_unlock()) {
          wgmma_planned_.at(b) = false;
          in_wgmma_.at(b) = false;
          cta_owner_a_.at(b) = -1;
        }
      #endif
        DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
        input.pop();
      }
    }
  }

  // Plan all line addresses required for the current WGMMA's A, B and
  // sparse-metadata tiles into the per-role caches inside TcuTbuf.
  // Lines already resident or in-flight are skipped (additive plan).
  void plan_wgmma_lines(uint32_t b, uint32_t wid,
                        uint32_t a_desc, uint32_t b_desc,
                        const IntrTcuArgs& args) {
    uint32_t fmt_s = args.fmt_s;
    bool is_sparse = args.is_sparse;
    bool is_a_smem = args.is_a_smem;
    uint32_t e_bits = elem_bits(fmt_s);
    if (e_bits < 8) return;
    uint32_t e_bytes = e_bits / 8;
    // NRC is encoded in args.cd_nregs (0→8, 1→16, 2→32). xtileN = NRC for the
    // canonical (xtileM=2*tcM) layout.
    uint32_t nrc      = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    uint32_t xtile_n  = nrc;

    lmem_desc_t sd_a{}, sd_b{};
    if (is_a_smem) {
      sd_a = {uint64_t(LMEM_BASE_ADDR) + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
      lmem_desc_[wid][0] = sd_a;
    }
    sd_b = {uint64_t(LMEM_BASE_ADDR) + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
    lmem_desc_[wid][1] = sd_b;

    // The gather walks the un-packed K element range:
    //   tileK = xtileK × i_ratio = xtileK × ratio   (i_ratio = 32 / e_bits)
    // Sparse compresses K *only on A*; B remains dense in K.
    uint32_t ratio  = 32 / e_bits;
    uint32_t tile_k = uint32_t(wg_cfg::xtileK) * ratio;
    uint32_t a_k    = is_sparse ? (tile_k / 2) : tile_k;

    // Block-major SMEM layout — descriptor stride field (ldm) is zero when
    // the kernel writes block-major: a-blocks/b-blocks sit on contiguous
    // bank-rows, and operand byte addresses use the block-decomposed offset
    // formula. Row-major path (non-zero ldm) preserved for legacy kernels.
    uint32_t k_blk_dim   = cfg::tcK * ratio;
    uint32_t a_blk_elems = cfg::tcM * k_blk_dim;
    uint32_t b_blk_elems = k_blk_dim * cfg::tcN;
    uint32_t n_steps     = xtile_n / cfg::tcN;

    auto& tbuf = simobject_->tbuf();

    // A: SS-mode only. xtileM rows × a_k columns (row-major-equivalent view).
    if (is_a_smem) {
      bool a_block_major = (sd_a.ldm == 0);
      std::vector<uint64_t> a_lines;
      a_lines.reserve(uint32_t(wg_cfg::xtileM) * a_k);
      for (uint32_t r = 0; r < wg_cfg::xtileM; ++r) {
        for (uint32_t c = 0; c < a_k; ++c) {
          uint64_t elem_off;
          if (a_block_major) {
            uint32_t m_blk = r / cfg::tcM;
            uint32_t i_in  = r % cfg::tcM;
            uint32_t k_blk = c / k_blk_dim;
            uint32_t k_in  = c % k_blk_dim;
            elem_off = (k_blk * wg_cfg::m_steps + m_blk) * a_blk_elems
                     + i_in * k_blk_dim + k_in;
          } else {
            elem_off = uint64_t(r) * sd_a.ldm + c;
          }
          uint64_t addr = sd_a.base + elem_off * e_bytes;
          a_lines.push_back(addr & ~uint64_t(MEM_BLOCK_SIZE - 1));
        }
      }
      tbuf->plan_a(b, a_lines);

      // Sparse metadata: kMetaBanks × meta_strd_words 32-bit words, packed
      // immediately after A in LMEM. Lifetime is the whole WGMMA, not the
      // k-stripe — kept in its own buffer so future A-side eviction policies
      // can't drop it mid-flight.
      if (is_sparse) {
        uint32_t rtl_i_ratio = 32 / e_bits;
        uint32_t meta_row_w  = cfg::tcK * 2 * rtl_i_ratio;
        uint32_t meta_strd_words = (cfg::tcM * meta_row_w + 31) / 32;
        uint32_t total_meta_words = kMetaBanks * meta_strd_words;
        // A region size in bytes: xtileM × a_k elements (row-major-equivalent).
        // For block-major: contiguous a-block layout, same total byte count.
        // For row-major:   xtileM rows × ldm-element stride.
        uint64_t a_region_bytes = a_block_major
            ? uint64_t(wg_cfg::xtileM) * a_k * e_bytes
            : uint64_t(wg_cfg::xtileM) * sd_a.ldm * e_bytes;
        uint64_t meta_base = sd_a.base + a_region_bytes;
        std::vector<uint64_t> m_lines;
        m_lines.reserve(total_meta_words);
        for (uint32_t w = 0; w < total_meta_words; ++w) {
          uint64_t addr = meta_base + uint64_t(w) * 4;
          m_lines.push_back(addr & ~uint64_t(MEM_BLOCK_SIZE - 1));
        }
        tbuf->plan_m(b, m_lines);
      }
    }

    // B: always SS, always dense in K. tileK rows × xtileN columns.
    bool b_block_major = (sd_b.ldm == 0);
    std::vector<uint64_t> b_lines;
    b_lines.reserve(tile_k * xtile_n);
    for (uint32_t r = 0; r < tile_k; ++r) {
      for (uint32_t c = 0; c < xtile_n; ++c) {
        uint64_t elem_off;
        if (b_block_major) {
          uint32_t k_blk = r / k_blk_dim;
          uint32_t r_in  = r % k_blk_dim;
          uint32_t n_blk = c / cfg::tcN;
          uint32_t n_in  = c % cfg::tcN;
          // Within-block layout: N outer, K inner — matches kernel
          // b_blockmajor_idx and tcu_core's b_off + j*TC_K + k.
          elem_off = (k_blk * n_steps + n_blk) * b_blk_elems
                   + n_in * k_blk_dim + r_in;
        } else {
          elem_off = uint64_t(r) * sd_b.ldm + c;
        }
        uint64_t addr = sd_b.base + elem_off * e_bytes;
        b_lines.push_back(addr & ~uint64_t(MEM_BLOCK_SIZE - 1));
      }
    }
    tbuf->plan_b(b_lines);
  }

  void meta_store(uint32_t wid,
                  uint32_t fmt_s,
                  uint32_t col_idx,
                  uint32_t meta_kind,
                  const std::vector<reg_data_t>& rs1_data) {
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
            bool is_sparse) {
    if (is_sparse) {
      if (!vt::sparse_format_supported(fmt_s)) {
        std::cout << "Error: WMMA_SP unsupported input format: "
                  << vt::fmt_string(fmt_s) << " (id=" << fmt_s
                  << "). Supported formats: i8, u8, fp8, bf8, fp16, bf16, i4, u4." << std::endl;
        std::abort();
      }
      if ((NUM_THREADS % cfg::b_block_size_sp) != 0) {
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
  }

  void wgmma(uint32_t wid,
             uint32_t fmt_s,
             uint32_t fmt_d,
             uint32_t step_m,
             uint32_t step_n,
             uint32_t step_k,
             uint32_t a_desc,
             uint32_t b_desc,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             bool is_sparse,
             uint32_t cd_nregs,
             uint32_t is_a_smem) {
    __unused(cd_nregs);
    if (is_sparse && !vt::sparse_format_supported(fmt_s)) {
      std::cout << "Error: WGMMA_SP unsupported input format: "
                << vt::fmt_string(fmt_s) << " (id=" << fmt_s << ")" << std::endl;
      std::abort();
    }

    uint32_t ratio   = elem_ratio(fmt_s);
    uint32_t k_words = cfg::tcK;
    uint32_t e_bytes = elem_bits(fmt_s) / 8;

    // Decode smem descriptors (B is always from smem, A optionally)
    lmem_desc_t sd_a, sd_b;
    if (step_k == 0 && step_m == 0 && step_n == 0) {
      if (is_a_smem) {
        sd_a = {uint64_t(LMEM_BASE_ADDR) + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
        lmem_desc_[wid][0] = sd_a;
      }
      sd_b = {uint64_t(LMEM_BASE_ADDR) + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
      lmem_desc_[wid][1] = sd_b;
    } else {
      sd_a = lmem_desc_[wid][0];
      sd_b = lmem_desc_[wid][1];
    }
    // load_lmem_word distinguishes A vs B by descriptor base.
    cur_a_desc_base_ = is_a_smem ? sd_a.base : ~uint64_t(0);
    // NRC drives B's xtileN. Encoding: cd_nregs 0/1/2 → NRC 8/16/32.
    cur_xtile_n_ = (cd_nregs == 0) ? 8 : (cd_nregs == 1) ? 16 : 32;

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
      // Bank encoding {step_m, step_k_half} with WMMA half-K width.
      // WGMMA sparse issues step_k=0, so m=1 maps to bank (cfg::k_steps/2).
      uint32_t wg_bank_rs = step_m * (cfg::k_steps / 2) + step_k;
      uint32_t wg_bank_ss = step_m * (wg_cfg::k_steps / 2) + step_k;
      uint32_t wg_bank = is_a_smem ? wg_bank_ss : wg_bank_rs;
      // For SS mode, read metadata from LMEM; for RS mode, use register data.
      // Metadata follows the A-tile in LMEM. For block-major (sd_a.ldm == 0)
      // the A region is xtileM × a_k contiguous elements; for row-major it
      // spans xtileM × ldm-stride elements.
      uint64_t meta_base = 0;
      if (is_a_smem) {
        uint32_t a_k_local = is_sparse ? (wg_cfg::xtileK * ratio / 2)
                                       : (wg_cfg::xtileK * ratio);
        uint64_t a_region_bytes = (sd_a.ldm == 0)
            ? uint64_t(wg_cfg::xtileM) * a_k_local * e_bytes
            : uint64_t(wg_cfg::xtileM) * sd_a.ldm * e_bytes;
        meta_base = sd_a.base + a_region_bytes;
      }
      auto meta_bit_wg = [&](uint32_t bit_idx) -> uint32_t {
        uint32_t word_val = 0;
        if (is_a_smem) {
          // SS-sparse metadata lives in LMEM at meta_base, packed into
          // kMetaBanks × meta_strd_words 32-bit words. Words are pre-fetched
          // into the per-block metadata buffer (see plan_wgmma_lines).
          uint32_t word_idx  = wg_bank * meta_strd_words + bit_idx / 32;
          uint64_t byte_addr = meta_base + uint64_t(word_idx) * 4;
          auto line = simobject_->tbuf()->read_m(cur_block_, byte_addr);
          if (!line) {
            std::cout << "Error: TCU metadata buffer miss at 0x"
                      << std::hex << byte_addr << std::dec << std::endl;
            std::abort();
          }
          uint32_t off = byte_addr & (MEM_BLOCK_SIZE - 1);
          word_val = (uint32_t((*line)[off    ]))        |
                     (uint32_t((*line)[off + 1]) << 8)   |
                     (uint32_t((*line)[off + 2]) << 16)  |
                     (uint32_t((*line)[off + 3]) << 24);
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

    fedp_tile(wid, step_m, step_n, step_k, fmt_s, fmt_d,
              a_tile, b_tile, rs3_data, rd_data);
    __unused(b_desc);
  }

  const PerfStats& perf_stats() const {
    // lmem_reads tracks actual MemReq traffic issued by TcuTbuf onto the
    // TCU LMEM port (sum across abuf×Q + mbuf×Q + bbuf).
    perf_stats_.lmem_reads = simobject_->tbuf()->reads();
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

  // Gather one 32-bit operand word from a TCU line cache.
  // `read_line` is supplied by the caller and routes to the right per-role
  // buffer inside TcuTbuf (A → read_a, B → read_b). For sub-32-bit formats,
  // packs `ratio = 32 / e_bits` adjacent elements along the K direction
  // (col for A, row for B via pack_along_row).
  template <typename ReadLine>
  uint32_t gather_word(ReadLine read_line, const lmem_desc_t& desc,
                       uint32_t row, uint32_t col,
                       uint32_t fmt_s, bool pack_along_row) const {
    uint32_t e_bits  = elem_bits(fmt_s);
    uint32_t ratio   = (e_bits >= 32) ? 1 : (32 / e_bits);
    uint32_t e_bytes = (e_bits >= 8)  ? (e_bits / 8) : 1;
    uint32_t result = 0;
    for (uint32_t r = 0; r < ratio; ++r) {
      uint32_t cur_row = pack_along_row ? (row + r) : row;
      uint32_t cur_col = pack_along_row ? col       : (col + r);
      uint64_t byte_addr;
      if (desc.ldm == 0) {
        // Block-major SMEM. K dimension is along col for A (pack_along_row
        // false) and along row for B (pack_along_row true).
        uint32_t k_blk_dim = cfg::tcK * ratio;
        if (pack_along_row) {
          // B: r is K coord, c is N coord. Within-block layout: N outer,
          // K inner — matches kernel b_blockmajor_idx and tcu_core's
          // b_off + j*TC_K + k.
          uint32_t k_blk = cur_row / k_blk_dim;
          uint32_t r_in  = cur_row % k_blk_dim;
          uint32_t n_blk = cur_col / cfg::tcN;
          uint32_t n_in  = cur_col % cfg::tcN;
          uint32_t b_blk_elems = k_blk_dim * cfg::tcN;
          uint32_t n_steps     = cur_xtile_n_ / cfg::tcN;
          uint64_t off = (k_blk * n_steps + n_blk) * b_blk_elems
                       + n_in * k_blk_dim + r_in;
          byte_addr = desc.base + off * e_bytes;
        } else {
          // A: r is M coord, c is K coord
          uint32_t m_blk = cur_row / cfg::tcM;
          uint32_t i_in  = cur_row % cfg::tcM;
          uint32_t k_blk = cur_col / k_blk_dim;
          uint32_t k_in  = cur_col % k_blk_dim;
          uint32_t a_blk_elems = cfg::tcM * k_blk_dim;
          uint64_t off = (k_blk * wg_cfg::m_steps + m_blk) * a_blk_elems
                       + i_in * k_blk_dim + k_in;
          byte_addr = desc.base + off * e_bytes;
        }
      } else if (desc.col_major) {
        byte_addr = desc.base + (uint64_t(cur_col) * desc.ldm + cur_row) * e_bytes;
      } else {
        byte_addr = desc.base + (uint64_t(cur_row) * desc.ldm + cur_col) * e_bytes;
      }
      auto line = read_line(byte_addr);
      if (!line) {
        // Caller must call ready() first; this is a programmer error.
        std::cout << "Error: TCU buffer miss at 0x" << std::hex << byte_addr
                  << std::dec << std::endl;
        std::abort();
      }
      uint32_t off = byte_addr & (MEM_BLOCK_SIZE - 1);
      if (e_bits == 32) {
        result = (uint32_t((*line)[off    ]))        |
                 (uint32_t((*line)[off + 1]) << 8)   |
                 (uint32_t((*line)[off + 2]) << 16)  |
                 (uint32_t((*line)[off + 3]) << 24);
        return result;
      } else if (e_bits == 16) {
        uint32_t val = uint32_t((*line)[off]) | (uint32_t((*line)[off + 1]) << 8);
        result |= (val & 0xFFFF) << (r * 16);
      } else if (e_bits == 8) {
        result |= uint32_t((*line)[off]) << (r * 8);
      } else {
        // 4-bit (int4/uint4/nvfp4) not supported.
        std::cout << "Error: TCU 4-bit operand gather not supported" << std::endl;
        std::abort();
      }
    }
    return result;
  }

  // Routes A reads through the current block's A buffer; B reads through
  // the shared B buffer. `cur_block_` is set by tick().
  uint32_t load_lmem_word(const lmem_desc_t& desc, uint32_t row, uint32_t col,
                          uint32_t fmt_s, bool pack_along_row) const {
    auto& tbuf = simobject_->tbuf();
    if (desc.base == cur_a_desc_base_) {
      uint32_t b = cur_block_;
      return gather_word([&](uint64_t addr) { return tbuf->read_a(b, addr); },
                         desc, row, col, fmt_s, pack_along_row);
    }
    return gather_word([&](uint64_t addr) { return tbuf->read_b(addr); },
                       desc, row, col, fmt_s, pack_along_row);
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

  TcuUnit*   simobject_;
  Core*         core_;
  std::vector<std::vector<uint32_t>> sparse_meta_;
  std::vector<std::array<uint32_t, kMxMetaWords>> mx_meta_;
  std::unordered_map<uint32_t, lmem_desc_t[2]> lmem_desc_;
  mutable PerfStats perf_stats_;
  // Per-block "execute already happened for this trace" guard. Reset on input.pop().
  std::array<bool, NUM_TCU_BLOCKS> exec_done_;
  // Per-block "WGMMA lines have been planned into the buffers" flag.
  std::array<bool, NUM_TCU_BLOCKS> wgmma_planned_;
  // Per-block "currently between first and last WGMMA uop" flag.
  // Used to gate the shared B buffer invalidation so it's only dropped when
  // no other block is mid-WGMMA.
  std::array<bool, NUM_TCU_BLOCKS> in_wgmma_;
  // Set by tick() to the current block index before delegating to wgmma() —
  // the gather helpers route reads through TcuTbuf for this block.
  uint32_t cur_block_ = 0;
  // Set by wgmma() once sd_a is decoded; load_lmem_word uses this to
  // distinguish A reads (per-block A buffer) from B reads (shared B).
  uint64_t cur_a_desc_base_ = ~uint64_t(0);
  // xtileN dimension of the active WGMMA (depends on the macro's NRC).
  uint32_t cur_xtile_n_ = 8;
  // Diagnostic: which CTA's data currently occupies each block's A/M and the
  // shared B buffer. -1 = unowned. Logged on transition.
  std::array<int32_t, NUM_TCU_BLOCKS> cta_owner_a_{};
  int32_t cta_owner_b_ = -1;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t TcuUnit::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::WGMMA: {
    uint32_t nrc = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
    std::string src_mode = std::string(args.is_a_smem ? "S" : "R") + "S";
    return {"WGMMA." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(nrc) + "." + src_mode
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  }
  case TcuType::META_STORE:
    return {"META_STORE." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::to_string(args.fmt_d), ""};
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
    uint32_t nrc = (args.cd_nregs == 0) ? 8 : (args.cd_nregs == 1) ? 16 : 32;
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
      // Sparse metadata-store uops (one per group of cols_per_load columns).
      constexpr uint32_t meta_reg0 = 14, meta_reg1 = 15;
      uint32_t group = uop_index;
      uint32_t reg_rs1 = (group > 0) ? meta_reg1 : meta_reg0;
      uop_instr->set_op_type(TcuType::META_STORE);
      uop_instr->set_args(IntrTcuArgs{false, 0, 0, fmt_s, group, 0, 0, 0, TCU_META_KIND_SPARSE});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else if (uop_index < total_meta_stores) {
      // MX scale metadata-store uops.
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
      uop_instr->set_args(IntrTcuArgs{0, 0, 0, fmt_s, mx_store, 0, 0, 0, TCU_META_KIND_MX});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else {
      // MMA uops.
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
        uop_instr->set_args(IntrTcuArgs{true, 0, 0, fmt_s, fmt_d, m_sp, actual_n, 0, 0});
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
        uop_instr->set_args(IntrTcuArgs{is_sparse, 0, 0, fmt_s, fmt_d, m, n, k, 0});
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
      uop_instr->set_args(IntrTcuArgs{false, 0, 0, fmt_s, group, 0, 0, 0, TCU_META_KIND_SPARSE_WG});
      uop_instr->set_src_reg(0, reg_rs1, RegType::Float);
    } else {
      // MMA phase
      uint32_t mma_idx = uop_index - meta_stores;
      uint32_t ra_base = is_a_smem ? 10 : 24;

      // Loop order: m (inner) -> n (middle) -> k (outer). K-outer maximizes
      // per-block A-buffer reuse: each A_w[m,k] is consumed across the entire
      // (n,m) inner sweep, and each shared B[k,n] is consumed for m_steps
      // consecutive uops.
      uint32_t mn = (total - meta_stores) / k_count;
      uint32_t k = mma_idx / mn;
      uint32_t rem = mma_idx % mn;
      uint32_t n = rem / m_steps;
      uint32_t m = rem % m_steps;
      uint32_t r = n * m_steps + m;

      uop_instr->set_op_type(TcuType::WGMMA);
      uop_instr->set_args(IntrTcuArgs{is_sparse, is_a_smem ? 1u : 0u, cd_nregs,
                                     fmt_s, fmt_d, m, n, k, 0});
      uop_instr->set_dest_reg(r, RegType::Float);
      if (mma_idx == 0) {
        if (is_a_smem) {
          uop_instr->set_src_reg(0, a0, RegType::Integer);
        } else {
          uint32_t rs1_off = m * k_count + k;
          uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
        }
        uop_instr->set_src_reg(1, a1, RegType::Integer);
      } else if (!is_a_smem) {
        uint32_t rs1_off = m * k_count + k;
        uop_instr->set_src_reg(0, ra_base + rs1_off, RegType::Float);
      }
      uop_instr->set_src_reg(2, r, RegType::Float);
    }
    // fu_lock on first uop, fu_unlock on last uop
    uop_instr->set_fu_lock(uop_index == 0);
    uop_instr->set_fu_unlock(uop_index == (total - 1));
  }
#endif

  return uop_instr;
}

///////////////////////////////////////////////////////////////////////////////

TcuUnit::TcuUnit(const SimContext &ctx, const char* name, Core* core)
	: FuncUnit(ctx, name, core)
	, impl_(new Impl(this, core))
{
  char sname[128];
  snprintf(sname, sizeof(sname), "%s-tbuf", name);
  tbuf_ = TcuTbuf::Create(sname);
}

TcuUnit::~TcuUnit() {
  delete impl_;
}

TcuTbuf::Ptr& TcuUnit::tbuf() {
  return tbuf_;
}

void TcuUnit::on_reset() {
  impl_->reset();
}

void TcuUnit::on_tick() {
  impl_->tick();
}

const TcuUnit::PerfStats &TcuUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TcuUnit::wmma(uint32_t wid,
                      uint32_t fmt_s,
                      uint32_t fmt_d,
                      uint32_t step_m,
                      uint32_t step_n,
                      uint32_t step_k,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      bool is_sparse) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, step_k,
              rs1_data, rs2_data, rs3_data,
              rd_data,
              is_sparse);
}

void TcuUnit::wgmma(uint32_t wid,
                       uint32_t fmt_s,
                       uint32_t fmt_d,
                       uint32_t step_m,
                       uint32_t step_n,
                       uint32_t step_k,
                       uint32_t a_desc,
                       uint32_t b_desc,
                       const std::vector<reg_data_t>& rs1_data,
                       const std::vector<reg_data_t>& rs3_data,
                       std::vector<reg_data_t>& rd_data,
                       bool is_sparse,
                       uint32_t cd_nregs,
                       uint32_t is_a_smem) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, a_desc, b_desc,
               rs1_data, rs3_data, rd_data,
               is_sparse, cd_nregs, is_a_smem);
}

void TcuUnit::meta_store(uint32_t wid,
                            uint32_t fmt_s,
                            uint32_t col_idx,
                            uint32_t meta_kind,
                            const std::vector<reg_data_t>& rs1_data) {
  impl_->meta_store(wid, fmt_s, col_idx, meta_kind, rs1_data);
}

