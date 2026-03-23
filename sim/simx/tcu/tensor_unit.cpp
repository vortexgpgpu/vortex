
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
#include <array>
#include <cmath>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg   = vt::wmma_config_t<NUM_THREADS>;
using wg_cfg = vt::wmma_config_t<NUM_THREADS, vt::fp32, vt::fp32, 4, 32>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

template <typename It, typename Ot>
struct FMA {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static otype eval(itype a, itype b, otype c) {
    return static_cast<otype>(a) * static_cast<otype>(b) + c;
  }
};

template <>
struct FMA<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_htof_s(a, 0, nullptr);
    auto xb = rv_htof_s(b, 0, nullptr);
    auto xc = rv_htof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoh_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto xa = rv_btof_s(a, 0, nullptr);
    auto xb = rv_btof_s(b, 0, nullptr);
    auto xc = rv_btof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftob_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::fp8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    auto xa = rv_e4m3tof_s(a, 0, nullptr);
    auto xb = rv_e4m3tof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::fp8, vt::fp8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    auto xa = rv_e4m3tof_s(a, 0, nullptr);
    auto xb = rv_e4m3tof_s(b, 0, nullptr);
    auto xc = rv_e4m3tof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoe4m3_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::bf8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    auto xa = rv_e5m2tof_s(a, 0, nullptr);
    auto xb = rv_e5m2tof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::bf8, vt::bf8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    auto xa = rv_e5m2tof_s(a, 0, nullptr);
    auto xb = rv_e5m2tof_s(b, 0, nullptr);
    auto xc = rv_e5m2tof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoe5m2_s(xd, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::tf32, vt::fp32> {
  static float eval(uint32_t a, uint32_t b, float c) {
    auto xa = rv_tf32tof_s(a, 0, nullptr);
    auto xb = rv_tf32tof_s(b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::tf32, vt::tf32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto xa = rv_tf32tof_s(a, 0, nullptr);
    auto xb = rv_tf32tof_s(b, 0, nullptr);
    auto xc = rv_tf32tof_s(c, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftotf32_s(xd, 0, nullptr);
    return xh;
  }
};

// TODO: temp arbitrarily hardcoded scale factors
constexpr uint8_t SCALE_FACTOR_E8M0_A = 129;  // val = 4, bias = 127
constexpr uint8_t SCALE_FACTOR_E8M0_B = 131;  // val = 16
constexpr uint8_t SCALE_FACTOR_E4M3_A = 0x41; // val = 2.25, bias = 7
constexpr uint8_t SCALE_FACTOR_E4M3_B = 0x33; // val = 0.6875

template <>
struct FMA<vt::mxfp8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E8M0_A; //TODO: input as parameter
    constexpr uint8_t sf_b = SCALE_FACTOR_E8M0_B;
    auto xa = rv_mxfp8tof_s(a, sf_a, 0, nullptr);
    auto xb = rv_mxfp8tof_s(b, sf_b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::mxfp8, vt::mxfp8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    constexpr uint8_t sf = SCALE_FACTOR_E8M0_A;
    auto xa = rv_mxfp8tof_s(a, sf, 0, nullptr);
    auto xb = rv_mxfp8tof_s(b, sf, 0, nullptr);
    auto xc = rv_mxfp8tof_s(c, sf, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftomxfp8_s(xd, sf, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::nvfp4, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E4M3_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E4M3_B;
    auto xa = rv_nvfp4tof_s(a, sf_a, 0, nullptr);
    auto xb = rv_nvfp4tof_s(b, sf_b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};

template <>
struct FMA<vt::nvfp4, vt::nvfp4> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    constexpr uint8_t sf = SCALE_FACTOR_E4M3_A;
    auto xa = rv_nvfp4tof_s(a, sf, 0, nullptr);
    auto xb = rv_nvfp4tof_s(b, sf, 0, nullptr);
    auto xc = rv_nvfp4tof_s(c, sf, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftonvfp4_s(xd, sf, 0, nullptr);
    return xh;
  }
};

template <>
struct FMA<vt::mxint8, vt::int32> {
  static int32_t eval(int8_t a, int8_t b, int32_t c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E8M0_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E8M0_B;
    //combined scale: block scale (2^(sf-127)) * implicit scale (2^(-6)) = 2^(sf-133)
    int32_t scale_exp_a = (int32_t)sf_a - 133;
    float scale_factor_a = std::ldexp(1.0f, scale_exp_a);
    int32_t scale_exp_b = (int32_t)sf_b - 133;
    float scale_factor_b = std::ldexp(1.0f, scale_exp_b);
    float product = (float)a * scale_factor_a * (float)b * scale_factor_b;
    return (int32_t)product + c;
  }
};

template <>
struct FMA<vt::mxint8, vt::mxint8> {
  static int8_t eval(int8_t a, int8_t b, int8_t c) {
    constexpr uint8_t sf = SCALE_FACTOR_E8M0_A;
    int32_t scale_exp = (int32_t)sf - 133;
    float scale_factor = std::ldexp(1.0f, scale_exp);
    float product = (float)a * (float)b * scale_factor * scale_factor;
    float result = product + (float)c;
    //clamp to int8 range [-127, 127]
    int32_t result_int = (int32_t)result;
    if (result_int > 127) result_int = 127;
    if (result_int < -127) result_int = -127;
    return (int8_t)result_int;
  }
};

template <typename It, typename Ot>
struct FEDP {
  using itype = typename It::dtype;
  using otype = typename Ot::dtype;
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
  static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
  auto acc = bit_cast<otype>(c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const itype *>(&a_row[z].u32);
    auto b = reinterpret_cast<const itype *>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<It, Ot>::eval(a[i], b[i], acc);
    }
  }
  return bit_cast<uint32_t>(acc);
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

template <>
struct FEDP<vt::nvfp4, vt::fp32>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E4M3_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E4M3_B;
    auto acc = bit_cast<float>(c_val);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        uint8_t a_val = (a >> (i * 4)) & 0xF;
        uint8_t b_val = (b >> (i * 4)) & 0xF;
        auto xa = rv_nvfp4tof_s(a_val, sf_a, 0, nullptr);
        auto xb = rv_nvfp4tof_s(b_val, sf_b, 0, nullptr);
        auto xab = rv_fmul_s(xa, xb, 0, nullptr);
        auto xc = bit_cast<uint32_t>(acc);
        auto xd = rv_fadd_s(xab, xc, 0, nullptr);
        acc = bit_cast<float>(xd);
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

template <>
struct FEDP<vt::nvfp4, vt::nvfp4>{
  static uint32_t eval(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val) {
    constexpr uint8_t sf = SCALE_FACTOR_E4M3_A;
    auto acc = bit_cast<uint8_t>(c_val & 0xFF);
    for (uint32_t z = 0; z < cfg::tcK; ++z) {
      auto a = a_row[z].u32;
      auto b = b_col[z].u32;
      for (uint32_t i = 0; i < 8; ++i) { // 8 * 4 bits = 32 bits
        uint8_t a_val = (a >> (i * 4)) & 0xF;
        uint8_t b_val = (b >> (i * 4)) & 0xF;
        auto xa = rv_nvfp4tof_s(a_val, sf, 0, nullptr);
        auto xb = rv_nvfp4tof_s(b_val, sf, 0, nullptr);
        auto xc = rv_nvfp4tof_s(acc, sf, 0, nullptr);
        auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
        acc = rv_ftonvfp4_s(xd, sf, 0, nullptr);
      }
    }
    return bit_cast<uint32_t>(acc);
  }
};

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
    case vt::mxfp8::id:
      return FEDP<vt::mxfp8, vt::fp32>::eval;
    case vt::nvfp4::id:
      return FEDP<vt::nvfp4, vt::fp32>::eval;
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
  case vt::mxfp8::id:
    switch (IT) {
    case vt::mxfp8::id:
      return FEDP<vt::mxfp8, vt::mxfp8>::eval;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::nvfp4::id:
    switch (IT) {
    case vt::nvfp4::id:
      return FEDP<vt::nvfp4, vt::nvfp4>::eval;
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
    case vt::mxint8::id:
      return FEDP<vt::mxint8, vt::int32>::eval;
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
// elements, collect those flagged by lo_mask/hi_mask respectively, and pack them
// sequentially into the output word. Works for elem_bits = 4, 8, or 16.
static inline uint32_t gather_sparse(uint32_t bword0, uint32_t bword1,
                                     uint32_t lo_mask, uint32_t hi_mask,
                                     uint32_t elem_bits) {
  uint32_t elem_count = 32 / elem_bits;
  uint32_t elem_mask  = (elem_bits < 32) ? ((1u << elem_bits) - 1u) : ~0u;
  assert(__builtin_popcount(lo_mask) + __builtin_popcount(hi_mask) == elem_count &&
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

class TensorUnit::Impl {
public:
  struct SmemTileMeta {
    uint64_t base = 0;
    uint32_t ldm = 0;
    bool col_major = false;
  };

  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
    , sparse_meta_(arch.num_warps(), std::vector<uint32_t>(kMetaBanks * kMaxMetaCols, 0))
  {
    //--
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
    for (auto& meta : sparse_meta_) {
      for (uint32_t bank = 0; bank < kMetaBanks; ++bank) {
        for (uint32_t col = 0; col < kMaxMetaCols; ++col) {
          meta[bank * kMaxMetaCols + col] = (bank & 1) ? 0xaaaaaaaau : 0x55555555u;
        }
      }
    }
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
        continue;
      auto trace = input.peek();
      auto tcu_type = std::get<TcuType>(trace->op_type);
      int delay = 0;
      switch (tcu_type) {
      case TcuType::WMMA:
      case TcuType::WGMMA:
      case TcuType::WMMA_SP:
        delay = 4;
        break;
      case TcuType::META_STORE:
        delay = 1;
        break;
      default:
        std::abort();
      }
      if (simobject_->Outputs.at(iw).try_send(trace, 2 + delay)) {
        DT(3, simobject_->name() << " execute: op=" << tcu_type << ", " << *trace);
        input.pop();
      }
    }
  }

  void meta_store(uint32_t wid,
                  uint32_t fmt_s,
                  uint32_t col_idx,
                  const std::vector<reg_data_t>& rs1_data,
                  ExeTraceData* trace_data) {
    __unused(trace_data);

    uint32_t num_cols = meta_num_cols(fmt_s);
    uint32_t total_stores = vt::sparse_meta_total_store_uops(fmt_s, cfg::stores_per_col, NUM_THREADS);
    if (num_cols == 0 || col_idx >= total_stores) {
      std::cout << "Error: META_STORE store index out of range: " << col_idx << std::endl;
      std::abort();
    }

    uint32_t actual_col = col_idx / cfg::stores_per_col;
    uint32_t sub_store = col_idx % cfg::stores_per_col;
    uint32_t bank_begin = sub_store * cfg::banks_per_store;
    uint32_t bank_end = std::min(bank_begin + cfg::banks_per_store, kMetaBanks);
    uint32_t thread_offset = (cfg::meta_cols_per_load > 1)
                           ? ((actual_col % cfg::meta_cols_per_load) * kMetaBanks)
                           : 0;

    for (uint32_t bank = bank_begin; bank < bank_end; ++bank) {
      uint32_t src_idx = (cfg::stores_per_col > 1)
                       ? (bank - bank_begin)
                       : (thread_offset + bank);
      sparse_meta_.at(wid).at(bank * kMaxMetaCols + actual_col) =
          rs1_data.at(src_idx).u32;
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
    __unused(trace_data);

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

    auto fedp  = select_FEDP(fmt_s, fmt_d);
    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = is_sparse
                   ? (step_n % cfg::b_sub_blocks_sp) * cfg::b_block_size_sp
                   : (step_n % cfg::b_sub_blocks)    * cfg::b_block_size;

    // Sparse-only: metadata bank index and per-element bit width
    constexpr uint32_t kCompression = 2;
    uint32_t bank      = step_m * (cfg::k_steps / 2) + step_k;
    uint32_t ebits     = is_sparse ? elem_bits(fmt_s) : 0;
    uint32_t meta_bits = is_sparse ? (32 / ebits) : 0;

    auto meta_bit = [&](uint32_t bit_idx) {
      return (sparse_meta_.at(wid).at(bank * kMaxMetaCols + bit_idx / 32) >> (bit_idx % 32)) & 1u;
    };

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;

        reg_data_t b_buf[cfg::tcK];
        const reg_data_t* b_col;
        if (is_sparse) {
          uint32_t row_base = i * meta_row_width(ebits);
          uint32_t j_sp = cfg::sym_sparse ? (j % (cfg::tcN / 2)) : j;
          for (uint32_t z = 0; z < cfg::tcK; ++z) {
            uint32_t b_idx = b_off + j_sp * cfg::tcK * kCompression + z * kCompression;
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < meta_bits; ++b) {
              lo |= meta_bit(row_base + meta_bits * z           + b) << b;
              hi |= meta_bit(row_base + meta_bits * (cfg::tcK + z) + b) << b;
            }
            b_buf[z].u32 = gather_sparse(rs2_data.at(b_idx).u32, rs2_data.at(b_idx + 1).u32, lo, hi, ebits);
          }
          b_col = b_buf;
        } else {
          b_col = rs2_data.data() + b_off + j * cfg::tcK;
        }

        auto d_val = fedp(a_row, b_col, c_val);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);

        DTH(3, simobject_->name() << (is_sparse ? " SP_FEDP" : " FEDP")
            << ": wid=" << wid << ", i=" << i << ", j=" << j
            << ", m=" << step_m << ", n=" << step_n << std::hex
            << ", a0=0x" << a_row[0].u32 << ", b0=0x" << b_col[0].u32
            << ", c=0x" << c_val << ", d=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  void wgmma(uint32_t wid,
             uint32_t fmt_s,
             uint32_t fmt_d,
             uint32_t step_m,
             uint32_t step_n,
             uint32_t a_desc,
             uint32_t b_desc,
             const std::vector<reg_data_t>& rs1_data,
             std::vector<reg_data_t>& rd_data,
             ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    auto fedp = select_FEDP(fmt_s, fmt_d);
    uint32_t ratio = elem_ratio(fmt_s);
    uint32_t k_words = wg_cfg::tcK;

    // Decode packed descriptors: bits[31:16] = ldm_bytes, bits[15:0] = smem offset
    uint32_t e_bytes = elem_bits(fmt_s) / 8;
    SmemTileMeta meta_a = {LMEM_BASE_ADDR + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
    SmemTileMeta meta_b = {LMEM_BASE_ADDR + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};

    // Seed accumulator from input fragC (rs1_data)
    rd_data = rs1_data;

    // Accumulate over all k_steps internally
    for (uint32_t k = 0; k < wg_cfg::k_steps; ++k) {
      for (uint32_t i = 0; i < wg_cfg::tcM; ++i) {
        for (uint32_t j = 0; j < wg_cfg::tcN; ++j) {
          reg_data_t a_row[wg_cfg::tcK];
          reg_data_t b_col[wg_cfg::tcK];
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t k_elem = (k * k_words + z) * ratio;
            uint32_t a_row_idx = step_m * wg_cfg::tcM + i;
            uint32_t b_col_idx = step_n * wg_cfg::tcN + j;
            a_row[z].u32 = load_smem_word(meta_a, a_row_idx, k_elem, fmt_s, false);
            b_col[z].u32 = load_smem_word(meta_b, k_elem, b_col_idx, fmt_s, true);
          }
          auto t = i * wg_cfg::tcN + j;
          auto d_val = fedp(a_row, b_col, rd_data.at(t).u32);
          rd_data.at(t).u64 = nan_box(d_val);

          DTH(3, simobject_->name() << " WGMMA FEDP: wid=" << wid << ", i=" << i << ", j=" << j
              << ", m=" << step_m << ", n=" << step_n << ", k=" << k << std::hex
              << ", a_row[0]=0x" << a_row[0].u32 << ", b_col[0]=0x" << b_col[0].u32
              << ", d=0x" << d_val << std::dec << std::endl);
        }
      }
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
    case vt::int8::id:
    case vt::uint8::id:
    case vt::mxint8::id:
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

  uint32_t load_smem_word(const SmemTileMeta& meta, uint32_t row, uint32_t col, uint32_t fmt_s, bool pack_along_row) const {
    auto bits = elem_bits(fmt_s);
    if (bits == 4) {
      // Direct-SMEM WGMMA path currently models byte/half/word formats.
      std::abort();
    }
    uint32_t elem_bytes = bits / 8;
    uint32_t elems_per_word = 4 / elem_bytes;
    uint32_t mask = (elem_bytes == 4) ? 0xffffffffu : ((1u << (8 * elem_bytes)) - 1u);

    uint32_t packed = 0;
    for (uint32_t e = 0; e < elems_per_word; ++e) {
      uint32_t rr = row + (pack_along_row ? e : 0);
      uint32_t cc = col + (pack_along_row ? 0 : e);
      if (meta.col_major) {
        std::swap(rr, cc);
      }
      uint64_t addr = meta.base + uint64_t(rr * meta.ldm + cc) * elem_bytes;
      uint32_t word = 0;
      core_->mem_read(&word, addr, elem_bytes);
      packed |= (word & mask) << (8 * elem_bytes * e);
    }
    return packed;
  }

  static constexpr uint32_t kSparseKSteps = cfg::k_steps / 2;
  static constexpr uint32_t kMetaBanks = cfg::m_steps * kSparseKSteps;
  static constexpr uint32_t kMaxMetaCols = NUM_THREADS / 2;

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
  std::vector<std::vector<uint32_t>> sparse_meta_;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::WGMMA:
    return {"WGMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  case TcuType::WMMA_SP:
    return {"WMMA_SP." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::META_STORE:
    return {"META_STORE." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::to_string(args.fmt_d), ""};
  default:
    std::abort();
  }
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
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, rs1_data, rs2_data, rs3_data, rd_data, trace_data, is_sparse);
}

void TensorUnit::wgmma(uint32_t wid,
                       uint32_t fmt_s,
                       uint32_t fmt_d,
                       uint32_t step_m,
                       uint32_t step_n,
                       uint32_t a_desc,
                       uint32_t b_desc,
                       const std::vector<reg_data_t>& rs1_data,
                       std::vector<reg_data_t>& rd_data,
                       ExeTraceData* trace_data) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, a_desc, b_desc, rs1_data, rd_data, trace_data);
}

void TensorUnit::meta_store(uint32_t wid,
                            uint32_t fmt_s,
                            uint32_t col_idx,
                            const std::vector<reg_data_t>& rs1_data,
                            ExeTraceData* trace_data) {
  impl_->meta_store(wid, fmt_s, col_idx, rs1_data, trace_data);
}
