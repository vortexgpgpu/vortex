
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
using wg_cfg = vt::wmma_config_t<NUM_THREADS, vt::fp32, vt::fp32, 32, 8>;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

static inline uint8_t unpack_u8(uint32_t word, uint32_t idx) {
  return (word >> (idx * 8)) & 0xffu;
}

#ifdef TCU_MX_ENABLE
static inline uint8_t unpack_mx_scale(uint32_t w0,
                                      uint32_t w1,
                                      uint32_t idx,
                                      const char* axis_name) {
  // dense-MX ABI carries two packed words per axis.
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
#endif

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

#ifdef TCU_MX_ENABLE
template <>
struct FMA<vt::mxfp8, vt::fp32> {
  static float eval(uint8_t, uint8_t, float) = delete;

  static float eval(uint8_t a, uint8_t b, float c, uint8_t sf_a, uint8_t sf_b) {
    auto xa = rv_mxfp8tof_s(a, sf_a, 0, nullptr);
    auto xb = rv_mxfp8tof_s(b, sf_b, 0, nullptr);
    auto xab= rv_fmul_s(xa, xb, 0, nullptr);
    auto xc = bit_cast<uint32_t>(c);
    auto xd = rv_fadd_s(xab, xc, 0, nullptr);
    return bit_cast<float>(xd);
  }
};
#endif

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

#ifdef TCU_MX_ENABLE
static uint32_t fedp_mxfp8_fp32_scaled(const reg_data_t* a_row,
                                       const reg_data_t* b_col,
                                       uint32_t c_val,
                                       uint8_t sf_a,
                                       uint8_t sf_b) {
  constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(uint8_t);
  auto acc = bit_cast<float>(c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const uint8_t*>(&a_row[z].u32);
    auto b = reinterpret_cast<const uint8_t*>(&b_col[z].u32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<vt::mxfp8, vt::fp32>::eval(a[i], b[i], acc, sf_a, sf_b);
    }
  }
  return bit_cast<uint32_t>(acc);
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

  struct lmem_desc_t {
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
            const std::vector<reg_data_t>& mx_a0_data,
            const std::vector<reg_data_t>& mx_a1_data,
            const std::vector<reg_data_t>& mx_b0_data,
            const std::vector<reg_data_t>& mx_b1_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data,
            bool is_sparse) {
    __unused(trace_data);
#ifndef TCU_MX_ENABLE
    __unused(mx_a0_data);
    __unused(mx_a1_data);
    __unused(mx_b0_data);
    __unused(mx_b1_data);
#endif

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

  #ifdef TCU_MX_ENABLE
    bool use_dynamic_scale_fedp = (!is_sparse)
                   && (fmt_s == vt::mxfp8::id)
                   && (fmt_d == vt::fp32::id);
  #else
    bool use_dynamic_scale_fedp = false;
  #endif
    PFN_FEDP fedp = nullptr;
    if (!use_dynamic_scale_fedp) {
      fedp = select_FEDP(fmt_s, fmt_d);
    }
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
              lo |= meta_bit(row_base + meta_bits * z + b) << b;
              hi |= meta_bit(row_base + meta_bits * (cfg::tcK + z) + b) << b;
            }
            b_buf[z].u32 = gather_sparse(rs2_data.at(b_idx).u32, rs2_data.at(b_idx + 1).u32, lo, hi, ebits);
          }
          b_col = b_buf;
        } else {
          b_col = rs2_data.data() + b_off + j * cfg::tcK;
        }

        uint32_t d_val;
        if (use_dynamic_scale_fedp) {
#ifdef TCU_MX_ENABLE
          constexpr uint32_t meta_lane = 0;
          if (mx_a0_data.size() <= meta_lane || mx_a1_data.size() <= meta_lane ||
              mx_b0_data.size() <= meta_lane || mx_b1_data.size() <= meta_lane) {
            std::cout << "Error: missing MX metadata registers for dynamic-scale WMMA" << std::endl;
            std::abort();
          }
          uint32_t row_idx = step_m * cfg::tcM + i;
          uint32_t col_idx = step_n * cfg::tcN + j;
          uint8_t sf_a = unpack_mx_scale(mx_a0_data.at(meta_lane).u32,
                                         mx_a1_data.at(meta_lane).u32,
                                         row_idx,
                                         "A");
          uint8_t sf_b = unpack_mx_scale(mx_b0_data.at(meta_lane).u32,
                                         mx_b1_data.at(meta_lane).u32,
                                         col_idx,
                                         "B");
          d_val = fedp_mxfp8_fp32_scaled(a_row, b_col, c_val, sf_a, sf_b);
#else
          std::abort();
#endif
        } else {
          d_val = fedp(a_row, b_col, c_val);
        }
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
             uint32_t step_k,
             uint32_t a_desc,
             uint32_t b_desc,
             const std::vector<reg_data_t>& rs1_data,
             std::vector<reg_data_t>& rd_data,
             ExeTraceData* trace_data,
             bool is_sparse) {
    __unused(trace_data);

    auto fedp    = select_FEDP(fmt_s, fmt_d);
    uint32_t ratio   = elem_ratio(fmt_s);
    uint32_t k_words = wg_cfg::tcK;
    uint32_t e_bytes = elem_bits(fmt_s) / 8;

    lmem_desc_t sd_a, sd_b;
    if (step_k == 0 && step_m == 0 && step_n == 0) {
      // Decode packed descriptors only on the very first uop of the WGMMA instruction.
      // (a_desc / b_desc are valid register values only for this uop per decode.cpp.)
      sd_a = {LMEM_BASE_ADDR + (a_desc & 0xFFFF), (a_desc >> 16) / e_bytes, false};
      sd_b = {LMEM_BASE_ADDR + (b_desc & 0xFFFF), (b_desc >> 16) / e_bytes, false};
      // Cache decoded descriptors for all subsequent uops of this WGMMA instruction.
      lmem_desc_[wid][0] = sd_a;
      lmem_desc_[wid][1] = sd_b;
    } else {
      sd_a = lmem_desc_[wid][0];
      sd_b = lmem_desc_[wid][1];
    }

    if (is_sparse) {
      if (!vt::sparse_format_supported(fmt_s)) {
        std::cout << "Error: WGMMA_SP unsupported input format: "
                  << vt::fmt_string(fmt_s) << " (id=" << fmt_s << ")" << std::endl;
        std::abort();
      }

      uint32_t ebits          = elem_bits(fmt_s);
      uint32_t rtl_i_ratio    = 32 / ebits;
      uint32_t meta_row_w     = k_words * 2 * rtl_i_ratio; // bits per tcM row
      uint32_t meta_strd_words = (wg_cfg::tcM * meta_row_w + 31) / 32; // words per bank
      // Metadata is stored in lmem immediately after the compressed A tile
      uint64_t meta_base = sd_a.base + uint64_t(wg_cfg::xtileM) * sd_a.ldm * e_bytes;
      uint32_t wg_bank   = step_m * (wg_cfg::k_steps / 2) + step_k;

      auto meta_bit_wg = [&](uint32_t bit_idx) -> uint32_t {
        uint32_t word_val = 0;
        core_->mem_read(&word_val,
          meta_base + uint64_t(wg_bank * meta_strd_words + bit_idx / 32) * 4, 4);
        return (word_val >> (bit_idx % 32)) & 1u;
      };

      for (uint32_t i = 0; i < wg_cfg::tcM; ++i) {
        for (uint32_t j = 0; j < wg_cfg::tcN; ++j) {
          reg_data_t a_row[wg_cfg::tcK];
          reg_data_t b_col[wg_cfg::tcK];
          uint32_t a_row_idx = step_m * wg_cfg::tcM + i;
          uint32_t b_col_idx = step_n * wg_cfg::tcN + j;
          uint32_t row_base  = i * meta_row_w;

          for (uint32_t z = 0; z < k_words; ++z) {
            // Load compressed A word from lmem
            uint32_t k_elem_a = (step_k * k_words + z) * ratio;
            a_row[z].u32 = load_lmem_word(sd_a, a_row_idx, k_elem_a, fmt_s, false);

            // Read lo/hi metadata masks for this row and compressed-K word
            uint32_t lo = 0, hi = 0;
            for (uint32_t b = 0; b < rtl_i_ratio; ++b) {
              lo |= meta_bit_wg(row_base + rtl_i_ratio * z              + b) << b;
              hi |= meta_bit_wg(row_base + rtl_i_ratio * (k_words + z) + b) << b;
            }

            // Load 2 consecutive dense B words and sparse-gather the selected elements
            constexpr uint32_t kCompression = 2;
            uint32_t k_elem_b0 = (step_k * k_words + z) * ratio * kCompression;
            uint32_t bword0 = load_lmem_word(sd_b, k_elem_b0,         b_col_idx, fmt_s, true);
            uint32_t bword1 = load_lmem_word(sd_b, k_elem_b0 + ratio, b_col_idx, fmt_s, true);
            b_col[z].u32 = gather_sparse(bword0, bword1, lo, hi, ebits);
          }

          auto t = i * wg_cfg::tcN + j;
          auto d_val = fedp(a_row, b_col, rs1_data.at(t).u32);
          rd_data.at(t).u64 = nan_box(d_val);

          DTH(3, simobject_->name() << " WGMMA_SP FEDP: wid=" << wid
              << ", i=" << i << ", j=" << j
              << ", m=" << step_m << ", n=" << step_n << ", k=" << step_k << std::hex
              << ", a_row[0]=0x" << a_row[0].u32 << ", b_col[0]=0x" << b_col[0].u32
              << ", c=0x" << rs1_data.at(t).u32 << ", d=0x" << d_val << std::dec << std::endl);
        }
      }
    } else {
      // Dense WGMMA path: RF (rs1_data) is the accumulator, updated each step_k uop.
      for (uint32_t i = 0; i < wg_cfg::tcM; ++i) {
        for (uint32_t j = 0; j < wg_cfg::tcN; ++j) {
          reg_data_t a_row[wg_cfg::tcK];
          reg_data_t b_col[wg_cfg::tcK];
          for (uint32_t z = 0; z < k_words; ++z) {
            uint32_t k_elem    = (step_k * k_words + z) * ratio;
            uint32_t a_row_idx = step_m * wg_cfg::tcM + i;
            uint32_t b_col_idx = step_n * wg_cfg::tcN + j;
            a_row[z].u32 = load_lmem_word(sd_a, a_row_idx, k_elem, fmt_s, false);
            b_col[z].u32 = load_lmem_word(sd_b, k_elem, b_col_idx, fmt_s, true);
          }
          auto t = i * wg_cfg::tcN + j;
          auto d_val = fedp(a_row, b_col, rs1_data.at(t).u32);
          rd_data.at(t).u64 = nan_box(d_val);

          DTH(3, simobject_->name() << " WGMMA FEDP: wid=" << wid
              << ", i=" << i << ", j=" << j
              << ", m=" << step_m << ", n=" << step_n << ", k=" << step_k << std::hex
              << ", a_row[0]=0x" << a_row[0].u32 << ", b_col[0]=0x" << b_col[0].u32
              << ", c=0x" << rs1_data.at(t).u32 << ", d=0x" << d_val << std::dec << std::endl);
        }
      }
    }

    // Performance counters.
    // wgmma_instrs: one per µop (each wgmma() call is one µop).
    ++perf_stats_.wgmma_instrs;
    // lmem_reads: in RTL each tile fetch issues (A_BANK_ROWS + B_BANK_ROWS) LMEM
    // transactions, one per warp per outer-loop iteration (first µop only).
    // Approximate in simx: count reads on first µop of each tile.
    if (step_m == 0 && step_n == 0 && step_k == 0) {
      constexpr uint32_t a_bank_rows = (wg_cfg::xtileM * wg_cfg::xtileK + LMEM_NUM_BANKS - 1) / LMEM_NUM_BANKS;
      constexpr uint32_t b_bank_rows = (wg_cfg::xtileK * wg_cfg::xtileN + LMEM_NUM_BANKS - 1) / LMEM_NUM_BANKS;
      perf_stats_.lmem_reads += a_bank_rows + b_bank_rows;
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
      return 8;
    case vt::int4::id:
    case vt::uint4::id:
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

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  std::vector<std::vector<uint32_t>> sparse_meta_;
  std::unordered_map<uint32_t, lmem_desc_t[2]> lmem_desc_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(args.is_sparse ? "SP." : "") + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n)
             + "." + std::to_string(args.step_k), ""};
  case TcuType::WGMMA:
    return {"WGMMA." + std::string(args.is_sparse ? "SP." : "") +  std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
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
                      const std::vector<reg_data_t>& mx_a0_data,
                      const std::vector<reg_data_t>& mx_a1_data,
                      const std::vector<reg_data_t>& mx_b0_data,
                      const std::vector<reg_data_t>& mx_b1_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data,
                      bool is_sparse) {
  impl_->wmma(wid,
              fmt_s,
              fmt_d,
              step_m,
              step_n,
              step_k,
              rs1_data,
              rs2_data,
              rs3_data,
              mx_a0_data,
              mx_a1_data,
              mx_b0_data,
              mx_b1_data,
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
                       const std::vector<reg_data_t>& rs3_data,
                       std::vector<reg_data_t>& rd_data,
                       ExeTraceData* trace_data,
                       bool is_sparse) {
  impl_->wgmma(wid, fmt_s, fmt_d, step_m, step_n, step_k, a_desc, b_desc, rs3_data, rd_data, trace_data, is_sparse);
}

void TensorUnit::meta_store(uint32_t wid,
                            uint32_t fmt_s,
                            uint32_t col_idx,
                            const std::vector<reg_data_t>& rs1_data,
                            ExeTraceData* trace_data) {
  impl_->meta_store(wid, fmt_s, col_idx, rs1_data, trace_data);
}
