
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
#include <cmath>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

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

static inline void gather_B8(uint8_t mask0,
                             uint8_t mask1,
                             uint32_t bword0,
                             uint32_t bword1,
                             uint32_t& b_gathered) {
  assert(__builtin_popcount(mask0 & 0x0f) == 2 && "mask0 must select exactly 2 of 4");
  assert(__builtin_popcount(mask1 & 0x0f) == 2 && "mask1 must select exactly 2 of 4");

  uint8_t out[4];
  uint32_t out_idx = 0;

  for (uint32_t i = 0; i < 4; ++i) {
    if (mask0 & (1u << i)) {
      out[out_idx++] = (bword0 >> (i * 8)) & 0xff;
    }
  }
  for (uint32_t i = 0; i < 4; ++i) {
    if (mask1 & (1u << i)) {
      out[out_idx++] = (bword1 >> (i * 8)) & 0xff;
    }
  }

  assert(out_idx == 4 && "gather_B8 must output exactly 4 elements");
  b_gathered = (uint32_t(out[0]) << 0)
             | (uint32_t(out[1]) << 8)
             | (uint32_t(out[2]) << 16)
             | (uint32_t(out[3]) << 24);
}

static inline void gather_B16(uint8_t mask,
                              uint32_t bword0,
                              uint32_t bword1,
                              uint32_t& b_gathered) {
  assert(__builtin_popcount(mask & 0x0f) == 2 && "mask must select exactly 2 of 4");

  uint16_t in[4] = {
      static_cast<uint16_t>((bword0 >> 0) & 0xffff),
      static_cast<uint16_t>((bword0 >> 16) & 0xffff),
      static_cast<uint16_t>((bword1 >> 0) & 0xffff),
      static_cast<uint16_t>((bword1 >> 16) & 0xffff),
  };

  uint16_t out[2];
  uint32_t out_idx = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    if (mask & (1u << i)) {
      out[out_idx++] = in[i];
    }
  }
  assert(out_idx == 2 && "gather_B16 must output exactly 2 elements");
  b_gathered = (uint32_t(out[0]) << 0) | (uint32_t(out[1]) << 16);
}

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
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
      case TcuType::WMMA_SP:
        delay = 4;
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

  void wmma(uint32_t wid,
            uint32_t fmt_s,
            uint32_t fmt_d,
            uint32_t step_m,
            uint32_t step_n,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    auto fedp = select_FEDP(fmt_s, fmt_d);

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
        auto d_val = fedp(a_row, b_col, c_val);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);

        DTH(3, simobject_->name() << " FEDP: wid=" << wid << ", i=" << i << ", j=" << j << ", m=" << step_m << ", n=" << step_n << ", a_row={" << std::hex);
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << a_row[q].u32);
        }
        DTN(3, "}, b_col={");
        for (uint32_t q = 0; q < cfg::tcK; ++q) {
          if (q) DTN(3, ", ");
          DTN(3, "0x" << b_col[q].u32);
        }
        DTN(3, "}, c_val=0x" << c_val << ", d_val=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  void wmma_sp(uint32_t wid,
               uint32_t fmt_s,
               uint32_t fmt_d,
               uint32_t step_m,
               uint32_t step_n,
               const std::vector<reg_data_t>& rs1_data,
               const std::vector<reg_data_t>& rs2_data,
               const std::vector<reg_data_t>& rs3_data,
               std::vector<reg_data_t>& rd_data,
               ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    auto fedp = select_FEDP(fmt_s, fmt_d);

    // Bring-up target: sparse path currently supports NT=8 and 8/16-bit input formats.
    // Do not silently fall back to dense WMMA, because sparse-packed A would
    // produce incorrect results under dense semantics.
    if (NUM_THREADS != 8) {
      std::cout << "Error: WMMA_SP unsupported for NUM_THREADS=" << NUM_THREADS
                << " (expected 8)." << std::endl;
      std::abort();
    }
    const bool is_8bit_sparse_fmt =
        (fmt_s == vt::int8::id)  ||
        (fmt_s == vt::uint8::id) ||
        (fmt_s == vt::fp8::id)   ||
        (fmt_s == vt::bf8::id)   ||
        (fmt_s == vt::mxfp8::id) ||
        (fmt_s == vt::mxint8::id);
    const bool is_16bit_sparse_fmt =
        (fmt_s == vt::fp16::id) ||
        (fmt_s == vt::bf16::id);
    if (!is_8bit_sparse_fmt && !is_16bit_sparse_fmt) {
      std::cout << "Error: WMMA_SP unsupported input format: "
                << vt::fmt_string(fmt_s) << " (id=" << fmt_s
                << "). Supported formats: i8, u8, fp8, bf8, mxfp8, mxi8, fp16, bf16." << std::endl;
      std::abort();
    }

    // Fixed masks for bring-up (2:4): keep lanes [0,3] from each 4-element B chunk.
    constexpr uint8_t kMask0 = 0x9;
    constexpr uint8_t kMask1 = 0x9;
    constexpr uint32_t kCompression = 2;
    constexpr uint32_t b_block_size_sp = cfg::b_block_size * kCompression;
    static_assert((NUM_THREADS % b_block_size_sp) == 0, "NUM_THREADS must be divisible by sparse B block size");
    constexpr uint32_t b_sub_blocks_sp = NUM_THREADS / b_block_size_sp;

    uint32_t a_off = (step_m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (step_n % b_sub_blocks_sp) * b_block_size_sp;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * (cfg::tcK * kCompression);
        auto c_val = rs3_data.at(i * cfg::tcN + j).u32;

        reg_data_t a_row_sparse[cfg::tcK];
        reg_data_t b_col_sparse[cfg::tcK];
        for (uint32_t z = 0; z < cfg::tcK; ++z) {
          a_row_sparse[z].u32 = a_row[z].u32;
          uint32_t b_gathered = 0;
          if (is_8bit_sparse_fmt) {
            gather_B8(kMask0, kMask1,
                      b_col[z * kCompression + 0].u32,
                      b_col[z * kCompression + 1].u32,
                      b_gathered);
          } else {
            gather_B16(kMask0,
                       b_col[z * kCompression + 0].u32,
                       b_col[z * kCompression + 1].u32,
                       b_gathered);
          }
          b_col_sparse[z].u32 = b_gathered;
        }

        auto d_val = fedp(a_row_sparse, b_col_sparse, c_val);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);

        DTH(3, simobject_->name() << " SP_FEDP: wid=" << wid << ", i=" << i << ", j=" << j
            << ", m=" << step_m << ", n=" << step_n << ", a0=0x" << std::hex << a_row_sparse[0].u32
            << ", a1=0x" << a_row_sparse[1].u32
            << ", bg0=0x" << b_col_sparse[0].u32
            << ", bg1=0x" << b_col_sparse[1].u32
            << ", c=0x" << c_val << ", d=0x" << d_val << std::dec << std::endl);
      }
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

op_string_t vortex::op_string(TcuType tcu_type, IntrTcuArgs args) {
  switch (tcu_type) {
  case TcuType::WMMA:
    return {"WMMA." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
  case TcuType::WMMA_SP:
    return {"WMMA_SP." + std::string(vt::fmt_string(args.fmt_s)) + "." + std::string(vt::fmt_string(args.fmt_d))
             + "." + std::to_string(args.step_m) + "." + std::to_string(args.step_n), ""};
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
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}

void TensorUnit::wmma_sp(uint32_t wid,
                         uint32_t fmt_s,
                         uint32_t fmt_d,
                         uint32_t step_m,
                         uint32_t step_n,
                         const std::vector<reg_data_t>& rs1_data,
                         const std::vector<reg_data_t>& rs2_data,
                         const std::vector<reg_data_t>& rs3_data,
                         std::vector<reg_data_t>& rd_data,
                         ExeTraceData* trace_data) {
  impl_->wmma_sp(wid, fmt_s, fmt_d, step_m, step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}
