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

#pragma once

// ONE definition of the tensor PE's arithmetic, for every unit that contains one.
//
// This is the value oracle -- the multiply-accumulate and the fused dot product that a
// tensor unit's PE performs. It lived inside tcu_unit.cpp, and dtcu.cpp carried a second
// copy of it under the header "FMA and FEDP definitions (copied from tensor_unit.cpp)".
// The copy then went stale, and not cosmetically:
//
//   * FMA<fp16,fp32> took `float c` and returned `float` there, `uint32_t` here.
//   * FEDP chained the incoming accumulator through EVERY multiply-add there. Here, for a
//     wide output type, the products are summed in fp32 first and C is added once at the
//     end. Different rounding, so the two units did not agree bit-for-bit on the same
//     GEMM -- they only agreed to within the harness's ULP tolerance, which is why it was
//     never caught.
//
// Both units now include this file. The DTCU is meant to be the same arithmetic in a
// different PLACE; a second implementation of the multiply was never part of the design,
// and it silently made the engine a different numerical machine from the core it is
// being compared against. Pipeline depth is shared the same way, in tcu_latency.h.
//
// The DTCU calls FEDP<>::eval in cfg::tcK-word chunks and chains the accumulator across
// them, so it needs no separate entry point; eval_n() is here for the TCU's variable-K
// uops.

#include <VX_config.h>
#include <bitmanip.h>
#include <rvfloats.h>
#include "types.h"
#include "tensor_cfg.h"
#include "tcu_latency.h"
#include <type_traits>

namespace vortex {
namespace tcu_pe {

namespace vt = vortex::tensor;
using vortex::tcu_timing::cfg;

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
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype);
    static_assert(i_ratio * sizeof(itype) == sizeof(uint32_t), "FEDP: tcK * i_ratio must be <= 32");
    if constexpr (std::is_same_v<Ot, vt::fp32>) {
      uint32_t acc = 0;
      for (uint32_t z = 0; z < k_words; ++z) {
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
      for (uint32_t z = 0; z < k_words; ++z) {
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
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < k_words; ++z) {
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
    return eval_n(a_row, b_col, c_val, cfg::tcK);
  }

  static uint32_t eval_n(const reg_data_t *a_row, const reg_data_t *b_col, uint32_t c_val, uint32_t k_words) {
    auto acc = bit_cast<int32_t>(c_val);
    for (uint32_t z = 0; z < k_words; ++z) {
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

// The type of FEDP<>::eval: one cfg::tcK-word chunk, accumulator in and out. The DTCU
// dispatches on the descriptor's runtime formats through a pointer of this type; the TCU
// uses the variable-K PFN_FEDP_N in tcu_unit.cpp for its uops.
using PFN_FEDP = uint32_t (*)(const reg_data_t*, const reg_data_t*, uint32_t);

} // namespace tcu_pe
} // namespace vortex
