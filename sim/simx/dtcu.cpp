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

#include "dtcu.h"
#include "dtcu_tma.h"
#include "dtcu_params.h"
#include "cluster.h"
#include "types.h"
#include "tensor_cfg.h"
#include <rvfloats.h>
#include <cmath>
#include <iostream>
#include <cstring>
#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <array>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

namespace {

constexpr uint32_t DTCU_TILE_K_WORDS = 8;

} // namespace

Dtcu::Dtcu(const SimContext& ctx,
                                   const char* name,
                                   Cluster* cluster,
                                   const Arch& arch,
                                   const DCRS& dcrs)
  : SimObject<Dtcu>(ctx, name)
  , cluster_(cluster)
  , arch_(arch)
  , dcrs_(dcrs)
  , state_(State::IDLE)
  , busy_(false)
  , done_(false)
  , desc_addr_(0)
  , desc_{}
  , a_buf_()
  , b_buf_()
  , accum_buf_()
  , tile_m_(0)
  , tile_n_(0)
  , tile_k_(0)
  , tile_m_idx_(0)
  , tile_n_idx_(0)
  , tile_k_idx_(0)
  , tiles_m_(1)
  , tiles_n_(1)
  , tiles_k_(1)
  , total_op_reqs_(0)
  , total_out_reqs_(0)
  , exec_cycles_left_(0)
{
  tma_ = std::make_unique<DtcuTma>(*this); // owns the L2 port + RAM
}

Dtcu::~Dtcu() {
  //--
}

void Dtcu::attach_ram(RAM* ram) {
  tma_->attach_ram(ram);
}

void Dtcu::reset() {
  state_ = State::IDLE;
  busy_  = false;
  done_  = false;
  desc_addr_ = 0;
  std::memset(&desc_, 0, sizeof(desc_));
  tma_->reset();
  a_buf_[0].clear();
  a_buf_[1].clear();
  b_buf_[0].clear();
  b_buf_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  dtcu_compute_cycles_ = 0;
  dtcu_wait_for_tma_cycles_ = 0;
  tma_mem_wait_cycles_ = 0;
  tma_wait_for_buffer_cycles_ = 0;
  tma_buffer_write_cycles_ = 0;
  tma_addrgen_cycles_ = 0;
  accum_buf_[0].clear();
  accum_buf_[1].clear();
  accum_compute_idx_ = 0;
  tile_m_ = 0;
  tile_n_ = 0;
  tile_k_ = 0;
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  tiles_m_ = 1;
  tiles_n_ = 1;
  tiles_k_ = 1;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
  exec_cycles_left_ = 0;
}

void Dtcu::start(uint64_t desc_addr) {
  if (busy_) {
    DP(2, this->name() << ": START ignored (busy)");
    return;
  }
  done_ = false;
  busy_ = true;
  desc_addr_ = desc_addr;
  state_ = State::DESC_REQ;
  tma_->reset();
  a_buf_[0].clear();
  a_buf_[1].clear();
  b_buf_[0].clear();
  b_buf_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  dtcu_compute_cycles_ = 0;
  dtcu_wait_for_tma_cycles_ = 0;
  tma_mem_wait_cycles_ = 0;
  tma_wait_for_buffer_cycles_ = 0;
  tma_buffer_write_cycles_ = 0;
  tma_addrgen_cycles_ = 0;
  accum_buf_[0].clear();
  accum_buf_[1].clear();
  accum_compute_idx_ = 0;
  tile_m_ = 0;
  tile_n_ = 0;
  tile_k_ = 0;
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  tiles_m_ = 1;
  tiles_n_ = 1;
  tiles_k_ = 1;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
  exec_cycles_left_ = 0;
}

uint32_t Dtcu::poll() const {
  return done_ ? 1u : 0u;
}


static inline uint32_t elem_size_bytes(uint32_t fmt_id) {
  switch (fmt_id) {
    case vt::fp32::id:  return 4;
    case vt::fp16::id:  return 2;
    case vt::bf16::id:  return 2;
    case vt::int32::id: return 4;
    case vt::int8::id:  return 1;
    case vt::uint8::id: return 1;
    case vt::int4::id:  return 1;
    case vt::uint4::id: return 1;
    default:            return 4;
  }
}

void Dtcu::init_tile_state_() {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);

  if (desc_.fmt_d != vt::fp32::id) {
    std::cout << "[DTCU] Error: Only supports fp32 output/accumulation" << std::endl;
    std::abort();
  }

  if (0 == in_sz || (4 % in_sz) != 0) {
    std::cout << "[DTCU] Error: Unsupported input element size: " << in_sz << std::endl;
    std::abort();
  }

  if (desc_.shape_n_size == 0) {
    std::cout << "[DTCU] Error: shape_n_size must explicitly select N-size" << std::endl;
    std::abort();
  }

  // Shape policy is TBD (just set to 0 for now)
  if (desc_.shape_policy != 0) {
    std::cout << "[DTCU] Error: Unsupported shape policy: " << uint32_t(desc_.shape_policy) << std::endl;
    std::abort();
  }

  tile_m_ = 64; // fixed tile M dimension
  tile_n_ = uint32_t(desc_.shape_n_size) * 16; // tile N dimension is determined by shape_n_size (in multiples of 16)
  tile_k_ = 8 * (4 / in_sz); // tile K dimension is determined by input element size (fp16 = 16 / fp32 = 8)

  if (tile_n_ < 16 || tile_n_ > 128 || (tile_n_ % 16) != 0) {
    std::cout << "[DTCU] Error: N-dimension must be in multiples of 16; maximum 128. Received: " << tile_n_ << std::endl;
    std::abort();
  }

  if ((desc_.M % tile_m_) != 0 || (desc_.N % tile_n_) != 0 || (desc_.K % tile_k_) != 0) {
    std::cout << "[DTCU] Error: Partial Tile not supported. M/N/K must be multiples of tile size. "
              << "M=" << desc_.M << ", N=" << desc_.N << ", K=" << desc_.K
              << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_ << std::endl;
    std::abort();
  }

  // Initialize internal buffers based on tile sizes
  a_buf_[0].assign(tile_m_ * 8, 0);
  a_buf_[1].assign(tile_m_ * 8, 0);
  b_buf_[0].assign(8 * tile_n_, 0);
  b_buf_[1].assign(8 * tile_n_, 0);
  accum_buf_[0].assign(tile_m_ * tile_n_, 0.0f);
  accum_buf_[1].assign(tile_m_ * tile_n_, 0.0f);

  // Calculate # of tiles required to cover the entire GEMM
  tiles_m_ = desc_.M / tile_m_;
  tiles_n_ = desc_.N / tile_n_;
  tiles_k_ = desc_.K / tile_k_;

  // Initialize tile indices to start from the first tile
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
}

bool Dtcu::advance_output_tile_() {
  tile_k_idx_ = 0;

  ++tile_n_idx_;
  if (tile_n_idx_ < tiles_n_)
    return true;

  tile_n_idx_ = 0;
  ++tile_m_idx_;
  if (tile_m_idx_ < tiles_m_)
    return true;

  return false;
}

uint32_t Dtcu::estimate_execute_cycles_() const {
  // Compute-phase latency for one K tile. Two parts:
  //  (1) MAC throughput: fixed array of DTCU_MACS_PER_CYCLE MAC/cycle over the
  //      tile_m*tile_n*tile_k MACs of the native tile.
  //  (2) Accumulator read-modify-write: each K tile reads the partial sums and
  //      writes the updated sums = 2*tile_m*tile_n words. The accumulator is the
  //      same kind of on-die SRAM as the operand buffers, so it uses the SAME
  //      bandwidth/latency (DTCU_BUF_BW / DTCU_BUF_LATENCY).
  // The functional execute_mma() loop stays sequential; this only models timing.
  const uint64_t tile_macs    = uint64_t(tile_m_) * tile_n_ * tile_k_;
  const uint64_t mac_cycles   = (tile_macs + DTCU_MACS_PER_CYCLE - 1) / DTCU_MACS_PER_CYCLE;
  const uint64_t accum_words  = 2ull * tile_m_ * tile_n_; // read partial + write updated
  const uint64_t accum_cycles = (accum_words + DTCU_BUF_BW - 1) / DTCU_BUF_BW + DTCU_BUF_LATENCY;
  return std::max(1u, uint32_t(mac_cycles + accum_cycles + DTCU_COMPUTE_LATENCY));
}


// --------------------- FMA and FEDP definitions (copied from tensor_unit.cpp) ---------------------
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


void Dtcu::execute_mma(uint32_t buf_idx) {
  auto fedp = select_FEDP(desc_.fmt_s, desc_.fmt_d);

  if ((DTCU_TILE_K_WORDS % cfg::tcK) != 0) {
    std::cout << "[DTCU] Error: Tile K is not divisible by FEDP width" << std::endl;
    std::abort();
  }

  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint32_t acc_bit;
      
      std::memcpy(&acc_bit, &accum_buf_[accum_compute_idx_][m * tile_n_ + n], 4); // Bitwise copy accumulator value in raw 32-bit representation

      for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; kw += cfg::tcK) {
        std::array<reg_data_t, cfg::tcK> a_words{};
        std::array<reg_data_t, cfg::tcK> b_words{};

        for (uint32_t z = 0; z < cfg::tcK; ++z) {
          a_words[z].u32 = a_buf_[buf_idx][m * DTCU_TILE_K_WORDS + kw + z];
          b_words[z].u32 = b_buf_[buf_idx][(kw + z) * tile_n_ + n];
        }

        acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);
      }

      std::memcpy(&accum_buf_[accum_compute_idx_][m * tile_n_ + n], &acc_bit, 4);
    }
  }
}

void Dtcu::tick() {
  // The TMA engine owns the L2 port: let it retire all responses this cycle.
  tma_->drain_responses();

  switch (state_) {
  case State::IDLE:
    break;

  case State::DESC_REQ:
    tma_->issue_desc_req(desc_addr_); // Read descriptor
    state_ = State::DESC_WAIT;
    break;

  case State::DESC_WAIT:
    if (tma_->main_done()) {
      tma_->read_desc(desc_addr_);
      init_tile_state_();

      // For debugging: print descriptor info
      std::cout << "[DTCU] " << "ptrA=0x" << std::hex << desc_.ptrA << " ptrB=0x" << desc_.ptrB << " ptrC=0x" << desc_.ptrC << " ptrD=0x" << desc_.ptrD //pointer
          << std::dec << " ldmA=" << desc_.ldmA << " ldmB=" << desc_.ldmB << " ldmC=" << desc_.ldmC << " ldmD=" << desc_.ldmD // leading dimension
          << " M=" << desc_.M << " N=" << desc_.N << " K=" << desc_.K // matrix size
          << " fmt_s=" << uint32_t(desc_.fmt_s) << " fmt_d=" << uint32_t(desc_.fmt_d) << " flags=" << uint32_t(desc_.flags) // metadata
          << " shape_n_size=" << uint32_t(desc_.shape_n_size) << " shape_policy=" << uint32_t(desc_.shape_policy) // N-dimension shape
          << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_ // Set Native Tile Size
          << std::endl;

      // Begin streaming: prefetch K0 of the first output tile into the compute buffer.
      tile_k_idx_ = 0;
      tma_->start_prefetch(compute_buf_, 0);
      state_ = State::FIRST_LOAD;
    }
    break;

  case State::FIRST_LOAD:
    // Fill the current compute buffer (K0 of this output tile) before computing.
    tma_->tick();
    if (buf_ready_[compute_buf_]) {
      exec_cycles_left_ = estimate_execute_cycles_();
      compute_done_ = false;
      // Start prefetching the next K tile into the other buffer (overlap).
      if (tile_k_idx_ + 1 < tiles_k_) {
        tma_->start_prefetch(compute_buf_ ^ 1, tile_k_idx_ + 1);
      }
      state_ = State::COMPUTE;
    }
    break;

  case State::COMPUTE:
    // Prefetch the next K tile concurrently with computing the current one.
    tma_->tick();

    // Prefetch is done-ahead but blocked: the next buffer is filled and a further
    // K tile exists, yet no buffer is free until the current compute consumes one.
    if (tma_->load_idle() && buf_ready_[compute_buf_ ^ 1]
        && (tile_k_idx_ + 2 < tiles_k_)) {
      ++tma_wait_for_buffer_cycles_;
    }

    if (exec_cycles_left_ > 0) {
      --exec_cycles_left_;
      ++dtcu_compute_cycles_;
      break; // still computing the current K tile
    }

    if (!compute_done_) {
      // Compute latency elapsed: run the MMA for the current K tile once.
      execute_mma(compute_buf_);
      compute_done_ = true;
    }

    if ((tile_k_idx_ + 1) < tiles_k_) {
      // More K tiles: advance only when the next operand buffer is prefetched.
      uint32_t next_buf = compute_buf_ ^ 1;
      if (buf_ready_[next_buf]) {
        buf_ready_[compute_buf_] = false; // release the consumed buffer
        compute_buf_ = next_buf;
        ++tile_k_idx_;
        exec_cycles_left_ = estimate_execute_cycles_();
        compute_done_ = false;
        // Kick prefetch of the following K tile.
        if (tile_k_idx_ + 1 < tiles_k_) {
          tma_->start_prefetch(compute_buf_ ^ 1, tile_k_idx_ + 1);
        }
      } else {
        // Compute finished but the next operand tile is not ready yet.
        ++dtcu_wait_for_tma_cycles_;
      }
    } else {
      // Last K tile done: hand the output (D) store to the TMA engine.
      tma_->begin_store();
      buf_ready_[compute_buf_] = false;
      state_ = State::OUT_REQ;
    }
    break;

  case State::OUT_REQ:
    // Issue exactly one output write request, then wait for its response.
    if (tma_->issue_next_out_req()) {
      state_ = State::OUT_WAIT;
    } else {
      tma_->clear_main_pending();
      state_ = State::OUT_WAIT;
    }
    break;

  case State::OUT_WAIT:
    if (tma_->main_done()) {
      if (tma_->store_has_more()) {
        // More output cache lines remain: issue next write request.
        state_ = State::OUT_REQ;
      } else {
        // All output write requests have completed.
        tma_->store_output();

        if (advance_output_tile_()) {
          // Next output tile: restart streaming from K0 (advance_output_tile_ reset tile_k_idx_).
          buf_ready_[0] = false;
          buf_ready_[1] = false;
          tma_->start_prefetch(compute_buf_, 0);
          state_ = State::FIRST_LOAD;
        } else {
          // Mem Req message moved to here
          std::cout << "[DTCU] L2 MemReq count: desc=1, op=" << total_op_reqs_
                    << ", output=" << total_out_reqs_
                    << ", total=" << (1 + total_op_reqs_ + total_out_reqs_)
                    << std::endl;

          // Overlap breakdown. dtcu_wait_for_tma_cycles is the key metric: a large
          // value means compute is starved by operand prefetch (memory-bound).
          std::cout << "[DTCU] overlap cycles: compute=" << dtcu_compute_cycles_
                    << ", wait_for_tma=" << dtcu_wait_for_tma_cycles_
                    << ", tma_mem_wait=" << tma_mem_wait_cycles_
                    << ", tma_wait_for_buffer=" << tma_wait_for_buffer_cycles_
                    << ", tma_buf_write=" << tma_buffer_write_cycles_
                    << ", tma_addrgen=" << tma_addrgen_cycles_
                    << std::endl;

          done_ = true;
          busy_ = false;
          state_ = State::DONE;
        }
      }
    }
    break;

  case State::DONE:
    break;

  default:
    break;
  }
}
