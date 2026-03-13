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

#include "d_tensor_core.h"
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

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

DTensorCore::DTensorCore(const SimContext& ctx,
                                   const char* name,
                                   Cluster* cluster,
                                   const Arch& arch,
                                   const DCRS& dcrs)
  : SimObject<DTensorCore>(ctx, name)
  , mem_req_out(this)
  , mem_rsp_in(this)
  , cluster_(cluster)
  , arch_(arch)
  , dcrs_(dcrs)
  , ram_(nullptr)
  , state_(State::IDLE)
  , busy_(false)
  , done_(false)
  , desc_addr_(0)
  , desc_{}
  , tag_alloc_(1)
  , pending_tag_(0)
  , fragA_(NUM_THREADS * cfg::NRA, 0.0f)
  , fragB_(NUM_THREADS * cfg::NRB, 0.0f)
  , fragC_(NUM_THREADS * cfg::NRC, 0.0f)
  , tile_m_idx_(0)
  , tile_n_idx_(0)
  , tile_k_idx_(0)
  , tiles_m_(1)
  , tiles_n_(1)
  , tiles_k_(1)
  , total_op_reqs_(0)
  , total_out_reqs_(0)
{
  //--
}

DTensorCore::~DTensorCore() {
  //--
}

void DTensorCore::attach_ram(RAM* ram) {
  ram_ = ram;
}

void DTensorCore::reset() {
  state_ = State::IDLE;
  busy_  = false;
  done_  = false;
  pending_tag_ = 0;
  desc_addr_ = 0;
  std::memset(&desc_, 0, sizeof(desc_));
  op_req_lines_.clear();
  out_req_lines_.clear();
  op_req_idx_ = 0;
  out_req_idx_ = 0;
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  tiles_m_ = 1;
  tiles_n_ = 1;
  tiles_k_ = 1;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
}

void DTensorCore::start(uint64_t desc_addr) {
  if (busy_) {
    DP(2, this->name() << ": START ignored (busy)");
    return;
  }
  done_ = false;
  busy_ = true;
  desc_addr_ = desc_addr;
  state_ = State::DESC_REQ;
  op_req_lines_.clear();
  out_req_lines_.clear();
  op_req_idx_ = 0;
  out_req_idx_ = 0;
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  tiles_m_ = 1;
  tiles_n_ = 1;
  tiles_k_ = 1;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
}

uint32_t DTensorCore::poll() const {
  return done_ ? 1u : 0u;
}

void DTensorCore::issue_mem_req(uint64_t addr, bool write) {
  MemReq req;
  req.addr  = addr;
  req.write = write;
  req.tag   = tag_alloc_++;
  req.cid   = 0;
  req.uuid  = 0;

  if (!write) {
    pending_tag_ = req.tag;
  } else {
    pending_tag_ = 0;
  }
  // 1-cycle latency for memory access
  // For same-level interconnect, other files (cache_sim) use a 1-cycle latency for the entire request-response round trip too.
  mem_req_out.send(req);
}

void DTensorCore::load_desc() {
  assert(ram_ && "RAM must be attached before DTensor use");
  ram_->read(&desc_, desc_addr_, sizeof(Desc));
  init_tile_state_();
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

void DTensorCore::init_tile_state_() {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint32_t i_ratio = 4 / in_sz;
  uint32_t tile_k_elems = cfg::tileK * i_ratio;

  uint32_t M = desc_.M ? desc_.M : cfg::tileM;
  uint32_t N = desc_.N ? desc_.N : cfg::tileN;
  uint32_t K = desc_.K ? desc_.K : tile_k_elems;

  if ((M % cfg::tileM) != 0 || (N % cfg::tileN) != 0 || (K % tile_k_elems) != 0) {
    std::cout << "[DTCU] Error: M/N/K must be multiples of tile size. "
              << "M=" << M << ", N=" << N << ", K=" << K
              << " tileM=" << cfg::tileM
              << " tileN=" << cfg::tileN
              << " tileK=" << tile_k_elems << std::endl;
    std::abort();
  }

  tiles_m_ = M / cfg::tileM;
  tiles_n_ = N / cfg::tileN;
  tiles_k_ = K / tile_k_elems;

  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
}

bool DTensorCore::advance_output_tile_() {
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

uint64_t DTensorCore::tile_ptrA_() const {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint32_t i_ratio = 4 / in_sz;
  uint64_t row = uint64_t(tile_m_idx_) * cfg::tileM;
  uint64_t col = uint64_t(tile_k_idx_) * cfg::tileK * i_ratio;
  return desc_.ptrA + (row * desc_.ldmA + col) * in_sz;
}

uint64_t DTensorCore::tile_ptrB_() const {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint32_t i_ratio = 4 / in_sz;
  uint64_t row = uint64_t(tile_k_idx_) * cfg::tileK * i_ratio;
  uint64_t col = uint64_t(tile_n_idx_) * cfg::tileN;
  return desc_.ptrB + (row + col * desc_.ldmB) * in_sz;
}

uint64_t DTensorCore::tile_ptrC_() const {
  uint32_t out_sz = elem_size_bytes(desc_.fmt_d);
  uint64_t row = uint64_t(tile_m_idx_) * cfg::tileM;
  uint64_t col = uint64_t(tile_n_idx_) * cfg::tileN;
  return desc_.ptrC + (row * desc_.ldmC + col) * out_sz;
}

uint64_t DTensorCore::tile_ptrD_() const {
  uint32_t out_sz = elem_size_bytes(desc_.fmt_d);
  uint64_t row = uint64_t(tile_m_idx_) * cfg::tileM;
  uint64_t col = uint64_t(tile_n_idx_) * cfg::tileN;
  return desc_.ptrD + (row * desc_.ldmD + col) * out_sz;
}

void DTensorCore::load_operands() {
  uint32_t fmt_s = desc_.fmt_s;
  uint32_t fmt_d = desc_.fmt_d;

  if (tile_k_idx_ == 0) {
    if (desc_.flags & 0x1) {
      std::fill(fragC_.begin(), fragC_.end(), 0.0f);
    }
  }

  // Load A (row_major), same mapping as kernel/include/vx_tensor.h
  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t block_idx   = (cfg::a_block_size == NUM_THREADS) ? 0 : (lane / cfg::a_block_size);
    uint32_t lane_in_blk = (cfg::a_block_size == NUM_THREADS) ? lane : (lane % cfg::a_block_size);
    uint32_t block_row   = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcM);
    uint32_t block_col   = (lane_in_blk % cfg::tcK);

    uint32_t i_ratio = 4 / elem_size_bytes(fmt_s);
    uint32_t m_stride  = cfg::a_sub_blocks * cfg::tcM;
    uint32_t k_stride  = cfg::tcK * i_ratio;

    uint64_t base_addr = tile_ptrA_() + (uint64_t)block_row * desc_.ldmA * elem_size_bytes(fmt_s)
                                     + (uint64_t)block_col * i_ratio * elem_size_bytes(fmt_s);

    for (uint32_t r = 0; r < cfg::NRA; ++r) {
      uint32_t block_m  = r / cfg::k_steps;
      uint32_t block_k  = r % cfg::k_steps;
      uint32_t elem_row = block_m * m_stride;
      uint32_t elem_col = block_k * k_stride;

      uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmA * elem_size_bytes(fmt_s)
                                + (uint64_t)elem_col * elem_size_bytes(fmt_s);

      float word = 0.0f;
      if (fmt_s == vt::fp32::id) {
        ram_->read(&word, addr, 4);
      } else if (fmt_s == vt::fp16::id || fmt_s == vt::bf16::id) {
        uint16_t v0, v1;
        ram_->read(&v0, addr, 2);
        ram_->read(&v1, addr + 2, 2);
        uint32_t u32 = (uint32_t(v1) << 16) | uint32_t(v0);
        std::memcpy(&word, &u32, 4);
      } else {
        ram_->read(&word, addr, 4);
      }

      fragA_[lane * cfg::NRA + r] = word;
    }
  }

  // Load B (col_major)
  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t block_idx   = (cfg::b_block_size == NUM_THREADS) ? 0 : (lane / cfg::b_block_size);
    uint32_t lane_in_blk = (cfg::b_block_size == NUM_THREADS) ? lane : (lane % cfg::b_block_size);
    uint32_t block_col   = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcN);
    uint32_t block_row   = (lane_in_blk % cfg::tcK);

    uint32_t i_ratio = 4 / elem_size_bytes(fmt_s);
    uint32_t n_stride  = cfg::b_sub_blocks * cfg::tcN;
    uint32_t k_stride  = cfg::tcK * i_ratio;

    uint64_t base_addr = tile_ptrB_() + (uint64_t)block_row * i_ratio * elem_size_bytes(fmt_s)
                                     + (uint64_t)block_col * desc_.ldmB * elem_size_bytes(fmt_s);

    for (uint32_t r = 0; r < cfg::NRB; ++r) {
      uint32_t block_k  = r / cfg::n_steps;
      uint32_t block_n  = r % cfg::n_steps;
      uint32_t elem_row = block_k * k_stride;
      uint32_t elem_col = block_n * n_stride;

      uint64_t addr = base_addr + (uint64_t)elem_row * elem_size_bytes(fmt_s)
                                + (uint64_t)elem_col * desc_.ldmB * elem_size_bytes(fmt_s);

      float word = 0.0f;
      if (fmt_s == vt::fp32::id) {
        ram_->read(&word, addr, 4);
      } else if (fmt_s == vt::fp16::id || fmt_s == vt::bf16::id) {
        uint16_t v0, v1;
        ram_->read(&v0, addr, 2);
        ram_->read(&v1, addr + 2, 2);
        uint32_t u32 = (uint32_t(v1) << 16) | uint32_t(v0);
        std::memcpy(&word, &u32, 4);
      } else {
        ram_->read(&word, addr, 4);
      }

      fragB_[lane * cfg::NRB + r] = word;
    }
  }

  // Load C accumulator (row_major) unless C_is_zero
  if (tile_k_idx_ == 0 && (desc_.flags & 0x1) == 0) {
    for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;

      uint64_t base_addr = tile_ptrC_() + (uint64_t)block_row * desc_.ldmC * elem_size_bytes(fmt_d)
                                       + (uint64_t)block_col * elem_size_bytes(fmt_d);

      for (uint32_t r = 0; r < cfg::NRC; ++r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * cfg::tcM;
        uint32_t elem_col = block_n * cfg::tcN;

        uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmC * elem_size_bytes(fmt_d)
                                  + (uint64_t)elem_col * elem_size_bytes(fmt_d);

        float word = 0.0f;
        ram_->read(&word, addr, 4);
        fragC_[lane * cfg::NRC + r] = word;
      }
    }
  }
}

// Start of FMA and FEDP definitions (copied from tensor_unit.cpp)
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
// End of FMA and FEDP definitions

void DTensorCore::execute_wmma() {
  // m, n, k calculation from decode.cpp
  uint32_t fmt_s = desc_.fmt_s;
  uint32_t fmt_d = desc_.fmt_d;

  std::vector<reg_data_t> rs1_data(NUM_THREADS);
  std::vector<reg_data_t> rs2_data(NUM_THREADS);
  std::vector<reg_data_t> rs3_data(NUM_THREADS);
  std::vector<reg_data_t> rd_data(NUM_THREADS);

  for (uint32_t k = 0; k < cfg::k_steps; ++k) {
    for (uint32_t m = 0; m < cfg::m_steps; ++m) {
      for (uint32_t n = 0; n < cfg::n_steps; ++n) {
        uint32_t rs1 = (m / cfg::a_sub_blocks) * cfg::k_steps + k;
        uint32_t rs2 = (k * cfg::n_steps + n) / cfg::b_sub_blocks;
        uint32_t rs3 = m * cfg::n_steps + n;

        for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
          float fa = fragA_[lane * cfg::NRA + rs1];
          float fb = fragB_[lane * cfg::NRB + rs2];
          float fc = fragC_[lane * cfg::NRC + rs3];
          uint32_t a_u32, b_u32, c_u32;
          std::memcpy(&a_u32, &fa, 4);
          std::memcpy(&b_u32, &fb, 4);
          std::memcpy(&c_u32, &fc, 4);
          rs1_data[lane].u32 = a_u32;
          rs2_data[lane].u32 = b_u32;
          rs3_data[lane].u32 = c_u32;
        }

        // Calls wmma function in TensorUnit with per-lane fragment data

        //tensor_unit() is getter from core.h which returns TensorUnit instance (TensorUnit::Create)
        //  ->wmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d, tpuArgs.step_m, tpuArgs.step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data.get());
        //tcu_->wmma(0, fmt_s, fmt_d, m, n, rs1_data, rs2_data, rs3_data, rd_data, nullptr);

        // Execute WMMA (copied from tensor_unit.cpp)
        auto fedp = select_FEDP(fmt_s, fmt_d);

        uint32_t a_off = (m % cfg::a_sub_blocks) * cfg::a_block_size;
        uint32_t b_off = (n % cfg::b_sub_blocks) * cfg::b_block_size;

        for (uint32_t i = 0; i < cfg::tcM; ++i) {
          for (uint32_t j = 0; j < cfg::tcN; ++j) {
            auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
            auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
            auto c_val = rs3_data.at(i * cfg::tcN + j).u32;
            auto d_val = fedp(a_row, b_col, c_val);
            rd_data.at(i * cfg::tcN + j).u64 = nan_box(d_val);
          }
        }
        
        for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
          uint32_t d_u32 = rd_data[lane].u32;
          float fd;
          std::memcpy(&fd, &d_u32, 4);
          fragC_[lane * cfg::NRC + rs3] = fd;
        }
      }
    }
  }
}

void DTensorCore::store_output() {
  uint32_t fmt_d = desc_.fmt_d;
  uint32_t out_sz = elem_size_bytes(fmt_d);

  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t block_row = lane / cfg::tcN;
    uint32_t block_col = lane % cfg::tcN;

    uint64_t base_addr = tile_ptrD_() + (uint64_t)block_row * desc_.ldmD * out_sz + (uint64_t)block_col * out_sz;

    for (uint32_t r = 0; r < cfg::NRC; ++r) {
      uint32_t block_m  = r / cfg::n_steps;
      uint32_t block_n  = r % cfg::n_steps;
      uint32_t elem_row = block_m * cfg::tcM;
      uint32_t elem_col = block_n * cfg::tcN;

      uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmD * out_sz
                                + (uint64_t)elem_col * out_sz;

      float fd = fragC_[lane * cfg::NRC + r];
      ram_->write(&fd, addr, 4);
    }
  }
}

// --------------------- L2 timing model for memory traffic -------------------
// Compute which cache lines are touched by A/B/C/D 
// then issue MemReq as per # of unique cache line

static inline uint64_t line_base(uint64_t addr) {
  return addr & ~uint64_t(L2_LINE_SIZE - 1);
}

// Similar to mem_coalescer
// Same addresses is combined together / unaligned accesses are split into multiple lines
static inline void coalesce_to_lines(const std::vector<uint64_t>& addrs, uint32_t bytes, std::unordered_set<uint64_t>& out_lines) {
  for (auto addr : addrs) {
    uint64_t l0 = line_base(addr);
    uint64_t l1 = line_base(addr + bytes - 1);
    out_lines.insert(l0);
    out_lines.insert(l1);
  }
}

void DTensorCore::build_req_lists_() {
  op_req_lines_.clear();
  out_req_lines_.clear();
  op_req_idx_ = 0;
  out_req_idx_ = 0;

  std::unordered_set<uint64_t> op_lines;
  std::unordered_set<uint64_t> out_lines;

  const uint32_t fmt_s  = desc_.fmt_s;
  const uint32_t fmt_d  = desc_.fmt_d;
  const uint32_t in_sz  = elem_size_bytes(fmt_s);
  const uint32_t out_sz = elem_size_bytes(fmt_d);

  // Match current RAM access granularity
  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> op_addrs;
  std::vector<uint64_t> out_addrs;

  // A - row_major
  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t block_idx   = (cfg::a_block_size == NUM_THREADS) ? 0 : (lane / cfg::a_block_size);
    uint32_t lane_in_blk = (cfg::a_block_size == NUM_THREADS) ? lane : (lane % cfg::a_block_size);
    uint32_t block_row   = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcM);
    uint32_t block_col   = (lane_in_blk % cfg::tcK);

    uint32_t i_ratio  = 4 / in_sz;
    uint32_t m_stride = cfg::a_sub_blocks * cfg::tcM;
    uint32_t k_stride = cfg::tcK * i_ratio;

    uint64_t base_addr = tile_ptrA_() + (uint64_t)block_row * desc_.ldmA * in_sz
                                   + (uint64_t)block_col * i_ratio * in_sz;

    for (uint32_t r = 0; r < cfg::NRA; ++r) {
      uint32_t block_m  = r / cfg::k_steps;
      uint32_t block_k  = r % cfg::k_steps;
      uint32_t elem_row = block_m * m_stride;
      uint32_t elem_col = block_k * k_stride;

      uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmA * in_sz
                                + (uint64_t)elem_col * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // B - col_major
  for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
    uint32_t block_idx   = (cfg::b_block_size == NUM_THREADS) ? 0 : (lane / cfg::b_block_size);
    uint32_t lane_in_blk = (cfg::b_block_size == NUM_THREADS) ? lane : (lane % cfg::b_block_size);
    uint32_t block_col   = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcN);
    uint32_t block_row   = (lane_in_blk % cfg::tcK);

    uint32_t i_ratio  = 4 / in_sz;
    uint32_t n_stride = cfg::b_sub_blocks * cfg::tcN;
    uint32_t k_stride = cfg::tcK * i_ratio;

    uint64_t base_addr = tile_ptrB_() + (uint64_t)block_row * i_ratio * in_sz
                                   + (uint64_t)block_col * desc_.ldmB * in_sz;

    for (uint32_t r = 0; r < cfg::NRB; ++r) {
      uint32_t block_k  = r / cfg::n_steps;
      uint32_t block_n  = r % cfg::n_steps;
      uint32_t elem_row = block_k * k_stride;
      uint32_t elem_col = block_n * n_stride;

      uint64_t addr = base_addr + (uint64_t)elem_row * in_sz
                                + (uint64_t)elem_col * desc_.ldmB * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // C - row_major
  if (tile_k_idx_ == 0 && (desc_.flags & 0x1) == 0) {
    for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;

      uint64_t base_addr = tile_ptrC_() + (uint64_t)block_row * desc_.ldmC * out_sz
                                     + (uint64_t)block_col * out_sz;

      for (uint32_t r = 0; r < cfg::NRC; ++r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * cfg::tcM;
        uint32_t elem_col = block_n * cfg::tcN;

        uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmC * out_sz
                                  + (uint64_t)elem_col * out_sz;
        op_addrs.push_back(addr);
      }
    }
  }

  // D output (row_major)
  if (tile_k_idx_ == (tiles_k_ - 1)) {
    for (uint32_t lane = 0; lane < NUM_THREADS; ++lane) {
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;

      uint64_t base_addr = tile_ptrD_() + (uint64_t)block_row * desc_.ldmD * out_sz
                                      + (uint64_t)block_col * out_sz;

      for (uint32_t r = 0; r < cfg::NRC; ++r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * cfg::tcM;
        uint32_t elem_col = block_n * cfg::tcN;

        uint64_t addr = base_addr + (uint64_t)elem_row * desc_.ldmD * out_sz
                                  + (uint64_t)elem_col * out_sz;
        out_addrs.push_back(addr);
      }
    }
  }

  // Coalesce in order to only calculate unique cache lines touched
  coalesce_to_lines(op_addrs,  WORD_BYTES, op_lines);
  coalesce_to_lines(out_addrs, WORD_BYTES, out_lines);

  op_req_lines_.assign(op_lines.begin(), op_lines.end());
  out_req_lines_.assign(out_lines.begin(), out_lines.end());

  total_op_reqs_ += op_req_lines_.size();
  total_out_reqs_ += out_req_lines_.size();
}


// Sequentially issues operand MemReq (one per cache line)
// Returns false when all operand read requests issued
bool DTensorCore::issue_next_op_req_() {
  if (op_req_idx_ >= op_req_lines_.size())
    return false;
  issue_mem_req(op_req_lines_[op_req_idx_++], false);
  return true;
}

// Sequentially issues output MemReq (one per cache line)
// Returns false when all output write requests issued
bool DTensorCore::issue_next_out_req_() {
  if (out_req_idx_ >= out_req_lines_.size())
    return false;
  issue_mem_req(out_req_lines_[out_req_idx_++], true);
  return true;
}

// Adapted from cache_sim.cpp for mem response handling
void DTensorCore::tick() {
  if (!mem_rsp_in.empty()) {
    auto rsp = mem_rsp_in.peek();
    if (rsp.tag == pending_tag_) {
      mem_rsp_in.pop();
      pending_tag_ = 0;
    }
  }

  switch (state_) {
  case State::IDLE:
    break;

  case State::DESC_REQ:
    issue_mem_req(desc_addr_, false); // Read descriptor
    state_ = State::DESC_WAIT;
    break;

  case State::DESC_WAIT:
    if (pending_tag_ == 0) {
      load_desc();

      // For debugging: print descriptor info
      std::cout << "[DTCU] " << "ptrA=0x" << std::hex << desc_.ptrA << " ptrB=0x" << desc_.ptrB << " ptrC=0x" << desc_.ptrC << " ptrD=0x" << desc_.ptrD //pointer
          << std::dec << " ldmA=" << desc_.ldmA << " ldmB=" << desc_.ldmB << " ldmC=" << desc_.ldmC << " ldmD=" << desc_.ldmD // leading dimension
          << " M=" << desc_.M << " N=" << desc_.N << " K=" << desc_.K // matrix size
          << " fmt_s=" << uint32_t(desc_.fmt_s) << " fmt_d=" << uint32_t(desc_.fmt_d) << " flags=" << uint32_t(desc_.flags) // metadata
          << std::endl;

      build_req_lists_();
      state_ = State::OP_REQ;
    }
    break;

  case State::OP_REQ:
    // Issue variable number of MemReq
    if (!issue_next_op_req_()) {
      pending_tag_ = 0;
    }
    state_ = State::OP_WAIT;
    break;

  case State::OP_WAIT:
    if (pending_tag_ == 0) {
      if (op_req_idx_ < op_req_lines_.size()) {
        state_ = State::OP_REQ;
      } else {
        load_operands();
        state_ = State::EXECUTE;
      }
    }
    break;

  case State::EXECUTE:
    execute_wmma();

    if ((tile_k_idx_ + 1) < tiles_k_) {
      // Repeat over tiles if there's tile left in K dimension
      ++tile_k_idx_;
      build_req_lists_();
      state_ = State::OP_REQ;
    } else {
      // Move to output store only if it's the last tile in K dimension
      state_ = State::OUT_REQ;
    }
    break;

  case State::OUT_REQ:
    // Issue variable number of MemReq
    if (!issue_next_out_req_()) {
      state_ = State::OUT_WAIT;
    }
    break;

  case State::OUT_WAIT:
    if (pending_tag_ == 0) {
      store_output();

      if (advance_output_tile_()) {
        // Move to next output tile if there's more tiles in M/N dimension
        build_req_lists_();
        state_ = State::OP_REQ;
      } else {
        // Mem Req message moved to here
        std::cout << "[DTCU] L2 MemReq count: desc=1, op=" << total_op_reqs_
                  << ", output=" << total_out_reqs_
                  << ", total=" << (1 + total_op_reqs_ + total_out_reqs_)
                  << std::endl;

        done_ = true;
        busy_ = false;
        state_ = State::DONE;
      }
    }
    break;

  case State::DONE:
    break;

  default:
    break;
  }
}
