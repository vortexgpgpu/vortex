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
#include <array>

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

namespace {

constexpr uint32_t DTCU_TILE_K_WORDS = 8;

} // namespace

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
  a_buf_[0].clear();
  a_buf_[1].clear();
  b_buf_[0].clear();
  b_buf_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  tma_state_ = TmaState::IDLE;
  tma_req_lines_.clear();
  tma_req_idx_ = 0;
  tma_pending_tag_ = 0;
  tma_target_buf_ = 0;
  tma_k_ = 0;
  accum_buf_.clear();
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
  a_buf_[0].clear();
  a_buf_[1].clear();
  b_buf_[0].clear();
  b_buf_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  tma_state_ = TmaState::IDLE;
  tma_req_lines_.clear();
  tma_req_idx_ = 0;
  tma_pending_tag_ = 0;
  tma_target_buf_ = 0;
  tma_k_ = 0;
  accum_buf_.clear();
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

  // Track both read and write requests until MemRsp arrives.
  pending_tag_ = req.tag;

  // 1-cycle latency for memory access
  // For same-level interconnect, other files (cache_sim) use a 1-cycle latency for the entire request-response round trip too.
  mem_req_out.send(req);
}

// Same as issue_mem_req but tracks the request under the TMA prefetch tag, so a
// prefetch can be in flight independently of the main (descriptor/output) path.
void DTensorCore::issue_mem_req_tma_(uint64_t addr, bool write) {
  MemReq req;
  req.addr  = addr;
  req.write = write;
  req.tag   = tag_alloc_++;
  req.cid   = 0;
  req.uuid  = 0;
  tma_pending_tag_ = req.tag;
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
  accum_buf_.assign(tile_m_ * tile_n_, 0.0f);

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

// Helper functions to calculate current tile's base addresses for A/B/C/D based on current tile indices and descriptor
uint64_t DTensorCore::calculate_base_A_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint64_t row = uint64_t(tile_m_idx_) * tile_m_;
  uint64_t col = uint64_t(k_idx) * tile_k_;
  return desc_.ptrA + (row * desc_.ldmA + col) * in_sz;
}

uint64_t DTensorCore::calculate_base_B_(uint32_t k_idx) const {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint64_t row = uint64_t(k_idx) * tile_k_;
  uint64_t col = uint64_t(tile_n_idx_) * tile_n_;
  return desc_.ptrB + (row + col * desc_.ldmB) * in_sz;
}

uint64_t DTensorCore::calculate_base_C_() const {
  uint32_t out_sz = elem_size_bytes(desc_.fmt_d);
  uint64_t row = uint64_t(tile_m_idx_) * tile_m_;
  uint64_t col = uint64_t(tile_n_idx_) * tile_n_;
  return desc_.ptrC + (row * desc_.ldmC + col) * out_sz;
}

uint64_t DTensorCore::calculate_base_D_() const {
  uint32_t out_sz = elem_size_bytes(desc_.fmt_d);
  uint64_t row = uint64_t(tile_m_idx_) * tile_m_;
  uint64_t col = uint64_t(tile_n_idx_) * tile_n_;
  return desc_.ptrD + (row * desc_.ldmD + col) * out_sz;
}

uint32_t DTensorCore::estimate_execute_cycles_() const {
  const uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  const uint32_t i_ratio = 4 / in_sz;

  // One in-core WMMA micro-op consumes cfg::tcK 32-bit words,
  const uint32_t wmma_uop_k = cfg::tcK * i_ratio;

  if ((tile_m_ % cfg::tcM) != 0 || (tile_n_ % cfg::tcN) != 0 || (tile_k_ % wmma_uop_k) != 0) {
    std::cout << "[DTCU] Error: Tile is not divisible by in-core WMMA micro-op shape"
              << " tile_m=" << tile_m_ << " tile_n=" << tile_n_ << " tile_k=" << tile_k_
              << " tcM=" << cfg::tcM << " tcN=" << cfg::tcN << " wmma_uop_k=" << wmma_uop_k
              << std::endl;
    std::abort();
  }

  const uint32_t wmma_uops_m = tile_m_ / cfg::tcM;
  const uint32_t wmma_uops_n = tile_n_ / cfg::tcN;
  const uint32_t wmma_uops_k = tile_k_ / wmma_uop_k;

  const uint32_t equivalent_wmma_uops = wmma_uops_m * wmma_uops_n * wmma_uops_k;

  // 4 is the constant used in sim/simx/tensor_unit.cpp
  // Each in-core WMMA micro-op has delay = 4 cycles.
  constexpr uint32_t WMMA_UOP_DELAY = 4;

  return std::max(1u, equivalent_wmma_uops * WMMA_UOP_DELAY);
}

void DTensorCore::load_operands_into(uint32_t buf_idx, uint32_t k_idx) {
  uint32_t in_sz = elem_size_bytes(desc_.fmt_s);
  uint32_t elems_per_word = 4 / in_sz;

  // Initialize Accumulators Buffer on the first K tile
  if (k_idx == 0) {
    if (desc_.flags & 0x1) {
      // No pre-load for accumulator
      std::fill(accum_buf_.begin(), accum_buf_.end(), 0.0f);
    } else {
      // Pre-loaded accumulator
      uint64_t baseC = calculate_base_C_();
      for (uint32_t m = 0; m < tile_m_; ++m) {
        for (uint32_t n = 0; n < tile_n_; ++n) {
          uint64_t addr = baseC + (uint64_t(m) * desc_.ldmC + n) * 4;
          float value = 0.0f;
          ram_->read(&value, addr, 4);
          accum_buf_[m * tile_n_ + n] = value;
        }
      }
    }
  }

  // Load A Buffer (row_major), same mapping as kernel/include/vx_tensor.h
  uint64_t baseA = calculate_base_A_(k_idx);
  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      uint64_t addr = baseA + (uint64_t(m) * desc_.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
      uint32_t word = 0;
      ram_->read(&word, addr, 4);
      a_buf_[buf_idx][m * DTCU_TILE_K_WORDS + kw] = word;
    }
  }

  // Load B Buffer (col_major)
  uint64_t baseB = calculate_base_B_(k_idx);
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc_.ldmB) * in_sz;
      uint32_t word = 0;
      ram_->read(&word, addr, 4);
      b_buf_[buf_idx][kw * tile_n_ + n] = word;
    }
  }
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


void DTensorCore::execute_mma(uint32_t buf_idx) {
  auto fedp = select_FEDP(desc_.fmt_s, desc_.fmt_d);

  if ((DTCU_TILE_K_WORDS % cfg::tcK) != 0) {
    std::cout << "[DTCU] Error: Tile K is not divisible by FEDP width" << std::endl;
    std::abort();
  }

  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint32_t acc_bit;
      
      std::memcpy(&acc_bit, &accum_buf_[m * tile_n_ + n], 4); // Bitwise copy accumulator value in raw 32-bit representation

      for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; kw += cfg::tcK) {
        std::array<reg_data_t, cfg::tcK> a_words{};
        std::array<reg_data_t, cfg::tcK> b_words{};

        for (uint32_t z = 0; z < cfg::tcK; ++z) {
          a_words[z].u32 = a_buf_[buf_idx][m * DTCU_TILE_K_WORDS + kw + z];
          b_words[z].u32 = b_buf_[buf_idx][(kw + z) * tile_n_ + n];
        }

        acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);
      }

      std::memcpy(&accum_buf_[m * tile_n_ + n], &acc_bit, 4);
    }
  }
}

void DTensorCore::store_output() {
  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint64_t addr = calculate_base_D_() + (uint64_t(m) * desc_.ldmD + n) * 4;
      float value = accum_buf_[m * tile_n_ + n];
      ram_->write(&value, addr, 4);
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
static inline void coalesce_to_lines(const std::vector<uint64_t>& addrs, uint32_t bytes, std::vector<uint64_t>& out_lines) {
  std::unordered_set<uint64_t> seen_lines;
  seen_lines.reserve(addrs.size() * 2);

  for (auto addr : addrs) {
    uint64_t l0 = line_base(addr);
    uint64_t l1 = line_base(addr + bytes - 1);

    if (seen_lines.insert(l0).second) {
      out_lines.push_back(l0);
    }

    if (l1 != l0 && seen_lines.insert(l1).second) {
      out_lines.push_back(l1);
    }
  }
}

// Build the operand (A/B/C) cache-line request list for a given K tile.
void DTensorCore::build_op_req_lines_(uint32_t k_idx, std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  const uint32_t in_sz  = elem_size_bytes(desc_.fmt_s);
  const uint32_t elems_per_word = 4 / in_sz;

  // Match current RAM access granularity
  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> op_addrs;
  op_addrs.reserve(tile_m_ * DTCU_TILE_K_WORDS + DTCU_TILE_K_WORDS * tile_n_ + tile_m_ * tile_n_);

  // A - row_major
  uint64_t baseA = calculate_base_A_(k_idx);
  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
      uint64_t addr = baseA + (uint64_t(m) * desc_.ldmA + uint64_t(kw) * elems_per_word) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // B - col_major
  uint64_t baseB = calculate_base_B_(k_idx);
  for (uint32_t kw = 0; kw < DTCU_TILE_K_WORDS; ++kw) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint64_t addr = baseB + (uint64_t(kw) * elems_per_word + uint64_t(n) * desc_.ldmB) * in_sz;
      op_addrs.push_back(addr);
    }
  }

  // C - row_major (only on the first K tile when accumulator is pre-loaded)
  if (k_idx == 0 && (desc_.flags & 0x1) == 0) {
    uint64_t baseC = calculate_base_C_();
    for (uint32_t m = 0; m < tile_m_; ++m) {
      for (uint32_t n = 0; n < tile_n_; ++n) {
        uint64_t addr = baseC + (uint64_t(m) * desc_.ldmC + n) * 4;
        op_addrs.push_back(addr);
      }
    }
  }

  // Coalesce while preserving first-touch order
  coalesce_to_lines(op_addrs, WORD_BYTES, out_lines);
}

// Build the output (D) cache-line request list for the current output tile.
void DTensorCore::build_out_req_lines_(std::vector<uint64_t>& out_lines) {
  out_lines.clear();

  constexpr uint32_t WORD_BYTES = 4;

  std::vector<uint64_t> out_addrs;
  out_addrs.reserve(tile_m_ * tile_n_);

  // D output (row_major)
  uint64_t baseD = calculate_base_D_();
  for (uint32_t m = 0; m < tile_m_; ++m) {
    for (uint32_t n = 0; n < tile_n_; ++n) {
      uint64_t addr = baseD + (uint64_t(m) * desc_.ldmD + n) * 4;
      out_addrs.push_back(addr);
    }
  }

  coalesce_to_lines(out_addrs, WORD_BYTES, out_lines);
}

// Start prefetching one K tile's operands (A/B and, on the first K tile, C) into
// the given buffer. Builds the cache-line request list and arms the TMA engine.
void DTensorCore::start_prefetch_(uint32_t buf_idx, uint32_t k_idx) {
  tma_target_buf_ = buf_idx;
  tma_k_ = k_idx;
  buf_ready_[buf_idx] = false;
  build_op_req_lines_(k_idx, tma_req_lines_);
  tma_req_idx_ = 0;
  total_op_reqs_ += tma_req_lines_.size();
  tma_state_ = TmaState::REQ;
}

// Advance the TMA prefetch engine by one cycle. Issues one operand cache-line
// request at a time (single outstanding); once all responses arrive it performs
// the functional buffer fill and marks the target buffer ready.
void DTensorCore::tick_tma_() {
  switch (tma_state_) {
  case TmaState::IDLE:
    break;
  case TmaState::REQ:
    if (tma_req_idx_ < tma_req_lines_.size()) {
      issue_mem_req_tma_(tma_req_lines_[tma_req_idx_], false);
      tma_state_ = TmaState::WAIT;
    } else {
      // All operand lines acknowledged: fill the buffer from RAM (functional).
      load_operands_into(tma_target_buf_, tma_k_);
      buf_ready_[tma_target_buf_] = true;
      tma_state_ = TmaState::IDLE;
    }
    break;
  case TmaState::WAIT:
    if (tma_pending_tag_ == 0) {
      ++tma_req_idx_;
      tma_state_ = TmaState::REQ;
    }
    break;
  }
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
    } else if (rsp.tag == tma_pending_tag_) {
      mem_rsp_in.pop();
      tma_pending_tag_ = 0;
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
          << " shape_n_size=" << uint32_t(desc_.shape_n_size) << " shape_policy=" << uint32_t(desc_.shape_policy) // N-dimension shape
          << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_ // Set Native Tile Size
          << std::endl;

      // Begin streaming: prefetch K0 of the first output tile into the compute buffer.
      tile_k_idx_ = 0;
      start_prefetch_(compute_buf_, 0);
      state_ = State::FIRST_LOAD;
    }
    break;

  case State::FIRST_LOAD:
    // Fill the current compute buffer (K0 of this output tile) before computing.
    tick_tma_();
    if (buf_ready_[compute_buf_]) {
      exec_cycles_left_ = estimate_execute_cycles_();
      compute_done_ = false;
      // Start prefetching the next K tile into the other buffer (overlap).
      if (tile_k_idx_ + 1 < tiles_k_) {
        start_prefetch_(compute_buf_ ^ 1, tile_k_idx_ + 1);
      }
      state_ = State::COMPUTE;
    }
    break;

  case State::COMPUTE:
    // Prefetch the next K tile concurrently with computing the current one.
    tick_tma_();

    if (exec_cycles_left_ > 0) {
      --exec_cycles_left_;
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
          start_prefetch_(compute_buf_ ^ 1, tile_k_idx_ + 1);
        }
      }
      // else: compute is waiting on the TMA prefetch (Phase 4 will count this).
    } else {
      // Last K tile done: build the output request list and store.
      build_out_req_lines_(out_req_lines_);
      total_out_reqs_ += out_req_lines_.size();
      out_req_idx_ = 0;
      buf_ready_[compute_buf_] = false;
      state_ = State::OUT_REQ;
    }
    break;

  case State::OUT_REQ:
    // Issue exactly one output write request, then wait for its response.
    if (issue_next_out_req_()) {
      state_ = State::OUT_WAIT;
    } else {
      pending_tag_ = 0;
      state_ = State::OUT_WAIT;
    }
    break;

  case State::OUT_WAIT:
    if (pending_tag_ == 0) {
      if (out_req_idx_ < out_req_lines_.size()) {
        // More output cache lines remain: issue next write request.
        state_ = State::OUT_REQ;
      } else {
        // All output write requests have completed.
        store_output();

        if (advance_output_tile_()) {
          // Next output tile: restart streaming from K0 (advance_output_tile_ reset tile_k_idx_).
          buf_ready_[0] = false;
          buf_ready_[1] = false;
          start_prefetch_(compute_buf_, 0);
          state_ = State::FIRST_LOAD;
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
    }
    break;

  case State::DONE:
    break;

  default:
    break;
  }
}
