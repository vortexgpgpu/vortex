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
using cfg = vt::wmma_config_t<VX_CFG_NUM_THREADS>;

namespace {

constexpr uint32_t DTCU_TILE_K_WORDS = 8;

} // namespace

Dtcu::Dtcu(const SimContext& ctx, const char* name, Cluster* cluster)
  : SimObject<Dtcu>(ctx, name)
  , cluster_(cluster)
  , state_(State::IDLE)
  , busy_(false)
  , done_(false)
  , desc_addr_(0)
  , desc_{}
  , shm_a_()
  , shm_b_()
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

void Dtcu::on_reset() {
  state_ = State::IDLE;
  busy_  = false;
  done_  = false;
  desc_addr_ = 0;
  std::memset(&desc_, 0, sizeof(desc_));
  tma_->reset();
  shm_a_[0].clear();
  shm_a_[1].clear();
  shm_b_[0].clear();
  shm_b_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  next_tile_load_issued_ = false;
  next_tile_load_buf_ = 0;
  dtcu_compute_cycles_ = 0;
  dtcu_next_k_load_stall_cycles_ = 0;
  tma_mem_wait_cycles_ = 0;
  tma_buf_starve_cycles_ = 0;
  tma_op_fill_cycles_ = 0;
  tma_addrgen_cycles_ = 0;
  tma_store_issue_stall_cycles_ = 0;
  dtcu_store_drain_cycles_ = 0;
  dtcu_smem_read_model_cycles_ = 0;
  dtcu_next_tile_load_stall_cycles_ = 0;
  dtcu_prev_tile_store_stall_cycles_ = 0;
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
  shm_a_[0].clear();
  shm_a_[1].clear();
  shm_b_[0].clear();
  shm_b_[1].clear();
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  next_tile_load_issued_ = false;
  next_tile_load_buf_ = 0;
  dtcu_compute_cycles_ = 0;
  dtcu_next_k_load_stall_cycles_ = 0;
  tma_mem_wait_cycles_ = 0;
  tma_buf_starve_cycles_ = 0;
  tma_op_fill_cycles_ = 0;
  tma_addrgen_cycles_ = 0;
  tma_store_issue_stall_cycles_ = 0;
  dtcu_store_drain_cycles_ = 0;
  dtcu_smem_read_model_cycles_ = 0;
  dtcu_next_tile_load_stall_cycles_ = 0;
  dtcu_prev_tile_store_stall_cycles_ = 0;
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
    case vt::fp8::id:   return 1;
    case vt::bf8::id:   return 1;
    case vt::tf32::id:  return 4;
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

  // Output/accumulation: fp32 (T1 sources) or int32 (T2 integer sources). Narrow
  // outputs (fp16/bf16/...) need out_sz-aware C-load/store packing — not yet (T3).
  if (desc_.fmt_d != vt::fp32::id && desc_.fmt_d != vt::int32::id) {
    std::cout << "[DTCU] Error: Only supports fp32/int32 output/accumulation" << std::endl;
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

  // Partial tiles are not supported, by design: like the in-core WMMA unit (which only
  // computes fixed-shape fragments), ragged edges are the caller's responsibility — the
  // kernel/host pads M/N/K up to the tile size before issuing the descriptor.
  if ((desc_.M % tile_m_) != 0 || (desc_.N % tile_n_) != 0 || (desc_.K % tile_k_) != 0) {
    std::cout << "[DTCU] Error: Partial Tile not supported. M/N/K must be multiples of tile size. "
              << "M=" << desc_.M << ", N=" << desc_.N << ", K=" << desc_.K
              << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_ << std::endl;
    std::abort();
  }

  // Fixed-size SRAM buffers, sized for the largest legal tile (Hopper-style fixed
  // SMEM capacity); a smaller tile_n_ uses only the leading prefix. Sizing is
  // independent of the descriptor so the physical buffer (and later its banks) is a
  // constant, not resized per GEMM. Compute still indexes with tile_m_/tile_n_.
  shm_a_[0].assign(DTCU_TILE_M * DTCU_TILE_K_WORDS, 0);
  shm_a_[1].assign(DTCU_TILE_M * DTCU_TILE_K_WORDS, 0);
  shm_b_[0].assign(DTCU_TILE_K_WORDS * DTCU_TILE_N_MAX, 0);
  shm_b_[1].assign(DTCU_TILE_K_WORDS * DTCU_TILE_N_MAX, 0);
  accum_buf_[0].assign(DTCU_TILE_M * DTCU_TILE_N_MAX, 0.0f);
  accum_buf_[1].assign(DTCU_TILE_M * DTCU_TILE_N_MAX, 0.0f);

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
  return next_tile_coord_(tile_m_idx_, tile_n_idx_, tiles_m_, tiles_n_);
}

namespace { constexpr uint32_t ct_log2(uint32_t x) { return x <= 1 ? 0 : 1 + ct_log2(x >> 1); } }

// Bank of a physical word index in the unified operand SRAM (A region then B region).
// Vortex MemCrossBar word-granular interleave: bank = word & (banks-1).
// With DTCU_SWIZZLE, XOR-permute the bank select by folding in the high bits (the
// row/K index, which for a B column lives above log2(DTCU_TILE_N_MAX)). A column read
// (stride DTCU_TILE_N_MAX) then maps to distinct banks instead of aliasing to one --
// the Hopper-TMA swizzle. Same map at fill+read in HW, so functional values are
// unchanged; here it only changes the timing bank distribution.
uint32_t Dtcu::bank_of_(uint32_t phys_word) const {
#if DTCU_SWIZZLE
  phys_word ^= (phys_word >> ct_log2(DTCU_TILE_N_MAX));
#endif
  return phys_word & (DTCU_SMEM_BANKS - 1);
}

// Operand-SRAM read cycles for one K tile, M2 (reuse-aware): the array reads each
// A-row once and each B-col once. A bank serves 1 word/cycle (MemCrossBar rule), so a
// K-word operand vector takes (max words landing on one bank) cycles -- conflict-free
// = 1. A is stride-1 (spreads across banks); B is stride DTCU_TILE_N_MAX (column read,
// the conflict site). This is the deterministic delivery time the crossbar would
// produce for this static read set (computed directly, same bank rule).
uint32_t Dtcu::operand_read_cycles_() const {
  const uint32_t Kw     = DTCU_TILE_K_WORDS;
  const uint32_t A_SIZE = DTCU_TILE_M * DTCU_TILE_K_WORDS; // A region size (B starts here)
  std::array<uint16_t, 64> hist{};
  uint32_t total = 0;
  // A-rows: physical word = m*Kw + kw (stride 1)
  for (uint32_t m = 0; m < tile_m_; ++m) {
    hist.fill(0);
    uint32_t mx = 0;
    for (uint32_t kw = 0; kw < Kw; ++kw)
      mx = std::max(mx, uint32_t(++hist[bank_of_(m * Kw + kw)]));
    total += mx;
  }
  // B-cols: physical word = A_SIZE + kw*DTCU_TILE_N_MAX + n (stride DTCU_TILE_N_MAX)
  for (uint32_t n = 0; n < tile_n_; ++n) {
    hist.fill(0);
    uint32_t mx = 0;
    for (uint32_t kw = 0; kw < Kw; ++kw)
      mx = std::max(mx, uint32_t(++hist[bank_of_(A_SIZE + kw * DTCU_TILE_N_MAX + n)]));
    total += mx;
  }
  return total;
}

uint32_t Dtcu::estimate_execute_cycles_() {
  // Compute-phase latency for one K tile, modeled as a 3-stage pipeline over the tile's
  // tile_m*tile_n output elements: (1) operand read -> (2) MAC/FEDP -> (3) accumulator
  // read-modify-write. FEDP fuses the accumulate (acc is the third MAC operand), so the
  // accumulator writeback streams alongside the MACs rather than running as a serial
  // phase after them. Pipeline throughput is therefore the SLOWEST stage, not the sum:
  //   cost = max(mac, read, accum) + COMPUTE_LATENCY (fill/drain).
  //  (1) MAC: fixed DTCU_MACS_PER_CYCLE MAC/cycle over tile_m*tile_n*tile_k MACs.
  //  (2) operand read: operand_read_cycles_() bank-conflict throughput (M2) PLUS
  //      DTCU_BUF_LATENCY base access latency (L1 dcache read-latency model);
  //      operand read and fill hit the same scratchpad SRAM, so they share it.
  //  (3) accumulator R/W: 2*tile_m*tile_n words at the accumulator SRAM rate
  //      (DTCU_ACC_BANKS) -- a separate SRAM from the operand scratchpad, no conflict.
  //      A 2-bank single-ported acc SRAM sustains one element RMW/cycle (1 read + 1
  //      write over 2 banks), so a 2048-element tile floors at ~2048 cycles regardless
  //      of MAC width -- a real bandwidth limit, not an optimistic hide.
  // Overlap assumes the acc SRAM ports are independent of the operand-read path (they
  // are: separate SRAM). If a future RTL shares a port between the two, stage (3) can no
  // longer overlap (1)/(2) and this must revert toward an additive term.
  // The functional execute_mma() stays the value oracle; this only models timing.
  const uint64_t tile_macs    = uint64_t(tile_m_) * tile_n_ * tile_k_;
  const uint64_t mac_cycles   = (tile_macs + DTCU_MACS_PER_CYCLE - 1) / DTCU_MACS_PER_CYCLE;
  const uint32_t read_cycles  = operand_read_cycles_() + DTCU_BUF_LATENCY;
  const uint64_t accum_words  = 2ull * tile_m_ * tile_n_; // read partial + write updated
  const uint64_t accum_cycles = (accum_words + DTCU_ACC_BANKS - 1) / DTCU_ACC_BANKS + DTCU_ACC_LATENCY;
  dtcu_smem_read_model_cycles_ += read_cycles; // report (swizzle on/off comparison)
  const uint64_t compute = std::max<uint64_t>({mac_cycles, read_cycles, accum_cycles}) + DTCU_COMPUTE_LATENCY;
  return std::max(1u, uint32_t(compute));
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

// NOTE (v3.0 port): microscaling formats (mxfp8/nvfp4) dropped — not in v3.0
// tensor_cfg.h / rvfloats. DTCU supports the v3.0 set: fp16/bf16/fp8/bf8/tf32->fp32,
// int8/uint8/int4/uint4->int32, fp32->fp32.

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
          a_words[z].u32 = shm_a_[buf_idx][m * DTCU_TILE_K_WORDS + kw + z];
          b_words[z].u32 = shm_b_[buf_idx][(kw + z) * DTCU_TILE_N_MAX + n]; // fixed physical row stride
        }

        acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);
      }

      std::memcpy(&accum_buf_[accum_compute_idx_][m * tile_n_ + n], &acc_bit, 4);
    }
  }
}

void Dtcu::on_tick() {
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

      // Descriptor dump (debug only; the same info is derivable from the MPM
      // class-DTCU counters + the host-side descriptor).
      DP(2, "[DTCU] ptrA=0x" << std::hex << desc_.ptrA << " ptrB=0x" << desc_.ptrB << " ptrC=0x" << desc_.ptrC << " ptrD=0x" << desc_.ptrD //pointer
          << std::dec << " ldmA=" << desc_.ldmA << " ldmB=" << desc_.ldmB << " ldmC=" << desc_.ldmC << " ldmD=" << desc_.ldmD // leading dimension
          << " M=" << desc_.M << " N=" << desc_.N << " K=" << desc_.K // matrix size
          << " fmt_s=" << uint32_t(desc_.fmt_s) << " fmt_d=" << uint32_t(desc_.fmt_d) << " flags=" << uint32_t(desc_.flags) // metadata
          << " shape_n_size=" << uint32_t(desc_.shape_n_size) << " shape_policy=" << uint32_t(desc_.shape_policy) // N-dimension shape
          << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_); // Set Native Tile Size

      // Begin streaming: prefetch K0 of the first output tile into the compute buffer.
      tile_k_idx_ = 0;
      tma_->start_prefetch(compute_buf_, tile_m_idx_, tile_n_idx_, 0, accum_compute_idx_);
      state_ = State::NEXT_TILE_LOAD;
    }
    break;

  case State::NEXT_TILE_LOAD:
    // Fill the current compute buffer (K0 of this output tile) before computing.
    tma_->tick();
    if (buf_ready_[compute_buf_]) {
      exec_cycles_left_ = estimate_execute_cycles_();
      compute_done_ = false;
      // Start prefetching the next K tile into the other buffer (overlap).
      if (tile_k_idx_ + 1 < tiles_k_) {
        tma_->start_prefetch(compute_buf_ ^ 1, tile_m_idx_, tile_n_idx_, tile_k_idx_ + 1, accum_compute_idx_);
      }
      state_ = State::COMPUTE;
    } else {
      ++dtcu_next_tile_load_stall_cycles_; // exposed K0 fetch wait
    }
    break;

  case State::COMPUTE:
    // Prefetch the next K tile concurrently with computing the current one.
    tma_->tick();

    // Cross-tile lookahead: the load channel idles during the last K tile's
    // compute -- issue the next tile's K0 now so its fetch hides under compute.
    if (!next_tile_load_issued_ && tile_k_idx_ + 1 == tiles_k_ && tma_->load_idle()) {
      uint32_t nm = tile_m_idx_, nn = tile_n_idx_; // peek via the shared walk
      if (next_tile_coord_(nm, nn, tiles_m_, tiles_n_)) {
        next_tile_load_buf_ = compute_buf_ ^ 1; // freed at the swap into the last K
        // C-preload into accum_buf_[^idx] is safe: start_store() snapshots its
        // payload synchronously (breaks if the store ever goes zero-copy).
        tma_->start_prefetch(next_tile_load_buf_, nm, nn, 0, accum_compute_idx_ ^ 1);
        next_tile_load_issued_ = true;
      }
    }

    // Prefetch is done-ahead but blocked: the next buffer is filled and a further
    // K tile exists, yet no buffer is free until the current compute consumes one.
    if (tma_->load_idle() && buf_ready_[compute_buf_ ^ 1]
        && (tile_k_idx_ + 2 < tiles_k_)) {
      ++tma_buf_starve_cycles_;
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
          tma_->start_prefetch(compute_buf_ ^ 1, tile_m_idx_, tile_n_idx_, tile_k_idx_ + 1, accum_compute_idx_);
        }
      } else {
        // Compute finished but the next operand tile is not ready yet.
        ++dtcu_next_k_load_stall_cycles_;
      }
    } else {
      // Last K tile of this output tile done: hand off the D store and move on.
      buf_ready_[compute_buf_] = false;
      state_ = State::TILE_STORE;
    }
    break;

  case State::TILE_STORE:
    // Single store channel: wait for the previous tile's store to drain, then hand
    // off this tile's store and immediately start the next tile (overlap).
    tma_->tick(); // progress any in-flight (previous) store
    if (tma_->store_active()) {
      ++dtcu_prev_tile_store_stall_cycles_; // handoff stalled on the prior store
      break;
    }
    // Hand off the just-computed tile's store; the kick carries its coordinates.
    tma_->start_store(accum_compute_idx_, tile_m_idx_, tile_n_idx_);
    if (advance_output_tile_()) {
      accum_compute_idx_ ^= 1; // next tile computes into the other accumulator buffer
      if (next_tile_load_issued_) {
        // Lookahead already fetching/fetched this tile's K0: adopt its buffer
        // (keep its buf_ready_; do NOT re-kick K0 -- traffic must not change).
        compute_buf_ = next_tile_load_buf_;
        buf_ready_[next_tile_load_buf_ ^ 1] = false;
        next_tile_load_issued_ = false;
      } else {
        buf_ready_[0] = false;
        buf_ready_[1] = false;
        // tile_k_idx_ already reset (and m/n advanced) by advance_output_tile_
        tma_->start_prefetch(compute_buf_, tile_m_idx_, tile_n_idx_, 0, accum_compute_idx_);
      }
      state_ = State::NEXT_TILE_LOAD;
    } else {
      state_ = State::FINAL_TILE_STORE;
    }
    break;

  case State::FINAL_TILE_STORE:
    // Final output tile: drain its background store before reporting done.
    tma_->tick();
    if (tma_->store_active()) {
      ++dtcu_store_drain_cycles_; // store not fully hidden under compute
      break;
    }
    {
      // Summary counters (debug only; canonical readout is MPM class DTCU via
      // vx_mpm_query -- see VX_CSR_MPM_DTCU_*). Labels match the CSR names.
      DP(2, "[DTCU] L2 lines: mem_reqs=" << (total_op_reqs_ + total_out_reqs_)
                << " (op=" << total_op_reqs_ << ", out=" << total_out_reqs_
                << "), +1 desc line excluded");

      // FSM family: mutually exclusive states -- these sum to the busy timeline.
      DP(2, "[DTCU] fsm cycles: compute=" << dtcu_compute_cycles_
                << ", next_k_load_stall=" << dtcu_next_k_load_stall_cycles_
                << ", next_tile_load_stall=" << dtcu_next_tile_load_stall_cycles_
                << ", prev_tile_store_stall=" << dtcu_prev_tile_store_stall_cycles_
                << ", store_drain=" << dtcu_store_drain_cycles_);
      // Engine family: concurrent with COMPUTE -- never add to the FSM values.
      DP(2, "[DTCU] tma cycles: tma_mem_wait=" << tma_mem_wait_cycles_
                << ", tma_buf_starve=" << tma_buf_starve_cycles_
                << ", tma_op_fill=" << tma_op_fill_cycles_
                << ", tma_addrgen=" << tma_addrgen_cycles_
                << ", tma_store_issue_stall=" << tma_store_issue_stall_cycles_
                << ", smem_read_model=" << dtcu_smem_read_model_cycles_);

      done_ = true;
      busy_ = false;
      state_ = State::DONE;
    }
    break;

  case State::DONE:
    break;

  default:
    break;
  }
}
