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
#include "tcu/tcu_latency.h"
#include "tcu/tcu_fedp.h"
#include "constants.h"
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

Dtcu::Dtcu(const SimContext& ctx, const char* name, int engine)
  : SimObject<Dtcu>(ctx, name)
  , engine_(engine)
  , smem_n_stride_(dtcu_tile_n_max_of(engine))
  , smem_a_words_(dtcu_tile_m_of(engine) * DTCU_TILE_K_WORDS)
  , state_(State::IDLE)
  , busy_(false)
  , submitted_(0)
  , completed_(0)
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

// Per-GEMM state only. Everything a descriptor needs to start from a clean slate, and
// nothing that accumulates across descriptors -- the perf counters deliberately do NOT
// belong here, or a second submission would erase the first one's numbers.
void Dtcu::begin_descriptor_(uint64_t desc_addr) {
  desc_addr_ = desc_addr;
  // Clear stale descriptor: flags carries a mode bit read via tma_enabled_().
  std::memset(&desc_, 0, sizeof(desc_));
  state_ = State::DESC_REQ;
  busy_ = true;
  tma_->reset();
  shm_a_[0].clear();
  shm_a_[1].clear();
  shm_b_[0].clear();
  shm_b_[1].clear();
  accum_buf_[0].clear();
  accum_buf_[1].clear();
  accum_compute_idx_ = 0;
  compute_buf_ = 0;
  buf_ready_[0] = false;
  buf_ready_[1] = false;
  compute_done_ = false;
  next_tile_load_issued_ = false;
  next_tile_load_buf_ = 0;
  tile_m_ = 0;
  tile_n_ = 0;
  tile_k_ = 0;
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
  tiles_m_ = 1;
  tiles_n_ = 1;
  tiles_k_ = 1;
  exec_cycles_left_ = 0;
}

void Dtcu::on_reset() {
  desc_queue_.clear();
  submitted_ = 0;
  completed_ = 0;
  begin_descriptor_(0);
  state_ = State::IDLE;
  busy_  = false;
  // Counters live for the whole device lifetime, like every other MPM counter, so this
  // is the only place they are cleared.
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
  dtcu_desc_wait_cycles_ = 0;
  dtcu_busy_cycles_ = 0;
  tma_acc_init_cycles_ = 0;
  dtcu_instr_tcu_ = 0;
  total_op_reqs_ = 0;
  total_out_reqs_ = 0;
}

// Non-blocking submit. Returns 0 when the queue is full so the caller can retry
// instead of losing the descriptor: the old code dropped it with only a debug print,
// which in a multi-core setting meant a second core's GEMM silently never ran.
uint32_t Dtcu::start(uint64_t desc_addr) {
  if (busy_) {
    // Depth scales with the number of cores that can submit to this engine, which
    // differs between the two placements (dtcu_params.h).
    const uint32_t depth = (engine_ == DTCU_ENGINE_CLUSTER) ? DTCU_CLUSTER_QUEUE_DEPTH
                                                            : DTCU_SOCKET_QUEUE_DEPTH;
    if (desc_queue_.size() >= depth) {
      DP(2, this->name() << ": START rejected (queue full)");
      return 0;
    }
    desc_queue_.push_back(desc_addr);
    return ++submitted_;
  }
  begin_descriptor_(desc_addr);
  return ++submitted_;
}

uint32_t Dtcu::poll() const {
  return completed_;
}


void Dtcu::init_tile_state_() {
  uint32_t in_sz = vt::elem_size_bytes(desc_.fmt_s);

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

  // Unknown flag bits fail loudly: guards against a stale simulator silently
  // ignoring a mode bit (e.g. FLAG_NO_TMA) and mislabeling a measurement.
  if (desc_.flags & ~uint8_t(DTENSOR_FLAG_ALL)) {
    std::cout << "[DTCU] Error: Unknown descriptor flags: 0x" << std::hex << uint32_t(desc_.flags) << std::dec << std::endl;
    std::abort();
  }

  // Geometry comes from the shared contract (sw/common/dtcu_cfg.h), so the host
  // traits and this path cannot disagree: M is build-time fixed PER ENGINE, N is
  // descriptor-driven and bounded by THIS engine's capacity, K is fixed in words
  // (shared by both engines) and widened by the input element size.
  tile_m_ = dtcu_tile_m_of(engine_);
  tile_n_ = dtcu_tile_n(desc_.shape_n_size);
  tile_k_ = dtcu_tile_k(in_sz);

  // Each engine validates against its OWN bound, so a cluster-sized shape_n_size
  // handed to the socket engine is rejected here rather than silently truncated into
  // a narrower physical buffer.
  if (!dtcu_tile_n_valid_of(engine_, tile_n_)) {
    std::cout << "[DTCU] Error: N-dimension must be in multiples of " << DTCU_TILE_N_GRAN
              << "; maximum " << dtcu_tile_n_max_of(engine_) << " for the "
              << (engine_ == DTCU_ENGINE_CLUSTER ? "cluster" : "socket")
              << " engine. Received: " << tile_n_ << std::endl;
    std::abort();
  }

  if (desc_.M == 0 || desc_.N == 0 || desc_.K == 0) {
    std::cout << "[DTCU] Error: empty GEMM. M=" << desc_.M << ", N=" << desc_.N
              << ", K=" << desc_.K << std::endl;
    std::abort();
  }

  // Fixed-size SRAM buffers, sized for the largest legal tile (Hopper-style fixed
  // SMEM capacity); a smaller tile_n_ uses only the leading prefix. Sizing is
  // independent of the descriptor so the physical buffer (and later its banks) is a
  // constant, not resized per GEMM. Compute still indexes with tile_m_/tile_n_.
  shm_a_[0].assign(smem_a_words_, 0);
  shm_a_[1].assign(smem_a_words_, 0);
  shm_b_[0].assign(DTCU_TILE_K_WORDS * smem_n_stride_, 0);
  shm_b_[1].assign(DTCU_TILE_K_WORDS * smem_n_stride_, 0);
  accum_buf_[0].assign(tile_m_ * smem_n_stride_, 0.0f);
  accum_buf_[1].assign(tile_m_ * smem_n_stride_, 0.0f);

  // The two invariants the operand SRAM depends on. Cheap, and they turn the
  // silent-corruption failure mode above into an immediate abort.
  assert(tile_n_ <= smem_n_stride_);
  assert(tile_m_ * DTCU_TILE_K_WORDS <= smem_a_words_);

  // Tiles needed to COVER the GEMM: round up, so M/N/K need not be tile multiples.
  // The trailing tile of each axis is partial and its out-of-matrix coordinates are
  // clamped out of the operand fetch and masked out of the D store by the TMA engine
  // (dtcu_tma.cpp, "ragged edges"). Padding is not the caller's job.
  tiles_m_ = (desc_.M + tile_m_ - 1) / tile_m_;
  tiles_n_ = (desc_.N + tile_n_ - 1) / tile_n_;
  tiles_k_ = (desc_.K + tile_k_ - 1) / tile_k_;

  // Initialize tile indices to start from the first tile. total_op_reqs_/total_out_reqs_
  // are NOT cleared here: they are perf counters and accumulate across every descriptor
  // in the launch, like the cycle counters.
  tile_m_idx_ = 0;
  tile_n_idx_ = 0;
  tile_k_idx_ = 0;
}

bool Dtcu::advance_output_tile_() {
  tile_k_idx_ = 0;
  return next_tile_coord_(tile_m_idx_, tile_n_idx_, tiles_m_, tiles_n_);
}

// Advance to the next output tile and kick its K0 fetch, or go drain the final
// store. Shared by the overlap (TILE_STORE) and blocking (TILE_STORE_BLOCK) paths.
void Dtcu::start_next_tile_or_drain_() {
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
}

namespace { constexpr uint32_t ct_log2(uint32_t x) { return x <= 1 ? 0 : 1 + ct_log2(x >> 1); } }

// Bank of a physical word index in the unified operand SRAM (A region then B region).
// Vortex MemCrossBar word-granular interleave: bank = word & (banks-1).
// With DTCU_SWIZZLE, XOR-permute the bank select by folding in the high bits (the
// row/K index, which for a B column lives above log2(smem_n_stride_)). A column read
// (stride smem_n_stride_) then maps to distinct banks instead of aliasing to one --
// the Hopper-TMA swizzle. Same map at fill+read in HW, so functional values are
// unchanged; here it only changes the timing bank distribution.
uint32_t Dtcu::bank_of_(uint32_t phys_word) const {
#if DTCU_SWIZZLE
  phys_word ^= (phys_word >> ct_log2(smem_n_stride_));
#endif
  return phys_word & (DTCU_SMEM_BANKS - 1);
}

// Operand-SRAM read cycles for one K tile, M2 (reuse-aware): the array reads each
// A-row once and each B-col once. A bank serves 1 word/cycle (MemCrossBar rule), so a
// K-word operand vector takes (max words landing on one bank) cycles -- conflict-free
// = 1. A is stride-1 (spreads across banks); B is stride smem_n_stride_ (column read,
// the conflict site). This is the deterministic delivery time the crossbar would
// produce for this static read set (computed directly, same bank rule).
uint32_t Dtcu::operand_read_cycles_() const {
  const uint32_t Kw     = DTCU_TILE_K_WORDS;
  const uint32_t A_SIZE = smem_a_words_; // A region size (B starts here), per-engine
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
  // B-cols: physical word = A_SIZE + kw*smem_n_stride_ + n (stride smem_n_stride_)
  for (uint32_t n = 0; n < tile_n_; ++n) {
    hist.fill(0);
    uint32_t mx = 0;
    for (uint32_t kw = 0; kw < Kw; ++kw)
      mx = std::max(mx, uint32_t(++hist[bank_of_(A_SIZE + kw * smem_n_stride_ + n)]));
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
  //  (1) MAC: this engine's MACS_PER_CYCLE over tile_m*tile_n*tile_k MACs.
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
  // Array width is per-engine: the cluster engine is twice the socket engine's.
  const uint32_t macs_per_cycle = (engine_ == DTCU_ENGINE_CLUSTER)
                                ? uint32_t(DTCU_CLUSTER_MACS_PER_CYCLE)
                                : uint32_t(DTCU_SOCKET_MACS_PER_CYCLE);
  const uint64_t tile_macs    = uint64_t(tile_m_) * tile_n_ * tile_k_;
  const uint64_t mac_cycles   = (tile_macs + macs_per_cycle - 1) / macs_per_cycle;
  const uint32_t read_cycles  = operand_read_cycles_() + DTCU_BUF_LATENCY;
  const uint64_t accum_words  = 2ull * tile_m_ * tile_n_; // read partial + write updated
  const uint64_t accum_cycles = (accum_words + DTCU_ACC_BANKS - 1) / DTCU_ACC_BANKS + DTCU_ACC_LATENCY;
  dtcu_smem_read_model_cycles_ += read_cycles; // report (swizzle on/off comparison)
  // Fill/drain depth follows the in-core TCU's, derived from VX_CFG_TCU_TYPE -- same
  // arithmetic, same pipeline, different place. See tcu/tcu_latency.h.
#ifdef DTCU_COMPUTE_LATENCY
  constexpr uint32_t kFillLatency = DTCU_COMPUTE_LATENCY;   // explicit -D override
#else
  constexpr uint32_t kFillLatency = vortex::tcu_timing::kMmaLatency;
#endif
  const uint64_t compute = std::max<uint64_t>({mac_cycles, read_cycles, accum_cycles}) + kFillLatency;
  return std::max(1u, uint32_t(compute));
}


// The PE arithmetic comes from tcu/tcu_fedp.h -- the SAME FMA and FEDP the in-core TCU
// uses. This file used to carry its own copy, taken from tensor_unit.cpp and then left
// behind: its FMA<fp16,fp32> passed the accumulator as `float` where the TCU passes
// uint32_t, and its FEDP chained C through every multiply-add where the TCU sums the
// products first and adds C once. The engine was quietly a different numerical machine
// from the core it is compared against. See that header.
using namespace vortex::tcu_pe;


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
          b_words[z].u32 = shm_b_[buf_idx][(kw + z) * smem_n_stride_ + n]; // per-engine physical row stride
        }

        acc_bit = fedp(a_words.data(), b_words.data(), acc_bit);
        ++dtcu_instr_tcu_; // one FEDP == one in-core-TCU MMA op (VX_CSR_MPM_INSTR_TCU)
      }

      std::memcpy(&accum_buf_[accum_compute_idx_][m * tile_n_ + n], &acc_bit, 4);
    }
  }
}

void Dtcu::on_tick() {
  if (busy_) ++dtcu_busy_cycles_; // accounting anchor: MCYCLE - busy = kernel-side

  // The TMA engine owns the L2 port: let it retire all responses this cycle.
  tma_->drain_responses();

  switch (state_) {
  case State::IDLE:
    break;

  case State::DESC_REQ:
    ++dtcu_desc_wait_cycles_;
    // Retry rather than send blind: SimChannel::send() asserts on a full channel, and
    // the socket engines share one L2 read port, so another engine's traffic can be
    // occupying it. The cluster engine owns its port outright and never retries, which
    // is why this is numerically inert there.
    if (tma_->issue_desc_req(desc_addr_)) // Read descriptor
      state_ = State::DESC_WAIT;
    break;

  case State::DESC_WAIT:
    if (!tma_->main_done()) {
      ++dtcu_desc_wait_cycles_;
    } else {
      tma_->read_desc(desc_addr_);
      init_tile_state_();

      // Descriptor dump (debug only; the same info is derivable from the MPM
      // class-DTCU counters + the host-side descriptor).
      DP(2, "[DTCU] ptrA=0x" << std::hex << desc_.ptrA << " ptrB=0x" << desc_.ptrB << " ptrC=0x" << desc_.ptrC << " ptrD=0x" << desc_.ptrD //pointer
          << std::dec << " ldmA=" << desc_.ldmA << " ldmB=" << desc_.ldmB << " ldmC=" << desc_.ldmC << " ldmD=" << desc_.ldmD // leading dimension
          << " M=" << desc_.M << " N=" << desc_.N << " K=" << desc_.K // matrix size
          << " fmt_s=" << uint32_t(desc_.fmt_s) << " fmt_d=" << uint32_t(desc_.fmt_d) << " flags=" << uint32_t(desc_.flags) // metadata
          << " shape_n_size=" << uint32_t(desc_.shape_n_size) << " shape_policy=" << uint32_t(desc_.shape_policy) // N-dimension shape
          << " tileM=" << tile_m_ << " tileN=" << tile_n_ << " tileK=" << tile_k_ // Set Native Tile Size
          << " engine=" << (engine_ == DTCU_ENGINE_CLUSTER ? "cluster" : "socket") // placement variant
          << " tma=" << (tma_enabled_() ? "on" : "off")); // overlap mode (FLAG_NO_TMA)

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
      // NO_TMA: no early kick; the fetch is deferred to consume time in COMPUTE.
      if (tma_enabled_() && tile_k_idx_ + 1 < tiles_k_) {
        tma_->start_prefetch(compute_buf_ ^ 1, tile_m_idx_, tile_n_idx_, tile_k_idx_ + 1, accum_compute_idx_);
      }
      state_ = State::COMPUTE;
    } else {
      ++dtcu_next_tile_load_stall_cycles_; // exposed K0 wait (both modes)
    }
    break;

  case State::COMPUTE:
    // Prefetch the next K tile concurrently with computing the current one.
    tma_->tick();

    // Cross-tile lookahead (overlap mode only): the load channel idles during the
    // last K tile's compute -- issue the next tile's K0 so it hides under compute.
    if (tma_enabled_() && !next_tile_load_issued_
        && tile_k_idx_ + 1 == tiles_k_ && tma_->load_idle()) {
      uint32_t nm = tile_m_idx_, nn = tile_n_idx_; // peek via the shared walk
      if (next_tile_coord_(nm, nn, tiles_m_, tiles_n_)) {
        next_tile_load_buf_ = compute_buf_ ^ 1; // freed at the swap into the last K
        // C-preload into accum_buf_[^idx] is safe: start_store() snapshots its
        // payload synchronously (breaks if the store ever goes zero-copy).
        tma_->start_prefetch(next_tile_load_buf_, nm, nn, 0, accum_compute_idx_ ^ 1);
        next_tile_load_issued_ = true;
      }
    }

    // Load channel starved (overlap mode): idle with the other buffer already
    // filled, yet more fetchable work exists -- a further K tile of this tile,
    // or the next tile's unissued K0. The headroom a third buffer would unlock.
    if (tma_enabled_() && tma_->load_idle() && buf_ready_[compute_buf_ ^ 1]) {
      uint32_t sm = tile_m_idx_, sn = tile_n_idx_;
      if ((tile_k_idx_ + 2 < tiles_k_)
          || (!next_tile_load_issued_ && next_tile_coord_(sm, sn, tiles_m_, tiles_n_)))
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
        // Kick prefetch of the following K tile (overlap mode only).
        if (tma_enabled_() && tile_k_idx_ + 1 < tiles_k_) {
          tma_->start_prefetch(compute_buf_ ^ 1, tile_m_idx_, tile_n_idx_, tile_k_idx_ + 1, accum_compute_idx_);
        }
      } else {
        // NO_TMA: fetch at consume time; load_idle() makes the kick fire once.
        if (!tma_enabled_() && tma_->load_idle()) {
          tma_->start_prefetch(next_buf, tile_m_idx_, tile_n_idx_, tile_k_idx_ + 1, accum_compute_idx_);
        }
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
    if (!tma_enabled_()) {
      // NO_TMA: serialize the store. The next tile's K0 kick moves after the
      // drain -- the load channel has port priority and would starve the store.
      state_ = State::TILE_STORE_BLOCK;
      break;
    }
    start_next_tile_or_drain_();
    break;

  case State::TILE_STORE_BLOCK:
    // NO_TMA blocking mode: drain this tile's D store fully before the next tile
    // (pre-TMA OUT_WAIT model). The drain measures store issue + acc-SRAM read
    // occupancy (TLM stores are fire-and-forget, no memory response to wait on).
    tma_->tick();
    if (tma_->store_active()) {
      ++dtcu_store_drain_cycles_;
      break;
    }
    // Final-tile exit enters FINAL_TILE_STORE with the store idle: epilogue only.
    start_next_tile_or_drain_();
    break;

  case State::FINAL_TILE_STORE:
    // Final output tile: drain its background store before reporting done.
    tma_->tick();
    if (tma_->store_active()) {
      ++dtcu_store_drain_cycles_; // store not fully hidden under compute
      break;
    }
    {
      // Every D line has been acknowledged by this point (store_active() is
      // response-based), so the data is visible before completion is announced.
      // Same shared-port retry as DESC_REQ: stay in this state until the flag store
      // has actually been accepted, or the GEMM would be marked complete without it.
      // Attempted before the summary prints so a retry does not repeat them.
      if (!tma_->issue_done_flag(desc_addr_))
        break;

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
                << ", store_drain=" << dtcu_store_drain_cycles_
                << ", desc_wait=" << dtcu_desc_wait_cycles_
                << ", busy=" << dtcu_busy_cycles_);
      // Engine family: concurrent with COMPUTE -- never add to the FSM values.
      DP(2, "[DTCU] tma cycles: tma_mem_wait=" << tma_mem_wait_cycles_
                << ", tma_buf_starve=" << tma_buf_starve_cycles_
                << ", tma_op_fill=" << tma_op_fill_cycles_
                << ", tma_acc_init=" << tma_acc_init_cycles_
                << ", tma_addrgen=" << tma_addrgen_cycles_
                << ", tma_store_issue_stall=" << tma_store_issue_stall_cycles_
                << ", smem_read_model=" << dtcu_smem_read_model_cycles_);

      ++completed_;
      if (!desc_queue_.empty()) {
        uint64_t next = desc_queue_.front();
        desc_queue_.pop_front();
        begin_descriptor_(next); // straight into the next GEMM, no idle gap
      } else {
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
