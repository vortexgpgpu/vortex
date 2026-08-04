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

#include <simobject.h>
#include <vector>
#include <array>
#include <memory>

#include "dtcu_cfg.h" // dtensor_desc_t + DTENSOR_FLAG_* (shared with host/device)

namespace vortex {

class Cluster;
class DtcuTma;

// Dtcu: the disaggregated tensor core compute engine. It owns the GEMM control
// FSM, the operand/accumulator scratchpad, and the MMA datapath. All memory
// movement (descriptor fetch, operand prefetch, output store) is delegated to the
// DtcuTma engine, which owns the L2 port. The two cooperate via the back-reference
// DtcuTma holds to this object (friend below).
class Dtcu : public SimObject<Dtcu> {
public:
  // The engine reads the descriptor straight out of device memory, so its layout is
  // an ABI shared with the host and the device kernel: one definition, in
  // sw/common/dtcu_cfg.h. This alias keeps the Dtcu::Desc spelling used throughout
  // the model.
  using Desc = dtensor_desc_t;

  // Descriptor flag bits (Desc::flags), aliased from the shared header.
  static constexpr uint8_t FLAG_ZERO_ACC = DTENSOR_FLAG_ZERO_ACC; // zero-accumulate (skip C preload)
  static constexpr uint8_t FLAG_NO_TMA   = DTENSOR_FLAG_NO_TMA;   // blocking mode: no operand-prefetch / store overlap

  Dtcu(const SimContext& ctx, const char* name, Cluster* cluster);

  ~Dtcu();

  // The TMA engine owns the L2 memory port; the Cluster binds it via tma().
  DtcuTma* tma() const { return tma_.get(); }

  void start(uint64_t desc_addr);

  uint32_t poll() const;

  // Perf counters surfaced to MPM CSRs (cluster-level engine; see claude_doc/DTCU Perf Stat).
  // Three families -- only same-family cycle counters may be summed:
  //  FSM (mutually-exclusive states, tile the busy timeline): compute, *_stall, store_drain
  //  engine (tma_*: run concurrently with COMPUTE -- never add to FSM values)
  //  model (smem_read_model: a component INSIDE compute -- never add to anything)
  // op_reqs/out_reqs are coalesced L2 cache-line counts, not cycles.
  struct PerfStats {
    uint64_t op_reqs;               // operand (A/B + K0's C) L2 cache-line requests
    uint64_t out_reqs;              // output (D) L2 cache-line requests
    uint64_t compute;               // compute-pipeline occupancy: max(mac, smem read, acc RMW) + latency
    uint64_t next_k_load_stall;     // FSM: compute stalled on the next K tile's operands (intra-tile)
    uint64_t tma_mem_wait;          // engine: load channel waited on memory (latency + throttle + contention)
    uint64_t tma_buf_starve;        // engine: load channel idle, work available, no free operand buffer
    uint64_t tma_op_fill;           // engine: buffer fill cycles (operand + acc init; split in a later step)
    uint64_t tma_addrgen;           // engine: AGU setup (fixed cost per kick)
    uint64_t tma_store_issue_stall; // engine: store line issue blocked (port yielded to loads / queue)
    uint64_t store_drain;           // FSM: exposed store drain (overlap: final tile; blocking: every tile)
    uint64_t smem_read_model;       // model: operand-SRAM read estimate, contained in compute
    uint64_t next_tile_load_stall;  // FSM: exposed K0 wait incl. tile 0 cold start
    uint64_t prev_tile_store_stall; // FSM: store handoff blocked by the previous tile's store
    uint64_t desc_wait;             // FSM: descriptor fetch window (DESC_REQ + DESC_WAIT)
    uint64_t busy;                  // total busy ticks; MCYCLE - busy = kernel-side overhead
    uint64_t tma_acc_init;          // engine: accumulator init on K0 fill (C-preload / zero-fill)
    // MMA ops issued to the matrix array, counted at the same granularity as the
    // in-core TCU's VX_CSR_MPM_INSTR_TCU: one per FEDP (the shared cfg::tcK-wide
    // dot-product primitive both units use), so the two are directly comparable.
    uint64_t instr_tcu;
  };
  PerfStats perf_stats() const {
    return PerfStats{ total_op_reqs_, total_out_reqs_, dtcu_compute_cycles_,
      dtcu_next_k_load_stall_cycles_, tma_mem_wait_cycles_, tma_buf_starve_cycles_,
      tma_op_fill_cycles_, tma_addrgen_cycles_, tma_store_issue_stall_cycles_,
      dtcu_store_drain_cycles_, dtcu_smem_read_model_cycles_,
      dtcu_next_tile_load_stall_cycles_, dtcu_prev_tile_store_stall_cycles_,
      dtcu_desc_wait_cycles_, dtcu_busy_cycles_, tma_acc_init_cycles_,
      dtcu_instr_tcu_ };
  }

protected:
  // v3.0 SimObject lifecycle (auto-driven by SimPlatform). Must stay protected.
  void on_reset();
  void on_tick();
  friend class SimObject<Dtcu>;

private:
  friend class DtcuTma; // TMA engine reaches scratchpad/geometry/counters via a Dtcu&

  enum class State {
    IDLE,
    DESC_REQ,
    DESC_WAIT,
    NEXT_TILE_LOAD,   // load the next output tile's K0 into the compute buffer
    COMPUTE,          // compute the current K tile while prefetching the next one
    TILE_STORE,       // hand off this tile's D store (after the prior store drains)
    TILE_STORE_BLOCK, // NO_TMA blocking mode: drain THIS tile's store before the next tile
    FINAL_TILE_STORE, // final tile: drain its background store, then done epilogue
    DONE
  };

  // TMA overlap policy from the descriptor: bit clear (default) = overlapped
  // prefetch + background store (current behavior); FLAG_NO_TMA = blocking mode.
  // Only valid after DESC_WAIT populates desc_ (all call sites are later states).
  bool tma_enabled_() const { return (desc_.flags & FLAG_NO_TMA) == 0; }

  Cluster*  cluster_;

  std::unique_ptr<DtcuTma> tma_; // tensor-memory engine (owns the L2 port)

  State     state_;
  bool      busy_;
  bool      done_;

  uint64_t  desc_addr_;
  Desc      desc_;

  // Operand shared memory A/B (shm_a_/shm_b_) — will be banked + (later) SIMT-shared
  // for fusion; currently private to the DTCU. Accumulator C stays a private
  // accumulator memory. All double-buffered (ping-pong). (element units, not bytes)
  std::array<std::vector<uint32_t>, 2> shm_a_;
  std::array<std::vector<uint32_t>, 2> shm_b_;
  std::array<std::vector<float>, 2> accum_buf_;
  uint32_t accum_compute_idx_ = 0; // accumulator buffer the current output tile computes into

  // Ping-pong operand buffers: compute_buf_ holds the K tile being computed; the
  // other buffer is filled by the TMA prefetch engine for the next K tile.
  uint32_t compute_buf_ = 0;
  bool     buf_ready_[2] = { false, false }; // buffer holds a valid loaded K tile
  bool     compute_done_ = false;            // current K tile's MMA already executed

  // Cross-tile lookahead: next output tile's K0 fetch issued during the current
  // tile's last-K compute; adopted at the TILE_STORE tile switch.
  bool     next_tile_load_issued_ = false;
  uint32_t next_tile_load_buf_ = 0;

  // Perf counters. dtcu_* = FSM observers (mutually exclusive, sum to the busy
  // timeline); tma_* = engine observers (concurrent with COMPUTE, not summable).
  uint64_t dtcu_compute_cycles_ = 0;               // compute-pipeline occupancy ticks
  uint64_t dtcu_next_k_load_stall_cycles_ = 0;     // COMPUTE stalled on next K tile's operands (intra-tile)
  uint64_t tma_mem_wait_cycles_ = 0;               // load FETCH waited on memory (latency + throttle + contention)
  uint64_t tma_buf_starve_cycles_ = 0;             // load channel idle with work available but no free buffer
  uint64_t tma_op_fill_cycles_ = 0;                // buffer fill (operand + acc init; split in a later step)
  uint64_t tma_addrgen_cycles_ = 0;                // AGU setup: fixed cost per kick (store AGU unmodeled)
  uint64_t tma_store_issue_stall_cycles_ = 0;      // store line issue blocked (port yielded to loads / queue)
  uint64_t dtcu_store_drain_cycles_ = 0;           // exposed store drain (overlap: final tile; blocking: every tile)
  uint64_t dtcu_smem_read_model_cycles_ = 0;       // modeled operand-SRAM read: component of compute, never additive
  uint64_t dtcu_next_tile_load_stall_cycles_ = 0;  // NEXT_TILE_LOAD: exposed K0 wait (incl. tile 0's cold start)
  uint64_t dtcu_prev_tile_store_stall_cycles_ = 0; // TILE_STORE: handoff blocked by the previous tile's store
  uint64_t dtcu_desc_wait_cycles_ = 0;             // descriptor fetch window (DESC_REQ + DESC_WAIT)
  uint64_t dtcu_busy_cycles_ = 0;                  // every tick busy_ is set (accounting anchor)
  uint64_t tma_acc_init_cycles_ = 0;               // K0 fill: accumulator init portion (separate SRAM)
  uint64_t dtcu_instr_tcu_ = 0;                    // MMA ops issued (one per FEDP), not a cycle count

  // Native tile, resolved per descriptor in init_tile_state_() from dtcu_cfg.h.
  uint32_t tile_m_ = 0; // build-time fixed (DTCU_TILE_M)
  uint32_t tile_n_ = 0; // descriptor-driven: shape_n_size * DTCU_TILE_N_GRAN, <= DTCU_TILE_N_MAX
  uint32_t tile_k_ = 0; // DTCU_TILE_K_WORDS words, widened to elements by the input format

  // Internal state for iterating through tiles
  uint32_t tile_m_idx_ = 0; // Internal index for current tile within big GEMM
  uint32_t tile_n_idx_ = 0;
  uint32_t tile_k_idx_ = 0;
  uint32_t tiles_m_ = 1; // # of tiles needed for the entire GEMM
  uint32_t tiles_n_ = 1;
  uint32_t tiles_k_ = 1;

  // Aggregate mem request counters (for the descriptor summary print); the TMA
  // engine accumulates these as it issues operand/output requests.
  uint64_t total_op_reqs_ = 0;
  uint64_t total_out_reqs_ = 0;

  // Execute latency modelling
  uint32_t exec_cycles_left_ = 0;
  uint32_t estimate_execute_cycles_(); // non-const: accumulates dtcu_smem_read_model_cycles_
  uint32_t operand_read_cycles_() const; // banked operand-SRAM read cycles for one K tile (M2)
  uint32_t bank_of_(uint32_t phys_word) const; // operand-SRAM bank of a physical word index

  void init_tile_state_();
  bool advance_output_tile_();
  void start_next_tile_or_drain_(); // advance + kick next tile's K0, or go drain

  // Single source of truth for the output-tile walk order (n-major): used by
  // both advance_output_tile_ and the lookahead peek so they cannot desync.
  static bool next_tile_coord_(uint32_t& m, uint32_t& n, uint32_t tiles_m, uint32_t tiles_n) {
    if (++n < tiles_n) return true;
    n = 0;
    return ++m < tiles_m;
  }

  void execute_mma(uint32_t buf_idx);
};

} // namespace vortex
