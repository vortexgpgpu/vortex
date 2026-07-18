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
  struct Desc {
    uint64_t ptrA;
    uint64_t ptrB;
    uint64_t ptrC;
    uint64_t ptrD;
    // Leading dimensions in number of elements (not bytes) for different element size
    uint32_t ldmA;
    uint32_t ldmB;
    uint32_t ldmC;
    uint32_t ldmD;
    uint16_t M;
    uint16_t N;
    uint16_t K;
    uint8_t  fmt_s;
    uint8_t  fmt_d;
    uint8_t  flags; // FLAG_* bits below
    uint8_t  shape_n_size;
    uint16_t shape_policy;
    uint32_t reserved2;
  };

  static_assert(sizeof(Desc) == 64, "Dtcu::Desc must be 64 bytes");

  // Descriptor flag bits (Desc::flags). Must match sw/kernel/include/vx_dtensor.h.
  static constexpr uint8_t FLAG_ZERO_ACC = 0x1; // zero-accumulate (skip C preload)
  static constexpr uint8_t FLAG_NO_TMA   = 0x2; // blocking mode: no operand-prefetch / store overlap

  Dtcu(const SimContext& ctx, const char* name, Cluster* cluster);

  ~Dtcu();

  // The TMA engine owns the L2 memory port; the Cluster binds it via tma().
  DtcuTma* tma() const { return tma_.get(); }

  void start(uint64_t desc_addr);

  uint32_t poll() const;

  // Perf counters surfaced to MPM CSRs (cluster-level engine; see claude_doc/DTCU Perf Stat).
  // op_reqs/out_reqs are coalesced L2 cache-line counts (same unit as core L2/memory req counters);
  // the rest are cycles.
  struct PerfStats {
    uint64_t op_reqs;     // TMA operand (A/B/C) L2 cache-line requests
    uint64_t out_reqs;    // TMA output (D) L2 cache-line requests
    uint64_t compute;     // MMA compute cycles
    uint64_t wait_tma;    // compute stalled on next operand tile (NO_TMA mode: the serialized k>=1 fetches)
    uint64_t mem_wait;    // prefetch waited on memory responses
    uint64_t wait_buf;    // prefetch idle (no free buffer)
    uint64_t buf_write;   // buffer (SRAM) fill cycles
    uint64_t addrgen;     // AGU address-gen setup cycles
    uint64_t store_wait;  // output store stalled cycles
    uint64_t store_drain; // unhidden store cycles (final tile only in overlap mode; every tile in NO_TMA mode)
    uint64_t opread;      // banked operand-SRAM read cycles
    uint64_t load_stall;  // NEXT_TILE_LOAD: exposed K0 fetch wait cycles (both modes)
    uint64_t store_stall; // TILE_STORE: handoff stalled on the prior store
  };
  PerfStats perf_stats() const {
    return PerfStats{ total_op_reqs_, total_out_reqs_, dtcu_compute_cycles_,
      dtcu_wait_for_tma_cycles_, tma_mem_wait_cycles_, tma_wait_for_buffer_cycles_,
      tma_buffer_write_cycles_, tma_addrgen_cycles_, tma_store_wait_cycles_,
      dtcu_store_drain_cycles_, dtcu_operand_read_cycles_,
      dtcu_next_tile_load_stall_cycles_, dtcu_curr_tile_store_stall_cycles_ };
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

  // Overlap counters (Phase 4). The TMA engine increments the tma_* ones via the
  // back-reference; the FSM here increments the compute/wait ones.
  uint64_t dtcu_compute_cycles_ = 0;        // cycles spent computing K tiles
  uint64_t dtcu_wait_for_tma_cycles_ = 0;   // cycles compute stalled waiting for next operand tile
  uint64_t tma_mem_wait_cycles_ = 0;        // cycles prefetch waited on memory responses
  uint64_t tma_wait_for_buffer_cycles_ = 0; // cycles prefetch idle (next buffer ready, no free buffer)
  uint64_t tma_buffer_write_cycles_ = 0;    // cycles writing fetched data into buffers (SRAM)
  uint64_t tma_addrgen_cycles_ = 0;         // cycles in AGU address-generation setup
  uint64_t tma_store_wait_cycles_ = 0;      // cycles output store stalled (port taken by load / waiting responses)
  uint64_t dtcu_store_drain_cycles_ = 0;    // unhidden store cycles (final tile in overlap mode; every tile in NO_TMA mode)
  uint64_t dtcu_operand_read_cycles_ = 0;   // cycles to read operands from the banked SRAM (M2 reuse; conflict-sensitive)
  uint64_t dtcu_next_tile_load_stall_cycles_ = 0;  // NEXT_TILE_LOAD: exposed K0 wait (incl. tile 0's cold start)
  uint64_t dtcu_curr_tile_store_stall_cycles_ = 0; // TILE_STORE: handoff stalled on prior store

  uint32_t tile_m_ = 0; // M dimension of native tile (=64)
  uint32_t tile_n_ = 0; // N dimension of native tile (multiple of 16, up to 128)
  uint32_t tile_k_ = 0; // K dimension of native tile (depends on data type)

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
  uint32_t estimate_execute_cycles_(); // non-const: accumulates dtcu_operand_read_cycles_
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
