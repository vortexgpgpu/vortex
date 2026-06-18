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
#include "mem_sim.h"
#include "arch.h"
#include "dcrs.h"
#include "mem.h"
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
    uint8_t  flags;
    uint8_t  shape_n_size;
    uint16_t shape_policy;
    uint32_t reserved2;
  };

  static_assert(sizeof(Desc) == 64, "Dtcu::Desc must be 64 bytes");

  Dtcu(const SimContext& ctx,
                   const char* name,
                   Cluster* cluster,
                   const Arch& arch,
                   const DCRS& dcrs);

  ~Dtcu();

  // The TMA engine owns the L2 memory port; the Cluster binds it via tma().
  DtcuTma* tma() const { return tma_.get(); }

  void attach_ram(RAM* ram);

  void start(uint64_t desc_addr);

  uint32_t poll() const;

  void reset();

  void tick();

private:
  friend class DtcuTma; // TMA engine reaches scratchpad/geometry/counters via a Dtcu&

  enum class State {
    IDLE,
    DESC_REQ,
    DESC_WAIT,
    FIRST_LOAD, // wait for the first K tile of an output tile to be prefetched
    COMPUTE,    // compute the current K tile while prefetching the next one
    OUT_REQ,
    OUT_WAIT,
    DONE
  };

  Cluster*  cluster_;
  const Arch& arch_;
  const DCRS& dcrs_;

  std::unique_ptr<DtcuTma> tma_; // tensor-memory engine (owns the L2 port + RAM)

  State     state_;
  bool      busy_;
  bool      done_;

  uint64_t  desc_addr_;
  Desc      desc_;

  // Internal operand buffers A/B and accumulator C, all double-buffered (ping-pong).
  // (in element units, not bytes)
  std::array<std::vector<uint32_t>, 2> a_buf_;
  std::array<std::vector<uint32_t>, 2> b_buf_;
  std::array<std::vector<float>, 2> accum_buf_;
  uint32_t accum_compute_idx_ = 0; // accumulator buffer the current output tile computes into

  // Ping-pong operand buffers: compute_buf_ holds the K tile being computed; the
  // other buffer is filled by the TMA prefetch engine for the next K tile.
  uint32_t compute_buf_ = 0;
  bool     buf_ready_[2] = { false, false }; // buffer holds a valid loaded K tile
  bool     compute_done_ = false;            // current K tile's MMA already executed

  // Overlap counters (Phase 4). The TMA engine increments the tma_* ones via the
  // back-reference; the FSM here increments the compute/wait ones.
  uint64_t dtcu_compute_cycles_ = 0;        // cycles spent computing K tiles
  uint64_t dtcu_wait_for_tma_cycles_ = 0;   // cycles compute stalled waiting for next operand tile
  uint64_t tma_mem_wait_cycles_ = 0;        // cycles prefetch waited on memory responses
  uint64_t tma_wait_for_buffer_cycles_ = 0; // cycles prefetch idle (next buffer ready, no free buffer)
  uint64_t tma_buffer_write_cycles_ = 0;    // cycles writing fetched data into buffers (SRAM)
  uint64_t tma_addrgen_cycles_ = 0;         // cycles in AGU address-generation setup

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
  uint32_t estimate_execute_cycles_() const;

  void init_tile_state_();
  bool advance_output_tile_();

  void execute_mma(uint32_t buf_idx);
};

} // namespace vortex
