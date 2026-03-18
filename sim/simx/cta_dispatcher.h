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

#include <cstdint>
#include "types.h"
#include "kmu.h"

namespace vortex {

class Core; // forward declaration

// Per-warp dispatch record — one VX_cta_dispatch.sv DISPATCH output.
struct cta_warp_record_t {
  Word       PC;
  ThreadMask tmask;
  Word       mscratch;       // lower XLEN bits of param (for VX_CSR_MSCRATCH)

  uint32_t   cta_id;         // global CTA index in the grid
  uint32_t   cta_rank;       // warp index within the CTA [0 .. cta_size-1]
  uint32_t   cta_size;       // number of warps in the CTA = ceil(block_size/NT)
  uint32_t   thread_idx[3];  // first thread's 3-D index within the block
  uint32_t   block_idx[3];   // block's 3-D position in the grid
  uint32_t   block_dim[3];   // block dimensions
  uint32_t   grid_dim[3];    // grid dimensions
  uint64_t   param;          // full 64-bit kernel argument pointer
  uint32_t   lmem_addr;      // local-memory base address for this CTA
};

// Stateful CTA dispatcher — software model of VX_cta_dispatch.sv.
//
// Holds a pointer to the processor's Kmu (no copy) and maintains its own
// per-core CTA iteration state so each core independently walks the grid
// using round-robin assignment (cta_id % total_cores == core_id).
//
// Usage:
//   CtaDispatcher(core)              — binds to Core to access the KMU
//   step(active_warps, &wid, &rec)  — called once per Emulator::step();
//                                      lazily starts dispatch on first call;
//                                      returns true when a warp was activated
//   running()                        — true while CTAs remain to dispatch
//   reset()                          — clears all dispatch state
class CtaDispatcher {
public:
  explicit CtaDispatcher(Core* core);

  // Clear all dispatch state; restarts lazily on the next step() call.
  void reset() {
    started_      = false;
    has_cta_      = false;
    lmem_tail_    = 0;
  }

  // Try to dispatch one CTA warp rank into a free warp slot.
  // Lazily starts on first call (reads KMU config from bound Core).
  // On success, sets *wid_out and *rec_out and returns true.
  bool step(const WarpMask& active_warps, uint32_t* wid_out, cta_warp_record_t* rec_out);

  // True while started and at least one more warp rank remains to dispatch.
  bool running() const {
    return started_ && has_cta_;
  }

private:
  // Initialize from the KMU stored in the bound Core (called lazily by step).
  void do_start();

  // Fill *out with the next warp rank and advance internal state.
  // Returns false when no CTA is loaded (should not happen after has_cta_ check).
  bool next_warp(cta_warp_record_t* out);

  // Pull the next CTA assigned to this core from the KMU grid (IDLE state).
  void load_next_cta();

  Core*         core_;
  const Kmu*    kmu_;          // pointer to processor's KMU — no copy
  bool          started_;
  uint32_t      total_cores_;
  uint32_t      core_id_;
  uint32_t      num_threads_;
  uint32_t      num_warps_;
  uint32_t      lmem_capacity_;
  uint32_t      lmem_tail_;

  // Per-core CTA grid iteration state
  uint32_t      iter_cta_id_;
  uint32_t      iter_block_idx_[3];
  bool          iter_running_;

  // Current CTA dispatch state (DISPATCH FSM registers)
  bool          has_cta_;
  kmu_req_t     cta_;
  uint32_t      cta_size_;
  uint32_t      rank_;
  uint32_t      block_size_rem_;
  uint32_t      thread_idx_[3];
  uint32_t      lmem_addr_;
};

} // namespace vortex
