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

#include <functional>
#include <memory>
#include <simobject.h>
#include "types.h"
#include "instr.h"
#include "lsu_unit.h"
#ifdef EXT_TCU_ENABLE
#include "tensor_unit.h"
#endif

namespace vortex {

class Core;
struct instr_trace_t;

///////////////////////////////////////////////////////////////////////////////

// Per-warp micro-op expander for multi-uop instructions (TCU WMMA/WGMMA).
// Simple instructions pass through unchanged. Mirrors RTL VX_uop_sequencer
// instantiated inside VX_ibuffer.sv.
class Sequencer : public SimObject<Sequencer> {
public:
  Sequencer(const SimContext& ctx, const char* name, Core* core, PoolAllocator<Instr, 64>& instr_pool);

  // Get current micro-op trace. Idempotent: returns same trace until advance().
  // For simple instructions, returns the input trace (pass-through).
  instr_trace_t* get(instr_trace_t* trace);

  // Advance to next micro-op. Returns true when all micro-ops have been issued.
  bool advance();

protected:
  void on_reset();

private:

  // uop generator callback: (macro_instr, uop_index) -> Instr::Ptr
  using GenFn = std::function<Instr::Ptr(const Instr&, uint32_t)>;

  struct State {
    bool     active;
    uint32_t uop_index;
    uint32_t uop_count;
    GenFn    gen_fn;
    instr_trace_t* current_uop;

    State() : active(false), uop_index(0), uop_count(0), current_uop(nullptr) {}

    void reset() {
      active = false;
      uop_index = 0;
      uop_count = 0;
      gen_fn = nullptr;
      current_uop = nullptr;
    }
  };

  Core* core_;
  State state_;
  LsuUopGen lsu_uop_gen_;
#ifdef EXT_TCU_ENABLE
  TcuUopGen tcu_uop_gen_;
#endif

  friend class SimObject<Sequencer>;
};

} // namespace vortex
