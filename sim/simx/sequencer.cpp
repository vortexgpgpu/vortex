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

#include <assert.h>
#include <util.h>

#include "sequencer.h"
#include "instr_trace.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

Sequencer::Sequencer(const SimContext& ctx, const char* name, Core* core, PoolAllocator<Instr, 64>& instr_pool)
  : SimObject<Sequencer>(ctx, name)
  , core_(core)
#ifdef EXT_TCU_ENABLE
  , tcu_uop_gen_(instr_pool)
#endif
{
  __unused(instr_pool);
}

void Sequencer::on_reset() {
  state_.reset();
}

instr_trace_t* Sequencer::get(instr_trace_t* trace) {
  // Idempotent: return cached micro-op if available
  if (state_.current_uop)
    return state_.current_uop;

  // Activate sequencing for macro-op instructions
  if (!state_.active && trace->instr_ptr->is_macro_op()) {
    state_.active = true;
    state_.uop_index = 0;

    // Bind the appropriate generator based on FU type
    switch (trace->instr_ptr->get_fu_type()) {
  #ifdef EXT_TCU_ENABLE
    case FUType::TCU:
      state_.uop_count = TcuUopGen::uop_count(*trace->instr_ptr);
      state_.gen_fn = [this](const Instr& m, uint32_t i) {
        return tcu_uop_gen_.get(m, i);
      };
      break;
  #endif
    // Future generators:
    // case FUType::VEC: ...
    default:
      std::abort();
    }
  }

  if (state_.active) {
    // Generate micro-op Instr
    auto uop_instr = state_.gen_fn(*trace->instr_ptr, state_.uop_index);

    // Allocate micro-op trace and fill metadata (like decode does)
    auto uop_trace = core_->trace_pool().allocate(1);
    new (uop_trace) instr_trace_t(uop_instr->get_uuid());
    uop_trace->cid       = trace->cid;
    uop_trace->wid       = trace->wid;
    uop_trace->PC        = trace->PC;
    uop_trace->tmask     = uop_instr->get_tmask().any() ? uop_instr->get_tmask() : trace->tmask;
    uop_trace->instr_ptr = uop_instr;
    uop_trace->fu_type   = uop_instr->get_fu_type();
    uop_trace->op_type   = uop_instr->get_op_type();
    uop_trace->dst_reg   = uop_instr->get_dest_reg();
    for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
      uop_trace->src_regs[i] = uop_instr->get_src_reg(i);
    }
    uop_trace->wb = (uop_instr->get_dest_reg().type != RegType::None);

    state_.current_uop = uop_trace;
    return uop_trace;
  }

  // Simple instruction — pass through
  state_.current_uop = trace;
  return trace;
}

bool Sequencer::advance() {
  state_.current_uop = nullptr;
  if (state_.active) {
    ++state_.uop_index;
    if (state_.uop_index >= state_.uop_count || state_.uop_index >= 255) {
      state_.active = false;
    }
  }
  return !state_.active;
}
