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

#include "sfu_unit.h"
#include "core.h"
#include "debug.h"

using namespace vortex;

SfuUnit::SfuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<NUM_SFU_BLOCKS>(ctx, name, core)
#ifdef EXT_DXA_ENABLE
	, dxa_req_out(this)
#endif
	, wctl_unit_(new WctlUnit(core))
	, csr_unit_(new CsrUnit(core))
#ifdef EXT_DXA_ENABLE
	, dxa_unit_(new DxaUnit(core, dxa_req_out))
#endif
{
}

uint32_t SfuUnit::latency_of(const instr_trace_t* /*trace*/) const {
	return 4;
}

void SfuUnit::on_tick() {
	// PE switch: peek input, route to the matching sub-unit (WCTL / CSR /
	// DXA) by op_type, gather to the single result port.
	for (uint32_t b = 0; b < NUM_SFU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		if (output.full())
			continue; // stall — no side effects this tick
		auto trace = input.peek();

		// WSYNC has a structural gate: cannot complete until prior insts retire.
		if (auto wctl_p = std::get_if<WctlType>(&trace->op_type)) {
			if (*wctl_p == WctlType::WSYNC) {
				if (!trace->eop || core_->has_pending_instrs(trace->wid))
					continue; // skip; do not pop
			}
		}

		bool release_warp = trace->fetch_stall;
		if (std::get_if<WctlType>(&trace->op_type)) {
			release_warp = wctl_unit_->process(trace);
		} else if (std::get_if<CsrType>(&trace->op_type)) {
			csr_unit_->process(trace);
#ifdef EXT_DXA_ENABLE
		} else if (std::get_if<DxaType>(&trace->op_type)) {
			// process() returns nullptr on backpressure (idempotent retry next
			// cycle) or the trace on success → fall through to send/pop.
			if (!dxa_unit_->process(trace)) {
				continue;
			}
#endif
		}

		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		if (trace->eop && release_warp) {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}
