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

#include "types.h"
#include <array>
#include "instr_trace.h"

namespace vortex {

class Core;

// Type-erased base used by Core to keep heterogeneous FuncUnit<N_BLOCKS>
// instances in a single container and to bind/probe their I/O channels.
class FuncUnitBase {
public:
	virtual ~FuncUnitBase() = default;
	virtual SimPortIn<instr_trace_t*>& input(uint32_t b) = 0;
	virtual SimPortIn<instr_trace_t*>& output(uint32_t b) = 0;
	virtual uint32_t num_blocks() const = 0;
};

// FuncUnit pipelines have NUM_BLOCKS physical lanes, not VX_CFG_ISSUE_WIDTH. The
// dispatcher upstream aggregates VX_CFG_ISSUE_WIDTH issue ports onto NUM_BLOCKS
// execution ports; commit downstream fans them back out to per-iw arbiters
// using trace->wid. A unit U with an on_tick() reads Inputs itself; any other
// takes each op through its handler U::issue(b, trace). Outputs land in result
// ports owned by the Core, which reads them.
template <typename U, uint32_t NUM_BLOCKS>
class FuncUnit : public FuncUnitBase, public SimObject<U> {
public:
	static constexpr uint32_t kNumBlocks = NUM_BLOCKS;

	SimPortInArray<instr_trace_t*> Inputs;
	SimPortOutArray<instr_trace_t*> Outputs;

	// reader: the Core, which owns the result ports.
	FuncUnit(const SimContext& ctx, const char* name, Core* core, SimObjectBase* reader,
	         uint32_t in_capacity, uint32_t out_capacity)
		: SimObject<U>(ctx, name)
		, Inputs(make_inputs(static_cast<U*>(this), in_capacity))
		, Outputs(this, NUM_BLOCKS)
		, core_(core)
	{
		for (uint32_t b = 0; b < NUM_BLOCKS; ++b) {
			results_.emplace_back(std::make_unique<SimPortIn<instr_trace_t*>>(reader, out_capacity));
			Outputs[b].bind(results_.back().get());
		}
	}

	virtual ~FuncUnit() {}

	// FuncUnitBase polymorphic accessors for the type-erased Core wiring.
	SimPortIn<instr_trace_t*>& input(uint32_t b) override { return Inputs[b]; }
	SimPortIn<instr_trace_t*>& output(uint32_t b) override { return *results_[b]; }
	uint32_t num_blocks() const override { return NUM_BLOCKS; }

protected:
	Core* core_;

private:
	static SimPortInArray<instr_trace_t*> make_inputs(U* self, uint32_t capacity) {
		if constexpr (U::kTicks) {
			return {self, NUM_BLOCKS, capacity};
		} else {
			return {self, NUM_BLOCKS, capacity, &U::issue};
		}
	}

	std::vector<std::unique_ptr<SimPortIn<instr_trace_t*>>> results_;
};

}
