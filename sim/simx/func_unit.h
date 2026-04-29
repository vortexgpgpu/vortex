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
#include <array>
#include "instr_trace.h"

namespace vortex {

class Core;

// Type-erased base used by Core to keep heterogeneous FuncUnit<N_BLOCKS>
// instances in a single container and to bind/probe their I/O channels.
class FuncUnitBase {
public:
	virtual ~FuncUnitBase() = default;
	virtual SimChannel<instr_trace_t*>& input(uint32_t b) = 0;
	virtual SimChannel<instr_trace_t*>& output(uint32_t b) = 0;
	virtual uint32_t num_blocks() const = 0;
};

// FuncUnit pipelines have NUM_BLOCKS physical lanes, not ISSUE_WIDTH. The
// dispatcher upstream aggregates ISSUE_WIDTH issue ports onto NUM_BLOCKS
// execution ports; commit downstream fans them back out to per-iw arbiters
// using trace->wid.
template <uint32_t NUM_BLOCKS>
class FuncUnit : public FuncUnitBase, public SimObject<FuncUnit<NUM_BLOCKS>> {
public:
	static constexpr uint32_t kNumBlocks = NUM_BLOCKS;

	std::array<SimChannel<instr_trace_t*>, NUM_BLOCKS> Inputs;
	std::array<SimChannel<instr_trace_t*>, NUM_BLOCKS> Outputs;

	FuncUnit(const SimContext& ctx, const char* name, Core* core)
		: SimObject<FuncUnit<NUM_BLOCKS>>(ctx, name)
		, Inputs(make_sim_channels<instr_trace_t*, NUM_BLOCKS>(this))
		, Outputs(make_sim_channels<instr_trace_t*, NUM_BLOCKS>(this))
		, core_(core)
	{}

	virtual ~FuncUnit() {}

	// FuncUnitBase polymorphic accessors for the type-erased Core wiring.
	SimChannel<instr_trace_t*>& input(uint32_t b) override { return Inputs[b]; }
	SimChannel<instr_trace_t*>& output(uint32_t b) override { return Outputs[b]; }
	uint32_t num_blocks() const override { return NUM_BLOCKS; }

protected:
	// SimObject<FuncUnit<N>>::on_tick is a non-virtual no-op — these
	// declarations introduce a new virtual that the CRTP do_tick()
	// resolves through, so derived FUs can override.
	virtual void on_reset() {}
	virtual void on_tick() = 0;

	friend class SimObject<FuncUnit<NUM_BLOCKS>>;

	Core* core_;
};

}
