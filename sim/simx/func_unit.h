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

class FuncUnit : public SimObject<FuncUnit> {
public:
	std::vector<SimPort<instr_trace_t*>> Inputs;
	std::vector<SimPort<instr_trace_t*>> Outputs;

	FuncUnit(const SimContext& ctx, Core* core, const char* name)
		: SimObject<FuncUnit>(ctx, name)
		, Inputs(ISSUE_WIDTH, this)
		, Outputs(ISSUE_WIDTH, this)
		, core_(core)
	{}

	virtual ~FuncUnit() {}

	virtual void reset() {}

	virtual void tick() = 0;

protected:
	Core* core_;
};

///////////////////////////////////////////////////////////////////////////////

class AluUnit : public FuncUnit {
public:
  AluUnit(const SimContext& ctx, Core*);

  void tick();
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public FuncUnit {
public:
  FpuUnit(const SimContext& ctx, Core*);

  void tick();
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public FuncUnit {
public:
	LsuUnit(const SimContext& ctx, Core*);
	~LsuUnit();

	void reset();
	void tick();

private:

 	struct pending_req_t {
		instr_trace_t* trace;
		BitVector<> mask;
	};

	struct lsu_state_t {
		HashTable<pending_req_t> pending_rd_reqs;
		instr_trace_t* fence_trace;
		bool fence_lock;

		lsu_state_t() : pending_rd_reqs(LSUQ_IN_SIZE) {}

		void clear() {
			this->pending_rd_reqs.clear();
			this->fence_trace = nullptr;
			this->fence_lock = false;
		}
	};

	std::array<lsu_state_t, NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;
};

///////////////////////////////////////////////////////////////////////////////

class TcuUnit : public FuncUnit {
public:
    TcuUnit(const SimContext& ctx, Core*);
    void tick();
};

///////////////////////////////////////////////////////////////////////////////

class SfuUnit : public FuncUnit {
public:
	SfuUnit(const SimContext& ctx, Core*);

	void tick();
};

///////////////////////////////////////////////////////////////////////////////

class VpuUnit : public FuncUnit {
public:
	VpuUnit(const SimContext& ctx, Core*);

	void tick();
};

// Simulate clock cycles depending on instruction type and element width and #lanes
// VSET = 1 cycle
// Vector instructions take the same amount of time as ALU instructions.
// In general there should be less overall instructions (hence the SIMD vector speedup).
// But, each vector instruction is bigger, and # of lanes greatly effects execution speed.

// Whenever we change VL using imm/VSET, we need to keep track of the new VL and SEW.
// By default, VL is set to MAXVL.
// After determining VL, we use VL and #lanes in order to determine overall cycle time.
// For example, for a vector add with VL=4 and #lanes=2, we will probably take 2 cycles,
// since we can only operate on two elements of the vector each cycle (limited by #lanes).
// SEW (element width) likely affects the cycle time, we can probably observe
// ALU operation cycle time in relation to element width to determine this though.

// The RTL implementation has an unroll and accumulate stage.
// The unroll stage sends vector elements to the appropriate functional unit up to VL,
// limited by the # lanes available.
// The accumulate stage deals with combining the results from the functional units,
// into the destination vector register.
// Which exact pipeline stage does the VPU unroll the vector (decode or execute)?
// Which exact pipeline stage does the VPU accumulate results?

// How do vector loads and stores interact with the cache?
// How about loading and storing scalars in vector registers?
// How does striding affect loads and stores?

}