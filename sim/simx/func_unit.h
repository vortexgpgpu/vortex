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

}