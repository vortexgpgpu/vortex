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
	std::vector<SimChannel<instr_trace_t*>> Inputs;
	std::vector<SimChannel<instr_trace_t*>> Outputs;

	FuncUnit(const SimContext& ctx, const char* name, Core* core)
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
  AluUnit(const SimContext& ctx, const char* name, Core*);

  void tick() override;
};

///////////////////////////////////////////////////////////////////////////////

class FpuUnit : public FuncUnit {
public:
  FpuUnit(const SimContext& ctx, const char* name, Core*);

  void tick() override;
};

///////////////////////////////////////////////////////////////////////////////

class LsuUnit : public FuncUnit {
public:
	LsuUnit(const SimContext& ctx, const char* name, Core*);
	~LsuUnit();

	void reset() override;
	void tick() override;

private:

 	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t count;
		bool eop;
	};

	struct lsu_state_t {
		HashTable<pending_req_t> pending_rd_reqs;
		instr_trace_t* fence_trace;
		bool fence_lock;

		lsu_state_t() : pending_rd_reqs(LSUQ_IN_SIZE) {}

		void reset() {
			this->pending_rd_reqs.clear();
			this->fence_trace = nullptr;
			this->fence_lock = false;
		}
	};

	std::array<lsu_state_t, NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;
	std::vector<mem_addr_size_t> addr_list_;
	uint32_t remain_addrs_;
};

///////////////////////////////////////////////////////////////////////////////

class SfuUnit : public FuncUnit {
public:
	SfuUnit(const SimContext& ctx, const char* name, Core*);

	void tick() override;

#ifdef EXT_TMA_ENABLE
private:
  struct tma_runtime_t {
    uint32_t desc_slot = 0;
    uint32_t bar_id = 0;
    uint32_t smem_addr = 0;
    uint32_t flags = 0;
    std::array<uint32_t, 5> coords = {0, 0, 0, 0, 0};
  };

  bool execute_tma_op(instr_trace_t* trace, TmaType tma_type, const TmaTraceData& tma_data);

  std::vector<tma_runtime_t> tma_runtime_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

#ifdef EXT_TCU_ENABLE

class TcuUnit : public FuncUnit {
public:
	TcuUnit(const SimContext& ctx, const char* name, Core*);

	void tick() override;
};

#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef EXT_V_ENABLE

class VpuUnit : public FuncUnit {
public:
	VpuUnit(const SimContext& ctx, const char* name, Core*);

	void tick() override;
};

#endif

}
