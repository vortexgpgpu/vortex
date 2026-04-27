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

#include <array>
#include "func_unit.h"
#include "instr.h"

namespace vortex {

class LsuUnit : public FuncUnit {
public:
	LsuUnit(const SimContext& ctx, const char* name, Core*);
	~LsuUnit();

protected:
	void on_reset() override;
	void on_tick() override;

private:
	// Per-unit functional execution for LsuType/AmoType. Called only from
	// this unit's tick() at first peek of a new trace.
	void execute(instr_trace_t* trace);

 	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t count;
		bool eop;
		// Per-lane addrs+sizes captured at issue time. The response handler
		// extracts bytes from the TLM payload using these.
		std::vector<mem_addr_size_t> lanes;
		// Load formatting captured at issue time (only meaningful for LOADs).
		IntrLsuArgs lsu_args;
		bool        is_load;
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

}
