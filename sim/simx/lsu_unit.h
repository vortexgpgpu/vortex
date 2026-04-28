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
#include <mempool.h>
#include "func_unit.h"
#include "instr.h"

namespace vortex {

// Micro-op generator for packed-load macro instructions (PACKLB.F / PACKLH.F).
class LsuUopGen {
public:
  LsuUopGen(PoolAllocator<Instr, 64>& pool) : pool_(pool) {}

  static uint32_t uop_count(const Instr& instr);

  Instr::Ptr get(const Instr& macro_instr, uint32_t uop_index);

private:
  PoolAllocator<Instr, 64>& pool_;
};

class LsuUnit : public FuncUnit<NUM_LSU_BLOCKS> {
public:
	LsuUnit(const SimContext& ctx, const char* name, Core*);
	~LsuUnit();

protected:
	void on_reset() override;
	void on_tick() override;

private:

	void compute_addrs(uint32_t b, instr_trace_t* trace);

	void process_response(uint32_t b);

	void process_request(uint32_t b);

  struct mem_addr_size_t {
		uint64_t addr;
		uint32_t size;
		uint64_t data;
		uint32_t tid;
	};

 	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t count;
		bool eop;
		std::vector<mem_addr_size_t> lanes;
		IntrLsuArgs lsu_args;
		bool        is_load;
	};

	struct lsu_state_t {
		HashTable<pending_req_t>     pending_rd_reqs;
		instr_trace_t*               fence_trace;
		bool                         fence_lock;
		std::vector<mem_addr_size_t> addr_list;
		uint32_t                     remain_addrs;

		lsu_state_t() : pending_rd_reqs(LSUQ_IN_SIZE), fence_trace(nullptr), fence_lock(false), remain_addrs(0) {}

		void reset() {
			this->pending_rd_reqs.clear();
			this->fence_trace = nullptr;
			this->fence_lock = false;
			this->addr_list.clear();
			this->remain_addrs = 0;
		}
	};

	std::array<lsu_state_t, NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;
};

}
