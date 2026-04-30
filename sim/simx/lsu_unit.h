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
#include <ringqueue.h>
#include "func_unit.h"
#include "instr.h"
#include "VX_config.h"

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

// Fence controller
class FenceController {
public:
  bool locked() const { return locked_; }

  void engage(instr_trace_t* trace) {
    trace_ = trace;
    locked_ = true;
  }

  // Try to release the lock. Returns true on success (lock cleared,
  // trace forwarded). Conditions: MSHR empty AND output channel can
  // accept the trace.
  bool try_release(SimChannel<instr_trace_t*>& out, bool mshr_empty) {
    if (!locked_) return false;
    if (!mshr_empty) return false;
    if (!out.try_send(trace_)) return false;
    locked_ = false;
    trace_ = nullptr;
    return true;
  }

  instr_trace_t* trace() const { return trace_; }

  void reset() { trace_ = nullptr; locked_ = false; }

private:
  instr_trace_t* trace_ = nullptr;
  bool locked_ = false;
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

	// Per-cycle handlers.
	void process_response_step(uint32_t b);
	void process_request_step(uint32_t b);

	// Drain Inputs[b] into req_queue.
	void ingest_inputs(uint32_t b);

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

	// Per-block LSU state. Each member is a named hardware sub-block.
	struct lsu_state_t {
		RingQueue<instr_trace_t*> req_queue{LSUQ_IN_SIZE};
		HashTable<pending_req_t>  mshr{LSUQ_IN_SIZE};
		FenceController           fence;
		std::vector<mem_addr_size_t> addr_list;
		uint32_t                  remain_addrs = 0;

		void reset() {
			this->req_queue.clear();
			this->mshr.clear();
			this->fence.reset();
			this->addr_list.clear();
			this->remain_addrs = 0;
		}
	};

	std::array<lsu_state_t, NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;
};

}
