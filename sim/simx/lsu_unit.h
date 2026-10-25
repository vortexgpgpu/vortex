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
#include <list>
#include <mempool.h>
#include <ringqueue.h>
#include "func_unit.h"
#include "instr.h"
#include "VX_config.h"

// The TCU metadata SRAM is present when any metadata-consuming mode (MX or
// sparse) is enabled. Internal derived macro — not a VX_CFG_* knob; same
// derivation as tcu_unit.h.
#if defined(VX_CFG_TCU_MX_ENABLE) || defined(VX_CFG_TCU_SPARSE_ENABLE)
#define TCU_META_ENABLE
#endif

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
  // trace forwarded). Conditions: pending-requests table empty AND
  // output channel can accept the trace.
  bool try_release(SimChannel<instr_trace_t*>& out, bool pending_empty) {
    if (!locked_) return false;
    if (!pending_empty) return false;
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

class LsuUnit : public FuncUnit<VX_CFG_NUM_LSU_BLOCKS> {
public:
	LsuUnit(const SimContext& ctx, const char* name, Core*);
	~LsuUnit();

#ifdef TCU_META_ENABLE
	// TCU metadata client port on block 0: TCU_LD load requests arbitrate
	// with the LSU's own traffic into lmem_switch; response fragments return
	// here with the client's own tag restored.
	SimChannel<LsuReq> TcuReqIn;
	SimChannel<LsuRsp> TcuRspOut;
#endif

	// Returns true when all LSU blocks have no in-flight requests.
	bool drained() const;

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
		uint64_t amo_cmp;
		uint32_t tid;
	};

	// A register-pair atomic in flight. A warp's pair uops reach its block in
	// program order, so each state is matched by (warp, packet) to the oldest
	// one still waiting for that uop: AMO_CMP opens it with the comparand, the
	// AMO claims it and fills `hi` from its responses, and AMO_HI retires the
	// high words once every beat of the AMO has returned.
	struct pair_state_t {
		uint32_t       wid;
		int            pid;
		std::vector<uint64_t> cmp;
		std::vector<uint32_t> hi;
		bool           has_amo  = false;
		uint32_t       beats    = 0;
		bool           issued   = false;
		bool           has_hi   = false;
		instr_trace_t* hi_trace = nullptr;
	};

	// Register-pair atomic support (64-bit atomics on a 32-bit hart).
	static bool is_pair_amo(const instr_trace_t* trace);
	bool process_pair_uop(uint32_t b, instr_trace_t* trace);
	bool process_pair_response(uint32_t b);

 	struct pending_req_t {
		instr_trace_t* trace;
		uint32_t count;
		bool eop;
		std::vector<mem_addr_size_t> lanes;
		IntrLsuArgs lsu_args;
		bool        is_load;
		// TCU metadata client entry: response fragments are forwarded to
		// TcuRspOut with client_tag restored instead of written to registers.
		bool        is_tcu     = false;
		uint32_t    client_tag = 0;
		// Register-pair atomic beat: one thread's 64-bit operation.
		pair_state_t* pair     = nullptr;
		// The fence's own request: its response completes the fence and
		// writes no register.
		bool        is_fence   = false;
	};

	// Per-block LSU state. Each member is a named hardware sub-block.
	struct lsu_state_t {
		RingQueue<instr_trace_t*> req_queue{VX_CFG_LSU_QUEUE_IN_SIZE};
		// In-flight LSU requests, keyed by the tag the LSU allocates on
		// issue. Stores the originating trace + per-lane (tid, addr, size)
		// info needed to write the response back to the right registers.
		// (Not a miss-status table — the cache has its own MSHR for that.)
		// Sized by the outstanding pool (MLP depth), decoupled from the
		// input staging queue above.
		HashTable<pending_req_t>  pending_reqs{VX_CFG_LSU_PENDING_SIZE};
		// Reads waiting in req_queue hold a pending_reqs slot from the moment
		// they enter it, so staged and outstanding reads share the pool.
		uint32_t                  staged_reads = 0;
		bool                      head_staged = false;
		FenceController           fence;
		std::vector<mem_addr_size_t> addr_list;
		uint32_t                  remain_addrs = 0;
		std::list<pair_state_t>   pairs;
		// A pair response retires in two beats, low half then high half; the
		// response stays at the head of the stream between them.
		bool                      rsp_hi_phase = false;
		pair_state_t*             rsp_pair = nullptr;
		// the pair state of the trace at the head of req_queue
		pair_state_t*             head_pair = nullptr;
		instr_trace_t*            rsp_hi_beat = nullptr;

		void reset() {
			this->req_queue.clear();
			this->pending_reqs.clear();
			this->staged_reads = 0;
			this->head_staged = false;
			this->fence.reset();
			this->addr_list.clear();
			this->remain_addrs = 0;
			this->pairs.clear();
			this->rsp_hi_phase = false;
			this->rsp_hi_beat = nullptr;
			this->rsp_pair = nullptr;
			this->head_pair = nullptr;
		}
	};

	std::array<lsu_state_t, VX_CFG_NUM_LSU_BLOCKS> states_;
	uint64_t pending_loads_;
};

}
