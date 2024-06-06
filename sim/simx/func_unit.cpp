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

#include "func_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "constants.h"
#include "cache_sim.h"

using namespace vortex;

AluUnit::AluUnit(const SimContext& ctx, Core* core) : FuncUnit(ctx, core, "ALU") {}

void AluUnit::tick() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		auto trace = input.front();
		switch (trace->alu_type) {
		case AluType::ARITH:
		case AluType::BRANCH:
		case AluType::SYSCALL:
		case AluType::IMUL:
			output.push(trace, LATENCY_IMUL+1);
			break;
		case AluType::IDIV:
			output.push(trace, XLEN+1);
			break;
		default:
			std::abort();
		}
		DT(3, "pipeline-execute: op=" << trace->alu_type << ", " << *trace);
		if (trace->eop && trace->fetch_stall) {
			core_->resume(trace->wid);
		}
		input.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(const SimContext& ctx, Core* core) : FuncUnit(ctx, core, "FPU") {}

void FpuUnit::tick() {
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		auto trace = input.front();
		switch (trace->fpu_type) {
		case FpuType::FNCP:
			output.push(trace, 2);
			break;
		case FpuType::FMA:
			output.push(trace, LATENCY_FMA+1);
			break;
		case FpuType::FDIV:
			output.push(trace, LATENCY_FDIV+1);
			break;
		case FpuType::FSQRT:
			output.push(trace, LATENCY_FSQRT+1);
			break;
		case FpuType::FCVT:
			output.push(trace, LATENCY_FCVT+1);
			break;
		default:
			std::abort();
		}
		DT(3, "pipeline-execute: op=" << trace->fpu_type << ", " << *trace);
		input.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////

LsuUnit::LsuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "LSU")
	, pending_loads_(0)
{}

LsuUnit::~LsuUnit()
{}

void LsuUnit::reset() {
	for (auto& state : states_) {
		state.clear();
	}
	pending_loads_ = 0;
}

void LsuUnit::tick() {
	core_->perf_stats_.load_latency += pending_loads_;

	// handle memory responses
	for (uint32_t r = 0; r < LSU_NUM_REQS; ++r) {
		auto& dcache_rsp_port = core_->lsu_demux_.at(r)->RspIn;
		if (dcache_rsp_port.empty())
			continue;
		uint32_t block_idx = r / LSU_CHANNELS;
		auto& state = states_.at(block_idx);
		auto& mem_rsp = dcache_rsp_port.front();
		auto& entry = state.pending_rd_reqs.at(mem_rsp.tag);
		auto trace = entry.trace;
		DT(3, "mem-rsp: tag=" << mem_rsp.tag << ", type=" << trace->lsu_type << ", rid=" << r << ", " << *trace);
		assert(entry.count);
		--entry.count; // track remaining addresses
		if (0 == entry.count) {
			int iw = trace->wid % ISSUE_WIDTH;
			Outputs.at(iw).push(trace, 1);
			state.pending_rd_reqs.release(mem_rsp.tag);
		}
		dcache_rsp_port.pop();
		--pending_loads_;
	}

	// handle LSU requests
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		uint32_t block_idx = iw % NUM_LSU_BLOCKS;
		auto& state = states_.at(block_idx);
		if (state.fence_lock) {
			// wait for all pending memory operations to complete
			if (!state.pending_rd_reqs.empty())
				continue;
			Outputs.at(iw).push(state.fence_trace, 1);
			state.fence_lock = false;
			DT(3, "fence-unlock: " << state.fence_trace);
		}

		// check input queue
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;

		auto& output = Outputs.at(iw);
		auto trace = input.front();

		if (trace->lsu_type == LsuType::FENCE) {
			// schedule fence lock
			state.fence_trace = trace;
			state.fence_lock = true;
			DT(3, "fence-lock: " << *trace);
			// remove input
			input.pop();
			continue;
		}

		bool is_write = (trace->lsu_type == LsuType::STORE);

		// check pending queue capacity
		if (!is_write && state.pending_rd_reqs.full()) {
			if (!trace->log_once(true)) {
				DT(4, "*** " << this->name() << "-queue-full: " << *trace);
			}
			continue;
		} else {
			trace->log_once(false);
		}

		uint32_t tag = 0;
		if (!is_write) {
			tag = state.pending_rd_reqs.allocate({trace, 0});
		}

		// send memory request
		auto num_reqs = this->send_requests(trace, block_idx, tag);

		if (!is_write) {
			state.pending_rd_reqs.at(tag).count = num_reqs;
		}

		// do not wait on writes
		if (is_write) {
			output.push(trace, 1);
		}

		// remove input
		input.pop();
	}
}

int LsuUnit::send_requests(instr_trace_t* trace, int block_idx, int tag) {
	int count = 0;

	auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);
	bool is_write = (trace->lsu_type == LsuType::STORE);
	auto t0 = trace->pid * NUM_LSU_LANES;

	for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
		uint32_t t = t0 + i;
		if (!trace->tmask.test(t))
			continue;

		int req_idx = block_idx * LSU_CHANNELS + (i % LSU_CHANNELS);
		auto& dcache_req_port = core_->lsu_demux_.at(req_idx)->ReqIn;

		auto mem_addr = trace_data->mem_addrs.at(t);
		auto type = get_addr_type(mem_addr.addr);

		MemReq mem_req;
		mem_req.addr  = mem_addr.addr;
		mem_req.write = is_write;
		mem_req.type  = type;
		mem_req.tag   = tag;
		mem_req.cid   = trace->cid;
		mem_req.uuid  = trace->uuid;

		dcache_req_port.push(mem_req, 1);
		DT(3, "mem-req: addr=0x" << std::hex << mem_req.addr << ", tag=" << tag
			<< ", lsu_type=" << trace->lsu_type << ", rid=" << req_idx << ", addr_type=" << mem_req.type << ", " << *trace);

		if (is_write) {
			++core_->perf_stats_.stores;
		} else {
			++core_->perf_stats_.loads;
			++pending_loads_;
		}

		++count;
	}
	return count;
}

///////////////////////////////////////////////////////////////////////////////

SfuUnit::SfuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "SFU")
{}

void SfuUnit::tick() {
	// check input queue
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		auto trace = input.front();
		auto sfu_type = trace->sfu_type;
		bool release_warp = trace->fetch_stall;

		switch  (sfu_type) {
		case SfuType::WSPAWN:
			output.push(trace, 1);
			if (trace->eop) {
				auto trace_data = std::dynamic_pointer_cast<SFUTraceData>(trace->data);
				release_warp = core_->wspawn(trace_data->arg1, trace_data->arg2);
			}
			break;
		case SfuType::TMC:
		case SfuType::SPLIT:
		case SfuType::JOIN:
		case SfuType::PRED:
		case SfuType::CSRRW:
		case SfuType::CSRRS:
		case SfuType::CSRRC:
			output.push(trace, 1);
			break;
		case SfuType::BAR: {
			output.push(trace, 1);
			if (trace->eop) {
				auto trace_data = std::dynamic_pointer_cast<SFUTraceData>(trace->data);
				release_warp = core_->barrier(trace_data->arg1, trace_data->arg2, trace->wid);
			}
		} break;
		default:
			std::abort();
		}

		DT(3, "pipeline-execute: op=" << trace->sfu_type << ", " << *trace);
		if (trace->eop && release_warp)  {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}
