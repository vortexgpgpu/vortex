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
#include "VX_types.h"

using namespace vortex;

AluUnit::AluUnit(const SimContext& ctx, Core* core) : FuncUnit(ctx, core, "alu-unit") {}

void AluUnit::tick() {
  for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		auto trace = input.front();
		int delay = 2;
		switch (trace->alu_type) {
		case AluType::ARITH:
		case AluType::BRANCH:
		case AluType::SYSCALL:
			output.push(trace, 2+delay);
			break;
		case AluType::IMUL:
			output.push(trace, LATENCY_IMUL+delay);
			break;
		case AluType::IDIV:
			output.push(trace, XLEN+delay);
			break;
		default:
			std::abort();
		}
		DT(3, this->name() << ": op=" << trace->alu_type << ", " << *trace);
		if (trace->eop && trace->fetch_stall) {
			core_->resume(trace->wid);
		}
		input.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////

FpuUnit::FpuUnit(const SimContext& ctx, Core* core) : FuncUnit(ctx, core, "fpu-unit") {}

void FpuUnit::tick() {
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		auto trace = input.front();
		int delay = 2;
		switch (trace->fpu_type) {
		case FpuType::FNCP:
			output.push(trace, 2+delay);
			break;
		case FpuType::FMA:
			output.push(trace, LATENCY_FMA+delay);
			break;
		case FpuType::FDIV:
			output.push(trace, LATENCY_FDIV+delay);
			break;
		case FpuType::FSQRT:
			output.push(trace, LATENCY_FSQRT+delay);
			break;
		case FpuType::FCVT:
			output.push(trace, LATENCY_FCVT+delay);
			break;
		default:
			std::abort();
		}
		DT(3,this->name() << ": op=" << trace->fpu_type << ", " << *trace);
		input.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////

LsuUnit::LsuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "lsu-unit")
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
	for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
		auto& lsu_rsp_port = core_->lmem_switch_.at(b)->RspIn;
		if (lsu_rsp_port.empty())
			continue;
		auto& state = states_.at(b);
		auto& lsu_rsp = lsu_rsp_port.front();
		DT(3, this->name() << "-mem-rsp: " << lsu_rsp);
		auto& entry = state.pending_rd_reqs.at(lsu_rsp.tag);
		auto trace = entry.trace;
		assert(!entry.mask.none());
		entry.mask &= ~lsu_rsp.mask; // track remaining
		if (entry.mask.none()) {
			// whole response received, release trace
			int iw = trace->wid % ISSUE_WIDTH;
			Outputs.at(iw).push(trace, 1);
			state.pending_rd_reqs.release(lsu_rsp.tag);
		}
		pending_loads_ -= lsu_rsp.mask.count();
		lsu_rsp_port.pop();
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
			DT(3, this->name() << "-fence-unlock: " << state.fence_trace);
		}

		// check input queue
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;

		auto trace = input.front();

		if (trace->lsu_type == LsuType::FENCE) {
			// schedule fence lock
			state.fence_trace = trace;
			state.fence_lock = true;
			DT(3, this->name() << "-fence-lock: " << *trace);
			// remove input
			input.pop();
			continue;
		}

		bool is_write = ((trace->lsu_type == LsuType::STORE) || (trace->lsu_type == LsuType::TCU_STORE));

		// check pending queue capacity
		if (!is_write && state.pending_rd_reqs.full()) {
			if (!trace->log_once(true)) {
				DT(4, "*** " << this->name() << "-queue-full: " << *trace);
			}
			continue;
		} else {
			trace->log_once(false);
		}

		// build memory request
		LsuReq lsu_req(NUM_LSU_LANES);
		lsu_req.write = is_write;
		{
			auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);
			auto t0 = trace->pid * NUM_LSU_LANES;
			for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
				if (trace->tmask.test(t0 + i)) {
					lsu_req.mask.set(i);
					lsu_req.addrs.at(i) = trace_data->mem_addrs.at(t0 + i).addr;
				}
			}
		}
		uint32_t tag = 0;

		if (!is_write) {
			tag = state.pending_rd_reqs.allocate({trace, lsu_req.mask});
		}
		lsu_req.tag  = tag;
		lsu_req.cid  = trace->cid;
		lsu_req.uuid = trace->uuid;

		// send memory request
		core_->lmem_switch_.at(block_idx)->ReqIn.push(lsu_req);
		DT(3, this->name() << "-mem-req: " << lsu_req);

		// update stats
		auto num_addrs = lsu_req.mask.count();
		if (is_write) {
			core_->perf_stats_.stores += num_addrs;
		} else {
			core_->perf_stats_.loads += num_addrs;
			pending_loads_ += num_addrs;
		}

		// do not wait on writes
		if (is_write) {
			Outputs.at(iw).push(trace, 1);
		}

		// remove input
		input.pop();
	}
}
/*  TO BE FIXED:Tensor_core code
    send_request is not used anymore. Need to be modified number of load
*/
/*
int LsuUnit::send_requests(instr_trace_t* trace, int block_idx, int tag) {
	int count = 0;

	auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);
	bool is_write = ((trace->lsu_type == LsuType::STORE) || (trace->lsu_type == LsuType::TCU_STORE));

	uint16_t req_per_thread = 1;
	if ((trace->lsu_type == LsuType::TCU_LOAD) || (trace->lsu_type == LsuType::TCU_STORE))
	{
 		req_per_thread= (1>(trace_data->mem_addrs.at(0).size)/4)? 1: ((trace_data->mem_addrs.at(0).size)/4);
	}

	auto t0 = trace->pid * NUM_LSU_LANES;

	for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
		uint32_t t = t0 + i;
		if (!trace->tmask.test(t))
			continue;

		int req_idx = block_idx * LSU_CHANNELS + (i % LSU_CHANNELS);
		auto& dcache_req_port = core_->lmem_switch_.at(req_idx)->ReqIn;

		auto mem_addr = trace_data->mem_addrs.at(t);
		auto type = get_addr_type(mem_addr.addr);
		// DT(3, "addr_type = " << type << ", " << *trace);
		uint32_t mem_bytes = 1;
		for (int i = 0; i < req_per_thread; i++)
		{
			MemReq mem_req;
			mem_req.addr  = mem_addr.addr + (i*mem_bytes);
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
	}
	return count;
}
*/

///////////////////////////////////////////////////////////////////////////////

TcuUnit::TcuUnit(const SimContext& ctx, Core* core)
    : FuncUnit(ctx, core, "TCU")
    {}

void TcuUnit::tick() {

	for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
        auto& input = Inputs.at(i);
        if (input.empty())
            continue;
        auto& output = Outputs.at(i);
        auto trace = input.front();
        uint32_t n_tiles = core_->emulator_.get_tiles();
		uint32_t tc_size = core_->emulator_.get_tc_size();

        switch (trace->tcu_type) {
            case TCUType::TCU_MUL:
            {    //mat size = n_tiles * tc_size
                int matmul_latency = (n_tiles * tc_size) + tc_size + tc_size;
                output.push(trace, matmul_latency);
				DT(3, "matmul_latency = " << matmul_latency << ", " << *trace);
                break;
            }
            default:
                std::abort();
        }
        DT(3, "pipeline-execute: op=" << trace->tcu_type << ", " << *trace);
        input.pop();
    }
}

///////////////////////////////////////////////////////////////////////////////

SfuUnit::SfuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "sfu-unit")
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
		int delay = 2;
		switch  (sfu_type) {
		case SfuType::WSPAWN:
			output.push(trace, 2+delay);
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
			output.push(trace, 2+delay);
			break;
		case SfuType::BAR: {
			output.push(trace, 2+delay);
			if (trace->eop) {
				auto trace_data = std::dynamic_pointer_cast<SFUTraceData>(trace->data);
				release_warp = core_->barrier(trace_data->arg1, trace_data->arg2, trace->wid);
			}
		} break;
		default:
			std::abort();
		}

		DT(3, this->name() << ": op=" << trace->sfu_type << ", " << *trace);
		if (trace->eop && release_warp)  {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}
