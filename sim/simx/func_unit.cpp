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
		int delay = 0;
		if (std::get_if<AluType>(&trace->op_type)) {
			auto alu_type = std::get<AluType>(trace->op_type);
			switch (alu_type) {
			case AluType::LUI:
			case AluType::AUIPC:
			case AluType::ADD:
			case AluType::SUB:
			case AluType::SLL:
			case AluType::SRL:
			case AluType::SRA:
			case AluType::SLT:
			case AluType::SLTU:
			case AluType::XOR:
			case AluType::AND:
			case AluType::OR:
			case AluType::CZERO:
				delay = 2;
				break;
			default:
				std::abort();
			}
			DT(3, this->name() << ": op=" << alu_type << ", " << *trace);
		} else if (std::get_if<VoteType>(&trace->op_type)) {
				delay = 2;
		} else if (std::get_if<ShflType>(&trace->op_type)) {
				delay = 2;
		} else if (std::	get_if<BrType>(&trace->op_type)) {
			auto br_type = std::get<BrType>(trace->op_type);
			switch (br_type) {
			case BrType::BR:
			case BrType::JAL:
			case BrType::JALR:
			case BrType::SYS:
				delay = 2;
				break;
			default:
				std::abort();
			}
			DT(3, this->name() << ": op=" << br_type << ", " << *trace);
		} else if (std::get_if<MdvType>(&trace->op_type)) {
			auto mdv_type = std::get<MdvType>(trace->op_type);
			switch (mdv_type) {
			case MdvType::MUL:
			case MdvType::MULHU:
			case MdvType::MULH:
			case MdvType::MULHSU:
				delay = 2;
				break;
			case MdvType::DIV:
			case MdvType::DIVU:
			case MdvType::REM:
			case MdvType::REMU:
				delay = XLEN+2;
				break;
			default:
				std::abort();
			}
			DT(3, this->name() << ": op=" << mdv_type << ", " << *trace);
		} else {
			std::abort();
		}
		output.push(trace, delay);
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
		auto fpu_type = std::get<FpuType>(trace->op_type);
		int delay = 2;
		switch (fpu_type) {
		case FpuType::FCMP:
		case FpuType::FSGNJ:
		case FpuType::FCLASS:
		case FpuType::FMVXW:
		case FpuType::FMVWX:
		case FpuType::FMINMAX:
			output.push(trace, 2+delay);
			break;
		case FpuType::FADD:
		case FpuType::FSUB:
		case FpuType::FMUL:
		case FpuType::FMADD:
		case FpuType::FMSUB:
		case FpuType::FNMADD:
		case FpuType::FNMSUB:
			output.push(trace, LATENCY_FMA+delay);
			break;
		case FpuType::FDIV:
			output.push(trace, LATENCY_FDIV+delay);
			break;
		case FpuType::FSQRT:
			output.push(trace, LATENCY_FSQRT+delay);
			break;
		case FpuType::F2I:
		case FpuType::I2F:
		case FpuType::F2F:
			output.push(trace, LATENCY_FCVT+delay);
			break;
		default:
			std::abort();
		}
		DT(3,this->name() << ": op=" << fpu_type << ", " << *trace);
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
		state.reset();
	}
	pending_loads_ = 0;
	remain_addrs_ = 0;
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
		assert(entry.count != 0);
		entry.count -= lsu_rsp.mask.count(); // track remaining
		if (entry.count == 0) {
			// full response batch received
			state.pending_rd_reqs.release(lsu_rsp.tag);
			// is last batch?
			if (entry.eop) {
				int iw = trace->wid % ISSUE_WIDTH;
				Outputs.at(iw).push(trace, 1);
			}
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

		bool is_fence = false;
		bool is_write = false;

		auto trace = input.front();
		if (std::get_if<LsuType>(&trace->op_type)) {
			auto lsu_type = std::get<LsuType>(trace->op_type);
			is_fence = (lsu_type == LsuType::FENCE);
			is_write = (lsu_type == LsuType::STORE);
		} else if (std::get_if<AmoType>(&trace->op_type)) {
			auto amp_type = std::get<AmoType>(trace->op_type);
			is_write = (amp_type != AmoType::LR);
		}
	#ifdef EXT_V_ENABLE
		else if (std::get_if<VlsType>(&trace->op_type)) {
			auto vls_type = std::get<VlsType>(trace->op_type);
			is_write = (vls_type == VlsType::VS
			         || vls_type == VlsType::VSS
							 || vls_type == VlsType::VSX);
		}
	#endif // EXT_V_ENABLE
		else {
			std::abort();
		}

		if (is_fence) {
			// schedule fence lock
			state.fence_trace = trace;
			state.fence_lock = true;
			DT(3, this->name() << "-fence-lock: " << *trace);
			// remove input
			input.pop();
			continue;
		}

		// check pending queue capacity
		if (!is_write && state.pending_rd_reqs.full()) {
			if (!trace->log_once(true)) {
				DT(4, "*** " << this->name() << "-queue-full: " << *trace);
			}
			continue;
		} else {
			trace->log_once(false);
		}

		if (remain_addrs_ == 0) {
			pending_addrs_.clear();
			if (trace->data) {
			#ifdef EXT_V_ENABLE
				if (std::get_if<VlsType>(&trace->op_type)) {
					auto trace_data = std::dynamic_pointer_cast<VecUnit::MemTraceData>(trace->data);
					for (uint32_t t = 0; t < trace_data->mem_addrs.size(); ++t) {
						if (!trace->tmask.test(t))
							continue;
						for (auto addr : trace_data->mem_addrs.at(t)) {
							pending_addrs_.push_back(addr);
						}
					}
				} else
			#endif
				{
					auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);
					for (uint32_t t = 0; t < trace_data->mem_addrs.size(); ++t) {
						if (!trace->tmask.test(t))
							continue;
						pending_addrs_.push_back(trace_data->mem_addrs.at(t));
					}
				}
				remain_addrs_ = pending_addrs_.size();
			}
		}

		if (remain_addrs_ != 0) {
			// setup memory request
			LsuReq lsu_req(NUM_LSU_LANES);
			lsu_req.write = is_write;
			uint32_t t0 = pending_addrs_.size() - remain_addrs_;
			for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
				lsu_req.mask.set(i);
				lsu_req.addrs.at(i) = pending_addrs_.at(t0 + i).addr;
				--remain_addrs_;
				if (remain_addrs_ == 0)
					break;
			}

			uint32_t count = lsu_req.mask.count();
			bool is_eop = (remain_addrs_ == 0);

			uint32_t tag = 0;
			if (!is_write) {
				tag = state.pending_rd_reqs.allocate({trace, count, is_eop});
			}
			lsu_req.tag  = tag;
			lsu_req.cid  = trace->cid;
			lsu_req.uuid = trace->uuid;

			// send memory request
			core_->lmem_switch_.at(block_idx)->ReqIn.push(lsu_req);
			DT(3, this->name() << "-mem-req: " << lsu_req);

			// update stats
			if (is_write) {
				core_->perf_stats_.stores += count;
			} else {
				core_->perf_stats_.loads += count;
				pending_loads_ += count;
			}
		}

		if (remain_addrs_ == 0) {
			// do not wait on writes
			if (is_write || 0 == pending_addrs_.size()) {
				Outputs.at(iw).push(trace, 1);
			}
			// remove input
			input.pop();
		}
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
		bool release_warp = trace->fetch_stall;
		int delay = 2;

		if (std::get_if<WctlType>(&trace->op_type)) {
			auto wctl_type = std::get<WctlType>(trace->op_type);
			switch (wctl_type) {
			case WctlType::WSPAWN:
				output.push(trace, 2+delay);
				if (trace->eop) {
					auto trace_data = std::dynamic_pointer_cast<SfuTraceData>(trace->data);
					release_warp = core_->wspawn(trace_data->arg1, trace_data->arg2);
				}
				break;
			case WctlType::TMC:
			case WctlType::SPLIT:
			case WctlType::JOIN:
			case WctlType::PRED:
				output.push(trace, 2+delay);
				break;
			case WctlType::BAR: {
				output.push(trace, 2+delay);
				if (trace->eop) {
					auto trace_data = std::dynamic_pointer_cast<SfuTraceData>(trace->data);
					release_warp = core_->barrier(trace_data->arg1, trace_data->arg2, trace->wid);
				}
			} break;
			default:
				std::abort();
			}
			DT(3, this->name() << ": op=" << wctl_type << ", " << *trace);
		} else if (std::get_if<CsrType>(&trace->op_type)) {
			auto csr_type = std::get<CsrType>(trace->op_type);
			switch  (csr_type) {
			case CsrType::CSRRW:
			case CsrType::CSRRS:
			case CsrType::CSRRC:
				output.push(trace, 2+delay);
				break;
			default:
				std::abort();
			}
			DT(3, this->name() << ": op=" << csr_type << ", " << *trace);
		} else {
			std::abort();
		}

		if (trace->eop && release_warp)  {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}

///////////////////////////////////////////////////////////////////////////////

#ifdef EXT_V_ENABLE

VpuUnit::VpuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "vpu-unit")
{
	// bind vector unit
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		this->Inputs.at(iw).bind(&core_->vec_unit()->Inputs.at(iw));
		core_->vec_unit()->Outputs.at(iw).bind(&this->Outputs.at(iw));
	}
}

void VpuUnit::tick() {
	// use vec_unit
}
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef EXT_TCU_ENABLE

TcuUnit::TcuUnit(const SimContext& ctx, Core* core)
	: FuncUnit(ctx, core, "tcu-unit")
{
	// bind tensor unit
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		this->Inputs.at(iw).bind(&core_->tensor_unit()->Inputs.at(iw));
		core_->tensor_unit()->Outputs.at(iw).bind(&this->Outputs.at(iw));
	}
}

void TcuUnit::tick() {
	// use tensor_unit
}
#endif
