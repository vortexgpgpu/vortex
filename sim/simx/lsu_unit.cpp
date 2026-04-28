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

#include "lsu_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "local_mem_switch.h"
#include "constants.h"
#include "mem_block_pool.h"
#include "VX_types.h"
#include "VX_config.h"

using namespace vortex;

uint32_t LsuUopGen::uop_count(const Instr& instr) {
  // PACKLB.F: width=0 (LB), 4 byte-elements
  // PACKLH.F: width=1 (LH), 2 halfword-elements
  auto args = std::get<IntrLsuArgs>(instr.get_args());
  return (args.width == 0) ? 4 : 2;
}

Instr::Ptr LsuUopGen::get(const Instr& macro_instr, uint32_t uop_index) {
  uint64_t parent_uuid = macro_instr.get_uuid();
  uint32_t total = uop_count(macro_instr);

  // Embed uop_index in the upper bits of the low half of the UUID so that
  // each uop has a distinct UUID for trace logging.
  uint32_t uuid_hi = (parent_uuid >> 32) & 0xffffffff;
  uint32_t uuid_lo = parent_uuid & 0xffffffff;
  uint32_t steps_shift = (total > 1) ? (32 - log2ceil(total)) : 0;
  uint64_t uop_uuid = (static_cast<uint64_t>(uuid_hi) << 32) | ((uop_index << steps_shift) | uuid_lo);

  auto args = std::get<IntrLsuArgs>(macro_instr.get_args());
  // Macro encodes elem-size as width: 0=byte (PACKLB), 1=halfword (PACKLH).
  // The uop is a regular unsigned load (LBU/LHU = width + 4) so the LSU
  // doesn't sign-extend the loaded byte/halfword before the bytesel write.
  uint32_t elem_bytes = 1u << args.width;
  uint32_t uop_width  = args.width + 4;          // LBU=4, LHU=5
  uint32_t byte_off   = uop_index * elem_bytes;  // byte offset in dst register
  // Bytesel: data bytes for this uop's elem + NaN-box bytes (4..7) for Float
  // dest. The LSU is generic — it reads bytesel and shifts the loaded data
  // into place; OpcUnit::writeback OR-merges by mask.
  uint8_t  data_mask  = uint8_t(((1u << elem_bytes) - 1u) << byte_off);
  bool     dst_float  = (macro_instr.get_dest_reg().type == RegType::Float);
  uint8_t  bytesel    = data_mask | (dst_float ? 0xF0 : 0);

  auto uop_instr = std::allocate_shared<Instr>(pool_, uop_uuid, FUType::LSU);
  uop_instr->set_parent_uuid(parent_uuid);
  uop_instr->set_op_type(LsuType::LOAD);
  uop_instr->set_dest_reg(macro_instr.get_dest_reg().idx, macro_instr.get_dest_reg().type);
  uop_instr->set_src_reg(0, macro_instr.get_src_reg(0).idx, RegType::Integer);
  uop_instr->set_src_reg(1, macro_instr.get_src_reg(1).idx, RegType::Integer);
  // Per-uop AGU input: stride = uop_index → addr = rs1 + uop_index*rs2 + 0.
  uop_instr->set_args(IntrLsuArgs{uop_width, /*stride*/ uop_index, /*offset*/ 0});
  uop_instr->set_dst_bytesel(bytesel);
  return uop_instr;
}

LsuUnit::LsuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit(ctx, name, core)
	, pending_loads_(0)
{}

LsuUnit::~LsuUnit()
{}

void LsuUnit::on_reset() {
	for (auto& state : states_) {
		state.reset();
	}
	pending_loads_ = 0;
	remain_addrs_ = 0;
}

void LsuUnit::compute_addrs(instr_trace_t* trace) {
	// AGU formula: addr[t] = rs1[t] + stride * rs2[t] + offset
	addr_list_.clear();
	auto lsu_type = std::get<LsuType>(trace->op_type);
	auto lsu_args = std::get<IntrLsuArgs>(trace->instr_ptr->get_args());
	auto& tmask = trace->tmask;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];
	uint32_t num_threads = NUM_THREADS;
	uint32_t data_bytes = 1u << (lsu_args.width & 0x3);
	bool is_write = (lsu_type == LsuType::STORE);
	int64_t  offset = lsu_args.offset;  // already signed via int32_t
	uint32_t stride = lsu_args.stride;
	for (uint32_t t = 0; t < num_threads; ++t) {
		if (!tmask.test(t)) continue;
		mem_addr_size_t e;
		e.addr = Word(rs1_data[t].i + (uint64_t)stride * rs2_data[t].u + offset);
		e.size = data_bytes;
		e.tid  = t;
		if (is_write) {
			e.data = rs2_data[t].u64;
		}
		addr_list_.push_back(e);
	}
	if (is_write && lsu_args.width > 3)
		std::abort();
	remain_addrs_ = addr_list_.size();
}

void LsuUnit::process_response(uint32_t b) {
	auto& lsu_rsp_in = core_->lmem_switch(b)->RspOut;
	if (lsu_rsp_in.empty())
		return;
	auto& state = states_.at(b);
	auto& lsu_rsp = lsu_rsp_in.peek();
	auto& entry = state.pending_rd_reqs.at(lsu_rsp.tag);
	auto trace = entry.trace;
	int iw = trace->wid % ISSUE_WIDTH;
	auto& output = Outputs.at(iw);
	if (output.full())
		return; // stall
	DT(3, this->name() << " mem-rsp: " << lsu_rsp);
	assert(entry.count != 0);

	if (entry.is_load) {
		bool dst_float = (trace->dst_reg.type == RegType::Float);
		uint32_t data_bytes = 1u << (entry.lsu_args.width & 0x3);
		uint32_t data_width = 8 * data_bytes;
		uint8_t  data_bs    = trace->dst_bytesel & 0x0F;
		uint32_t byte_off   = (data_bs == 0 || data_bs == 0x0F) ? 0
		                    : (uint32_t)__builtin_ctz(data_bs);
		bool nan_box = dst_float && (data_bytes < 8);

		for (uint32_t lane = 0; lane < lsu_rsp.mask.size(); ++lane) {
			if (!lsu_rsp.mask.test(lane))
				continue;
			const auto& lane_info = entry.lanes.at(lane);
			assert(lsu_rsp.data.at(lane) && "LOAD response must carry line payload");
			uint32_t off = lane_info.addr & (MEM_BLOCK_SIZE - 1);
			uint64_t read_data = 0;
			std::memcpy(&read_data, lsu_rsp.data.at(lane)->data() + off, data_bytes);
			// Format the loaded value at low bits per RISC-V load semantics.
			uint64_t formatted = 0;
			switch (entry.lsu_args.width) {
			case 0: // LB
			case 1: // LH
				formatted = (uint64_t)(int64_t)sext((Word)read_data, data_width);
				break;
			case 2: // LW (sign-ext for Integer dest; raw bits for Float dest, NaN-boxed below)
				formatted = dst_float ? read_data
				                       : (uint64_t)(int64_t)sext((Word)read_data, data_width);
				break;
			case 3: // LD
			case 4: // LBU
			case 5: // LHU
			case 6: // LWU
				formatted = read_data;
				break;
			default:
				std::abort();
			}
			// Place at bytesel-specified position; OR-in NaN-box for Float dest.
			auto& dst = entry.trace->dst_data.at(lane_info.tid);
			dst.u64 = (formatted << (8 * byte_off))
			        | (nan_box ? 0xFFFFFFFF00000000ull : 0);
		}
	}
	entry.count -= lsu_rsp.mask.count(); // track remaining
	if (entry.count == 0) {
		state.pending_rd_reqs.release(lsu_rsp.tag);
		if (entry.eop) {
			output.send(trace, 1);
		}
	}
	pending_loads_ -= lsu_rsp.mask.count();
	lsu_rsp_in.pop();
}

void LsuUnit::process_request(uint32_t iw) {
	uint32_t block_idx = iw % NUM_LSU_BLOCKS;
	auto& state = states_.at(block_idx);
	if (state.fence_lock) {
		// wait for all pending memory operations to complete
		if (!state.pending_rd_reqs.empty())
			return;
		if (!Outputs.at(iw).try_send(state.fence_trace))
			return;
		state.fence_lock = false;
		DT(3, this->name() << " fence-unlock: " << state.fence_trace);
	}

	// check input queue
	auto& input = Inputs.at(iw);
	if (input.empty())
		return;

	auto trace = input.peek();
	if (!std::get_if<LsuType>(&trace->op_type)) {
		// AMO ops are unsupported in this build (EXT_A_ENABLE=false). The
		// LSU only talks through lmem_switch_; no functional bypass.
		std::abort();
	}
	auto lsu_type = std::get<LsuType>(trace->op_type);
	bool is_fence = (lsu_type == LsuType::FENCE);
	bool is_write = (lsu_type == LsuType::STORE);

	if (is_fence) {
		// schedule fence lock
		state.fence_trace = trace;
		state.fence_lock = true;
		DT(3, this->name() << " fence-lock: " << *trace);
		input.pop();
		return;
	}

	// check pending queue capacity
	if (!is_write && state.pending_rd_reqs.full()) {
		if (!trace->log_once(true)) {
			DT(4, this->name() << " queue-full: " << *trace);
		}
		return;
	} else {
		trace->log_once(false);
	}

	// First time we see this trace: derive the per-lane addr/size/data
	// list via compute_addrs(). Persists across multi-batch dispatch via
	// remain_addrs_; rebuilt only when the previous trace is fully drained.
	if (remain_addrs_ == 0) {
		this->compute_addrs(trace);
	}

	// check output backpressure
	bool direct_commit = (is_write || 0 == addr_list_.size());
	if (direct_commit && remain_addrs_ <= NUM_LSU_LANES) {
		if (Outputs.at(iw).full())
			return; // stall
	}

	if (remain_addrs_ != 0) {
		// check lmem switch backpressure
		if (core_->lmem_switch(block_idx)->ReqIn.full())
			return; // stall

		// setup memory request
		LsuReq lsu_req(NUM_LSU_LANES);
		lsu_req.write = is_write;
		uint32_t t0 = addr_list_.size() - remain_addrs_;
		std::vector<mem_addr_size_t> lane_entries(NUM_LSU_LANES);
		for (uint32_t i = 0; i < NUM_LSU_LANES; ++i) {
			auto& entry = addr_list_.at(t0 + i);
			lsu_req.mask.set(i);
			lsu_req.addrs.at(i) = entry.addr;
			lane_entries.at(i) = entry;
			if (is_write && entry.size > 0) {
				// Package the lane's write value into a per-lane block + byteen.
				auto block = make_mem_block();
				std::memset(block->data(), 0, block->size());
				uint32_t off = entry.addr & (MEM_BLOCK_SIZE - 1);
				for (uint32_t b = 0; b < entry.size; ++b) {
					(*block)[off + b] = uint8_t((entry.data >> (8 * b)) & 0xff);
				}
				lsu_req.data.at(i) = block;
				lsu_req.byteen.at(i) = ((1ull << entry.size) - 1) << off;
			}
			--remain_addrs_;
			if (remain_addrs_ == 0)
				break;
		}

		uint32_t count = lsu_req.mask.count();
		bool is_eop = (remain_addrs_ == 0);

		uint32_t tag = 0;
		if (!is_write) {
			IntrLsuArgs lsu_args = std::get<IntrLsuArgs>(trace->instr_ptr->get_args());
			tag = state.pending_rd_reqs.allocate({trace, count, is_eop, std::move(lane_entries), lsu_args, true});
		}
		lsu_req.tag  = tag;
		lsu_req.cid  = trace->cid;
		lsu_req.uuid = trace->uuid;

		// send memory request
		core_->lmem_switch(block_idx)->ReqIn.send(lsu_req);
		DT(3, this->name() << " mem-req: " << lsu_req);

		// update stats
		if (is_write) {
			core_->perf_stats().stores += count;
		} else {
			core_->perf_stats().loads += count;
			pending_loads_ += count;
		}
	}

	if (remain_addrs_ == 0) {
		if (direct_commit) {
			Outputs.at(iw).send(trace);
		}
		input.pop();
	}
}

void LsuUnit::on_tick() {
	core_->perf_stats().load_latency += pending_loads_;

	for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
		this->process_response(b);
	}

	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		this->process_request(iw);
	}
}
