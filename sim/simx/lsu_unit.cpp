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
#include "socket.h"
#include "cluster.h"
#include "processor_impl.h"
#include "constants.h"
#include "cache.h"
#include "VX_types.h"
#include "VX_config.h"

using namespace vortex;

namespace {
inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

// Functional RAM access used by the LSU's synchronous AMO and packed-load
// fast-paths. Global addrs hit the processor-wide RAM; shared addrs hit the
// per-core LocalMem. The cache hierarchy is bypassed.
void func_read(Core* core, void* data, uint64_t addr, uint32_t size) {
  if (get_addr_type(addr) == AddrType::Shared) {
    core->local_mem()->read(data, addr, size);
  } else {
    core->processor()->ram()->read(data, addr, size);
  }
}

void func_write(Core* core, const void* data, uint64_t addr, uint32_t size) {
  if (get_addr_type(addr) == AddrType::Shared) {
    core->local_mem()->write(data, addr, size);
  } else {
    core->processor()->ram()->write(data, addr, size);
  }
}
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

void LsuUnit::execute(instr_trace_t* trace) {
	// Use the trace's snapshot tmask captured at issue, not the live warp.tmask.
	// Divergent control flow may have changed warp.tmask after this trace was
	// issued; the on_tick handler builds its lsu_req from trace->tmask, so any
	// mismatch here yields lanes whose addr/data were never populated.
	auto& tmask = trace->tmask;
	auto& instr = *trace->instr_ptr;
	auto instrArgs = instr.get_args();
	uint32_t num_threads = NUM_THREADS;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];

	uint32_t thread_start = 0;
	for (; thread_start < num_threads; ++thread_start) {
		if (tmask.test(thread_start)) break;
	}

	trace->dst_data.assign(num_threads, reg_data_t{});
	auto& rd_data = trace->dst_data;

	if (std::get_if<LsuType>(&trace->op_type)) {
		auto lsu_type = std::get<LsuType>(trace->op_type);
		auto lsuArgs = std::get<IntrLsuArgs>(instrArgs);
		switch (lsu_type) {
		case LsuType::LOAD: {
			auto trace_data = std::make_shared<LsuTraceData>(num_threads);
			trace->data = trace_data;
			uint32_t data_bytes = 1 << (lsuArgs.width & 0x3);
			Word offset = sext<Word>(lsuArgs.offset, 32);
			if (lsuArgs.pack != 0) {
				// Packed-load is a Vortex-only multi-element bulk read; keep its
				// synchronous functional path. The on_tick rsp handler is told
				// not to overwrite dst_data via the is_load=false flag.
				uint32_t elem_bytes = (lsuArgs.pack == 1) ? 1 : 2;
				uint32_t num_elems  = (lsuArgs.pack == 1) ? 4 : 2;
				uint32_t elem_mask  = (elem_bytes == 1) ? 0xffu : 0xffffu;
				for (uint32_t t = thread_start; t < num_threads; ++t) {
					if (!tmask.test(t))
						continue;
					uint64_t base   = rs1_data[t].u;
					uint64_t stride = rs2_data[t].u;
					uint32_t packed = 0;
					for (uint32_t i = 0; i < num_elems; ++i) {
						uint64_t elem_addr = base + i * stride;
						uint64_t elem_data = 0;
						func_read(core_,&elem_data, elem_addr, elem_bytes);
						packed |= (uint32_t)(elem_data & elem_mask) << (8 * elem_bytes * i);
						trace_data->mem_addrs.at(t) = {elem_addr, elem_bytes};
					}
					rd_data[t].u64 = nan_box(packed);
				}
				break;
			}
			// Async path: record per-thread addrs only; rsp handler fills dst_data.
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t))
					continue;
				uint64_t mem_addr = rs1_data[t].i + offset;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
			}
		} break;
		case LsuType::STORE: {
			auto trace_data = std::make_shared<LsuTraceData>(num_threads);
			trace->data = trace_data;
			uint32_t data_bytes = 1 << (lsuArgs.width & 0x3);
			Word offset = sext<Word>(lsuArgs.offset, 32);
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t))
					continue;
				uint64_t mem_addr = rs1_data[t].i + offset;
				uint64_t write_data = rs2_data[t].u64;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes, write_data};
				if (lsuArgs.width > 3)
					std::abort();
			}
		} break;
		case LsuType::FENCE:
			// no compute
			break;
		default:
			std::abort();
		}
	} else if (std::get_if<AmoType>(&trace->op_type)) {
		auto amo_type = std::get<AmoType>(trace->op_type);
		auto amoArgs = std::get<IntrAmoArgs>(instrArgs);
		auto trace_data = std::make_shared<LsuTraceData>(num_threads);
		trace->data = trace_data;
		uint32_t data_bytes = 1 << (amoArgs.width & 0x3);
		uint32_t data_width = 8 * data_bytes;
		switch (amo_type) {
		case AmoType::LR: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				if (get_addr_type(mem_addr) == AddrType::Global)
					core_->processor()->amo_reserve(mem_addr);
				rd_data[t].i = sext((Word)read_data, data_width);
			}
		} break;
		case AmoType::SC: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				if ((get_addr_type(mem_addr) == AddrType::Global && core_->processor()->amo_check(mem_addr))) {
					func_write(core_,&rs2_data[t].u64, mem_addr, data_bytes);
					trace_data->mem_addrs.at(t) = {mem_addr, data_bytes, rs2_data[t].u64};
					rd_data[t].i = 0;
				} else {
					// reservation invalid — record the addr but mark the lane as
					// "no actual write" via size=0 so the cache path skips it.
					trace_data->mem_addrs.at(t) = {mem_addr, 0, 0};
					rd_data[t].i = 1;
				}
			}
		} break;
		case AmoType::AMOADD: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
				uint64_t result = read_data_i + rs1_data_i;
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOSWAP: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = rs1_data_u;
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOXOR: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto read_data_u = zext((Word)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = read_data_u ^ rs1_data_u;
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOOR: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto read_data_u = zext((Word)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = read_data_u | rs1_data_u;
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOAND: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto read_data_u = zext((Word)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = read_data_u & rs1_data_u;
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOMIN: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
				uint64_t result = std::min(read_data_i, rs1_data_i);
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOMAX: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
				uint64_t result = std::max(read_data_i, rs1_data_i);
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOMINU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto read_data_u = zext((Word)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = std::min(read_data_u, rs1_data_u);
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		case AmoType::AMOMAXU: {
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				uint64_t mem_addr = rs1_data[t].u;
				trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
				uint64_t read_data = 0;
				func_read(core_,&read_data, mem_addr, data_bytes);
				auto read_data_i = sext((WordI)read_data, data_width);
				auto read_data_u = zext((Word)read_data, data_width);
				auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
				uint64_t result = std::max(read_data_u, rs1_data_u);
				func_write(core_,&result, mem_addr, data_bytes);
				trace_data->mem_addrs.at(t).data = result;
				rd_data[t].i = read_data_i;
			}
		} break;
		default:
			std::abort();
		}
#ifdef EXT_V_ENABLE
	} else if (std::get_if<VlsType>(&trace->op_type)) {
		auto vls_type = std::get<VlsType>(trace->op_type);
		auto trace_data = std::make_shared<VecUnit::MemTraceData>(num_threads);
		trace->data = trace_data;
		switch (vls_type) {
		case VlsType::VL:
		case VlsType::VLS:
		case VlsType::VLX:
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				core_->vec_unit()->load(instr, trace->wid, t, rs1_data, rs2_data, trace_data.get());
			}
			break;
		case VlsType::VS:
		case VlsType::VSS:
		case VlsType::VSX:
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				if (!tmask.test(t)) continue;
				core_->vec_unit()->store(instr, trace->wid, t, rs1_data, rs2_data, trace_data.get());
			}
			break;
		default:
			std::abort();
		}
#endif
	} else {
		std::abort();
	}
}

void LsuUnit::on_tick() {
	core_->perf_stats_.load_latency += pending_loads_;

	// handle memory responses
	for (uint32_t b = 0; b < NUM_LSU_BLOCKS; ++b) {
		auto& lsu_rsp_in = core_->lmem_switch_.at(b)->RspOut;
		if (lsu_rsp_in.empty())
			continue;
		auto& state = states_.at(b);
		auto& lsu_rsp = lsu_rsp_in.peek();
		auto& entry = state.pending_rd_reqs.at(lsu_rsp.tag);
		auto trace = entry.trace;
		int iw = trace->wid % ISSUE_WIDTH;
		auto& output = Outputs.at(iw);
		if (output.full())
			continue; // stall
		DT(3, this->name() << " mem-rsp: " << lsu_rsp);
		assert(entry.count != 0);

		// Async LOAD: extract per-lane bytes from the TLM payload and format
		// into the trace's dst_data using the captured lsu_args.
		if (entry.is_load) {
			uint32_t data_bytes = 1u << (entry.lsu_args.width & 0x3);
			uint32_t data_width = 8 * data_bytes;
			for (uint32_t lane = 0; lane < lsu_rsp.mask.size(); ++lane) {
				if (!lsu_rsp.mask.test(lane))
					continue;
				const auto& lane_info = entry.lanes.at(lane);
				assert(lsu_rsp.data.at(lane) && "LOAD response must carry line payload");
				uint32_t off = lane_info.addr & (MEM_BLOCK_SIZE - 1);
				uint64_t read_data = 0;
				std::memcpy(&read_data, lsu_rsp.data.at(lane)->data() + off, data_bytes);
				auto& dst = entry.trace->dst_data.at(lane_info.tid);
				switch (entry.lsu_args.width) {
				case 0: // LB
				case 1: // LH
					dst.i = sext((Word)read_data, data_width);
					break;
				case 2:
					if (entry.lsu_args.is_float) {
						// FLW: NaN-box single-precision float in 64-bit slot.
						dst.u64 = read_data | 0xffffffff00000000ull;
					} else {
						// LW
						dst.i = sext((Word)read_data, data_width);
					}
					break;
				case 3: // LD
				case 4: // LBU
				case 5: // LHU
				case 6: // LWU
					dst.u64 = read_data;
					break;
				default:
					std::abort();
				}
			}
		}
		entry.count -= lsu_rsp.mask.count(); // track remaining
		if (entry.count == 0) {
			// full response batch received
			state.pending_rd_reqs.release(lsu_rsp.tag);
			// is last batch?
			if (entry.eop) {
				output.send(trace, 1);
			}
		}
		pending_loads_ -= lsu_rsp.mask.count();
		lsu_rsp_in.pop();
	}

	// handle LSU requests
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		uint32_t block_idx = iw % NUM_LSU_BLOCKS;
		auto& state = states_.at(block_idx);
		if (state.fence_lock) {
			// wait for all pending memory operations to complete
			if (!state.pending_rd_reqs.empty())
				continue;
			if (!Outputs.at(iw).try_send(state.fence_trace))
				continue;
			state.fence_lock = false;
			DT(3, this->name() << " fence-unlock: " << state.fence_trace);
		}

		// check input queue
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;

		bool is_fence = false;
		bool is_write = false;

		auto trace = input.peek();
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
			DT(3, this->name() << " fence-lock: " << *trace);
			// remove input
			input.pop();
			continue;
		}

		// check pending queue capacity
		if (!is_write && state.pending_rd_reqs.full()) {
			if (!trace->log_once(true)) {
				DT(4, this->name() << " queue-full: " << *trace);
			}
			continue;
		} else {
			trace->log_once(false);
		}

		if (remain_addrs_ == 0) {
			addr_list_.clear();
			// Functional execute for LSU/AMO types.
			if (std::get_if<LsuType>(&trace->op_type) || std::get_if<AmoType>(&trace->op_type)) {
				if (!trace->data) {
					this->execute(trace);
				}
			}
			if (trace->data) {
			#ifdef EXT_V_ENABLE
				if (std::get_if<VlsType>(&trace->op_type)) {
					auto trace_data = std::dynamic_pointer_cast<VecUnit::MemTraceData>(trace->data);
					for (uint32_t t = 0; t < trace_data->mem_addrs.size(); ++t) {
						if (!trace->tmask.test(t))
							continue;
						for (auto addr : trace_data->mem_addrs.at(t)) {
							addr.tid = t;
							addr_list_.push_back(addr);
						}
					}
				} else
			#endif
				{
					auto trace_data = std::dynamic_pointer_cast<LsuTraceData>(trace->data);
					for (uint32_t t = 0; t < trace_data->mem_addrs.size(); ++t) {
						if (!trace->tmask.test(t))
							continue;
						auto entry = trace_data->mem_addrs.at(t);
						entry.tid = t;
						addr_list_.push_back(entry);
					}
				}
				remain_addrs_ = addr_list_.size();
			}
		}

		// check output backpressure
		bool direct_commit = (is_write || 0 == addr_list_.size());
		if (direct_commit && remain_addrs_ <= NUM_LSU_LANES) {
			if (Outputs.at(iw).full())
				continue; // stall
		}

		if (remain_addrs_ != 0) {
			// check lmem switch backpressure
			if (core_->lmem_switch_.at(block_idx)->ReqIn.full())
				continue; // stall

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
					auto block = std::make_shared<mem_block_t>();
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
				IntrLsuArgs lsu_args{};
				bool is_load = false;
				if (std::get_if<LsuType>(&trace->op_type)
				 && std::get<LsuType>(trace->op_type) == LsuType::LOAD) {
					lsu_args = std::get<IntrLsuArgs>(trace->instr_ptr->get_args());
					// Packed-loads were already fulfilled synchronously in execute();
					// rsp handler must not touch dst_data for them.
					is_load = (lsu_args.pack == 0);
				}
				tag = state.pending_rd_reqs.allocate({trace, count, is_eop, std::move(lane_entries), lsu_args, is_load});
			}
			lsu_req.tag  = tag;
			lsu_req.cid  = trace->cid;
			lsu_req.uuid = trace->uuid;

			// send memory request
			core_->lmem_switch_.at(block_idx)->ReqIn.send(lsu_req);
			DT(3, this->name() << " mem-req: " << lsu_req);

			// update stats
			if (is_write) {
				core_->perf_stats_.stores += count;
			} else {
				core_->perf_stats_.loads += count;
				pending_loads_ += count;
			}
		}

		if (remain_addrs_ == 0) {
			if (direct_commit) {
				Outputs.at(iw).send(trace);
			}
			// remove input
			input.pop();
		}
	}
}
