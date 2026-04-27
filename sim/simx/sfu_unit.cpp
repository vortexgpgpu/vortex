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

#include "sfu_unit.h"
#include <iostream>
#include <iomanip>
#include <string.h>
#include <assert.h>
#include <util.h>
#include "debug.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "constants.h"
#include "VX_types.h"

using namespace vortex;

SfuUnit::SfuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit(ctx, name, core)
{}

uint32_t SfuUnit::latency_of(const instr_trace_t* /*trace*/) const {
	// All SFU operations (WCTL, CSR, DXA) share a fixed pipeline latency.
	return 4;
}

void SfuUnit::execute(instr_trace_t* trace) {
	auto& sched = core_->scheduler();
	auto& warp = sched.warp(trace->wid);
	auto& instr = *trace->instr_ptr;
	auto instrArgs = instr.get_args();
	uint32_t num_threads = NUM_THREADS;
	auto& rs1_data = trace->src_data[0];
	auto& rs2_data = trace->src_data[1];

	uint32_t thread_start = 0;
	for (; thread_start < num_threads; ++thread_start) {
		if (warp.tmask.test(thread_start)) break;
	}
	int32_t thread_last = num_threads - 1;
	for (; thread_last >= 0; --thread_last) {
		if (warp.tmask.test(thread_last)) break;
	}

	trace->dst_data.assign(num_threads, reg_data_t{});
	auto& rd_data = trace->dst_data;

	if (std::get_if<WctlType>(&trace->op_type)) {
		auto wctl_type = std::get<WctlType>(trace->op_type);
		auto wctlArgs = std::get<IntrWctlArgs>(instrArgs);
		Word next_pc = trace->PC + 4;
		switch (wctl_type) {
		case WctlType::TMC: {
			ThreadMask next_tmask(num_threads);
			for (uint32_t t = 0; t < num_threads; ++t) {
				next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
			}
			trace->data = std::make_shared<SfuTraceData>(next_tmask.to_ulong(), 0);
		} break;
		case WctlType::WSPAWN: {
			trace->data = std::make_shared<SfuTraceData>(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
		} break;
		case WctlType::SPLIT: {
			auto stack_size = warp.ipdom_stack.size();
			ThreadMask then_tmask(num_threads);
			ThreadMask else_tmask(num_threads);
			auto not_pred = wctlArgs.is_cond_neg;
			for (uint32_t t = 0; t < num_threads; ++t) {
				auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
				then_tmask[t] = warp.tmask.test(t) && cond;
				else_tmask[t] = warp.tmask.test(t) && !cond;
			}
			ThreadMask next_tmask = warp.tmask;
			bool is_divergent = then_tmask.any() && else_tmask.any();
			if (is_divergent) {
				if (stack_size == sched.ipdom_size()) {
					std::cout << "IPDOM stack is full! size=" << stack_size << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
					std::abort();
				}
				if (then_tmask.count() <= else_tmask.count()) {
					next_tmask = then_tmask;
				} else {
					next_tmask = else_tmask;
				}
				warp.ipdom_stack.emplace(warp.tmask, next_pc);
				core_->perf_stats().divergence += 1;
			}
			for (uint32_t t = thread_start; t < num_threads; ++t) {
				rd_data[t].i = stack_size;
			}
			trace->data = std::make_shared<SfuTraceData>(next_tmask.to_ulong(), 0);
		} break;
		case WctlType::JOIN: {
			auto stack_ptr = rs1_data.at(thread_last).u;
			auto stack_size = warp.ipdom_stack.size();
			ThreadMask next_tmask = warp.tmask;
			if (stack_ptr != stack_size) {
				if (warp.ipdom_stack.empty()) {
					std::cout << "IPDOM stack is empty!\n" << std::flush;
					std::abort();
				}
				if (warp.ipdom_stack.top().fallthrough) {
					next_tmask = warp.ipdom_stack.top().orig_tmask;
					warp.ipdom_stack.pop();
				} else {
					next_tmask = ~warp.tmask & warp.ipdom_stack.top().orig_tmask;
					warp.PC = warp.ipdom_stack.top().else_PC;
					warp.ipdom_stack.top().fallthrough = true;
				}
			}
			trace->data = std::make_shared<SfuTraceData>(next_tmask.to_ulong(), 0);
		} break;
		case WctlType::BAR: {
			uint32_t arg1 = rs1_data[thread_last].u;
			uint32_t arg2 = rs2_data[thread_last].u;
			uint32_t bar_id = bar_decode_id(arg1, NUM_BARRIERS);
			trace->data = std::make_shared<BarTraceData>(bar_id, arg2, (bool)wctlArgs.is_sync_bar);
			if (wctlArgs.is_bar_arrive) {
				uint32_t phase = sched.barrier_unit().get_phase(bar_id);
				for (uint32_t t = thread_start; t < num_threads; ++t) {
					if (!warp.tmask.test(t)) continue;
					rd_data[t].i = phase;
				}
			}
		} break;
		case WctlType::PRED: {
			ThreadMask pred(num_threads);
			auto not_pred = wctlArgs.is_cond_neg;
			for (uint32_t t = 0; t < num_threads; ++t) {
				auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
				pred[t] = warp.tmask.test(t) && cond;
			}
			ThreadMask next_tmask = warp.tmask;
			if (pred.any()) {
				next_tmask &= pred;
			} else {
				next_tmask = ThreadMask(num_threads, rs2_data.at(thread_last).u);
			}
			trace->data = std::make_shared<SfuTraceData>(next_tmask.to_ulong(), 0);
		} break;
		case WctlType::WSYNC:
			break;
		default:
			std::abort();
		}
		DT(3, this->name() << " execute: op=" << wctl_type << ", " << *trace);
#ifdef EXT_DXA_ENABLE
	} else if (std::get_if<DxaType>(&trace->op_type)) {
		// wgather DXA: args packed into 4 lanes (lmem_addr/meta/coords[0..4]/cta_mask)
		uint64_t lmem_addr  = static_cast<uint64_t>(rs1_data.at(0).u);
		uint32_t meta       = rs1_data.at(1).u;
		uint32_t coords[5]  = { static_cast<uint32_t>(rs1_data.at(2).u),
		                        static_cast<uint32_t>(rs1_data.at(3).u),
		                        static_cast<uint32_t>(rs2_data.at(0).u),
		                        static_cast<uint32_t>(rs2_data.at(1).u),
		                        static_cast<uint32_t>(rs2_data.at(2).u) };
		uint32_t cta_mask   = rs2_data.at(3).u;  // lane 3 rs2 = cta_mask (multicast)
		uint32_t desc_slot  = meta & 0x0fu;
		uint32_t raw_bar    = (meta >> 4) & 0x07ffffffu;
		uint32_t bar_id     = bar_decode_id(raw_bar, core_->arch().num_barriers());
		auto dxa_core = core_->socket()->cluster()->dxa_core();
		auto td = dxa_core->execute_copy(core_, desc_slot, lmem_addr, coords);
		td->bar_id   = bar_id;
		td->cta_mask = cta_mask;
		trace->data  = td;
#endif
	} else {
		// CsrType is owned by CsrUnit (FUType::CSR); anything else here is unexpected.
		std::abort();
	}
}

void SfuUnit::on_tick() {
	for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
		auto& input = Inputs.at(iw);
		if (input.empty())
			continue;
		auto& output = Outputs.at(iw);
		if (output.full())
			continue; // stall
		auto trace = input.peek();

		// WSYNC has a structural gate: cannot complete until prior insts retire
		if (std::get_if<WctlType>(&trace->op_type)) {
			auto wctl_type = std::get<WctlType>(trace->op_type);
			if (wctl_type == WctlType::WSYNC) {
				if (!trace->eop || core_->has_pending_instrs(trace->wid))
					continue; // skip; do not pop
			}
		}

#ifdef EXT_DXA_ENABLE
		// DXA submission may backpressure on its request queue; gate before execute
		if (std::get_if<DxaType>(&trace->op_type)) {
			// Pre-check: build the trace_data via execute, then attempt submit
			// The execute() above populates trace->data with DxaCore::TraceData.
		}
#endif

		// Functional execution at ACCEPT (guarded so DXA submit retry doesn't re-execute)
		if (!trace->data) {
			this->execute(trace);
		}

#ifdef EXT_DXA_ENABLE
		if (std::get_if<DxaType>(&trace->op_type)) {
			auto td = std::dynamic_pointer_cast<DxaCore::TraceData>(trace->data);
			assert(td);
			auto dxa_core = core_->socket()->cluster()->dxa_core();
			if (!dxa_core->submit(core_, td)) {
				// Backpressure: retry next cycle. execute_copy already mutated
				// DXA state and cannot be fully rolled back, so the queue
				// capacity acts as a hard pre-condition for issue.
				continue;
			}
			core_->barrier_event_attach(td->bar_id);
		}
#endif

		bool release_warp = trace->fetch_stall;
		if (std::get_if<WctlType>(&trace->op_type)) {
			auto wctl_type = std::get<WctlType>(trace->op_type);
			switch (wctl_type) {
			case WctlType::WSPAWN:
				if (trace->eop) {
					auto trace_data = std::dynamic_pointer_cast<SfuTraceData>(trace->data);
					release_warp = core_->wspawn(trace_data->arg1, trace_data->arg2);
				}
				break;
			case WctlType::TMC:
			case WctlType::SPLIT:
			case WctlType::JOIN:
			case WctlType::PRED:
				if (trace->eop) {
					auto trace_data = std::dynamic_pointer_cast<SfuTraceData>(trace->data);
					ThreadMask tmask(NUM_THREADS, trace_data->arg1);
					release_warp = core_->setTmask(trace->wid, tmask);
				}
				break;
			case WctlType::WSYNC:
				release_warp = true;
				break;
			case WctlType::BAR: {
				if (trace->eop) {
					auto trace_data = std::dynamic_pointer_cast<BarTraceData>(trace->data);
					if (trace->wb || trace_data->is_sync_bar) {
						core_->barrier_arrive(trace_data->bar_id, trace_data->count, trace->wid, trace_data->is_sync_bar);
						if (trace_data->is_sync_bar) {
							release_warp = false;
						}
					} else {
						release_warp = !core_->barrier_wait(trace_data->bar_id, trace_data->count, trace->wid);
					}
				}
			} break;
			default:
				break;
			}
		}

		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		if (trace->eop && release_warp) {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}
