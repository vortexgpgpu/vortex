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
#include "scheduler.h"
#include "socket.h"
#include "cluster.h"
#include "constants.h"
#include "VX_types.h"

using namespace vortex;

SfuUnit::SfuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<NUM_SFU_BLOCKS>(ctx, name, core)
{}

uint32_t SfuUnit::latency_of(const instr_trace_t* /*trace*/) const {
	// All SFU operations (WCTL, CSR, DXA) share a fixed pipeline latency.
	return 4;
}

void SfuUnit::on_tick() {
	for (uint32_t b = 0; b < NUM_SFU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		if (output.full())
			continue; // stall — no side effects this tick
		auto trace = input.peek();

		// WSYNC has a structural gate: cannot complete until prior insts retire.
		if (auto wctl_p = std::get_if<WctlType>(&trace->op_type)) {
			if (*wctl_p == WctlType::WSYNC) {
				if (!trace->eop || core_->has_pending_instrs(trace->wid))
					continue; // skip; do not pop
			}
		}

#ifdef EXT_DXA_ENABLE
		// DXA: execute_copy mutates DXA state and is non-idempotent, so we
		// run it once per trace and retain the resulting td across submit
		// retries via dxa_pending_[b]. submit() can backpressure on the
		// DXA queue.
		if (std::get_if<DxaType>(&trace->op_type)) {
			auto& slot = dxa_pending_.at(b);
			if (!slot) {
				auto& rs1_data = trace->src_data[0];
				auto& rs2_data = trace->src_data[1];
				uint64_t lmem_addr  = static_cast<uint64_t>(rs1_data.at(0).u);
				uint32_t meta       = rs1_data.at(1).u;
				uint32_t coords[5]  = { static_cast<uint32_t>(rs1_data.at(2).u),
				                        static_cast<uint32_t>(rs1_data.at(3).u),
				                        static_cast<uint32_t>(rs2_data.at(0).u),
				                        static_cast<uint32_t>(rs2_data.at(1).u),
				                        static_cast<uint32_t>(rs2_data.at(2).u) };
				uint32_t cta_mask   = rs2_data.at(3).u;
				uint32_t desc_slot  = meta & 0x0fu;
				uint32_t raw_bar    = (meta >> 4) & 0x07ffffffu;
				uint32_t bar_id     = bar_decode_id(raw_bar, NUM_BARRIERS);
				auto dxa_core = core_->socket()->cluster()->dxa_core();
				slot = dxa_core->execute_copy(core_, desc_slot, lmem_addr, coords);
				slot->bar_id   = bar_id;
				slot->cta_mask = cta_mask;
			}
			auto dxa_core = core_->socket()->cluster()->dxa_core();
			if (!dxa_core->submit(core_, slot)) {
				continue; // queue backpressure — retry next cycle
			}
			core_->barrier_event_attach(slot->bar_id);
			slot.reset();
		}
#endif

		// Apply WctlType side effects + compute release_warp inline. For
		// non-DXA we reach here only when output is not full (gated above)
		// and submit succeeded — so this body runs at most once per trace,
		// matching the original execute()-once semantics.
		bool release_warp = trace->fetch_stall;
		if (auto wctl_p = std::get_if<WctlType>(&trace->op_type)) {
			auto wctl_type = *wctl_p;
			auto& sched = core_->scheduler();
			auto& warp = sched.warp(trace->wid);
			auto instrArgs = trace->instr_ptr->get_args();
			auto wctlArgs = std::get<IntrWctlArgs>(instrArgs);
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

			switch (wctl_type) {
			case WctlType::TMC: {
				ThreadMask next_tmask(num_threads);
				for (uint32_t t = 0; t < num_threads; ++t) {
					next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
				}
				if (trace->eop) {
					release_warp = core_->setTmask(trace->wid, next_tmask);
				}
			} break;
			case WctlType::WSPAWN: {
				if (trace->eop) {
					release_warp = core_->wspawn(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
				}
			} break;
			case WctlType::SPLIT: {
				Word next_pc = trace->PC + 4;
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
				auto stack_size = warp.ipdom_stack.size();
				// stack_size captured pre-push is what each pid writes to its
				// dst_data — it's the kernel-visible value at this PC.
				for (uint32_t t = thread_start; t < num_threads; ++t) {
					trace->dst_data[t].i = stack_size;
				}
				// ipdom_stack push + tmask update
				if (trace->eop) {
					if (is_divergent) {
						if (stack_size == sched.ipdom_size()) {
							std::cout << "IPDOM stack is full! size=" << stack_size << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
							std::abort();
						}
						next_tmask = (then_tmask.count() <= else_tmask.count()) ? then_tmask : else_tmask;
						warp.ipdom_stack.emplace(warp.tmask, next_pc);
						core_->perf_stats().divergence += 1;
					}
					release_warp = core_->setTmask(trace->wid, next_tmask);
				}
			} break;
			case WctlType::JOIN: {
				auto stack_ptr = rs1_data.at(thread_last).u;
				auto stack_size = warp.ipdom_stack.size();
				ThreadMask next_tmask = warp.tmask;
        // ipdom_stack pop + tmask update
				if (trace->eop) {
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
					release_warp = core_->setTmask(trace->wid, next_tmask);
				}
			} break;
			case WctlType::BAR: {
				uint32_t arg1 = rs1_data[thread_last].u;
				uint32_t arg2 = rs2_data[thread_last].u;
				uint32_t bar_id = bar_decode_id(arg1, NUM_BARRIERS);
				bool is_sync_bar = (bool)wctlArgs.is_sync_bar;
				if (wctlArgs.is_bar_arrive) {
					uint32_t phase = sched.barrier_unit().get_phase(bar_id);
					for (uint32_t t = thread_start; t < num_threads; ++t) {
						if (!warp.tmask.test(t)) continue;
						trace->dst_data[t].i = phase;
					}
				}
				if (trace->eop) {
					if (trace->wb || is_sync_bar) {
						core_->barrier_arrive(bar_id, arg2, trace->wid, is_sync_bar);
						if (is_sync_bar) {
							release_warp = false;
						}
					} else {
						release_warp = !core_->barrier_wait(bar_id, arg2, trace->wid);
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
				if (trace->eop) {
					release_warp = core_->setTmask(trace->wid, next_tmask);
				}
			} break;
			case WctlType::WSYNC:
				release_warp = true;
				break;
			default:
				std::abort();
			}
			DT(3, this->name() << " execute: op=" << wctl_type << ", " << *trace);
		} else {
#ifndef EXT_DXA_ENABLE
			// CsrType is owned by CsrUnit; only DXA reaches here when enabled.
			std::abort();
#endif
		}

		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		if (trace->eop && release_warp) {
			core_->resume(trace->wid);
		}

		input.pop();
	}
}
