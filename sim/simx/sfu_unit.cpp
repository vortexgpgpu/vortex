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
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "scheduler.h"
#include "debug.h"
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om/om_core.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include "raster/raster_core.h"
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
#include <VX_types.h>
#endif

using namespace vortex;

SfuUnit::SfuUnit(const SimContext& ctx, const char* name, Core* core)
	: FuncUnit<VX_CFG_NUM_SFU_BLOCKS>(ctx, name, core)
#ifdef VX_CFG_EXT_DXA_ENABLE
	, dxa_req_out(this)
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
	, tex_req_out(this)
	, tex_rsp_in(this)
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
	, om_req_out(this)
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
	, raster_req_out(this)
	, raster_rsp_in(this)
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
	, rtu_req_out(this)
	, rtu_rsp_in(this)
#endif
	, wctl_unit_(new WctlUnit(core))
	, csr_unit_(new CsrUnit(core))
#ifdef VX_CFG_EXT_DXA_ENABLE
	, dxa_unit_(new DxaUnit(core, dxa_req_out))
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
	, tex_unit_(new TexUnit(core, tex_req_out))
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
	, om_unit_(new OmUnit(core, om_req_out))
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
	, raster_unit_(new RasterUnit(core, raster_req_out))
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
	, rtu_unit_(new RtuUnit(core, rtu_req_out))
	, rtu_trap_slot_(VX_CFG_NUM_WARPS, uint32_t(-1))
#endif
{
}

uint32_t SfuUnit::latency_of(const instr_trace_t* /*trace*/) const {
	return 4;
}

#ifdef VX_CFG_EXT_RTU_ENABLE
void SfuUnit::set_rtu_core(RtuCore* core) {
	rtu_unit_->set_rtu_core(core);
}
#endif

void SfuUnit::on_tick() {
#ifdef VX_CFG_EXT_RTU_ENABLE
	// Drain RTU rsps. Two flavors:
	//   TERMINAL — the ray finished; apply hit attrs into the RTU regfile,
	//              write per-lane status into trace->dst_data, forward the
	//              parked trace (the TRACE instr) to writeback.
	//   CB_YIELD — the ray yielded to AHS/IS. Stage candidate-hit attrs +
	//              cb_type into the yielded lanes' RTU regs and raise an
	//              async trap on the warp. The trace stays parked in
	//              RtuCore; a later TERMINAL drains it via the path above.
	//              See proposal §4.6 (option-c: reuse existing mtvec/MRET).
	while (!rtu_rsp_in.empty()) {
		auto& rsp = rtu_rsp_in.peek();
		if (rsp.kind == RtuRspKind::CB_YIELD) {
			auto& sched = core_->scheduler();
			// Phase 3-A2 divergent-SBT: this warp may be running a
			// previous dispatcher and not yet have executed `mret`.
			// raising another async-trap on the warp now would
			// clobber mepc/mtvec, losing the resume PC. Defer until
			// the in-flight trap is retired.
			if (sched.in_async_trap(rsp.warp_id)) break;
			rtu_unit_->apply_callback_payload(rsp);
			auto& warp  = sched.warp(rsp.warp_id);
			ThreadMask yielded(VX_CFG_NUM_THREADS);
			for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
				if ((rsp.cb_active_mask >> t) & 1u) yielded.set(t);
			}
			// mepc = current fetch PC. The warp's pipeline still has the
			// pre-trap instructions (incl. the parked TRACE's scoreboard
			// dependency on rd) in-flight; the dispatcher's instructions
			// fire alongside but don't touch TRACE's rd, so they make
			// progress while the post-WAIT kernel ops stay stalled on
			// TRACE until the final TERMINAL rsp.
			constexpr Word TRAP_CAUSE_RTU_CALLBACK = VX_TRAP_CAUSE_RTU_CALLBACK;
			sched.raise_async_trap(rsp.warp_id, TRAP_CAUSE_RTU_CALLBACK,
			                       warp.PC, yielded);
			// Remember which ray's dispatcher is now running (cb_handle =
			// slot, uniform across yielded lanes) so only THIS ray's TERMINAL
			// is held off until mret — a recursive traceRay the dispatcher
			// itself fires must complete normally.
			for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
				if ((rsp.cb_active_mask >> t) & 1u) {
					rtu_trap_slot_.at(rsp.warp_id) = rsp.cb_handle[t];
					break;
				}
			}
			DT(3, "rtu-cb_yield: core=" << core_->id() << ", wid=" << rsp.warp_id
			      << ", mask=0x" << std::hex << rsp.cb_active_mask << std::dec);
			rtu_rsp_in.pop();
			continue;
		}
		// Defer THIS ray's TERMINAL writeback while its callback dispatcher
		// is still running. A high-pressure dispatcher (e.g. an FP
		// intersection shader) saves/restores the WAIT's rd register; its
		// scoreboard reservation was lifted at trap entry and re-installed
		// at mret, so the status word must not land until after mret —
		// otherwise the epilogue restore would clobber it. Only the
		// trap-triggering ray (rtu_trap_slot_) is held off; a nested
		// recursive traceRay must drain normally or the dispatcher (blocked
		// on the nested wait) would deadlock.
		if (core_->scheduler().in_async_trap(rsp.warp_id)
		    && rsp.slot_idx == rtu_trap_slot_.at(rsp.warp_id)) break;
		// §8.6 TERMINAL: route to the parked WAIT trace (if WAIT
		// already issued) or latch into pending_terminals_ (if
		// WAIT hasn't issued yet — slot is short-lived enough
		// that TERMINAL beat WAIT to the SFU). The TRACE trace
		// is NOT used here — TRACE's writeback already happened
		// synchronously at vx_rt_trace dispatch (its dst_data
		// carries the slot handle). Pre-check output.full() before
		// calling on_terminal_rsp because the latter is destructive
		// (frees the slot and erases the parked entry).
		uint32_t bid = 0;
		if (rtu_unit_->terminal_would_writeback(rsp, &bid)
		    && Outputs.at(bid).full()) {
			break;  // backpressure: retry next tick
		}
		auto wb = rtu_unit_->on_terminal_rsp(rsp);
		if (wb.trace) {
			Outputs.at(wb.block_id).send(wb.trace, this->latency_of(wb.trace));
			DT(3, "rtu-rsp deliver: core=" << core_->id()
				 << ", wid=" << wb.trace->wid << ", slot=" << rsp.slot_idx);
		} else {
			DT(3, "rtu-rsp latch: core=" << core_->id()
				 << ", wid=" << rsp.warp_id << ", slot=" << rsp.slot_idx);
		}
		rtu_rsp_in.pop();
	}
#endif

#ifdef VX_CFG_EXT_TEX_ENABLE
	// Drain TEX completions FIRST. TexCore returns each finished trace via
	// tex_rsp_in; copy filtered texels into dst_data and forward the trace
	// onto the originally-recorded writeback output lane.
	while (!tex_rsp_in.empty()) {
		auto& rsp = tex_rsp_in.peek();
		auto& output = Outputs.at(rsp.block_id);
		if (output.full())
			break;
		instr_trace_t* trace = rsp.trace;
		for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
			if (trace->tmask.test(t)) {
				trace->dst_data[t].i = rsp.texels[t];
			}
		}
		output.send(trace, this->latency_of(trace));
		DT(3, "tex-rsp deliver: core=" << core_->id() << ", wid=" << trace->wid);
		tex_rsp_in.pop();
	}
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
	// Drain RASTER completions. RasterCore returns one stamp per active lane
	// per request; deliver pos_mask to the trace's dst_data (vx_rast result),
	// and latch pid + bcoords into per-warp+thread CSR storage so the kernel
	// can read VX_CSR_RASTER_PID + VX_CSR_RASTER_BCOORD_*.
	while (!raster_rsp_in.empty()) {
		auto& rsp = raster_rsp_in.peek();
		auto& output = Outputs.at(rsp.block_id);
		if (output.full())
			break;
		instr_trace_t* trace = rsp.trace;
		for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
			if (!trace->tmask.test(t)) continue;
			const auto& s = rsp.stamps[t];
			trace->dst_data[t].i = s.pos_mask;
			RasterCsrs csrs;
			csrs.pos_mask = s.pos_mask;
			csrs.pid      = s.pid;
			csrs.bcoords  = s.bcoords;
			csr_unit_->set_raster_csrs(trace->wid, t, csrs);
		}
		output.send(trace, this->latency_of(trace));
		DT(3, "raster-rsp deliver: core=" << core_->id() << ", wid=" << trace->wid);
		raster_rsp_in.pop();
	}
#endif

	// PE switch: peek input, route to the matching sub-unit (WCTL / CSR /
	// DXA / TEX / OM / RASTER) by op_type, gather to the single result port.
	for (uint32_t b = 0; b < VX_CFG_NUM_SFU_BLOCKS; ++b) {
		auto& input = Inputs.at(b);
		if (input.empty())
			continue;
		auto& output = Outputs.at(b);
		auto trace = input.peek();

#ifdef VX_CFG_EXT_TEX_ENABLE
		// TEX path is async: don't gate on output.full() yet — that check
		// happens on completion. Submit only.
		if (std::get_if<TexType>(&trace->op_type)) {
			if (!tex_unit_->process(trace, b))
				continue; // backpressure — leave trace in input, retry next cycle
			input.pop();
			continue;
		}
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
		// RTU dispatch. §8.6 (async ray pool):
		//   SET / GET           — synchronous (RTU register-file
		//                          updates / reads).
		//   TRACE               — synchronous writeback of the slot
		//                          handle; ray walks async in RtuCore.
		//   WAIT                — fast path (short-circuit) when the
		//                          TERMINAL already landed; otherwise
		//                          parked in RtuUnit and consumed by
		//                          the TERMINAL drain above.
		//   CB_RET              — async (TEX-shape): submit, drop input;
		//                          RtuCore owns the action until the
		//                          matching CB_YIELD/TERMINAL arrives.
		if (auto rtu_p = std::get_if<RtuType>(&trace->op_type)) {
			if (*rtu_p == RtuType::TRACE) {
				// §8.6: TRACE writes the pre-allocated slot index back
				// synchronously through the standard SFU writeback so
				// kernel code that does `uint32_t h = vx_rt_trace(...)`
				// can use h immediately (e.g., pass it to vx_rt_wait
				// for a different ray). process_trace returns nullptr
				// on bus full OR pool full.
				if (output.full()) continue;
				if (!rtu_unit_->process_trace(trace, b))
					continue;
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			if (*rtu_p == RtuType::CB_RET) {
				// Phase 2: send the per-lane action to RtuCore via the bus
				// and retire the CB_RET op synchronously (no rd). The
				// dispatcher follows up with `mret` to resume the kernel
				// at the post-WAIT PC.
				if (!rtu_unit_->process_cb_ret(trace, b))
					continue; // backpressure
				if (output.full()) continue;
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			if (*rtu_p == RtuType::WAIT) {
				// §8.6 WAIT: pre-check whether the short-circuit path
				// (TERMINAL already latched in pending_terminals_)
				// would need an output slot. If yes and output is
				// full, backpressure. If not (will park), output is
				// not consumed; the matching TERMINAL drain delivers
				// the writeback later via on_terminal_rsp.
				uint32_t slot = rtu_unit_->wait_handle(trace);
				if (rtu_unit_->wait_would_short_circuit(trace->wid, slot)
				    && output.full()) {
					continue;  // backpressure
				}
				instr_trace_t* wb = rtu_unit_->process_wait(trace, b);
				if (wb) {
					output.send(wb, this->latency_of(wb));
				}
				input.pop();
				continue;
			}
			// SET / GET use synchronous SFU writeback below.
			if (output.full()) continue;
			if (*rtu_p == RtuType::SET) {
				rtu_unit_->process_set(trace);
			} else /* GET */ {
				rtu_unit_->process_get(trace);
			}
			output.send(trace, this->latency_of(trace));
			input.pop();
			continue;
		}
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
		// RASTER path. POP is async (same shape as TEX) — RasterCore
		// owns the trace from accept until rsp arrives. BEGIN is the
		// per-frame fetch trigger: pulse the cluster RasterCore here
		// and complete the SFU op synchronously via the path below —
		// no quad data to return, no RasterReq to send.
		// Idempotent — concurrent pulses from multiple warps/cores
		// collapse via RasterCore's has_begun_ flag.
		if (auto raster_p = std::get_if<RasterType>(&trace->op_type)) {
			if (*raster_p == RasterType::POP) {
				if (!raster_unit_->process(trace, b))
					continue;
				input.pop();
				continue;
			}
			if (*raster_p == RasterType::BEGIN) {
				core_->socket()->cluster()->raster_core()->begin();
				// fall through to synchronous SFU completion
			}
		}
#endif

		if (output.full())
			continue; // stall — no side effects this tick

		// WSYNC has a structural gate: cannot complete until prior insts retire.
		// BAR (vx_barrier and vx_barrier_arrive) drains LSU before continuing —
		// implements CUDA __syncthreads / OpenCL barrier(CLK_LOCAL_MEM_FENCE) semantic.
		if (auto wctl_p = std::get_if<WctlType>(&trace->op_type)) {
			if (trace->eop) {
				if (*wctl_p == WctlType::WSYNC) {
					if (core_->has_pending_instrs(trace->wid))
						continue; // wait for the warp's prior instrs to retire
				} else if (*wctl_p == WctlType::BAR) {
					if (!core_->lsu_drained())
						continue; // drain LSU before the barrier
				}
			}
		}

		bool release_warp = trace->fetch_stall;
		if (std::get_if<WctlType>(&trace->op_type)) {
			release_warp = wctl_unit_->process(trace);
		} else if (std::get_if<CsrType>(&trace->op_type)) {
			csr_unit_->process(trace);
#ifdef VX_CFG_EXT_DXA_ENABLE
		} else if (std::get_if<DxaType>(&trace->op_type)) {
			// process() returns nullptr on backpressure (idempotent retry next
			// cycle) or the trace on success → fall through to send/pop.
			if (!dxa_unit_->process(trace)) {
				continue;
			}
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
		} else if (std::get_if<OmType>(&trace->op_type)) {
			// process() returns nullptr on backpressure (idempotent retry next
			// cycle) or the trace on success → fall through to send/pop.
			if (!om_unit_->process(trace)) {
				continue;
			}
#endif
		}

		uint32_t delay = this->latency_of(trace);
		output.send(trace, delay);
		// Warp-control refines the default (fetch_stall) release decision: a
		// sync-barrier, a not-yet-last barrier arrival, a deferred wspawn, or a
		// warp that disabled itself (tmask=0) keeps the warp parked — it is
		// released by the barrier/spawn machinery rather than at this commit.
		trace->resume_warp = release_warp;

		input.pop();
	}
}
