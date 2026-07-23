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
#include "mem/local_mem.h"
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
#ifdef VX_CFG_EXT_RTU_ENABLE
	, rtu_unit_(new RtuUnit(core, rtu_req_out, rtu_window_))
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

bool SfuUnit::rtu_trace2_reserve_slot(uint32_t wid) {
	return rtu_unit_->trace2_reserve_slot(wid);
}
#endif


void SfuUnit::on_tick() {
#ifdef VX_CFG_EXT_RTU_ENABLE
	// Drain RTU rsps. Two flavors, both completing the warp's parked WAIT
	// through the same writeback path (candidate-return, no async trap):
	//   TERMINAL — the ray finished; apply hit attrs into the RTU regfile,
	//              write the terminal status into trace->dst_data, free the
	//              slot, forward the parked WAIT trace to writeback.
	//   CB_YIELD — a non-opaque candidate (AHS / procedural) is returned to
	//              the issuing warp; stage candidate attrs into the yielded
	//              lanes' RTU regs and complete the parked WAIT with a YIELD
	//              status. The slot stays live; the warp reads the candidate,
	//              decides, and issues vx_rt_continue (CB_ACTION) to resume.
	while (!rtu_rsp_in.empty()) {
		auto& rsp = rtu_rsp_in.peek();
		const bool is_candidate = (rsp.kind == RtuRspKind::CB_YIELD);
		// Both paths complete the parked WAIT: pre-check output.full() before
		// the destructive on_*_rsp() (which erases the parked entry / frees
		// the slot). If no WAIT is parked yet, the rsp is latched and picked
		// up when WAIT issues.
		uint32_t bid = 0;
		const bool would_wb = is_candidate
			? rtu_unit_->candidate_would_writeback(rsp, &bid)
			: rtu_unit_->terminal_would_writeback(rsp, &bid);
		if (would_wb && Outputs.at(bid).full()) {
			break;  // backpressure: retry next tick
		}
		auto wb = is_candidate ? rtu_unit_->on_candidate_rsp(rsp)
		                       : rtu_unit_->on_terminal_rsp(rsp);
		if (wb.trace) {
			Outputs.at(wb.block_id).send(wb.trace, this->latency_of(wb.trace));
			DT(3, "rtu-rsp deliver: core=" << core_->id()
				 << ", wid=" << wb.trace->wid << ", cand=" << is_candidate);
		} else {
			DT(3, "rtu-rsp latch: core=" << core_->id()
				 << ", wid=" << rsp.warp_id << ", cand=" << is_candidate);
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
			if (!trace->tmask.test(t)) continue;
			trace->dst_data[t].i = rsp.texels[t];   // rd = texel
		}
		{
			// Unit latency is already modeled by the TEX pipeline; charge only
			// the gather/writeback hop.
			output.send(trace, 2);
			DT(3, "tex-rsp deliver: core=" << core_->id() << ", wid=" << trace->wid);
		}
		tex_rsp_in.pop();
	}
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
	{
		// Fragment dispatch is PUSH. The per-core fragment work distributor pulls
		// covered-quad waves from the cluster RasterCore autonomously (no kernel op):
		// each tick post RasterReqs while the producer is armed and has request
		// budget, then convert each RasterRsp into a FwdWave the scheduler launches as
		// a fragment warp. An all-zero (pos_mask==0) rsp is the drained sentinel.
		auto& sched = core_->scheduler();

		// 1) Autonomous wave-pull: keep the producer fed while armed.
		while (sched.fwd_armed() && sched.fwd_can_request()
		    && !sched.fwd_wave_queue_full() && !raster_req_out.full()) {
			RasterReq req;
			req.uuid       = 0;
			req.tag        = 0;
			req.core_id    = core_->id();
			req.trace      = nullptr;   // autonomous pull — no kernel trace
			req.block_id   = 0;
			req.tmask_bits = (VX_CFG_NUM_THREADS >= 32)
			               ? 0xffffffffu : ((1u << VX_CFG_NUM_THREADS) - 1u);
			raster_req_out.send(req);
			sched.fwd_on_request();
		}

		// 2) Drain responses, compacting covered quads across responses into full
		//    warps: launch one warp per full/flushed pack, not one per sparse
		//    response. Image-neutral.
		//
		//    A quad owns four adjacent lanes, so the flush expands each buffered stamp
		//    into four per-lane payloads. All four lanes are thread-active, including
		//    those the primitive misses: they run as helper lanes so their covered
		//    neighbours can shuffle a value out of them for a derivative. The `covered`
		//    bit, not the thread mask, is what gates the export.
		constexpr uint32_t kQuadLanes = VX_FRAG_QUAD_LANES;
		constexpr uint32_t kPosBits   = VX_RASTER_DIM_BITS - 1;
		constexpr uint32_t kPosMask   = (1u << kPosBits) - 1u;

		auto fwd_flush_pack = [&]() {
			if (fwd_pack_count_ == 0) {
				return;
			}
			Scheduler::FwdWave wave;
			for (uint32_t q = 0; q < fwd_pack_count_; ++q) {
				const auto& s = fwd_pack_buf_[q];
				uint32_t qx = (s.pos_mask >> 4) & kPosMask;
				uint32_t qy = (s.pos_mask >> (4 + kPosBits)) & kPosMask;
				for (uint32_t sub = 0; sub < kQuadLanes; ++sub) {
					uint32_t l = q * kQuadLanes + sub;
					uint32_t x = 2 * qx + (sub & 1);
					uint32_t y = 2 * qy + (sub >> 1);
					uint32_t covered = (s.pos_mask >> sub) & 1;
					wave.tmask.set(l);
					wave.payload[l].pos = x | (y << 16) | (covered << 31);
					wave.payload[l].pid = s.pid;
				}
			}
			sched.fwd_push_wave(wave);
			fwd_pack_count_ = 0;
		};
		while (!raster_rsp_in.empty()) {
			auto& rsp = raster_rsp_in.peek();
			bool drained = true;
			for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
				if (rsp.stamps[t].pos_mask != 0) {
					drained = false;
				}
			}
			if (drained) {
				fwd_flush_pack();               // flush the tail partial warp
				sched.fwd_mark_drained();
			} else {
				for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
					const auto& s = rsp.stamps[t];
					// Skip uncovered quads (coverage nibble empty): block batches carry
					// mask=0 fillers with valid positions that must not occupy a slot.
					if ((s.pos_mask & 0xf) == 0) {
						continue;
					}
					// Never co-pack two quads at the same (pos_x,pos_y): flush first so
					// same-pixel fragments land in distinct, ordered warps.
					bool collide = false;
					for (uint32_t j = 0; j < fwd_pack_count_; ++j) {
						if ((fwd_pack_buf_[j].pos_mask >> 4) == (s.pos_mask >> 4)) {
							collide = true;
						}
					}
					if (collide || fwd_pack_count_ == FWD_PACK_QUADS) {
						fwd_flush_pack();
					}
					fwd_pack_buf_[fwd_pack_count_] = s;
					if (++fwd_pack_count_ == FWD_PACK_QUADS) {
						fwd_flush_pack();
					}
				}
			}
			sched.fwd_on_response();
			raster_rsp_in.pop();
		}

		// 3) Epoch complete (producer drained AND every launched wave retired):
		//    return the core to idle so run()/busy can settle.
		if (sched.fwd_done()) {
			sched.fwd_disarm();
		}
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
			// vx_tex: u/v/lod are already in src_data[0..2] (rs1/rs2/rs3). TEX does
			// not read or write the hit window, so there is nothing to
			// stage and nothing to sequence -- one request, one response.
			if (!tex_unit_->process(trace, b))
				continue; // backpressure — leave trace in input, retry next cycle
			input.pop();
			continue;
		}
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
		// vx_om_export: one packet for the whole warp. Each lane holds its aperture
		// address, colour and depth in registers -- no window read, no sub-pixel
		// loop. The address stays UNDECODED: recovering (x, y, face) needs the
		// aperture DCRs, which are cluster state, so OmCore does it (the SimX
		// counterpart of VX_om_ingress).
		if (std::get_if<OmType>(&trace->op_type)) {
			// Every stall check must come BEFORE the export: process_export SENDS the
			// fragment, and a uop that cannot retire stays at the head of the input
			// queue and is re-run next cycle. Testing `output` afterwards exported the
			// same fragment twice — invisible for a plain colour or depth write, which
			// is idempotent, but a blend reads the destination first, so the second
			// fragment blended the pixel against itself.
			if (output.full())
				continue;
			auto omArgs = std::get<IntrOmArgs>(trace->instr_ptr->get_args());
			if (!om_unit_->process_export(trace, omArgs.export_mask))
				continue;   // OM back-pressure — nothing was sent; retry
			output.send(trace, 1);
			input.pop();
			continue;
		}
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
		// Graphics-window / RTU dispatch. SETW (write) and GETW/GETWF (windowed
		// read) are pure register-window ops, available whenever any FF consumer
		// is built. The RTU-specific ops (CB_RET / TRACE / WAIT) are gated on
		// VX_CFG_EXT_RTU_ENABLE — they are only ever decoded with the RTU built,
		// and they touch rtu_unit_ which does not exist otherwise.
		//   SETW / GETW[F]      — synchronous graphics-window updates / reads.
		//   TRACE              — synchronous writeback of the slot handle; the
		//                          ray walks async in RtuCore.
		//   WAIT               — fast path (short-circuit) when the TERMINAL
		//                          already landed; otherwise parked in RtuUnit.
		//   CB_RET              — async (TEX-shape): submit, drop input.
		if (auto rtu_p = std::get_if<GfxwType>(&trace->op_type)) {
#ifdef VX_CFG_EXT_RTU_ENABLE
			if (*rtu_p == GfxwType::CB_RET) {
				// Send the per-lane action to RtuCore via the bus and retire the
				// CB_RET op synchronously (no rd). The dispatcher follows up with
				// `mret` to resume the kernel at the post-WAIT PC.
				//
				// Both stall checks come BEFORE the send: process_cb_ret puts the
				// action packet on the bus, and a uop that cannot retire is re-run
				// from the head of the input queue next cycle — which would resolve
				// the same candidate twice.
				if (output.full()) continue;
				if (!rtu_unit_->process_cb_ret(trace, b))
					continue; // bus backpressure — nothing was sent
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			// Each TRACE/WAIT macro-op
			// arrives here already expanded by the per-warp sequencer into
			// micro-ops; args.uop is the micro-op index.
			if (*rtu_p == GfxwType::TRACE) {
				// All 4 uops complete synchronously (the async traversal kicks
				// off when uop 3 arms the slot). Backpressure: pool full at
				// uop 0, bus full at uop 3 — retry the same uop next cycle.
				auto args = std::get<IntrGfxwArgs>(trace->instr_ptr->get_args());
				if (output.full()) continue;
				if (!rtu_unit_->process_trace_uop(trace, b, args.uop))
					continue;
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			if (*rtu_p == GfxwType::WAIT) {
				// single-op block. Identical park / short-circuit to v1
				// WAIT, so it survives an async callback trap (parked traces are
				// revived by on_terminal_rsp; a macro-op could not be). The hit
				// window is delivered by the separate WAIT_WB that follows.
				uint32_t slot = rtu_unit_->wait_handle(trace);
				if (rtu_unit_->wait_would_short_circuit(trace->wid, slot)
				    && output.full()) {
					continue;
				}
				instr_trace_t* wb = rtu_unit_->process_wait(trace, b);
				if (wb) {
					output.send(wb, this->latency_of(wb));
				}
				input.pop();
				continue;
			}
#endif // VX_CFG_EXT_RTU_ENABLE
			// GETWF / GETW: FP / GP windowed read, expanded by the
			// sequencer into one synchronous uop per window slot (args.uop = slot
			// offset). Reads are synchronous; any ordering vs terminal is enforced
			// by the optional rs1 scoreboard chain (vx_rt_wait sets it to status).
			if (*rtu_p == GfxwType::GETWF || *rtu_p == GfxwType::GETW) {
				auto args = std::get<IntrGfxwArgs>(trace->instr_ptr->get_args());
				if (output.full()) continue;
				rtu_window_.process_getw_uop(trace, args.uop, *rtu_p == GfxwType::GETWF);
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
		}
#endif // VX_CFG_EXT_RTU_ENABLE

		// Fragment dispatch is push, not pull: there is no kernel-side raster op.
		// The fragment work distributor (above + scheduler) launches fragment
		// warps directly from the autonomously-pulled covered-quad waves.

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
