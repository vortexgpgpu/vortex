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
#include <vx_tex_lod.h>   // vx_tex_quad_lod — shared HW-LOD formula (vx_tex4 quad)
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
	, rtu_unit_(new RtuUnit(core, rtu_req_out, gfx_window_))
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

#ifdef VX_CFG_EXT_RASTER_ENABLE
void SfuUnit::stage_fwd_window(uint32_t wid, const Scheduler::FwdWave& wave) {
#ifdef VX_GFX_WINDOW_ENABLE
	// P2: the record is just {pos_mask, pid}; the FS recomputes per-corner edge
	// values from the primitive edges + the quad origin (no bcoords seeded).
	constexpr uint32_t B = GfxWindow::FRAG_SLOT_BASE;
	for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
		if (!wave.tmask.test(t)) continue;
		const auto& p = wave.payload[t];
		gfx_window_.set(wid, t, B + 0, p.pos_mask);
		gfx_window_.set(wid, t, B + 1, p.pid);
	}
#else
	(void)wid; (void)wave;
#endif
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
		// Single (and legacy vx_tex) retire on their one response; a quad retires
		// only on its 4th fragment — frags 0..2 just land their texel in the window.
		bool retire = !rsp.is_quad || (rsp.frag == 3);
		auto& output = Outputs.at(rsp.block_id);
		if (retire && output.full())
			break;
		instr_trace_t* trace = rsp.trace;
		for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
			if (!trace->tmask.test(t)) continue;
#ifdef VX_GFX_WINDOW_ENABLE
			// vx_tex4: land this fragment's texel in the window at out_slot+frag.
			if (rsp.is_tex4)
				gfx_window_.set(trace->wid, t, (rsp.out_slot + rsp.frag) & 0x1f, rsp.texels[t]);
#endif
			if (retire)
				trace->dst_data[t].i = rsp.texels[t];   // rd = scoreboard sync handle
		}
		if (rsp.is_quad) {
			// advance the per-block fragment sequencer; the input was held across
			// the four issues and is released here on the last response.
			q_issued_[rsp.block_id] = 0;
			if (rsp.frag == 3) { q_frag_[rsp.block_id] = 0; Inputs.at(rsp.block_id).pop(); }
			else ++q_frag_[rsp.block_id];
		}
		if (retire) {
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
		// RASTER dispatch v2 (push). The per-core fragment work distributor pulls
		// covered-quad waves from the cluster RasterCore autonomously (no kernel
		// op): each tick post RasterReqs while the producer is armed and has
		// request budget, then convert each RasterRsp into a FwdWave the scheduler
		// launches as a fragment warp (payload seeded into the warp's register
		// window at launch). An all-zero (pos_mask==0) rsp is the drained sentinel.
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
		//    NUM_THREADS warps (mirror of VX_raster_packer): launch one warp per
		//    full/flushed pack, not one per sparse response. Image-neutral.
		auto fwd_flush_pack = [&]() {
			if (fwd_pack_count_ == 0) return;
			Scheduler::FwdWave wave;
			for (uint32_t j = 0; j < fwd_pack_count_; ++j) {
				wave.tmask.set(j);
				wave.payload[j] = fwd_pack_buf_[j];
			}
			sched.fwd_push_wave(wave);
			fwd_pack_count_ = 0;
		};
		while (!raster_rsp_in.empty()) {
			auto& rsp = raster_rsp_in.peek();
			bool drained = true;
			for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t)
				if (rsp.stamps[t].pos_mask != 0) drained = false;
			if (drained) {
				fwd_flush_pack();               // flush the tail partial warp
				sched.fwd_mark_drained();
			} else {
				for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
					const auto& s = rsp.stamps[t];
					// Skip uncovered quads (coverage nibble empty): block batches
					// carry mask=0 fillers with valid positions that must not
					// occupy a wave lane.
					if ((s.pos_mask & 0xf) == 0) continue;
					// Never co-pack two quads at the same (pos_x,pos_y): flush first
					// so same-pixel fragments land in distinct, ordered warps.
					bool collide = false;
					for (uint32_t j = 0; j < fwd_pack_count_; ++j)
						if ((fwd_pack_buf_[j].pos_mask >> 4) == (s.pos_mask >> 4)) collide = true;
					if (collide || fwd_pack_count_ == VX_CFG_NUM_THREADS)
						fwd_flush_pack();
					fwd_pack_buf_[fwd_pack_count_].pos_mask = s.pos_mask;
					fwd_pack_buf_[fwd_pack_count_].pid      = s.pid;
					if (++fwd_pack_count_ == VX_CFG_NUM_THREADS)
						fwd_flush_pack();
				}
			}
			sched.fwd_on_response();
			raster_rsp_in.pop();
		}

		// 3) Epoch complete (producer drained AND every launched wave retired):
		//    return the core to idle so run()/busy can settle.
		if (sched.fwd_done())
			sched.fwd_disarm();
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
#ifdef VX_GFX_WINDOW_ENABLE
			// vx_tex4: source the payload from the shared graphics window (staged by
			// SETW) so TexUnit::process sees the legacy operand layout (u=src0,
			// v=src1, lod=src2). src_data is always NUM_SRC_REGS-wide.
			auto targs = std::get<IntrTexArgs>(trace->instr_ptr->get_args());
			if (targs.is_tex4 && targs.mode) {
				// quad mode: one fragment in flight. Cache rs1(dims)/rs2(in_slot) at
				// fragment 0 (src_data is overwritten per fragment below), compute the
				// integer LOD from the quad derivatives, and issue fragment F. The
				// frag-3 response retires the op and pops the input.
				if (q_issued_[b]) continue;
				uint32_t F = q_frag_[b];
				if (F == 0) {
					for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
						if (!trace->tmask.test(t)) continue;
						q_in_slot_[b] = trace->src_data[1].at(t).u & 0x1f;
						q_dims_[b]    = trace->src_data[0].at(t).u;
						break;
					}
				}
				uint32_t logw = q_dims_[b] & 0xffff, logh = q_dims_[b] >> 16;
				for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
					if (!trace->tmask.test(t)) continue;
					int32_t u[4], v[4];
					for (int k = 0; k < 4; ++k) {
						u[k] = (int32_t)gfx_window_.get(trace->wid, t, (q_in_slot_[b] + k) & 0x1f);
						v[k] = (int32_t)gfx_window_.get(trace->wid, t, (q_in_slot_[b] + 4 + k) & 0x1f);
					}
					uint32_t lod = vx_tex_quad_lod(u, v, logw, logh);
					trace->src_data[0].at(t).u = (uint32_t)u[F];
					trace->src_data[1].at(t).u = (uint32_t)v[F];
					trace->src_data[2].at(t).u = lod;
				}
				if (!tex_unit_->process(trace, b, F))
					continue; // backpressure
				q_issued_[b] = 1;
				continue;     // do NOT pop — the frag-3 response pops the input
			}
			if (targs.is_tex4) {
				// single mode: u at in_slot, v at in_slot+1, lod from rs1.
				for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
					if (!trace->tmask.test(t)) continue;
					uint32_t in_slot = trace->src_data[1].at(t).u & 0x1f;
					uint32_t lod     = trace->src_data[0].at(t).u;
					trace->src_data[0].at(t).u = gfx_window_.get(trace->wid, t, in_slot);
					trace->src_data[1].at(t).u = gfx_window_.get(trace->wid, t, (in_slot + 1) & 0x1f);
					trace->src_data[2].at(t).u = lod;
				}
			}
#endif
			if (!tex_unit_->process(trace, b))
				continue; // backpressure — leave trace in input, retry next cycle
			input.pop();
			continue;
		}
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
		// vx_om4: one thread owns a 2x2 quad. Emit one OmReq per covered
		// sub-pixel F (0..3), skipping sub-pixels no lane covers, reading
		// colour[F]/depth[F] from the shared window; retire (send+pop, no rd)
		// after the last sub-pixel.
		if (std::get_if<OmType>(&trace->op_type)) {
#ifdef VX_GFX_WINDOW_ENABLE
			if (!om_last_sent_[b]) {
				uint32_t F = om_q_frag_[b];
				// Capture desc/base ONCE per op (see om_captured_): the loop below
				// overwrites src_data in place, so a re-entry must not re-read it.
				if (!om_captured_[b]) {
					om_captured_[b] = 1;
					for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t)
						om_desc_[b][t] = trace->src_data[0].at(t).u;
					for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
						if (!trace->tmask.test(t)) continue;
						om_base_[b] = trace->src_data[1].at(t).u & 0x1f;
						break;
					}
					// Latch the full colour/depth payload now: the op has no
					// completion handle, so the window can be re-seeded for the
					// next fragment CTA before later sub-pixels are emitted.
					for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
						if (!trace->tmask.test(t)) continue;
						for (uint32_t k = 0; k < 4; ++k) {
							om_color_[b][t][k] = (uint32_t)gfx_window_.get(trace->wid, t, (om_base_[b] + k) & 0x1f);
							om_depth_[b][t][k] = (uint32_t)gfx_window_.get(trace->wid, t, (om_base_[b] + 4 + k) & 0x1f);
						}
					}
				}
				uint32_t fmask = 0;
				for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
					if (!trace->tmask.test(t)) continue;
					uint32_t desc = om_desc_[b][t];
					if (!((desc >> F) & 0x1)) continue;   // lane not covered for F
					uint32_t qx   = (desc >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
					uint32_t qy   = (desc >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 2)) - 1);
					uint32_t face = (desc >> 31) & 0x1;
					uint32_t pos_x = (qx << 1) | (F & 1);
					uint32_t pos_y = (qy << 1) | ((F >> 1) & 1);
					trace->src_data[0].at(t).u = (pos_y << 16) | (pos_x << 1) | face;
					trace->src_data[1].at(t).u = om_color_[b][t][F];
					trace->src_data[2].at(t).u = om_depth_[b][t][F];
					fmask |= (1u << t);
				}
				if (fmask != 0 && !om_unit_->process(trace, fmask))
					continue; // OM bus backpressure — retry this sub-pixel
				if (F < 3) { om_q_frag_[b] = F + 1; continue; }
				om_last_sent_[b] = 1;
			}
			if (output.full())
				continue; // last sub-pixel submitted; retire when output frees
			om_q_frag_[b]    = 0;
			om_last_sent_[b] = 0;
			om_captured_[b]  = 0;
			output.send(trace, this->latency_of(trace));
			input.pop();
			continue;
#endif
		}
#endif

#ifdef VX_GFX_WINDOW_ENABLE
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
				gfx_window_.process_getw_uop(trace, args.uop, *rtu_p == GfxwType::GETWF);
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			// GETWS: GP windowed read indexed by rs1 (block_idx) — the FWD-v2
			// fragment-record read (single-slot; block_idx recovered from CTA_BLOCK_ID).
			if (*rtu_p == GfxwType::GETWS) {
				auto args = std::get<IntrGfxwArgs>(trace->instr_ptr->get_args());
				if (output.full()) continue;
				gfx_window_.process_getws_uop(trace, args.uop);
				output.send(trace, this->latency_of(trace));
				input.pop();
				continue;
			}
			// SETW: synchronous regfile write (callback writeback).
			if (output.full()) continue;
			gfx_window_.process_set(trace);
			output.send(trace, this->latency_of(trace));
			input.pop();
			continue;
		}
#endif // VX_GFX_WINDOW_ENABLE

		// RASTER dispatch v2 is push, not pull: there is no kernel-side raster op.
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
