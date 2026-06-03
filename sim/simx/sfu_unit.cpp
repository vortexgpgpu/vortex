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
#include "debug.h"
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om/om_core.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include "raster/raster_core.h"
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
{
}

uint32_t SfuUnit::latency_of(const instr_trace_t* /*trace*/) const {
	return 4;
}

void SfuUnit::on_tick() {
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
