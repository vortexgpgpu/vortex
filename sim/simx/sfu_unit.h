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

#pragma once

#include <array>
#include <memory>
#include "func_unit.h"
#include "wctl_unit.h"
#include "csr_unit.h"
#include "gfx_window.h"   // GfxWindow — shared SETW/GETW/GETWF slot file
#ifdef VX_CFG_EXT_DXA_ENABLE
#include "dxa/dxa_unit.h"
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
#include "tex/tex_unit.h"
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om/om_unit.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
#include "raster/raster_unit.h"
#include "scheduler.h"   // Scheduler::FwdWave — RASTER push-dispatch payload
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
#include "rtu/rtu_unit.h"
#endif

namespace vortex {

class TexCore;
class OmCore;
class RasterCore;
class RtuCore;

// SFU has a single dispatch port that fans out to per-op sub-units
// (WCTL / CSR / DXA / TEX / OM / RASTER) by op_type, then gathers their
// results back to a single result port. Sub-units are plain non-SimObject
// helpers owned here.
//
// TEX takes the DXA-style fire-and-wait path: TexUnit posts a TexReq onto
// `tex_req_out` and the SFU does NOT push the trace onto its writeback
// output — TexCore owns the trace until it returns it via `tex_rsp_in`,
// at which point on_tick() forwards it to the original writeback lane.
class SfuUnit : public FuncUnit<VX_CFG_NUM_SFU_BLOCKS> {
public:
	SfuUnit(const SimContext& ctx, const char* name, Core*);

	CsrUnit& csr_unit() { return *csr_unit_; }

#ifdef VX_CFG_EXT_DXA_ENABLE
	// Outbound DXA request channel — bound by Cluster to
	// DxaCore::dxa_req_in[cid]. Owned here (SfuUnit is the SimObject;
	// DxaUnit is a plain helper sub-class).
	SimChannel<DxaReq> dxa_req_out;
#endif

#ifdef VX_CFG_EXT_TEX_ENABLE
	// Outbound TEX request / inbound TEX response channels. Cluster binds
	// these to the cluster-level TexBus arbiter (which fans into TexCore).
	SimChannel<TexReq> tex_req_out;
	SimChannel<TexRsp> tex_rsp_in;
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
	// Outbound OM request channel. Cluster binds to OmCore::om_req_in[cid].
	// vx_om has no return value — there is no rsp channel; OmCore drives
	// the R-M-W asynchronously through the ocache.
	SimChannel<OmReq> om_req_out;
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
	// Outbound RASTER request / inbound response channels. Cluster binds
	// these to the cluster-level RasterBus arbiter (which fans into RasterCore).
	SimChannel<RasterReq> raster_req_out;
	SimChannel<RasterRsp> raster_rsp_in;
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
	// Outbound RTU request / inbound response channels (PRISM Phase 1).
	// Cluster binds these to the cluster-level RtuBus arbiter (which fans
	// into RtuCore). Trace ownership follows the TEX shape: from
	// vx_rt_trace acceptance until RtuRsp arrival, the trace is owned by
	// RtuCore. On rsp arrival, SfuUnit applies the response into
	// RtuUnit's register file and forwards the trace to writeback with
	// the terminal status word.
	SimChannel<RtuReq> rtu_req_out;
	SimChannel<RtuRsp> rtu_rsp_in;
	// §8.6 async ray pool: Cluster calls this after RtuCore is created
	// so RtuUnit can pre-allocate slot handles at vx_rt_trace time and
	// free them at vx_rt_wait completion (the alloc/free path is a
	// direct C++ call, not a SimChannel hop).
	void set_rtu_core(RtuCore* core);

	// Claim this warp's ray-pool slot for a TRACE2 macro head; false when the
	// pool is full (see RtuUnit::trace2_reserve_slot).
	bool rtu_trace2_reserve_slot(uint32_t wid);
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
	// Seed an injected fragment warp's per-lane payload into the gfx register
	// window (FWD-5 launch-time window write). Called by the scheduler's
	// fragment work distributor at warp launch; the FS reads it back with GETW.
	void stage_fwd_window(uint32_t wid, const Scheduler::FwdWave& wave);
#endif

protected:
	void on_tick() override;

private:
	uint32_t latency_of(const instr_trace_t* trace) const;

	std::unique_ptr<WctlUnit> wctl_unit_;
	std::unique_ptr<CsrUnit>  csr_unit_;
#ifdef VX_GFX_WINDOW_ENABLE
	// Shared graphics register window (SETW/GETW/GETWF slot file) used by the RTU
	// ray/hit stream, TEX (vx_tex4) u,v payload + texel, and OM (vx_om4) payload.
	// Declared before rtu_unit_ so it outlives the RtuUnit that borrows it.
	GfxWindow gfx_window_;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
	// RASTER dispatch v2 packer (SimX mirror of VX_raster_packer): compact covered
	// quads across raster responses into full NUM_THREADS warps before launch, so
	// the cycle model matches the RTL's one-launch-per-full-warp rate. Image-neutral
	// (same fragments, regrouped); same-quad co-packing is avoided to preserve OM
	// submission order. graphics::frag_payload_t comes from scheduler.h (RASTER-only).
	std::array<graphics::frag_payload_t, VX_CFG_NUM_THREADS> fwd_pack_buf_{};
	uint32_t fwd_pack_count_ = 0;
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
	std::unique_ptr<DxaUnit>  dxa_unit_;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
	std::unique_ptr<TexUnit>  tex_unit_;
	// vx_tex4 quad sequencer: one fragment in flight per SFU block. q_in_slot_ /
	// q_dims_ cache the rs2/rs1 operands at fragment 0 (src_data is overwritten
	// per fragment with that fragment's u/v/lod for TexUnit::process).
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> q_frag_{};
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> q_issued_{};
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> q_in_slot_{};
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> q_dims_{};
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
	std::unique_ptr<OmUnit>   om_unit_;
	// vx_om4 quad sequencer: one sub-pixel in flight per SFU block. om_desc_ /
	// om_base_ cache the per-lane rs1 descriptor and the rs2 window base at
	// sub-pixel 0; om_last_sent_ holds the op after sub-pixel 3 submits until the
	// output port accepts the (data-less) retirement.
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> om_q_frag_{};
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> om_last_sent_{};
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> om_base_{};
	std::array<std::array<uint32_t, VX_CFG_NUM_THREADS>, VX_CFG_NUM_SFU_BLOCKS> om_desc_{};
	// The whole colour/depth payload is latched with desc/base at capture: the
	// sub-pixel sequence spans many cycles and vx_om4 has no completion handle,
	// so the issuing warp can retire and its window be re-seeded (next fragment
	// CTA) while later sub-pixels are still pending — a mid-sequence window
	// read would pick up the next quad's colours.
	std::array<std::array<std::array<uint32_t, 4>, VX_CFG_NUM_THREADS>, VX_CFG_NUM_SFU_BLOCKS> om_color_{};
	std::array<std::array<std::array<uint32_t, 4>, VX_CFG_NUM_THREADS>, VX_CFG_NUM_SFU_BLOCKS> om_depth_{};
	// The vx_om4 dispatch overwrites trace->src_data in place (the operand
	// slots double as the per-sub-pixel pos/colour/depth carrier to OmUnit),
	// so the desc/base capture must run exactly ONCE per op — a second entry
	// (the op is re-presented to the block before it retires) would re-read the
	// clobbered operands and corrupt the cache, dropping fragments under
	// multi-warp x multi-thread (the gfx_om §3.1 hazard). Cleared at retire.
	std::array<uint32_t, VX_CFG_NUM_SFU_BLOCKS> om_captured_{};
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
	std::unique_ptr<RtuUnit>    rtu_unit_;
#endif
};

}
