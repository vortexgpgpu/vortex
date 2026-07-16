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
#include "rtu/rtu_window.h"   // RtuWindow — the RTU hit-window slot file
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
	// Outbound RTU request / inbound response channels.
	// Cluster binds these to the cluster-level RtuBus arbiter (which fans
	// into RtuCore). Trace ownership follows the TEX shape: from
	// vx_rt_trace acceptance until RtuRsp arrival, the trace is owned by
	// RtuCore. On rsp arrival, SfuUnit applies the response into
	// RtuUnit's register file and forwards the trace to writeback with
	// the terminal status word.
	SimChannel<RtuReq> rtu_req_out;
	SimChannel<RtuRsp> rtu_rsp_in;
	// Async ray pool: Cluster calls this after RtuCore is created
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
#endif

protected:
	void on_tick() override;

private:
	uint32_t latency_of(const instr_trace_t* trace) const;

	std::unique_ptr<WctlUnit> wctl_unit_;
	std::unique_ptr<CsrUnit>  csr_unit_;
#ifdef VX_RTU_WINDOW_ENABLE
	// Shared graphics register window (SETW/GETW/GETWF slot file) used by the RTU
	// ray/hit stream, TEX (vx_tex4) u,v payload + texel, and OM (vx_om4) payload.
	// Declared before rtu_unit_ so it outlives the RtuUnit that borrows it.
	RtuWindow rtu_window_;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
	// Fragment warp packer: compact covered quads across raster responses into full
	// warps before launch, so one warp launches per full pack rather than per sparse
	// response. Image-neutral (same fragments, regrouped); same-quad co-packing is
	// avoided to preserve OM submission order.
	//
	// A quad owns four adjacent lanes (one pixel each), so a full warp is
	// NUM_THREADS/4 quads and the buffer holds stamps, not per-lane payloads: the
	// expansion to one pixel per lane happens at flush.
	static_assert(VX_CFG_NUM_THREADS >= VX_FRAG_QUAD_LANES
	           && (VX_CFG_NUM_THREADS % VX_FRAG_QUAD_LANES) == 0,
	              "a pixel quad occupies four adjacent lanes, so a warp must hold whole quads");
	static constexpr uint32_t FWD_PACK_QUADS = VX_CFG_NUM_THREADS / VX_FRAG_QUAD_LANES;
	std::array<RasterStamp, FWD_PACK_QUADS> fwd_pack_buf_{};
	uint32_t fwd_pack_count_ = 0;
#endif
#ifdef VX_CFG_EXT_DXA_ENABLE
	std::unique_ptr<DxaUnit>  dxa_unit_;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
	std::unique_ptr<TexUnit>  tex_unit_;
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
	std::unique_ptr<OmUnit>   om_unit_;
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
	std::unique_ptr<RtuUnit>    rtu_unit_;
#endif
};

}
