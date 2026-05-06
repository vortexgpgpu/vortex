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
#ifdef EXT_DXA_ENABLE
#include "dxa/dxa_unit.h"
#endif
#ifdef EXT_TEX_ENABLE
#include "tex/tex_unit.h"
#endif
#ifdef EXT_OM_ENABLE
#include "om/om_unit.h"
#endif
#ifdef EXT_RASTER_ENABLE
#include "raster/raster_unit.h"
#endif

namespace vortex {

class TexCore;
class OmCore;
class RasterCore;

// SFU has a single dispatch port that fans out to per-op sub-units
// (WCTL / CSR / DXA / TEX / OM / RASTER) by op_type, then gathers their
// results back to a single result port. Sub-units are plain non-SimObject
// helpers owned here.
//
// TEX takes the DXA-style fire-and-wait path: TexUnit posts a TexReq onto
// `tex_req_out` and the SFU does NOT push the trace onto its writeback
// output — TexCore owns the trace until it returns it via `tex_rsp_in`,
// at which point on_tick() forwards it to the original writeback lane.
class SfuUnit : public FuncUnit<NUM_SFU_BLOCKS> {
public:
	SfuUnit(const SimContext& ctx, const char* name, Core*);

	CsrUnit& csr_unit() { return *csr_unit_; }

#ifdef EXT_DXA_ENABLE
	// Outbound DXA request channel — bound by Cluster to
	// DxaCore::dxa_req_in[cid]. Owned here (SfuUnit is the SimObject;
	// DxaUnit is a plain helper sub-class).
	SimChannel<DxaReq> dxa_req_out;
#endif

#ifdef EXT_TEX_ENABLE
	// Outbound TEX request / inbound TEX response channels. Cluster binds
	// these to the cluster-level TexBus arbiter (which fans into TexCore).
	SimChannel<TexReq> tex_req_out;
	SimChannel<TexRsp> tex_rsp_in;
#endif

#ifdef EXT_OM_ENABLE
	// Outbound OM request channel. Cluster binds to OmCore::om_req_in[cid].
	// vx_om has no return value — there is no rsp channel; OmCore drives
	// the R-M-W asynchronously through the ocache.
	SimChannel<OmReq> om_req_out;
#endif

#ifdef EXT_RASTER_ENABLE
	// Outbound RASTER request / inbound response channels. Cluster binds
	// these to the cluster-level RasterBus arbiter (which fans into RasterCore).
	SimChannel<RasterReq> raster_req_out;
	SimChannel<RasterRsp> raster_rsp_in;
#endif

protected:
	void on_tick() override;

private:
	uint32_t latency_of(const instr_trace_t* trace) const;

	std::unique_ptr<WctlUnit> wctl_unit_;
	std::unique_ptr<CsrUnit>  csr_unit_;
#ifdef EXT_DXA_ENABLE
	std::unique_ptr<DxaUnit>  dxa_unit_;
#endif
#ifdef EXT_TEX_ENABLE
	std::unique_ptr<TexUnit>  tex_unit_;
#endif
#ifdef EXT_OM_ENABLE
	std::unique_ptr<OmUnit>   om_unit_;
#endif
#ifdef EXT_RASTER_ENABLE
	std::unique_ptr<RasterUnit> raster_unit_;
#endif
};

}
