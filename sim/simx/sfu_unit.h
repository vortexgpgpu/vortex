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

namespace vortex {

// SFU has a single dispatch port that fans out to per-op sub-units
// (WCTL / CSR / DXA) by op_type, then gathers their results back to a
// single result port. Sub-units are plain non-SimObject helpers owned
// here.
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

protected:
	void on_tick() override;

private:
	uint32_t latency_of(const instr_trace_t* trace) const;

	std::unique_ptr<WctlUnit> wctl_unit_;
	std::unique_ptr<CsrUnit>  csr_unit_;
#ifdef EXT_DXA_ENABLE
	std::unique_ptr<DxaUnit>  dxa_unit_;
#endif
};

}
