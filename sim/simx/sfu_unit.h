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
#include "func_unit.h"
#ifdef EXT_DXA_ENABLE
#include "dxa_core.h"
#endif

namespace vortex {

class SfuUnit : public FuncUnit<NUM_SFU_BLOCKS> {
public:
	SfuUnit(const SimContext& ctx, const char* name, Core*);

protected:
	void on_tick() override;

private:
	uint32_t latency_of(const instr_trace_t* trace) const;

#ifdef EXT_DXA_ENABLE
	// Per-block DXA pending slot. When non-empty, a previous tick already ran
	// execute_copy() for this trace and is retrying submit() on backpressure.
	std::array<DxaCore::TraceData::Ptr, NUM_SFU_BLOCKS> dxa_pending_;
#endif
};

}
