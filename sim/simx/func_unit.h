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

#include <simobject.h>
#include <vector>
#include "instr_trace.h"

namespace vortex {

class Core;

class FuncUnit : public SimObject<FuncUnit> {
public:
	std::vector<SimChannel<instr_trace_t*>> Inputs;
	std::vector<SimChannel<instr_trace_t*>> Outputs;

	FuncUnit(const SimContext& ctx, const char* name, Core* core)
		: SimObject<FuncUnit>(ctx, name)
		, Inputs(ISSUE_WIDTH, this)
		, Outputs(ISSUE_WIDTH, this)
		, core_(core)
	{}

	virtual ~FuncUnit() {}

protected:
	virtual void on_reset() {}
	virtual void on_tick() = 0;

	friend class SimObject<FuncUnit>;

	Core* core_;
};

}
