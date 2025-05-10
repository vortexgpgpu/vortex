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

#include "instr_trace.h"
#include <queue>
#include <vector>

namespace vortex {

class Core;

class Dispatcher : public SimObject<Dispatcher> {
public:
	std::vector<SimPort<instr_trace_t*>> Outputs;
	std::vector<SimPort<instr_trace_t*>> Inputs;

	Dispatcher(const SimContext& ctx, Core* core, uint32_t buf_size, uint32_t block_size, uint32_t num_lanes);

	virtual ~Dispatcher();

	virtual void reset();

	virtual void tick();

private:
	const Arch& arch_;
	Core*    core_;
	uint32_t buf_size_;
	uint32_t block_size_;
	uint32_t num_lanes_;
	uint32_t num_blocks_;
	uint32_t num_packets_;
	uint32_t batch_idx_;
	std::vector<int> block_pids_;
};

}
