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
	std::vector<SimChannel<instr_trace_t*>> Inputs;
	std::vector<SimChannel<instr_trace_t*>> Outputs;
	// An op leaving its slot queue, which returns its issue credit.
	SimChannel<instr_trace_t*> ReleaseOut;

	// out_delay: cycles from the slot queue's head to the unit taking the op,
	// past the lane-dispatch buffer when the unit has one.
	Dispatcher(const SimContext& ctx, const char* name, Core* core, uint32_t queue_size, uint32_t block_size, uint32_t num_lanes, uint32_t out_delay);

	virtual ~Dispatcher();

	// The unit takes ops straight off the slot queue, so the credit returns as
	// the unit accepts the op, from the unit itself.
	void set_release_on_accept() { release_on_accept_ = true; }

protected:
	virtual void on_reset();
	virtual void on_tick();

private:
	Core*    core_;
	uint32_t block_size_;
	uint32_t num_lanes_;
	uint32_t num_blocks_;
	uint32_t num_packets_;
	uint32_t batch_idx_;
	uint32_t out_delay_;
	std::vector<int> block_pids_;
	// first cycle a block's lane iterator can present a new op
	std::vector<uint64_t> block_free_;
	// the slot queue a block's iterator still holds a last packet of, or -1
	std::vector<int> held_input_;
	// A SIMD slice wider than the fanout limit (8 lanes, plus half) splits
	// into packets through a registered iterator.
	bool iter_out_reg_;
	bool release_on_accept_ = false;

	friend class SimObject<Dispatcher>;
};

}
