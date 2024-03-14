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

class Dispatcher : public SimObject<Dispatcher> {
public:
	std::vector<SimPort<instr_trace_t*>> Outputs;

	Dispatcher(const SimContext& ctx, const Arch& arch, uint32_t buf_size, uint32_t block_size, uint32_t num_lanes) 
		: SimObject<Dispatcher>(ctx, "Dispatcher") 
		, Outputs(ISSUE_WIDTH, this)
		, Inputs_(ISSUE_WIDTH, this)
		, arch_(arch)
		, queues_(ISSUE_WIDTH, std::queue<instr_trace_t*>())
		, buf_size_(buf_size)
		, block_size_(block_size)
		, num_lanes_(num_lanes)
		, batch_count_(ISSUE_WIDTH / block_size)
		, pid_count_(arch.num_threads() / num_lanes)
		, batch_idx_(0)
		, start_p_(block_size, 0)
	{}
	
	virtual ~Dispatcher() {}

	virtual void reset() {
		batch_idx_ = 0;
		for (uint32_t b = 0; b < block_size_; ++b) {
			start_p_.at(b) = 0;
		}
	}

	virtual void tick() {
		for (uint32_t i = 0; i < ISSUE_WIDTH; ++i) {
			auto& queue = queues_.at(i);
			if (queue.empty())
				continue;
			auto trace = queue.front();
			Inputs_.at(i).push(trace, 1);
			queue.pop();
		}

		uint32_t block_sent = 0;
		for (uint32_t b = 0; b < block_size_; ++b) {
			uint32_t i = batch_idx_ * block_size_ + b;
			auto& input = Inputs_.at(i);
			if (input.empty()) {
				++block_sent;
				continue;
			}
			auto& output = Outputs.at(i);
			auto trace = input.front();
			auto new_trace = trace;
			if (pid_count_ != 1) {
				auto start_p = start_p_.at(b);
				if (start_p == -1) {
					++block_sent;
					continue; 
				} 
				int start(-1), end(-1);
				for (uint32_t j = start_p * num_lanes_, n = arch_.num_threads(); j < n; ++j) {
					if (!trace->tmask.test(j))
						continue;
					if (start == -1)
						start = j;
					end = j;
				}
				start /= num_lanes_;
				end /= num_lanes_;
				if (start != end) {
					new_trace = new instr_trace_t(*trace);
					new_trace->eop = false;
					start_p_.at(b) = start + 1;
				} else {
					start_p_.at(b) = -1;
					input.pop();
					++block_sent;
				}
				new_trace->pid = start;
				new_trace->sop = (0 == start_p);
				ThreadMask tmask;
				for (int j = start * num_lanes_, n = j + num_lanes_; j < n; ++j) {
					tmask[j] = trace->tmask[j];
				}
				new_trace->tmask = tmask;
			} else {
				new_trace->pid = 0;
				input.pop();
				++block_sent;
			}
			DT(3, "pipeline-dispatch: " << *new_trace);
			output.push(new_trace, 1);
		}
		if (block_sent == block_size_) {
			batch_idx_ = (batch_idx_ + 1) % batch_count_;
			for (uint32_t b = 0; b < block_size_; ++b) {
				start_p_.at(b) = 0;
			}
		}
	};

	bool push(uint32_t issue_index, instr_trace_t* trace) {
		auto& queue = queues_.at(issue_index);
		if (queue.size() >= buf_size_)
			return false;
		queue.push(trace);
		return true;
	}

private:
	std::vector<SimPort<instr_trace_t*>> Inputs_;
	const Arch& arch_;
	std::vector<std::queue<instr_trace_t*>> queues_;
	uint32_t buf_size_;
	uint32_t block_size_;
	uint32_t num_lanes_;
	uint32_t batch_count_;
	uint32_t pid_count_;
	uint32_t batch_idx_;
	std::vector<int> start_p_;
};

}
