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

#include "dispatcher.h"
#include "core.h"

using namespace vortex;

Dispatcher::Dispatcher(const SimContext& ctx, Core* core, uint32_t buf_size, uint32_t block_size, uint32_t num_lanes)
  : SimObject<Dispatcher>(ctx, "dispatcher")
  , Outputs(ISSUE_WIDTH, this)
  , Inputs(ISSUE_WIDTH, this)
  , arch_(core->arch())
  , core_(core)
  , buf_size_(buf_size)
  , block_size_(block_size)
  , num_lanes_(num_lanes)
  , num_blocks_(ISSUE_WIDTH / block_size)
  , num_packets_(core->arch().num_threads() / num_lanes)
  , batch_idx_(0)
  , block_pids_(block_size, 0)
{}

Dispatcher::~Dispatcher() {}

void Dispatcher::reset() {
  batch_idx_ = 0;
  for (auto& bp : block_pids_) {
    bp = 0;
  }
}

void Dispatcher::tick() {
  // process inputs
  uint32_t block_sent = 0;
  for (uint32_t b = 0; b < block_size_; ++b) {
    uint32_t i = batch_idx_ * block_size_ + b;
    auto& input = Inputs.at(i);
    if (input.empty()) {
      ++block_sent;
      continue;
    }
    auto& output = Outputs.at(i);
    auto trace = input.front();

    // check if trace should be split
    auto new_trace = trace;
    if (num_packets_ != 1) {
      auto block_pid = block_pids_.at(b);
      // check if current block has already been processed
      if (block_pid == -1) {
        ++block_sent;
        continue;
      }
      // calculate current packet start and end
      int start(-1), end(-1);
      for (uint32_t j = block_pid * num_lanes_, n = arch_.num_threads(); j < n; ++j) {
        if (!trace->tmask.test(j))
          continue;
        if (start == -1)
          start = j;
        end = j;
      }
      start /= num_lanes_;
      end /= num_lanes_;

      // issue partial trace
      if (start != end) {
        auto trace_alloc = core_->trace_pool().allocate(1);
        new_trace = new (trace_alloc) instr_trace_t(*trace);
        block_pids_.at(b) = start + 1;
      } else {
        block_pids_.at(b) = -1; // mark block as processed
        input.pop();
        ++block_sent;
      }
      ThreadMask tmask(arch_.num_threads());
      for (int j = start * num_lanes_, n = j + num_lanes_; j < n; ++j) {
        tmask[j] = trace->tmask[j];
      }
      new_trace->tmask = tmask;
      new_trace->pid = start;
      new_trace->sop = trace->sop && (0 == block_pid);
      new_trace->eop = trace->eop && (start == end);
    } else {
      // issue the trace
      input.pop();
      ++block_sent;
    }
    DT(3, "pipeline-dispatch: " << *new_trace);
    output.push(new_trace, 1);
  }

  // we move to the next batch only when all blocks in current batch have been processed
  if (block_sent == block_size_) {
    // round-robin batch selection
    batch_idx_ = (batch_idx_ + 1) % num_blocks_;
    for (auto& bp : block_pids_) {
      bp = 0;
    }
  }
};
