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

Dispatcher::Dispatcher(const SimContext& ctx, const char* name, Core* core, uint32_t queue_size, uint32_t block_size, uint32_t num_lanes, uint32_t out_delay)
  : SimObject<Dispatcher>(ctx, name)
  // One dispatch queue per issue slot. The extra entry holds the op still in
  // the collector's output register, which the channel counts as occupancy.
  , Inputs(VX_CFG_ISSUE_WIDTH, SimChannel<instr_trace_t*>(this, queue_size + 1))
  // Bound to the unit's per-block inputs.
  , Outputs(block_size, this)
  , ReleaseOut(this, VX_CFG_ISSUE_WIDTH)
  , core_(core)
  , block_size_(block_size)
  , num_lanes_(num_lanes)
  , num_blocks_(VX_CFG_ISSUE_WIDTH / block_size)
  , num_packets_(VX_CFG_NUM_THREADS / num_lanes)
  , batch_idx_(0)
  , out_delay_(out_delay)
  , block_pids_(block_size, 0)
{}

Dispatcher::~Dispatcher() {}

void Dispatcher::on_reset() {
  batch_idx_ = 0;
  for (auto& bp : block_pids_) {
    bp = 0;
  }
}

void Dispatcher::on_tick() {
  // Batches holding an instruction this cycle, sampled before any is popped.
  uint32_t valid_batches = 0;
  if (num_blocks_ != 1) {
    for (uint32_t i = 0; i < VX_CFG_ISSUE_WIDTH; ++i) {
      if (!Inputs.at(i).empty()) {
        valid_batches |= 1u << (i / block_size_);
      }
    }
  }

  // process inputs
  uint32_t block_sent = 0;
  for (uint32_t b = 0; b < block_size_; ++b) {
    uint32_t i = batch_idx_ * block_size_ + b;
    auto& input = Inputs.at(i);
    if (input.empty()) {
      ++block_sent;
      continue;
    }

    // input[batch_idx*block_size + b] aggregates onto output[b]; the op
    // leaves its slot queue only when the unit's input has room.
    auto& output = Outputs.at(b);
    if (output.full())
      continue;

    auto trace = input.peek();

    // check if trace should be split
    auto new_trace = trace;
    if (num_packets_ != 1) {
      auto block_pid = block_pids_.at(b);
      // check if current block has already been processed
      if (block_pid == -1) {
        ++block_sent;
        continue;
      }
      // Compute the total number of packets we'll emit on the first call
      // for this trace (block_pid==0). Each SIMD group with any active
      // lane gets one packet; sparse divergent tmasks emit fewer than
      // num_packets_. Commit uses num_pkts to defer the scoreboard
      // release until every packet's writeback has applied — without it,
      // an eop packet that completes ahead of its peers (cache responses
      // arrive out of order) would release the destination while some
      // lanes are still stale.
      if (block_pid == 0) {
        uint32_t n_pkts = 0;
        for (uint32_t j = 0; j < VX_CFG_NUM_THREADS; j += num_lanes_) {
          for (uint32_t k = 0; k < num_lanes_; ++k) {
            if (trace->tmask.test(j + k)) { ++n_pkts; break; }
          }
        }
        trace->num_pkts = n_pkts == 0 ? 1 : n_pkts;
      }
      // calculate current packet start and end
      int start(-1), end(-1);
      for (uint32_t j = block_pid * num_lanes_, n = VX_CFG_NUM_THREADS; j < n; ++j) {
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
        ReleaseOut.send(trace, 0);
        ++block_sent;
      }
      ThreadMask tmask(VX_CFG_NUM_THREADS);
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
      ReleaseOut.send(trace, 0);
      ++block_sent;
    }
    DT(3, this->name() << "-pipeline dispatch: " << *new_trace);
    output.send(new_trace, out_delay_);
  }

  // advance to next batch once all blocks in the current batch have been processed
  if (block_sent == block_size_) {
    // Priority grant to the lowest batch with an instruction: an empty issue
    // slot costs no dispatch cycle. With none, the grant rests on the last.
    batch_idx_ = valid_batches ? __builtin_ctz(valid_batches) : (num_blocks_ - 1);
    for (auto& bp : block_pids_) {
      bp = 0;
    }
  }
};
