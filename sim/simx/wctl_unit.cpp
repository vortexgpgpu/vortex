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

#include "wctl_unit.h"
#include <iostream>
#include "core.h"
#include "scheduler.h"
#include "constants.h"
#include "debug.h"

using namespace vortex;

bool WctlUnit::process(instr_trace_t* trace) {
  bool release_warp = trace->fetch_stall;
  auto wctl_type    = std::get<WctlType>(trace->op_type);
  auto& sched       = core_->scheduler();
  auto& warp        = sched.warp(trace->wid);
  auto instrArgs    = trace->instr_ptr->get_args();
  auto wctlArgs     = std::get<IntrWctlArgs>(instrArgs);
  uint32_t num_threads = NUM_THREADS;
  auto& rs1_data    = trace->src_data[0];
  auto& rs2_data    = trace->src_data[1];

  uint32_t thread_start = 0;
  for (; thread_start < num_threads; ++thread_start) {
    if (warp.tmask.test(thread_start)) break;
  }
  int32_t thread_last = num_threads - 1;
  for (; thread_last >= 0; --thread_last) {
    if (warp.tmask.test(thread_last)) break;
  }

  switch (wctl_type) {
  case WctlType::TMC: {
    ThreadMask next_tmask(num_threads);
    for (uint32_t t = 0; t < num_threads; ++t) {
      next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
    }
    if (trace->eop) {
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::WSPAWN: {
    if (trace->eop) {
      release_warp = core_->wspawn(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
    }
  } break;
  case WctlType::SPLIT: {
    Word next_pc = trace->PC + 4;
    ThreadMask then_tmask(num_threads);
    ThreadMask else_tmask(num_threads);
    auto not_pred = wctlArgs.is_cond_neg;
    for (uint32_t t = 0; t < num_threads; ++t) {
      auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
      then_tmask[t] = warp.tmask.test(t) && cond;
      else_tmask[t] = warp.tmask.test(t) && !cond;
    }
    ThreadMask next_tmask = warp.tmask;
    bool is_divergent = then_tmask.any() && else_tmask.any();
    auto stack_size = warp.ipdom_stack.size();
    // stack_size captured pre-push is what each pid writes to its
    // dst_data — it's the kernel-visible value at this PC.
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      trace->dst_data[t].i = stack_size;
    }
    // ipdom_stack push + tmask update
    if (trace->eop) {
      if (is_divergent) {
        if (stack_size == sched.ipdom_size()) {
          std::cout << "IPDOM stack is full! size=" << stack_size << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
          std::abort();
        }
        next_tmask = (then_tmask.count() <= else_tmask.count()) ? then_tmask : else_tmask;
        warp.ipdom_stack.emplace(warp.tmask, next_pc);
        core_->perf_stats().divergence += 1;
      }
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::JOIN: {
    auto stack_ptr = rs1_data.at(thread_last).u;
    auto stack_size = warp.ipdom_stack.size();
    ThreadMask next_tmask = warp.tmask;
    // ipdom_stack pop + tmask update
    if (trace->eop) {
      if (stack_ptr != stack_size) {
        if (warp.ipdom_stack.empty()) {
          std::cout << "IPDOM stack is empty!\n" << std::flush;
          std::abort();
        }
        if (warp.ipdom_stack.top().fallthrough) {
          next_tmask = warp.ipdom_stack.top().orig_tmask;
          warp.ipdom_stack.pop();
        } else {
          next_tmask = ~warp.tmask & warp.ipdom_stack.top().orig_tmask;
          warp.PC = warp.ipdom_stack.top().else_PC;
          warp.ipdom_stack.top().fallthrough = true;
        }
      }
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::BAR: {
    uint32_t arg1 = rs1_data[thread_last].u;
    uint32_t arg2 = rs2_data[thread_last].u;
    uint32_t bar_id = bar_decode_id(arg1, NUM_BARRIERS);
    bool is_sync_bar = (bool)wctlArgs.is_sync_bar;
    if (wctlArgs.is_bar_arrive) {
      uint32_t phase = sched.barrier_unit().get_phase(bar_id);
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t)) continue;
        trace->dst_data[t].i = phase;
      }
    }
    if (trace->eop) {
      if (trace->wb || is_sync_bar) {
        core_->barrier_arrive(bar_id, arg2, trace->wid, is_sync_bar);
        if (is_sync_bar) {
          release_warp = false;
        }
      } else {
        release_warp = !core_->barrier_wait(bar_id, arg2, trace->wid);
      }
    }
  } break;
  case WctlType::PRED: {
    ThreadMask pred(num_threads);
    auto not_pred = wctlArgs.is_cond_neg;
    for (uint32_t t = 0; t < num_threads; ++t) {
      auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
      pred[t] = warp.tmask.test(t) && cond;
    }
    ThreadMask next_tmask = warp.tmask;
    if (pred.any()) {
      next_tmask &= pred;
    } else {
      next_tmask = ThreadMask(num_threads, rs2_data.at(thread_last).u);
    }
    if (trace->eop) {
      release_warp = core_->setTmask(trace->wid, next_tmask);
    }
  } break;
  case WctlType::WSYNC:
    release_warp = true;
    break;
  default:
    std::abort();
  }
  return release_warp;
}
