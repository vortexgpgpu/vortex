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

#include "barrier_unit.h"
#include <cassert>
#include "core.h"
#include "scheduler.h"
#include "socket.h"
#include "debug.h"

using namespace vortex;

BarrierUnit::BarrierUnit(const SimContext& ctx, const char* name, Core* core, Scheduler* scheduler)
  : SimObject<BarrierUnit>(ctx, name)
  , core_(core)
  , scheduler_(scheduler)
  , barriers_(NUM_WARPS * NUM_BARRIERS)
{}

BarrierUnit::~BarrierUnit() {}

void BarrierUnit::on_reset() {
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
}

uint32_t BarrierUnit::get_phase(uint32_t bar_id) const {
  auto bar_index = bar_id & 0x7fffffff;
  return barriers_.at(bar_index).phase;
}

void BarrierUnit::arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x7fffffff;

  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();

  DP(4, "*** Barrier arrive: core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id << ", phase=" << barrier.phase << ", wait_mask=" << barrier.wait_mask << ", count=" << barrier.count << ", events=" << barrier.events);

  if (is_global) {
    barrier.count = count; // save core count
    barrier.wait_mask.set(wid); // mark warp arrival
    if (barrier.wait_mask.count() == active_warps.count() && barrier.events == 0) {
      // notify global barrier
      core_->socket()->global_barrier_arrive(bar_id, count, core_->id());
      barrier.reset();
    }
  } else {
    if (is_sync_bar) {
      barrier.wait_mask.set(wid);
    }
    // local barrier handling
    auto barrier_count_p1 = barrier.count + 1;
    if (barrier_count_p1 == count && barrier.events == 0) {
      // resume waiting warps
      for (uint32_t i = 0; i < NUM_WARPS; ++i) {
        if (barrier.wait_mask.test(i)) {
          DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
          scheduler_->resume(i);
        }
      }
      // reset barrier and advance phase
      barrier.wait_mask.reset();
      ++barrier.phase;
    }
    // update count and wrap around
    if (count == 0) {
      std::cout << "Error: barrier_arrive count=0: core=" << core_->id() << " wid=" << wid << " bar_id=0x" << std::hex << bar_id << std::dec << " bar_index=" << bar_index << " is_sync=" << is_sync_bar << std::endl;
      std::abort();
    }
    barrier.count = barrier_count_p1 % count;
  }
}

bool BarrierUnit::wait(uint32_t bar_id, uint32_t phase, uint32_t wid) {
  auto bar_index = bar_id & 0x7fffffff;

  auto& barrier = barriers_.at(bar_index);
  bool wait = (barrier.phase == phase);

  DP(4, "*** Barrier wait: core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id << ", wait=" << wait);

  if (wait) {
    // add warp to wait list
    barrier.wait_mask.set(wid);
    DP(3, "*** Suspend core #" << core_->id() << ", warp #" << wid << " at barrier #" << bar_id);
  }

  return wait;
}

void BarrierUnit::global_resume(uint32_t bar_id) {
  auto bar_index = bar_id & 0x7fffffff;
  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();
  for (uint32_t i = 0; i < NUM_WARPS; ++i) {
    if (active_warps.test(i)) {
      DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
      scheduler_->resume(i);
    }
  }
  barrier.wait_mask.reset();
  ++barrier.phase;
}

void BarrierUnit::event_attach(uint32_t bar_id) {
  auto bar_index = bar_id & 0x7fffffff;
  auto& barrier = barriers_.at(bar_index);
  ++barrier.events;
}

void BarrierUnit::event_release(uint32_t bar_id) {
  bool is_global = (bar_id >> 31);
  auto bar_index = bar_id & 0x7fffffff;
  auto& barrier = barriers_.at(bar_index);
  const auto& active_warps = scheduler_->active_warps();
  assert(barrier.events > 0);
  --barrier.events;
  if (barrier.events == 0) {
    if (is_global) {
      if (barrier.wait_mask.count() == active_warps.count()) {
        uint32_t num_cores = barrier.count; // was saved in barrier_arrive
        // notify global barrier
        core_->socket()->global_barrier_arrive(bar_id, num_cores, core_->id());
        barrier.reset();
      }
    } else {
      if (barrier.count == 0) {
        // resume waiting warps
        for (uint32_t i = 0; i < NUM_WARPS; ++i) {
          if (barrier.wait_mask.test(i)) {
            DP(3, "*** Resume core #" << core_->id() << ", warp #" << i << " at barrier #" << bar_id);
            scheduler_->resume(i);
          }
        }
        // reset barrier and advance phase
        barrier.wait_mask.reset();
        ++barrier.phase;
      }
    }
  }
}
