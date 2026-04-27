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

#include <vector>
#include <simobject.h>
#include "types.h"

namespace vortex {

class Core;
class Scheduler;

// Mirrors the RTL VX_bar_unit module instantiated inside VX_scheduler.sv.
// Owns per-warp barrier state; resumes waiting warps via the parent Scheduler;
// participates in cross-core global barriers via Socket.
class BarrierUnit : public SimObject<BarrierUnit> {
public:
  BarrierUnit(const SimContext& ctx, const char* name, Core* core, Scheduler* scheduler);
  ~BarrierUnit();

  uint32_t get_phase(uint32_t bar_id) const;
  void arrive(uint32_t bar_id, uint32_t count, uint32_t wid, bool is_sync_bar);
  bool wait(uint32_t bar_id, uint32_t phase, uint32_t wid);
  void global_resume(uint32_t bar_id);
  void event_attach(uint32_t bar_id);
  void event_release(uint32_t bar_id);

protected:
  void on_reset();

private:
  Core*       core_;
  Scheduler*  scheduler_;
  std::vector<warp_barrier_t> barriers_;

  friend class SimObject<BarrierUnit>;
};

}
