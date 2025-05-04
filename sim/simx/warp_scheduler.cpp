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

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <util.h>
#include <bitset>
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#include <rvfloats.h>

#ifdef EXT_V_ENABLE
#include "processor_impl.h"
#endif
#include "VX_types.h"

#include "emulator.h"
#include "instr_trace.h"
#include "instr.h"
#include "dcrs.h"
#include "core.h"
#include "socket.h"
#include "cluster.h"
#include "local_mem.h"
#include "types.h"

using namespace vortex;

int Emulator::schedule_warp(SchedulerPolicy policy) {
  switch (policy) {
  case SchedulerPolicy::ROUND_ROBIN:
    return Emulator::schedule_RR();

  case SchedulerPolicy::ROUND_ROBIN_ROUND_ROBIN:
    return Emulator::schedule_RR_RR();

  case SchedulerPolicy::ROUND_ROBIN_GREEDY_THEN_OLDEST:
    return Emulator::schedule_RR_GTO();

  case SchedulerPolicy::GREEDY_THEN_OLDEST_GREEDY_THEN_OLDEST:
    return Emulator::schedule_GTO_GTO();

  case SchedulerPolicy::GREEDY_THEN_OLDEST_ROUND_ROBIN:
    return Emulator::schedule_GTO_RR();

  case SchedulerPolicy::GREEDY_THEN_OLDEST:
    return Emulator::schedule_GTO();

  default:
    return Emulator::schedule_RR();
  }
}

int Emulator::schedule_RR(){
  int scheduled_warp = -1;
  for (size_t wid = 0, nw = arch_.num_warps(); wid < nw; ++wid) {
    bool warp_active = active_warps_.test(wid);
    bool warp_stalled = stalled_warps_.test(wid);
    if (warp_active && !warp_stalled) {
        scheduled_warp = wid;
        break;
    }
  }
  return scheduled_warp;
}

int Emulator::schedule_RR_RR(){
    return schedule_RR(); //Temporary
}

int Emulator::schedule_RR_GTO(){
    return schedule_RR(); //Temporary
}

int Emulator::schedule_GTO_GTO(){
    return schedule_RR(); //Temporary
}

int Emulator::schedule_GTO_RR(){
    return schedule_RR(); //Temporary
}

int Emulator::schedule_GTO(){
    uint32_t oldest_warp_age = 0;

    if(oldest_warp_ != -1) {
        bool warp_active = active_warps_.test(oldest_warp_);
        bool warp_stalled = stalled_warps_.test(oldest_warp_);
        if (warp_active && !warp_stalled) {
            return oldest_warp_;
        }
        oldest_warp_ = -1;
    }

    for (size_t wid = 0, nw = arch_.num_warps(); wid < nw; ++wid) {
        bool warp_active = active_warps_.test(wid);
        bool warp_stalled = stalled_warps_.test(wid);
        if (warp_active && !warp_stalled) {
            warps_.at(wid).age++;
            if (warps_.at(wid).age >= oldest_warp_age) {
                oldest_warp_ = wid;
                oldest_warp_age = warps_.at(wid).age;
                continue;
            }
        }
    }
    
    if (oldest_warp_ != -1) {
        warps_.at(oldest_warp_).age = 0;
    }
    return oldest_warp_;
}