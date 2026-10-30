// Copyright © 2019-2026
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

#include <cstdint>
#include <VX_types.h>
#include "types.h"
#include "instr_trace.h"

namespace vortex {

// MUFU sub-unit of the SFU. The SFU's lanes are serialized onto the MUFU
// datapath lanes, one batch per cycle, and a warp is accepted on its last batch.
class MufuUnit {
public:
  // Cycles a result spends in the datapath after the warp is accepted.
  static constexpr uint32_t kLatency = VX_MUFU_LATENCY;

  // Returns false while the warp's batches are still being issued; on the
  // accepting cycle computes the results and returns true.
  bool process(instr_trace_t* trace);

private:
  static uint32_t issue_cycles(const instr_trace_t* trace);

  const instr_trace_t* head_ = nullptr;
  uint32_t remaining_ = 0;
};

}
