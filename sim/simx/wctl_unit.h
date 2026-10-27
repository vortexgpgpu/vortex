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

namespace vortex {

class Core;

// Warp-control sub-unit of the SFU. Handles WctlType ops (TMC, WSPAWN,
// SPLIT, JOIN, BAR, PRED, WSYNC). Plain (non-SimObject) class owned by
// SfuUnit; its tick is driven through SfuUnit's per-block input/output
// channels.
class WctlUnit {
public:
  explicit WctlUnit(Core* core) : core_(core) {}

  // Execute the WctlType side effects for `trace`. Returns whether the
  // warp should be released after this trace's eop fires; `exit` is set
  // when a TMC disables every thread, leaving the warp's deactivation to
  // the caller. Caller is responsible for the input/output channel push and
  // the latency.
  bool process(instr_trace_t* trace, bool* exit);

  // Whether `trace` needs a core-wide warp-control resource (the barrier unit,
  // wspawn, or warp retirement), which takes one request per cycle across all
  // SFU blocks. No side effects.
  bool is_shared(const instr_trace_t* trace) const;

private:
  // Thread mask a TMC or PRED installs.
  ThreadMask next_tmask(const instr_trace_t* trace) const;

  Core* core_;
};

} // namespace vortex
