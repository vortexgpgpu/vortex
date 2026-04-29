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

#include <simobject.h>
#include "instr_trace.h"
#include <unordered_map>
#include <vector>

namespace vortex {

// Tracks per-warp register reservations between issue and writeback.
class Scoreboard : public SimObject<Scoreboard> {
public:
  struct reg_use_t {
    RegType  reg_type;
    uint32_t reg_id;
    FUType   fu_type;
    OpType   op_type;
    uint64_t uuid;
  };

  Scoreboard(const SimContext& ctx, const char* name);
  ~Scoreboard();

  bool in_use(instr_trace_t* trace) const;

  std::vector<reg_use_t> get_uses(instr_trace_t* trace) const;

  void reserve(instr_trace_t* trace);

  void release(instr_trace_t* trace);

protected:
  void on_reset();

private:
  static uint32_t get_reg_id(const RegOpd& reg, uint32_t wid) {
    return (wid << RegOpd::ID_BITS) | reg.id();
  }

  std::vector<std::vector<RegMask>> in_use_regs_;
  std::unordered_map<uint32_t, instr_trace_t*> owners_;

  friend class SimObject<Scoreboard>;
};

}
