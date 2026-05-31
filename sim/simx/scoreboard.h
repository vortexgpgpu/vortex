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

  // RTU callback-trap support (proposal §4.6). When a warp parked at
  // vx_rt_wait traps into its callback dispatcher, the parked WAIT's
  // destination-register reservation would deadlock the dispatcher's
  // register save/restore (a high-pressure FP intersection shader spills
  // the WAIT's rd, and `sw <rd>` sources a register the WAIT can't release
  // until the dispatcher's own cb_ret runs). snapshot_warp() lifts the
  // warp's outstanding reservations at trap entry so the dispatcher runs
  // freely; restore_warp() re-installs them at mret so the resumed kernel's
  // vx_rt_get_after ordering still holds. If a register is transiently
  // owned by an in-flight dispatcher writeback at restore time, the
  // re-reservation is deferred to that op's release() (pending_reserve_).
  std::vector<instr_trace_t*> snapshot_warp(uint32_t wid);
  void restore_warp(const std::vector<instr_trace_t*>& snapshot);

  // Per-packet commit notifier. Returns true when every SIMD-split packet
  // for the instruction owning this destination register has committed
  // (i.e., when the caller should call release()). For non-split traces
  // (num_pkts==1) this returns true on the first call. Independent of the
  // dispatcher's `eop` flag — eop fires on the last DISPATCHED packet,
  // but cache responses may complete out of order, so we count COMMITS.
  bool commit_packet(instr_trace_t* trace);

protected:
  void on_reset();

private:
  static uint32_t get_reg_id(const RegOpd& reg, uint32_t wid) {
    return (wid << RegOpd::ID_BITS) | reg.id();
  }

  std::vector<std::vector<RegMask>> in_use_regs_;
  std::unordered_map<uint32_t, instr_trace_t*> owners_;
  std::unordered_map<uint32_t, uint32_t> commit_counts_;
  // reg_id -> trace whose reservation must be re-installed once the reg's
  // current in-flight owner releases (RTU callback-trap restore handoff).
  std::unordered_map<uint32_t, instr_trace_t*> pending_reserve_;

  friend class SimObject<Scoreboard>;
};

}
