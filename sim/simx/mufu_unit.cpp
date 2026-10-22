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

#include "mufu_unit.h"
#include <mufu.h>
#include <nan_box.h>
#include "debug.h"

using namespace vortex;

uint32_t MufuUnit::issue_cycles(const instr_trace_t* trace) {
  constexpr uint32_t kPacketLanes = VX_CFG_NUM_SFU_LANES;
  constexpr uint32_t kBatches = VX_CFG_NUM_SFU_LANES / VX_CFG_NUM_MUFU_LANES;
  static_assert(VX_CFG_NUM_SFU_LANES % VX_CFG_NUM_MUFU_LANES == 0,
                "MUFU lanes must divide the SFU lanes");
  uint32_t cycles = 0;
  for (uint32_t base = 0; base < VX_CFG_NUM_THREADS; base += kPacketLanes) {
    for (uint32_t t = base; t < base + kPacketLanes; ++t) {
      if (trace->tmask.test(t)) {
        cycles += kBatches;
        break;
      }
    }
  }
  return cycles;
}

bool MufuUnit::process(instr_trace_t* trace) {
  if (head_ != trace) {
    head_ = trace;
    remaining_ = issue_cycles(trace);
  }
  if (remaining_ > 1) {
    --remaining_;
    return false;
  }
  head_ = nullptr;
  remaining_ = 0;

  auto fn = uint32_t(std::get<MufuType>(trace->op_type));
  auto& rs1_data = trace->src_data[0];
  trace->dst_data.assign(VX_CFG_NUM_THREADS, reg_data_t{});
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (trace->tmask.test(t)) {
      uint32_t x = nanbox::unbox32(rs1_data[t].u64, mufu::QNAN_F32);
      trace->dst_data[t].u64 = nanbox::box32(mufu::eval(fn, x));
    }
  }
  DT(3, "mufu-unit process: " << *trace);
  return true;
}
