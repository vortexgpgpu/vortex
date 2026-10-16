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
#include "debug.h"
#include "core.h"

using namespace vortex;

namespace {
// SimX keeps every single-precision FP register value NaN-boxed (see fpu_unit.cpp).
inline uint32_t unbox(uint64_t value) {
  if (uint32_t(value >> 32) != 0xffffffff)
    return mufu::QNAN_F32;
  return uint32_t(value);
}
inline uint64_t box(uint32_t value) {
  return value | 0xffffffff00000000;
}
}

MufuUnit::MufuUnit(const SimContext& ctx, const char* name, Core* core)
  // The RTL pipeline has no tag store: what is in flight is bounded by the
  // pipeline depth plus its two-entry output buffer.
  : FuncUnit<VX_CFG_NUM_MUFU_BLOCKS>(ctx, name, core, VX_CFG_MUFU_LATENCY + 2)
{}

void MufuUnit::execute(instr_trace_t* trace) {
  auto& tmask = trace->tmask;
  auto mufu_type = std::get<MufuType>(trace->op_type);
  auto& rs1_data = trace->src_data[0];
  trace->dst_data.assign(VX_CFG_NUM_THREADS, reg_data_t{});
  auto& rd_data = trace->dst_data;
  for (uint32_t t = 0; t < VX_CFG_NUM_THREADS; ++t) {
    if (!tmask.test(t))
      continue;
    rd_data[t].u64 = box(mufu::eval(uint32_t(mufu_type), unbox(rs1_data[t].u64)));
  }
  DT(3, this->name() << " execute: op=" << mufu_type << ", " << *trace);
}

void MufuUnit::on_tick() {
  bool idle = true;
  for (uint32_t b = 0; b < VX_CFG_NUM_MUFU_BLOCKS; ++b) {
    auto& input = Inputs.at(b);
    if (!input.empty()) {
      auto& output = Outputs.at(b);
      if (!output.full()) {
        auto trace = input.peek();
        this->execute(trace);
        // Without a lane-gather stage commit reads this port directly, so the
        // result also takes the registered stage other units cross into commit.
        uint32_t delay = VX_CFG_MUFU_LATENCY + (kGather ? 0 : 1);
        output.send(trace, delay);
        input.pop();
      }
    }
    idle &= (input.size() == 0);
  }
  if (idle) {
    this->tick_sleep();
  }
}
