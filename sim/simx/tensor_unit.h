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
#include "pipeline.h"

namespace vortex {

enum class TensorFormat : int {
  Int4  = 0,
  Int8  = 1,
  Int16 = 2,
  Int32 = 3,
  Int64 = 4,
  FP16  = 5,
  FP32  = 6,
  FP64  = 7,
  BF16  = 8,
  _MAX  = 9
};

class TensorUnit : public SimObject<TensorUnit> {
public:
  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    uint32_t tc_idx;
    uint32_t latency;
  };

  struct PerfStats {
    uint64_t latency;

    PerfStats()
      : latency(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->latency += rhs.latency;
      return *this;
    }
  };

  SimPort<instr_trace_t*> Input;
  SimPort<instr_trace_t*> Output;

  TensorUnit(const SimContext& ctx, const char* name, uint32_t tile_size);
  ~TensorUnit();

  void reset();

  void tick();

  void mmadd(TensorFormat from_format,
             TensorFormat to_format,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs2_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             TensorUnit::TraceData::Ptr trace_data);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}