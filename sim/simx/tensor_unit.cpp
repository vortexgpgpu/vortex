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

#include "tensor_unit.h"
#include "mem.h"
#include <VX_config.h>
#include <algorithm>

using namespace vortex;

class TensorCore : public SimObject<TensorCore> {
public:
  struct PerfStats {
    uint64_t reads;
    uint64_t writes;
    uint64_t latency;
    uint64_t stalls;

    PerfStats()
      : reads(0)
      , writes(0)
      , latency(0)
      , stalls(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads   += rhs.reads;
      this->writes  += rhs.writes;
      this->latency += rhs.latency;
      this->stalls  += rhs.stalls;
      return *this;
    }
  };

  SimPort<instr_trace_t*> Input;
  SimPort<instr_trace_t*> Output;

  TensorCore(const SimContext& ctx, const char* name);
  ~TensorCore();

  void reset();

  void tick();

  void attach_ram(RAM* mem);

  void mmadd(TensorUnit::TraceData::Ptr trace_data);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

///////////////////////////////////////////////////////////////////////////////

class TensorUnit::Impl {
public:

  Impl(TensorUnit* simobject)
    : simobject_(simobject)
  {
    this->reset();
  }

  ~Impl() {}

  void reset() {
    //--
  }

  void tick() {
    //--
  }

  void mmadd(const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs2_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             TensorUnit::TraceData::Ptr& trace_data) {
    //--
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit* simobject_;
  PerfStats perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext& ctx, const char* name)
  : SimObject<TensorUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

void TensorUnit::mmadd(const std::vector<reg_data_t>& rs1_data,
                        const std::vector<reg_data_t>& rs2_data,
                        const std::vector<reg_data_t>& rs3_data,
                        std::vector<reg_data_t>& rd_data,
                        TensorUnit::TraceData::Ptr& trace_data) {
  impl_->mmadd(rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
