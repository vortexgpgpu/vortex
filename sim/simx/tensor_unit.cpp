
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

using namespace vortex;

template <typename T>
class FMAD : public SimObject<FMAD<T>> {
public:
  SimPort<T> Input;
	SimPort<T> Output;

  FMAD(const SimContext &ctx, const char* name)
    : SimObject<FMAD<T>>(ctx, name)
    , Input(this)
    , Output(this)
  {}

  virtual ~FMAD() {}

  void reset() {
    //--
  }

  void tick() {
    //--
  }
};

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Config& config, Core* core)
    : simobject_(simobject)
    , config_(config)
    , core_(core)
    , perf_stats_()
  {}

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
  }

  void tick() {
    // Implement the tick logic here
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  TensorUnit*   simobject_;
  Config        config_;
  Core*         core_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Config& config, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(config.num_ports, this)
	, Outputs(config.num_ports, this)
	, impl_(new Impl(this, config, core))
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

const TensorUnit::PerfStats &TensorUnit::perf_stats() const {
	return impl_->perf_stats();
}