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
#include <rvfloats.h>
#include <algorithm>

using namespace vortex;

union flaot_uint32_t {
  float f;
  uint32_t u;
};

inline uint32_t read_element(const std::vector<reg_data_t>& reg_data, int index, TensorFormat format) {
  switch (format) {
  case TensorFormat::Int4: {
    return reg_data.at(index / 8).u >> (index % 8);
  }
  case TensorFormat::Int8: {
    return reg_data.at(index / 4).u >> (index % 4);
  }
  case TensorFormat::FP16: {
    return reg_data.at(index / 2).u >> (index % 2);
  }
  case TensorFormat::BF16: {
    return reg_data.at(index / 2).u >> (index % 2);
  }
  default: assert(false);
    return 0;
  }
}

inline void write_element(std::vector<reg_data_t>& reg_data, int index, uint32_t value, TensorFormat format) {
  switch (format) {
  case TensorFormat::Int32:
  case TensorFormat::FP32: {
    reg_data.at(index).i = value;
    break;
  }
  default: assert(false);
  }
}

inline float type_to_float(uint32_t value, TensorFormat format) {
  switch (format) {
  case TensorFormat::Int4: {
    flaot_uint32_t u2f;
    u2f.u = rv_itof_s(value, 0, nullptr);
    return u2f.f;
  }
  case TensorFormat::Int8: {
    flaot_uint32_t u2f;
    u2f.u = rv_itof_s(value, 0, nullptr);
    return u2f.f;
  }
  case TensorFormat::FP16: {
    flaot_uint32_t u2f;
    u2f.u = rv_htof_s(value, 0, nullptr);
    return u2f.f;
  }
  case TensorFormat::BF16: {
    flaot_uint32_t u2f;
    u2f.u = rv_btof_s(value, 0, nullptr);
    return u2f.f;
  }
  default: assert(false);
  }
  return 0;
}

inline  uint32_t float_to_type(float value, TensorFormat format) {
  switch (format) {
  case TensorFormat::Int32: {
    flaot_uint32_t f2u;
    f2u.f = value;
    return rv_ftoi_s(f2u.u, 0, nullptr);
  }
  case TensorFormat::FP32: {
    flaot_uint32_t f2u;
    f2u.f = value;
    return f2u.u;
  }
  default: assert(false);
  }
  return 0;
}

class TensorCore : public SimObject<TensorCore> {
public:
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

  TensorCore(const SimContext& ctx, const char* name, uint32_t tile_size)
    : SimObject<TensorCore>(ctx, name)
    , Input(this)
    , Output(this)
    , tile_size_(tile_size)
  {}

  ~TensorCore() {
    this->reset();
  }

  void reset() {
    //--
  }

  void tick() {
    //--
  }

  void mmadd(TensorFormat from_format,
             TensorFormat to_format,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs2_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             TensorUnit::TraceData::Ptr trace_data) {
    assert(rd_data.size() <= tile_size_);
    trace_data->latency = 2 + tile_size_;
    // matrix multiplication and accumulation
    for (uint32_t i = 0; i < tile_size_; i++) {
      for (uint32_t j = 0; j < tile_size_; j++) {
        float sum = type_to_float(read_element(rs3_data, i * tile_size_ + j, to_format), to_format);
        for (uint32_t k = 0; k < tile_size_; k++) {
          auto a = type_to_float(read_element(rs1_data, i * tile_size_ + k, from_format), from_format);
          auto b = type_to_float(read_element(rs2_data, k * tile_size_ + j, from_format), from_format);
          sum += a * b;
        }
        write_element(rd_data, i * tile_size_ + j, float_to_type(sum, to_format), to_format);
      }
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  PerfStats perf_stats_;
  uint32_t tile_size_;
};

///////////////////////////////////////////////////////////////////////////////

class TensorUnit::Impl {
public:

  Impl(TensorUnit* simobject, uint32_t tile_size)
    : simobject_(simobject)
    , tensor_cores_(NUM_TENSOR_CORES)
    , tc_sel_(0)
  {
    char sname[100];
    for (uint32_t i = 0; i < NUM_TENSOR_CORES; i++) {
      snprintf(sname, 100, "%s-core%d", simobject->name().c_str(), i);
      tensor_cores_[i] = TensorCore::Create(sname, tile_size);
    }

    this->reset();
  }

  ~Impl() {}

  void reset() {
    //--
  }

  void tick() {
    // forward input to tensor cores
    auto& input = simobject_->Input;
    if (input.empty())
      return;
    auto trace = input.front();
    auto trace_data = std::dynamic_pointer_cast<TraceData>(trace->data);
    tensor_cores_.at(trace_data->tc_idx)->Input.push(trace, 1);
    input.pop();
  }

  void mmadd(TensorFormat from_format,
             TensorFormat to_format,
             const std::vector<reg_data_t>& rs1_data,
             const std::vector<reg_data_t>& rs2_data,
             const std::vector<reg_data_t>& rs3_data,
             std::vector<reg_data_t>& rd_data,
             TensorUnit::TraceData::Ptr trace_data) {
    tensor_cores_.at(tc_sel_)->mmadd(from_format, to_format, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
    trace_data->tc_idx = tc_sel_;
    tc_sel_ = (tc_sel_ + 1) % NUM_TENSOR_CORES;
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit* simobject_;
  std::vector<TensorCore::Ptr> tensor_cores_;
  uint32_t tc_sel_;
  PerfStats perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext& ctx, const char* name, uint32_t tile_size)
  : SimObject<TensorUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , impl_(new Impl(this, tile_size))
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

void TensorUnit::mmadd(TensorFormat from_format,
                       TensorFormat to_format,
                       const std::vector<reg_data_t>& rs1_data,
                       const std::vector<reg_data_t>& rs2_data,
                       const std::vector<reg_data_t>& rs3_data,
                       std::vector<reg_data_t>& rd_data,
                       TensorUnit::TraceData::Ptr trace_data) {
  impl_->mmadd(from_format, to_format, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}

const TensorUnit::PerfStats& TensorUnit::perf_stats() const {
  return impl_->perf_stats();
}
