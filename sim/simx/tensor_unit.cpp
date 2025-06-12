
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
#include "tensor_cfg.h"
#include <softfloat_types.h>
#include <rvfloats.h>
#include "core.h"

using namespace vortex;

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;

union fp32_u32_t {
  float    f;
  uint32_t u;
};

inline uint32_t floatToBits(float f) noexcept {
  fp32_u32_t pun;
  pun.f = f;
  return pun.u;
}

inline float bitsToFloat(uint32_t u) noexcept {
  fp32_u32_t pun;
  pun.u = u;
  return pun.f;
}

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

template <typename Ot, typename It>
class FMA {
public:
  Ot operator()(const It& a, const It& b, const Ot& c) {
    return static_cast<Ot>(a) * static_cast<Ot>(b) + c;
  }
};

template <>
class FMA<float, float16_t> {
public:
  float operator()(float16_t a, float16_t b, float c) {
    auto xa = rv_htof_s(a.v, 0, nullptr);
    auto xb = rv_htof_s(b.v, 0, nullptr);
    auto xc = floatToBits(c);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    return bitsToFloat(xd);
  }
};

template <>
class FMA<float16_t, float16_t> {
public:
  float16_t operator()(float16_t a, float16_t b, float16_t c) {
    auto xa = rv_htof_s(a.v, 0, nullptr);
    auto xb = rv_htof_s(b.v, 0, nullptr);
    auto xc = rv_htof_s(c.v, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftoh_s(xd, 0, nullptr);
    return float16_t{xh};
  }
};

template <>
class FMA<float, bfloat16_t> {
public:
  float operator()(bfloat16_t a, bfloat16_t b, float c) {
    auto xa = rv_btof_s(a.v, 0, nullptr);
    auto xb = rv_btof_s(b.v, 0, nullptr);
    auto xc = floatToBits(c);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    return bitsToFloat(xd);
  }
};

template <>
class FMA<bfloat16_t, bfloat16_t> {
public:
  bfloat16_t operator()(bfloat16_t a, bfloat16_t b, bfloat16_t c) {
    auto xa = rv_btof_s(a.v, 0, nullptr);
    auto xb = rv_btof_s(b.v, 0, nullptr);
    auto xc = rv_btof_s(c.v, 0, nullptr);
    auto xd = rv_fmadd_s(xa, xb, xc, 0, nullptr);
    auto xh = rv_ftob_s(xd, 0, nullptr);
    return bfloat16_t{xh};
  }
};

template <typename Ot, typename It>
float FEDP(const reg_data_t *a_row, const reg_data_t *b_col, float c_val) {
  constexpr uint32_t i_ratio = sizeof(float) / sizeof(It);
  static_assert(i_ratio * sizeof(It) == sizeof(float), "FEDP: tcK * i_ratio must be <= 32");
DISABLE_WARNING_PUSH
DISABLE_WARNING_STRICT_ALIASING
  Ot acc = *reinterpret_cast<const Ot*>(&c_val);
  for (uint32_t z = 0; z < cfg::tcK; ++z) {
    auto a = reinterpret_cast<const It *>(&a_row[z].f32);
    auto b = reinterpret_cast<const It *>(&b_col[z].f32);
    for (uint32_t i = 0; i < i_ratio; ++i) {
      acc = FMA<Ot, It>()(a[i], b[i], acc);
    }
  }
  float ret(0);
  *reinterpret_cast<Ot*>(&ret) = acc;
DISABLE_WARNING_POP
  return ret;
}

using PFN_FEDP = float (*)(const reg_data_t*, const reg_data_t*, float);

static PFN_FEDP select_FEDP(uint32_t IT, uint32_t OT) {
  switch (OT) {
  case vt::fp32::id:
    switch (IT) {
    case vt::fp32::id:
      return FEDP<float, float>;
    case vt::fp16::id:
      return FEDP<float, float16_t>;
    case vt::bf16::id:
      return FEDP<float, bfloat16_t>;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::fp16::id:
    switch (IT) {
    case vt::fp16::id:
      return FEDP<float16_t, float16_t>;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::bf16::id:
    switch (IT) {
    case vt::bf16::id:
      return FEDP<bfloat16_t, bfloat16_t>;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  case vt::int32::id:
    switch (IT) {
    case vt::int32::id:
      return FEDP<int32_t, int32_t>;
    case vt::int16::id:
      return FEDP<int32_t, int16_t>;
    case vt::int8::id:
      return FEDP<int32_t, int8_t>;
    default:
      std::cout << "Error: unsupported mma format: " << IT << " -> " << OT << "!" << std::endl;
      std::abort();
    }
    break;
  default:
    std::cout << "Error: unsupported output type: " << OT << "!" << std::endl;
    std::abort();
  }
}

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
  {
    //--
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
          return;
      auto trace = input.front();
      int delay = 0;
      switch (trace->tpu_type) {
      case TpuType::WMMA:
        delay = 4;
        break;
      default:
        std::abort();
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << trace->tpu_type << ", " << *trace);
      input.pop();
    }
  }

  void wmma(uint32_t wid,
            uint32_t fmt,
            uint32_t step,
            const std::vector<reg_data_t>& rs1_data,
            const std::vector<reg_data_t>& rs2_data,
            const std::vector<reg_data_t>& rs3_data,
            std::vector<reg_data_t>& rd_data,
            ExeTraceData* trace_data) {
    __unused(wid);
    __unused(trace_data);

    uint32_t fmt_d = fmt >> 4;
    uint32_t fmt_s = fmt & 0xf;
    auto fedp = select_FEDP(fmt_s, fmt_d);

    uint32_t m = step >> 4;
    uint32_t n = step & 0xf;
    uint32_t a_off = (m % cfg::a_sub_blocks) * cfg::a_block_size;
    uint32_t b_off = (n % cfg::b_sub_blocks) * cfg::b_block_size;

    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      for (uint32_t j = 0; j < cfg::tcN; ++j) {
        auto a_row = rs1_data.data() + a_off + i * cfg::tcK;
        auto b_col = rs2_data.data() + b_off + j * cfg::tcK;
        auto c = rs3_data.at(i * cfg::tcN + j).f32;
        auto d = fedp(a_row, b_col, c);
        rd_data.at(i * cfg::tcN + j).u64 = nan_box(floatToBits(d));
      }
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:

  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
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

void TensorUnit::wmma(uint32_t wid,
                      uint32_t fmt,
                      uint32_t step,
                      const std::vector<reg_data_t>& rs1_data,
                      const std::vector<reg_data_t>& rs2_data,
                      const std::vector<reg_data_t>& rs3_data,
                      std::vector<reg_data_t>& rd_data,
                      ExeTraceData* trace_data) {
  impl_->wmma(wid, fmt, step, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}