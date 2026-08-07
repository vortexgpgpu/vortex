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

// ONE definition of the tensor PE's pipeline depth, for every unit that contains one.
//
// This used to live inside tcu_unit.cpp as a file-local constant, which was fine while
// the in-core TCU was the only tensor datapath in the machine. It is not any more: the
// DTCU has its own MAC array, and it had its own hardcoded latency (6) chosen by hand
// against whatever the TCU happened to cost at the time. When upstream replaced the TCU's
// hardcoded `delay = 4` with the derived value below, the DTCU did not follow, and the
// two units' pipeline depths silently stopped agreeing -- catastrophically so for any
// VX_CFG_TCU_TYPE but the default, where the TCU's depth ranges up to 54 cycles and the
// DTCU's stayed at 6.
//
// Both now include this header. A change to the PE type moves both, which is the point:
// the DTCU is meant to be the SAME arithmetic in a different PLACE, so its compute
// pipeline has no business being a different depth from the TCU's.

#include <VX_config.h>
#include <bitmanip.h>
#include "tensor_cfg.h"

namespace vortex {
namespace tcu_timing {

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<VX_CFG_NUM_THREADS>;

// Dot-product pipeline depth of the configured tensor-PE type
// (multiply / align / accumulate-reduce / round stage sum).
//
// VX_CFG_TCU_TYPE selects which FEDP (fused elementwise dot-product) hardware the
// tensor unit is built from -- one of DPI | DSP | BHF | TFR | FPNEW, TFR by default.
// It is a HARDWARE choice, not a numerical one: every type computes the same product,
// and they differ in how deep a pipeline it takes. That is why the depth is derived
// here rather than written down.
#if defined(VX_CFG_TCU_TYPE_DSP)
static constexpr uint32_t kFedpLatency = 1 + 8 + log2ceil(2 * cfg::tcK + 1) * 11;
#elif defined(VX_CFG_TCU_TYPE_BHF)
static constexpr uint32_t kFedpLatency = (2 + 1) + 1 + log2ceil(2 * cfg::tcK + 1) * (2 + 1);
#elif defined(VX_CFG_TCU_TYPE_FPNEW)
static constexpr uint32_t kFedpLatency = 6 + 1 + log2ceil(2 * cfg::tcK) * 7 + 7;
#elif defined(VX_CFG_TCU_TYPE_DPI)
static constexpr uint32_t kFedpLatency = 2 + 2;
#else // TFR
static constexpr uint32_t kFedpLatency = 1 + 1 + 1 + 1;
#endif

// End-to-end MMA cost: dispatch plus the dot-product pipeline.
static constexpr uint32_t kMmaLatency = 1 + kFedpLatency;

} // namespace tcu_timing
} // namespace vortex
