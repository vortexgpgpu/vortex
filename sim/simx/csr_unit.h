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

#include <array>
#include "instr_trace.h"
#include "constants.h"

namespace vortex {

class Core;

#ifdef EXT_RASTER_ENABLE
// Per-warp + per-thread snapshot of the most recent vx_rast() pop's
// raster_csrs_t. Latched by SfuUnit when a RasterRsp is delivered;
// surfaced to the kernel via VX_CSR_RASTER_POS_MASK + VX_CSR_RASTER_PID +
// VX_CSR_RASTER_BCOORD_{X,Y,Z}{0..3}.
struct RasterCsrs {
  uint32_t pos_mask = 0;
  uint32_t pid      = 0;
  std::array<std::array<uint32_t, 4>, 3> bcoords = {}; // [axis][corner]
};
#endif

// CSR sub-unit of the SFU. Owns CSR semantics (get/set CSR, FPU rounding
// mode lookup, fflags update). Plain (non-SimObject) class owned by
// SfuUnit; per-warp fcsr / cta_csrs / mscratch live on the Scheduler
// and are reached via core->scheduler().warp(wid).
class CsrUnit {
public:
  explicit CsrUnit(Core* core) : core_(core) {}

  // Execute the CsrType side effects for `trace` (CSRRW / CSRRS / CSRRC).
  // Caller (SfuUnit) is responsible for the input/output channel push and
  // the latency.
  void process(instr_trace_t* trace);

  // CSR access surface used by FpuUnit (fcsr lookup / fflags update).
  Word get_csr(uint32_t addr, uint32_t wid, uint32_t tid);
  void set_csr(uint32_t addr, Word value, uint32_t wid, uint32_t tid);
  uint32_t get_fpu_rm(uint32_t funct3, uint32_t wid, uint32_t tid);
  void update_fcrs(uint32_t fflags, uint32_t wid, uint32_t tid);

#ifdef EXT_RASTER_ENABLE
  // Latch raster CSRs for one (warp, thread) lane on RasterRsp arrival.
  void set_raster_csrs(uint32_t wid, uint32_t tid, const RasterCsrs& csrs) {
    raster_csrs_.at(wid).at(tid) = csrs;
  }
#endif

private:
  Core* core_;
#ifdef EXT_RASTER_ENABLE
  std::array<std::array<RasterCsrs, NUM_THREADS>, NUM_WARPS> raster_csrs_{};
#endif
};

} // namespace vortex
