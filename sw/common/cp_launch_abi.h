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

// Launch descriptors the command processor reads from device memory when the
// command executes, so a kernel may also produce them. Both are a header
// followed by 8-byte {u32 dcr_addr, u32 value} pairs that never straddle a
// 64-byte line.
//
// Both descriptors start on a 64-byte line. A command whose descriptor address
// is not line aligned is not executed: it retires and sets CP_Q_ERROR_DESC_ALIGN
// in its queue's Q_ERROR register.
//
// QMD (CMD_LAUNCH_QMD): {u32 count, u32 reserved, count x pair}.
//
// ILD (CMD_LAUNCH_INDIRECT): a 64-byte header line, then qmd_count pairs:
//
//   predicate : exec = (*pred_addr > pred_ref) ^ PRED_INV          (HAS_PRED)
//   grid      : v_d = (sel_d == ILD_SEL_ONE) ? 1 : record[sel_d]   (HAS_RECORD)
//               x only: v_x = sat_sub(v_x, x_bias), then v_x /= x_div (x_div > 1)
//               g_d = (v_d + 2^sh_d - 1) >> sh_d; any g_d == 0 clears exec
//   not exec  : retire, and retire the next skip_n commands without executing
//   NO_LAUNCH : retire after the predicate; no DCR write and no start
//   otherwise : replay the pairs (GRID_DIM_X/Y/Z replaced by g when
//               HAS_RECORD), then start the kernel and drain it

#ifndef VORTEX_CP_LAUNCH_ABI_H
#define VORTEX_CP_LAUNCH_ABI_H

#include <cstddef>
#include <cstdint>

namespace vortex {

struct cp_ild_header_t {
  uint32_t ctl;
  uint16_t skip_n;
  uint16_t reserved0;
  uint64_t record_addr;   // 4-byte aligned; the selected words span at most two lines
  uint64_t pred_addr;     // 4-byte aligned
  uint32_t pred_ref;
  uint32_t qmd_count;
  uint32_t x_bias;
  uint32_t x_div;         // 0 or 1 bypasses the divide
  uint32_t reserved1[6];
};
// Byte offsets of the header fields; the hardware decodes the same positions.
constexpr uint32_t CP_ILD_OFF_CTL      = 0;
constexpr uint32_t CP_ILD_OFF_SKIP     = 4;
constexpr uint32_t CP_ILD_OFF_RECORD   = 8;
constexpr uint32_t CP_ILD_OFF_PRED     = 16;
constexpr uint32_t CP_ILD_OFF_PRED_REF = 24;
constexpr uint32_t CP_ILD_OFF_COUNT    = 28;
constexpr uint32_t CP_ILD_OFF_X_BIAS   = 32;
constexpr uint32_t CP_ILD_OFF_X_DIV    = 36;

static_assert(sizeof(cp_ild_header_t) == 64, "ILD header is one cache line");
static_assert(offsetof(cp_ild_header_t, ctl)         == CP_ILD_OFF_CTL, "ILD layout");
static_assert(offsetof(cp_ild_header_t, skip_n)      == CP_ILD_OFF_SKIP, "ILD layout");
static_assert(offsetof(cp_ild_header_t, record_addr) == CP_ILD_OFF_RECORD, "ILD layout");
static_assert(offsetof(cp_ild_header_t, pred_addr)   == CP_ILD_OFF_PRED, "ILD layout");
static_assert(offsetof(cp_ild_header_t, pred_ref)    == CP_ILD_OFF_PRED_REF, "ILD layout");
static_assert(offsetof(cp_ild_header_t, qmd_count)   == CP_ILD_OFF_COUNT, "ILD layout");
static_assert(offsetof(cp_ild_header_t, x_bias)      == CP_ILD_OFF_X_BIAS, "ILD layout");
static_assert(offsetof(cp_ild_header_t, x_div)       == CP_ILD_OFF_X_DIV, "ILD layout");

constexpr uint32_t CP_LAUNCH_DESC_ALIGN  = 64;
constexpr uint32_t CP_Q_ERROR_DESC_ALIGN = 1u << 0;

constexpr uint32_t CP_QMD_PAIRS_OFFSET = 8;
constexpr uint32_t CP_ILD_HEADER_BYTES = 64;
constexpr uint32_t CP_LAUNCH_MAX_PAIRS = 64;

constexpr uint32_t CP_ILD_SEL_ONE      = 0xF;     // grid source: constant 1
constexpr uint32_t CP_ILD_SEL_X_SH     = 0;
constexpr uint32_t CP_ILD_SEL_Y_SH     = 4;
constexpr uint32_t CP_ILD_SEL_Z_SH     = 8;
constexpr uint32_t CP_ILD_SHIFT_X_SH   = 12;
constexpr uint32_t CP_ILD_SHIFT_Y_SH   = 16;
constexpr uint32_t CP_ILD_SHIFT_Z_SH   = 20;
constexpr uint32_t CP_ILD_HAS_RECORD   = 1u << 24;
constexpr uint32_t CP_ILD_HAS_PRED     = 1u << 25;
constexpr uint32_t CP_ILD_PRED_INV     = 1u << 26;
constexpr uint32_t CP_ILD_NO_LAUNCH    = 1u << 27;

inline uint32_t cp_ild_sel(uint32_t ctl, int dim) {
  return (ctl >> (CP_ILD_SEL_X_SH + 4 * dim)) & 0xF;
}

inline uint32_t cp_ild_shift(uint32_t ctl, int dim) {
  return (ctl >> (CP_ILD_SHIFT_X_SH + 4 * dim)) & 0xF;
}

} // namespace vortex

#endif // VORTEX_CP_LAUNCH_ABI_H
