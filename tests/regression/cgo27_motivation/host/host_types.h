#ifndef _CGO27_HOST_TYPES_H_
#define _CGO27_HOST_TYPES_H_

// Host-side types shared by the driver and anything that reads its output: the
// element-format conversions, the per-run counter record, and the ULP comparison the
// verify pass uses.
//
// Everything here is inline or a type: this header is included by more than one
// translation unit and must not define storage.

#include "common.h"
#include <cmath>
#include <cstdint>
#include <tensor_cfg.h>

namespace vt = vortex::tensor;
using cfg     = vt::wmma_config_t<NUM_THREADS>;
// Workgroup MMA geometry (modes 12/13). DIFFERENT tile shape from `cfg` -- xtileM/xtileN
// are derived from NRC, not from the per-warp WMMA tile -- so the host must size the
// grid, the CTA and the DXA descriptors with THESE and not with cfg's. Getting that
// wrong is silent: the kernel tiles one way, the host another, and D comes out wrong
// with no error.
#ifndef WGMMA_NRC
#define WGMMA_NRC 8
#endif
using wgcfg   = vt::wgmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;
// MOTI_WG_KSTEPS and MOTI_WG_NCOLS come from common.h -- one definition for both sides.
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

// ---- type conversion (host side) ----
template <typename T> struct Convert;
template <> struct Convert<vt::fp32> {
  using dtype = float;
  static inline dtype from_float(float f) { return f; }
  static inline float to_float(dtype x) { return x; }
};
template <> struct Convert<vt::fp16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_htof_s(x, 0, nullptr)); }
};
template <> struct Convert<vt::bf16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) { return rv_ftob_s(bit_cast<uint32_t>(f), 0, nullptr); }
  static inline float to_float(dtype x) { return bit_cast<float>(rv_btof_s(x, 0, nullptr)); }
};

struct Stats {
  uint64_t cycles = 0, instrs = 0;
  // Core counters, MPM class 1. The whole class is collected so the in-core paths get
  // the same depth of accounting the DTCU class already gives modes 7/8.
  //
  // NOTE on the WMMA fragment loads: there is no counter dedicated to them. A dense
  // ctx::load_matrix_sync() compiles to ordinary per-lane loads (vx_tensor.h), so its
  // traffic lands in loads / instr_lsu / load_lt / stall_lsu like any other load, and
  // instr_tcu counts only the mma ops. In the WMMA kernels here almost every load IS
  // fragment traffic, which makes instr_lsu a usable proxy — but it is a proxy.
  uint64_t instr_alu = 0, instr_fpu = 0, instr_lsu = 0, instr_sfu = 0, instr_tcu = 0;
  uint64_t stall_alu = 0, stall_fpu = 0, stall_lsu = 0, stall_sfu = 0, stall_tcu = 0;
  uint64_t branches = 0, divergence = 0;
  // *_lt are latency SUMS, not per-access values: divide by the matching request
  // count for the average (load_lt / loads, ifetch_lt / ifetches).
  uint64_t ifetches = 0, ifetch_lt = 0, loads = 0, load_lt = 0, stores = 0;
  uint64_t l2_reads = 0, l2_writes = 0, mem_reads = 0, mem_writes = 0;
  double   host_ms = 0.0;
  // Set when run_case() bailed because the path's engine is not in this build. The
  // case produced no output, so it must be excluded from verification and reporting
  // instead of being compared against the CPU reference as a zero matrix.
  bool     skipped = false;
  // DTCU engine counters (modes 7/8/14/15), MPM class 9 for the cluster engine and 10
  // for the socket engines. Labels match the CSRs; the dtcu_* FSM family sums to busy,
  // the tma_* engine family overlaps compute.
  //
  // d_engines is how many engines EXIST at this scope and were summed over;
  // d_engines_active is how many actually did any work. They differ, and the difference
  // is the point: a short row split can leave an engine with no nonempty descriptor.
  // Per-unit throughput has to divide by the active count; provisioning efficiency has
  // to divide by the existing count.
  // With more than
  // one, the CYCLE counters are engine-cycles and are NOT comparable to MCYCLE --
  // d_busy_max (the busiest single engine) is the one that is. Counts (op_reqs,
  // out_reqs, instr_tcu) sum correctly either way.
  uint32_t d_engines = 1;
  uint32_t d_engines_active = 0;
  uint64_t d_busy_max = 0;
  uint64_t d_op_reqs = 0, d_out_reqs = 0, d_compute = 0, d_next_k_load_stall = 0, d_tma_mem_wait = 0,
           d_tma_buf_starve = 0, d_tma_op_fill = 0, d_tma_addrgen = 0, d_tma_store_issue_stall = 0,
           d_store_drain = 0, d_smem_read_model = 0, d_next_tile_load_stall = 0, d_prev_tile_store_stall = 0,
           d_desc_wait = 0, d_busy = 0, d_tma_acc_init = 0, d_instr_tcu = 0;
};

inline int ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isinf(a) || std::isinf(b)) return (a == b) ? 0 : 0x7fffffff;
  int ia, ib;
  std::memcpy(&ia, &a, sizeof(int));
  std::memcpy(&ib, &b, sizeof(int));
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  return std::abs(ia - ib);
}

// Selected app (1..6 or 9) — see epilogue.h. Declared here because
// run_case() stamps it into the kernel arg and the CPU reference applies the same
// map; -a sets it in parse_args().
static uint32_t g_app = MOTI_APP;   // the build decides; -a only checks

// Run one GEMM path on a fresh device. Fills out[] with D and records stats.

#endif // _CGO27_HOST_TYPES_H_
