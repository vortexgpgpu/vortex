// cgo27_motivation: run the SAME GEMM (D = C + A*B) on the same input through every
// execution path in one build, and report per-path cycles / MPM counters. This is the
// motivation harness for the CGO'27 paper: it measures how the optimal compute/memory
// unit for one GEMM shifts with shape and with the hardware available.
//
//   mode 0     in-core SIMT              scalar MAC loop
//   mode 1     in-core TCU               WMMA
//   mode 2     in-core TCU + DXA         WMMA fed by DXA-staged smem, single buffer
//   mode 3     in-core TCU               2-stage smem pipeline, LSU-staged
//   mode 4     in-core TCU               3-stage smem pipeline, LSU-staged
//   mode 5     in-core TCU + DXA         2-stage smem pipeline, DXA-staged
//   mode 6     in-core TCU + DXA         3-stage smem pipeline, DXA-staged
//   mode 7     DTCU_socket               descriptor engine, D -> that socket's L1
//   mode 8     DTCU_cluster              descriptor engine, D -> L2
//   mode 9,10,11  hetero                 cores + engine(s) on one GEMM (not built)
//
// The map is grouped by what executes rather than packed, so a path can be added next
// to its relatives without renumbering the rest. 3/5 and 4/6 differ ONLY in who stages
// the operands, which is what makes the DXA engine's contribution measurable.
//
// Snippet provenance:
//   - run harness / MPM queries / CPU ref / compare : dtcu_compare/main.cpp
//   - mode 0 scalar loop                            : sgemm/kernel.cpp
//   - mode 1 WMMA                                   : dtcu_compare/kernel.cpp (mode 0)
//   - mode 2 DXA host programming (program_2d)      : sgemm_tcu_wg_dxa/main.cpp
//   - mode 3/4 LSU-staged pipelines                 : kernel_modes/kernel_m3.cpp
//   - mode 7/8 DTCU descriptor                      : dtcu_compare (mode 1)

#include "common.h"
#include "epilogue.h"   // app id -> epilogue, shared with the kernel

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <dtcu_cfg.h>   // DTCU descriptor + native-tile geometry (modes 7/8)
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>
#include <dxa.h>   // host-side DXA descriptor programming (mode 2)

#define FLOAT_ULP 6
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 != _ret) {                                         \
      std::cerr << "Runtime Error: " << #_expr               \
                << " returned " << _ret << std::endl;        \
      return _ret;                                           \
    }                                                        \
  } while (false)

using namespace vortex;
namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<NUM_THREADS>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

// GEMM shape, set by -M / -N / -K. There used to be a -s multiplier that expanded to
// `mult * the DTCU native tile`, but with two DTCU engines whose tiles differ (cluster
// 64x32, socket 32x16) there is no single tile left for a multiplier to multiply, and
// the harness's premise is that every mode runs the SAME GEMM. The ladder now lives in
// the sweep scripts, which expand each rung to explicit -M/-N/-K. These defaults
// reproduce what -s 2 used to produce.
static const uint32_t kDefaultM = 128, kDefaultN = 64, kDefaultK = 32;
static uint32_t g_M = kDefaultM, g_N = kDefaultN, g_K = kDefaultK;

// Strict positive-integer parse. atoi() stops at the first non-digit and reports no
// error, so `-K 1.7` silently became `-K 1` and then failed with a confusing complaint
// about a size the caller never asked for. Reject the input instead of guessing at it.
// allow_zero because 0 is a legal MODE (SIMT, the baseline) but not a legal matrix
// dimension. Rejecting it everywhere made `-m 0` unselectable: the SIMT baseline could
// only be reached through `-m all`, and the error it printed named the wrong problem.
static uint32_t parse_u32(const char* s, const char* flag, bool allow_zero = false) {
  errno = 0;
  char* end = nullptr;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || errno != 0 || (v == 0 && !allow_zero)
      || v > 0xFFFFFFFFul) {
    std::cerr << "cgo27_motivation: invalid " << flag << " '" << s << "' (expected a "
              << (allow_zero ? "non-negative" : "positive") << " decimal integer)"
              << std::endl;
    exit(-1);
  }
  return (uint32_t)v;
}

#include "host_modes.h"
#include "run_modes.h"

#include "host_types.h"
#include "host_run.h"
#include "host_args.h"

int main(int argc, char** argv) {
  parse_args(argc, argv);

  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM;
  const uint32_t tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * tcu_i_ratio;

  // No DTCU tile appears here any more. tile-M and tile-K are build-fixed inside each
  // engine, ragged edges are clamped in hardware, and the one axis software picks
  // (tile-N) is per engine and now chosen in run_case(). The two things that used a
  // DTCU tile at this level -- the -s expansion and the NO_TMA tripwire -- are gone.
  const uint32_t M = g_M, N = g_N, K = g_K; // parse_args() applied the defaults

  if (!check_shape(M, N, K, tcu_tileM, tcu_tileN, tcu_tileK))
    return -1;

  std::vector<itype_t> hA(M * K), hB(K * N);
  std::vector<otype_t> hC(M * N);
  std::vector<float>   hRef(M * N);

  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t k = 0; k < K; ++k) {
      float v = float((i * 13 + k * 7) % 11) - 5.0f;
      hA[i * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(v);
    }
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t j = 0; j < N; ++j) {
      float v = float((k * 5 + j * 17) % 9) - 4.0f;
      hB[j * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(v); // col-major
    }
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float v = float((i * 9 + j * 11) % 13) - 6.0f;
      hC[i * N + j] = (otype_t)Convert<vt::OTYPE>::from_float(v);
    }

  // CPU reference: D = C + A*B
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float acc = Convert<vt::OTYPE>::to_float(hC[i * N + j]);
      for (uint32_t k = 0; k < K; ++k)
        acc += Convert<vt::ITYPE>::to_float(hA[i * K + k]) * Convert<vt::ITYPE>::to_float(hB[j * K + k]);
      // Same epilogue the kernel applies (shared epilogue.h) so the comparison is
      // against identical arithmetic, not a second implementation.
      hRef[i * N + j] = epi_apply(g_app, acc);
    }

  const char* names[NUM_MODES] = {
      "SIMT (no tensor unit)",
      "TCU (in-core WMMA)",
      "TCU + DXA-staged operands",
      "TCU workgroup + DXA (warp-specialised)",
      "TCU workgroup, cooperative SW load",
      "TCU workgroup + DXA, A resident in LMEM",
      "<reserved>",
      "DTCU_socket (engine per socket)",
      "DTCU_cluster (engine per cluster)",
      "hetero: TCU + DTCU_socket",
      "hetero: TCU + DTCU_cluster",
      "hetero: TCU + both engines",
  };

  // App 6 is a row-wise softmax, which is not a float->float map and so cannot go through
  // epi_apply(). The device runs it as a separate pass over D for every mode; the
  // reference makes the same pass here, using the same helpers out of epilogue/softmax.h
  // so the two agree bit-for-bit.
  if (epi_needs_row_pass(g_app)) {
    // MIRROR THE KERNEL'S REDUCTION ORDER, not just its arithmetic. The max is
    // order-independent, but the sum of exponentials is not: the kernel has NUM_THREADS
    // lanes each stride-summing its own slice and then a log2 tree over the lanes, while a
    // plain left-to-right loop here rounds differently. At N = 64 the gap stayed inside
    // the harness's ULP <= 6; at N = 384 it did not, and modes 1, 7 and 8 all reported
    // exactly 3,770 mismatches -- the same count from three different modes, which is what
    // pointed at the shared reference rather than at any one of them.
    const uint32_t NT = NUM_THREADS;
    std::vector<float> red(NT);
    for (uint32_t i = 0; i < M; ++i) {
      float* r = &hRef[i * N];

      for (uint32_t t = 0; t < NT; ++t) {
        float m = -3.4028235e38f;
        for (uint32_t j = t; j < N; j += NT) m = epi_softmax_max(m, r[j]);
        red[t] = m;
      }
      for (uint32_t s = NT >> 1; s > 0; s >>= 1)
        for (uint32_t t = 0; t < s; ++t) red[t] = epi_softmax_max(red[t], red[t + s]);
      const float row_max = red[0];

      for (uint32_t t = 0; t < NT; ++t) {
        float acc = 0.0f;
        for (uint32_t j = t; j < N; j += NT) acc = epi_softmax_addexp(acc, r[j], row_max);
        red[t] = acc;
      }
      for (uint32_t s = NT >> 1; s > 0; s >>= 1)
        for (uint32_t t = 0; t < s; ++t) red[t] += red[t + s];
      const float row_sum = red[0];

      for (uint32_t j = 0; j < N; ++j) r[j] = epi_softmax_norm(r[j], row_max, row_sum);
    }
  }
  std::vector<otype_t> out[NUM_MODES];
  Stats stats[NUM_MODES];
  int mode_errors[NUM_MODES] = {0};

  for (uint32_t m = 0; m < NUM_MODES; ++m) {
    if (!run_this(m)) continue;
    if (mode_state(m) != ModeState::Implemented) {
      // Reserved slots stay invisible under -m all; planned ones report themselves so a
      // sweep records "asked for, not built" rather than silently omitting a column.
      if (mode_state(m) == ModeState::Planned)
        stats[m].skipped = true;
      continue;
    }
    out[m].assign(M * N, 0);
    std::cout << "cgo27_motivation: ---------- Running mode " << m << " (" << names[m] << ") ----------" << std::endl;
    RT_CHECK(run_case(m, M, N, K, tcu_tileM, tcu_tileN, tcu_tileK, hA, hB, hC, out[m], stats[m]));
  }

  // ---------- verify each mode against the CPU reference ----------
  std::cout << "cgo27_motivation: ---------- RESULT ----------" << std::endl;
  std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
  for (uint32_t m = 0; m < NUM_MODES; ++m) {
    // out[m] is only sized for modes that actually ran; indexing a reserved or planned
    // slot here would read past an empty vector.
    if (!run_this(m) || stats[m].skipped) continue;
    if (mode_state(m) != ModeState::Implemented) continue;
    for (uint32_t idx = 0; idx < M * N; ++idx) {
      float got = Convert<vt::OTYPE>::to_float(out[m][idx]);
      if (ulp_diff(got, hRef[idx]) > FLOAT_ULP) {
        if (mode_errors[m] < 3)
          std::cerr << "  " << names[m] << " mismatch D[" << idx << "]: got=" << got
                    << " exp=" << hRef[idx] << "\n";
        ++mode_errors[m];
      }
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  for (uint32_t m = 0; m < NUM_MODES; ++m) {
    if (!run_this(m)) continue;
    if (mode_state(m) == ModeState::Reserved) continue; // a hole, not a result
    if (stats[m].skipped) {
      // Still emit a [MOTI] line so the sweep scripts see the mode and can tell a
      // missing engine apart from a mode that simply was not requested.
      std::cout << "[MOTI] app=" << g_app
                << " M=" << M << " N=" << N << " K=" << K
                << " mode=" << m << " name=" << kShortNames[m]
                << " cycles=0 errors=0 skipped=1" << std::endl;
      // Two different reasons reach this line and conflating them sent an earlier reader
      // looking for a missing -DVX_CFG_EXT_* that was never the problem.
      std::cout << "[" << names[m] << "] skipped ("
                << (mode_state(m) == ModeState::Planned
                      ? "mode is numbered but not implemented"
                      : "engine not present in this build")
                << ")" << std::endl;
      continue;
    }
    const Stats& s = stats[m];
    // Machine-parseable line for the sweep scripts (sweep_exp1.py / sweep_exp2.py).
    // `size=` is gone with -s: the shape is always explicit now, so M/N/K say it all.
    // `name=` lets the scripts cross-check their own mode table against the binary
    // rather than silently mislabelling a column when the modes get renumbered.
    std::cout << "[MOTI] app=" << g_app
              << " M=" << M << " N=" << N << " K=" << K
              << " mode=" << m << " name=" << kShortNames[m]
              << " cycles=" << s.cycles
              << " errors=" << mode_errors[m] << std::endl;
    std::cout << "[" << names[m] << "]"
              << " cycles=" << s.cycles << " instrs=" << s.instrs
              << " IPC=" << (s.cycles ? double(s.instrs) / double(s.cycles) : 0.0)
              << " errors=" << mode_errors[m] << std::endl;
    const auto avg = [](uint64_t total, uint64_t n) { return n ? double(total) / double(n) : 0.0; };
    std::cout << "    instr: alu=" << s.instr_alu << " fpu=" << s.instr_fpu
              << " lsu=" << s.instr_lsu << " sfu=" << s.instr_sfu << " tcu=" << s.instr_tcu
              << " branches=" << s.branches << " divergence=" << s.divergence << std::endl;
    std::cout << "    stall: alu=" << s.stall_alu << " fpu=" << s.stall_fpu
              << " lsu=" << s.stall_lsu << " sfu=" << s.stall_sfu << " tcu=" << s.stall_tcu << std::endl;
    // load_lt/ifetch_lt are sums; the per-access averages are what the paper compares
    // against the DTCU's tma_mem_wait.
    std::cout << "    lsu:   loads=" << s.loads << " stores=" << s.stores
              << " load_lt=" << s.load_lt << " (avg " << avg(s.load_lt, s.loads) << " cyc)"
              << " ifetches=" << s.ifetches
              << " ifetch_lt=" << s.ifetch_lt << " (avg " << avg(s.ifetch_lt, s.ifetches) << " cyc)"
              << std::endl;
    std::cout << "    mem:  l2_reads=" << s.l2_reads << " l2_writes=" << s.l2_writes
              << " mem_reads=" << s.mem_reads << " mem_writes=" << s.mem_writes << std::endl;
    if (uses_engine(m)) {
      // instr_tcu is counted per FEDP, the same primitive the in-core TCU's
      // VX_CSR_MPM_INSTR_TCU counts, so mode 1/2/5/6's tcu= is directly comparable.
      // With engines>1 the cycle fields are engine-cycles summed over the engines;
      // busy_max is the busiest single engine and is what compares to cycles= above.
      std::cout << "    dtcu: engines=" << s.d_engines
                << " active=" << s.d_engines_active
                << " instr_tcu=" << s.d_instr_tcu << " compute=" << s.d_compute << " next_k_load_stall=" << s.d_next_k_load_stall
                << " next_tile_load_stall=" << s.d_next_tile_load_stall
                << " prev_tile_store_stall=" << s.d_prev_tile_store_stall
                << " store_drain=" << s.d_store_drain << " desc_wait=" << s.d_desc_wait
                << " busy=" << s.d_busy << " busy_max=" << s.d_busy_max << std::endl;
      std::cout << "    dtcu: tma_mem_wait=" << s.d_tma_mem_wait << " tma_buf_starve=" << s.d_tma_buf_starve
                << " tma_op_fill=" << s.d_tma_op_fill << " tma_acc_init=" << s.d_tma_acc_init
                << " tma_addrgen=" << s.d_tma_addrgen << " tma_store_issue_stall=" << s.d_tma_store_issue_stall
                << " smem_read_model=" << s.d_smem_read_model
                << " op_reqs=" << s.d_op_reqs << " out_reqs=" << s.d_out_reqs << std::endl;
    }
  }

  int total_errors = 0;
  for (uint32_t m = 0; m < NUM_MODES; ++m) total_errors += mode_errors[m];

  // The mode-3-vs-4 tripwire is gone with the modes it guarded. It asserted that
  // blocking (NO_TMA) could never beat overlapped, which was a meaningful invariant
  // when 3 and 4 were the same engine differing only in that flag. Now they are two
  // different engines with different tiles and different output targets, and neither
  // ordering is required -- which is exactly the question the comparison exists to
  // answer, so asserting an answer here would be begging it. DTENSOR_FLAG_NO_TMA stays
  // in the ISA; only the harness mode is retired.

  if (total_errors) { std::cout << "FAILED! total_errors=" << total_errors << std::endl; return total_errors; }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
