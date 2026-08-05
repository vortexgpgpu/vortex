// cgo27_motivation: run the SAME GEMM (D = C + A*B) on the same input through
// FIVE execution paths in one build, and report per-path cycles / MPM counters.
// This is the motivation harness for the CGO'27 paper: it lets us measure how the
// optimal compute/memory unit for one GEMM shifts across hardware paths.
//
//   mode 0 : in-core SIMT        (scalar MAC loop)
//   mode 1 : in-core TCU         (WMMA)
//   mode 2 : in-core TCU + DXA   (WMMA fed by DXA-staged smem)
//   mode 3 : DTCU (no TMA)       (descriptor engine, blocking loads/stores)
//   mode 4 : DTCU + DTCU_TMA     (descriptor engine + TMA overlap; see note below)
//
// Snippet provenance:
//   - run harness / MPM queries / CPU ref / compare : dtcu_compare/main.cpp
//   - mode 0 scalar loop                            : sgemm/kernel.cpp
//   - mode 1 WMMA                                   : dtcu_compare/kernel.cpp (mode 0)
//   - mode 2 DXA host programming (program_2d)      : sgemm_tcu_wg_dxa/main.cpp
//   - mode 3/4 DTCU descriptor                      : dtcu_compare (mode 1)
//
// NOTE on modes 3 vs 4: both fire the same DTCU descriptor path; the difference
// is the descriptor's DTENSOR_FLAG_NO_TMA bit. Mode 3 sets it -> blocking
// baseline (each K tile's operands are fetched at consume time and each tile's D
// store drains before the next tile starts). Mode 4 leaves it clear -> the
// default overlapped engine (double-buffered TMA prefetch + background store).
// Traffic (op/out cache-line requests) is identical; only overlap timing differs.
// If mode 3 and mode 4 report IDENTICAL cycles, the simulator predates the flag
// (a stale build silently ignores it) -- the harness fails loudly on that below.

#include "common.h"
#include "epilogue.h"   // app id -> epilogue, shared with the kernel

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <dtcu_cfg.h>   // DTCU descriptor + native-tile geometry (modes 3/4)
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>
#include <dxa.h>   // host-side DXA descriptor programming (mode 2)

#define FLOAT_ULP 6
#define MAX_ERRORS 100

#ifndef SIZE_MULT
#define SIZE_MULT 2
#endif

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

static uint32_t g_size_mult = SIZE_MULT;

// Explicit GEMM shape (-M / -N / -K). 0 means "not given -> derive from -s", so the
// original multiple-of-native-tile behavior (and sweep_exp1.py / sweep_exp2.py, which
// drive -s) keeps working unchanged. Each dimension falls back independently, so
// `-M 1024 -N 512` takes K from -s.
static uint32_t g_M = 0, g_N = 0, g_K = 0;

// Strict positive-integer parse. atoi() stops at the first non-digit and reports no
// error, so `-s 1.7` silently became `-s 1` and then failed with a confusing "size
// unsupported" complaint about a size the caller never asked for. Reject the input
// instead of guessing at it.
static uint32_t parse_u32(const char* s, const char* flag) {
  errno = 0;
  char* end = nullptr;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || errno != 0 || v == 0 || v > 0xFFFFFFFFul) {
    std::cerr << "cgo27_motivation: invalid " << flag << " '" << s
              << "' (expected a positive decimal integer)" << std::endl;
    exit(-1);
  }
  return (uint32_t)v;
}

// Which HW path(s) to run. MODE_ALL runs all NUM_MODES (0..6); otherwise a
// single mode selected on the command line with -m.
static const uint32_t NUM_MODES = 7;
static const uint32_t MODE_ALL  = 0xFFFFFFFFu;
static uint32_t g_mode = MODE_ALL;
static inline bool run_this(uint32_t m) { return g_mode == MODE_ALL || g_mode == m; }



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
  // the same depth of accounting the DTCU class already gives modes 3/4.
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
  // DTCU engine counters, MPM class 9 (modes 3/4 only). Labels match the CSRs;
  // dtcu_* FSM family sums to busy, tma_* engine family overlaps compute.
  uint64_t d_op_reqs = 0, d_out_reqs = 0, d_compute = 0, d_next_k_load_stall = 0, d_tma_mem_wait = 0,
           d_tma_buf_starve = 0, d_tma_op_fill = 0, d_tma_addrgen = 0, d_tma_store_issue_stall = 0,
           d_store_drain = 0, d_smem_read_model = 0, d_next_tile_load_stall = 0, d_prev_tile_store_stall = 0,
           d_desc_wait = 0, d_busy = 0, d_tma_acc_init = 0, d_instr_tcu = 0;
};

static inline int ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return 0;
  if (std::isinf(a) || std::isinf(b)) return (a == b) ? 0 : 0x7fffffff;
  int ia, ib;
  std::memcpy(&ia, &a, sizeof(int));
  std::memcpy(&ib, &b, sizeof(int));
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  return std::abs(ia - ib);
}

// Selected app (prologue/epilogue), 1..8 — see epilogue.h. Declared here because
// run_case() stamps it into the kernel arg and the CPU reference applies the same
// map; -a sets it in parse_args().
static uint32_t g_app = 1;

// Run one GEMM path on a fresh device. Fills out[] with D and records stats.
static int run_case(uint32_t mode,
                    uint32_t M, uint32_t N, uint32_t K,
                    uint32_t tcu_tileM, uint32_t tcu_tileN, uint32_t tcu_tileK,
                    uint8_t shape_n_size,
                    const std::vector<itype_t>& hA,
                    const std::vector<itype_t>& hB,
                    const std::vector<otype_t>& hC,
                    std::vector<otype_t>& out,
                    Stats& stats) {
  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Each path needs its engine present. Modes 2/5/6 stage operands through DXA;
  // modes 3/4 hand the whole GEMM to the DTCU. Marking the case skipped (rather
  // than just returning) keeps its all-zero output out of the verify pass, which
  // would otherwise report M*N mismatches and fail a run that never executed.
  {
    const uint64_t need =
        (mode == 2 || mode == 5 || mode == 6) ? VX_ISA_EXT_DXA          :
        (mode == 3 || mode == 4)              ? VX_ISA_EXT_DTCU_CLUSTER : 0;
    if (need != 0) {
      uint64_t isa_flags = 0;
      RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
      if ((isa_flags & need) == 0) {
        std::cerr << "  (skipped: " << ((need == VX_ISA_EXT_DXA) ? "DXA" : "DTCU")
                  << " ISA extension disabled)" << std::endl;
        stats.skipped = true;
        vx_queue_release(queue); vx_device_release(device);
        return 0;
      }
    }
  }

  kernel_arg_t karg{};
  karg.mode = mode; karg.app = g_app; karg.M = M; karg.N = N; karg.K = K;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, D_buf = nullptr, desc_buf = nullptr;
  RT_CHECK(vx_buffer_create(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &karg.B_addr));
  RT_CHECK(vx_buffer_create(device, hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf));
  RT_CHECK(vx_buffer_address(C_buf, &karg.C_addr));
  RT_CHECK(vx_buffer_create(device, out.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &karg.D_addr));

  // DTCU descriptor (modes 3, 4).
  dtensor_desc_t desc{};
  if (mode == 3 || mode == 4) {
    desc.ptrA = karg.A_addr; desc.ptrB = karg.B_addr; desc.ptrC = karg.C_addr; desc.ptrD = karg.D_addr;
    desc.ldmA = K; desc.ldmB = K; desc.ldmC = N; desc.ldmD = N;
    desc.M = M; desc.N = N; desc.K = K;
    desc.fmt_s = vt::ITYPE::id; desc.fmt_d = vt::OTYPE::id;
    // D = C + A*B in both modes; mode 3 additionally disables the TMA overlap.
    desc.flags = (mode == 3) ? DTENSOR_FLAG_NO_TMA : 0x0;
    desc.shape_n_size = shape_n_size; desc.shape_policy = 0;
    desc.done = 0; // engine sets this once D is visible; dtensor_check() reads it
    RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ_WRITE, &desc_buf));
    RT_CHECK(vx_buffer_address(desc_buf, &karg.desc_addr));
  }

  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, C_buf, 0, hC.data(), hC.size() * sizeof(otype_t), 0, nullptr, nullptr));
  if (desc_buf)
    RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, &desc, sizeof(dtensor_desc_t), 0, nullptr, nullptr));

  // mode 2: program DXA descriptors (source layout -> smem tile).
  //   A: row-major [M x K], tile [tcu_tileM x tcu_tileK], row stride K.
  //   B: col-major [K x N] stored as [N x K] row-major, tile [tcu_tileN x tcu_tileK], row stride K.
  if (mode == 2 || mode == 5 || mode == 6) {
    RT_CHECK(vortex::dxa::program_2d(device, DESC_A, karg.A_addr,
      /*size0=*/K, /*size1=*/M, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/tcu_tileK, /*tile1=*/tcu_tileM, /*elem_bytes=*/sizeof(itype_t)));
    RT_CHECK(vortex::dxa::program_2d(device, DESC_B, karg.B_addr,
      /*size0=*/K, /*size1=*/N, /*stride0_bytes=*/K * sizeof(itype_t),
      /*tile0=*/tcu_tileK, /*tile1=*/tcu_tileN, /*elem_bytes=*/sizeof(itype_t)));
  }

  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  // Each HW path is a separate kernel entry (see kernel.cpp); select by name so
  // a WMMA launch never loads a binary containing dtensor_start.
  const char* kentry =
      (mode == 0) ? "moti_simt"         :
      (mode == 1) ? "moti_tcu"          :
      (mode == 2) ? "moti_tcu_dxa"      :
      (mode == 5) ? "moti_tcu_dxa_pipe3" :
      (mode == 6) ? "moti_tcu_dxa_pipe" :
                    "moti_dtcu";  // modes 3 and 4 (flag differs in the descriptor)
  RT_CHECK(vx_module_get_kernel(module_, kentry, &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel; li.args_host = &karg; li.args_size = sizeof(karg);
  if (mode == 0) {
    // SIMT: one thread per output element; warp = NUM_THREADS cols of one row.
    li.ndim = 2;
    li.grid_dim[0]  = N / NUM_THREADS; li.grid_dim[1]  = M;
    li.block_dim[0] = NUM_THREADS;     li.block_dim[1] = 1;
  } else if (mode == 1 || mode == 2 || mode == 5 || mode == 6) {
    // WMMA modes: one block (one warp) per output tile.
    li.ndim = 2;
    li.grid_dim[0]  = N / tcu_tileN; li.grid_dim[1]  = M / tcu_tileM;
    li.block_dim[0] = NUM_THREADS;   li.block_dim[1] = 1;
    const uint32_t stage_bytes = (tcu_tileM * tcu_tileK + tcu_tileN * tcu_tileK) * sizeof(itype_t);
    if (mode == 2)      li.lmem_size = stage_bytes;      // single-buffer DXA
    else if (mode == 6) li.lmem_size = 2 * stage_bytes;  // 2-stage smem pipeline
    else if (mode == 5) li.lmem_size = 3 * stage_bytes;  // 3-stage smem pipeline
  } else {
    // DTCU: a single thread fires the whole-GEMM descriptor.
    li.ndim = 1;
    li.grid_dim[0] = 1; li.block_dim[0] = 1;
  }

  // DTCU epilogue pass (modes 3/4). The engine is GEMM-only, so an elementwise
  // epilogue cannot be fused into it the way the in-core modes fuse theirs; it runs
  // as a SECOND launch over the whole matrix. That extra M*N round-trip is the cost
  // asymmetry the app sweep measures, and it is deliberately inside the timed
  // region so the reported cycles include it.
  const bool dtcu_needs_epi = (mode == 3 || mode == 4) && epi_is_elementwise(g_app);
  vx_kernel_h epi_kernel = nullptr;
  vx_launch_info_t epi_li = {};
  if (dtcu_needs_epi) {
    RT_CHECK(vx_module_get_kernel(module_, "moti_epilogue", &epi_kernel));
    epi_li.struct_size = sizeof(epi_li);
    epi_li.kernel = epi_kernel; epi_li.args_host = &karg; epi_li.args_size = sizeof(karg);
    epi_li.ndim = 2;                                   // same geometry as moti_simt
    epi_li.grid_dim[0]  = N / NUM_THREADS; epi_li.grid_dim[1]  = M;
    epi_li.block_dim[0] = NUM_THREADS;     epi_li.block_dim[1] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr, read_ev = nullptr, epi_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  if (dtcu_needs_epi) {
    RT_CHECK(vx_enqueue_launch(queue, &epi_li, 1, &launch_ev, &epi_ev));
    RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &epi_ev, &read_ev));
  } else {
    RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &launch_ev, &read_ev));
  }
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MCYCLE, 0, &stats.cycles));
  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MINSTRET, 0, &stats.instrs));
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_CORE;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_ALU,  0, &stats.instr_alu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_FPU,  0, &stats.instr_fpu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_LSU,  0, &stats.instr_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_SFU,  0, &stats.instr_sfu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_TCU,  0, &stats.instr_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_ALU,  0, &stats.stall_alu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_FPU,  0, &stats.stall_fpu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_LSU,  0, &stats.stall_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_SFU,  0, &stats.stall_sfu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_TCU,  0, &stats.stall_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_BRANCHES,   0, &stats.branches));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DIVERGENCE, 0, &stats.divergence));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_IFETCHES,   0, &stats.ifetches));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_IFETCH_LT,  0, &stats.ifetch_lt));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOADS,      0, &stats.loads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOAD_LT,    0, &stats.load_lt));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STORES,     0, &stats.stores));
  }
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_MEM;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_READS,  0, &stats.l2_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_L2CACHE_WRITES, 0, &stats.l2_writes));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_READS,      0, &stats.mem_reads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_MEM_WRITES,     0, &stats.mem_writes));
  }
  // DTCU engine counters (cluster-level, own MPM class) -- dtcu_compare pattern.
  if (mode == 3 || mode == 4) {
    const uint32_t cls = VX_DCR_MPM_CLASS_DTCU;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_OP_REQS,     0, &stats.d_op_reqs));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_OUT_REQS,    0, &stats.d_out_reqs));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_COMPUTE,     0, &stats.d_compute));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_NEXT_K_LOAD_STALL,    0, &stats.d_next_k_load_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_MEM_WAIT,    0, &stats.d_tma_mem_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_BUF_STARVE,    0, &stats.d_tma_buf_starve));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_OP_FILL,   0, &stats.d_tma_op_fill));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_ADDRGEN,     0, &stats.d_tma_addrgen));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_STORE_ISSUE_STALL,  0, &stats.d_tma_store_issue_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_STORE_DRAIN, 0, &stats.d_store_drain));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_SMEM_READ_MODEL,      0, &stats.d_smem_read_model));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_NEXT_TILE_LOAD_STALL,  0, &stats.d_next_tile_load_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_PREV_TILE_STORE_STALL, 0, &stats.d_prev_tile_store_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_DESC_WAIT,   0, &stats.d_desc_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_BUSY,        0, &stats.d_busy));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_TMA_ACC_INIT, 0, &stats.d_tma_acc_init));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_INSTR_TCU,   0, &stats.d_instr_tcu));
  }

  vx_event_release(read_ev); vx_event_release(launch_ev);
  if (epi_ev) vx_event_release(epi_ev);
  vx_buffer_release(A_buf); vx_buffer_release(B_buf); vx_buffer_release(C_buf); vx_buffer_release(D_buf);
  if (desc_buf) vx_buffer_release(desc_buf);
  vx_kernel_release(kernel); vx_module_release(module_);
  vx_queue_release(queue); vx_device_release(device);
  return 0;
}



static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "s:a:m:M:N:K:h")) != -1) {
    switch (c) {
    case 's': g_size_mult = parse_u32(optarg, "-s"); break;
    case 'a': g_app = parse_u32(optarg, "-a"); break;
    case 'M': g_M = parse_u32(optarg, "-M"); break;
    case 'N': g_N = parse_u32(optarg, "-N"); break;
    case 'K': g_K = parse_u32(optarg, "-K"); break;
    case 'm':
      if (0 == strcmp(optarg, "all")) {
        g_mode = MODE_ALL;
      } else {
        uint32_t m = parse_u32(optarg, "-m");
        if (m >= NUM_MODES) {
          std::cerr << "cgo27_motivation: invalid -m '" << optarg
                    << "' (expected 0.." << (NUM_MODES - 1) << " or 'all')\n";
          exit(-1);
        }
        g_mode = m;
      }
      break;
    case 'h':
      std::cout << "Usage: [-M m] [-N n] [-K k] [-s size_mult] [-a app_id] [-m mode]\n"
                   "  -M m   GEMM M (rows of A/C/D)      -- absolute size, overrides -s\n"
                   "  -N n   GEMM N (cols of B/C/D)      -- absolute size, overrides -s\n"
                   "  -K k   GEMM K (reduction depth)    -- absolute size, overrides -s\n"
                   "         e.g. -M 1024 -N 512 -K 64. Dimensions not given fall back to -s.\n"
                   "         The DTCU (modes 3/4) takes any shape -- it clamps ragged edges in\n"
                   "         hardware. The in-core modes do not: for those, each dimension must\n"
                   "         be a multiple of their tile (the harness prints the requirement).\n"
                   "  -s N   GEMM size = N * DTCU native tile (default " << SIZE_MULT << ")\n"
                   "  -a N   app id 1..8 (epilogue; default 1)\n"
                   "  -m X   which HW path to run: 'all' (default) or one mode:\n"
                   "           0=in-core SIMT    1=in-core TCU      2=in-core TCU+DXA\n"
                   "           3=DTCU (no TMA)   4=DTCU+DTCU_TMA\n"
                   "           5=TCU+DXA pipelined (3-stage)   6=TCU+DXA pipelined (2-stage)\n";
      exit(0);
    default: exit(-1);
    }
  }
}

// ---------------------------------------------------------------------------
// Shape validation.
//
// Only the DTCU handles a ragged edge. The in-core modes do not, and each breaks
// differently when M/N/K are not exact multiples of its tile:
//   * modes 3/4 -- FINE for any shape. The engine rounds its tile counts up and the
//     TMA clamps the trailing tile: operands past the matrix are never fetched (the
//     scratchpad is zero-filled instead) and the D store leaves those bytes disabled.
//     Only the descriptor's uint16_t M/N/K field width binds.
//   * modes 1/2/5/6 -- `for (i = 0; i < K; i += tileK)` overruns K on the last step,
//     and the grid `(N / tileN, M / tileM)` truncates, leaving output tiles never
//     written.
//   * mode 0 -- the grid `(N / NUM_THREADS, M)` truncates the same way.
// The truncating cases produce a VERIFICATION MISMATCH rather than a diagnostic, so
// they need an up-front check -- but only against the modes that will actually run:
// `-m 4 -M 100 -N 48 -K 20` is legal, while the same shape on `-m 1` is not.
// ---------------------------------------------------------------------------
static uint32_t gcd_u32(uint32_t a, uint32_t b) { while (b) { uint32_t t = a % b; a = b; b = t; } return a; }
static uint32_t lcm_u32(uint32_t a, uint32_t b) { return (a / gcd_u32(a, b)) * b; }

static bool check_shape(uint32_t M, uint32_t N, uint32_t K,
                        uint32_t tcu_tileM, uint32_t tcu_tileN, uint32_t tcu_tileK,
                        uint32_t dtcu_tileM, uint32_t dtcu_tileN, uint32_t dtcu_tileK) {
  bool ok = true;
  uint32_t need_M = 1, need_N = 1, need_K = 1;   // running LCM of every active constraint

  auto need = [&](uint32_t v, uint32_t mult, const char* dim, uint32_t* acc, const char* who) {
    *acc = lcm_u32(*acc, mult);
    if (v % mult) {
      std::cerr << "cgo27_motivation: " << dim << "=" << v << " is not a multiple of "
                << mult << " (" << who << "); nearest legal " << dim << ": ";
      if (v > mult) std::cerr << (v / mult) * mult << " or ";   // 0 is not a size
      std::cerr << (v / mult + 1) * mult << std::endl;
      ok = false;
    }
  };

  // mode 0, and the DTCU epilogue pass, which reuses the mode-0 launch geometry.
  const bool simt_geom = run_this(0)
      || ((run_this(3) || run_this(4)) && epi_is_elementwise(g_app));
  if (simt_geom)
    need(N, NUM_THREADS, "N", &need_N, "SIMT grid width NUM_THREADS -- mode 0 / DTCU epilogue pass");

  // modes 1, 2, 5, 6: one warp per output tile, K stepped by the WMMA tile.
  if (run_this(1) || run_this(2) || run_this(5) || run_this(6)) {
    need(M, tcu_tileM, "M", &need_M, "in-core TCU tileM");
    need(N, tcu_tileN, "N", &need_N, "in-core TCU tileN");
    need(K, tcu_tileK, "K", &need_K, "in-core TCU tileK");
  }

  // modes 3, 4: the DTCU rounds its tile counts UP and handles the ragged trailing
  // tile in hardware -- the operand fetch clamps past the matrix and zero-fills, and
  // the D store masks the bytes outside D (sim/simx/dtcu/dtcu_tma.cpp). So M/N/K need
  // no relation to the native tile; only the descriptor's field width binds.
  (void)dtcu_tileM; (void)dtcu_tileN; (void)dtcu_tileK;
  if (run_this(3) || run_this(4)) {
    // dtensor_desc_t holds M/N/K as uint16_t (dtcu_cfg.h), so a larger GEMM would
    // wrap silently and the engine would compute a different shape than we verify.
    if (M > 0xFFFFu || N > 0xFFFFu || K > 0xFFFFu) {
      std::cerr << "cgo27_motivation: M/N/K must each be <= 65535 for modes 3/4"
                   " (dtensor_desc_t stores them as uint16_t)" << std::endl;
      ok = false;
    }
  }

  if (!ok)
    std::cerr << "cgo27_motivation: for the selected mode(s), M must be a multiple of "
              << need_M << ", N of " << need_N << ", K of " << need_K << std::endl;
  return ok;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM;
  const uint32_t tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * tcu_i_ratio;

  // DTCU geometry from the shared traits. tileM/tileK are hardware-fixed, so they are
  // read, not restated; tileN is genuinely ours to pick out of the engine's legal
  // range, and shape_n_size is how the descriptor asks for it.
  using dcfg = vt::dtcu_config_t<vt::ITYPE>;
  constexpr uint32_t dtcu_tileN = 32;
  static_assert(dcfg::tileN_valid(dtcu_tileN), "dtcu_tileN is not a legal DTCU native tile-N");
  const uint32_t dtcu_tileM = dcfg::tileM;
  const uint32_t dtcu_tileK = dcfg::tileK;
  const uint8_t  shape_n_size = dcfg::shape_n_size_for(dtcu_tileN);

  // Absolute -M/-N/-K win per dimension; anything omitted falls back to the
  // size_mult x native_tile form that -s (and the sweep scripts) use.
  const bool explicit_shape = (g_M != 0) || (g_N != 0) || (g_K != 0);
  const uint32_t M = g_M ? g_M : g_size_mult * dtcu_tileM;
  const uint32_t N = g_N ? g_N : g_size_mult * dtcu_tileN;
  const uint32_t K = g_K ? g_K : g_size_mult * dtcu_tileK;

  if (!check_shape(M, N, K, tcu_tileM, tcu_tileN, tcu_tileK,
                   dtcu_tileM, dtcu_tileN, dtcu_tileK))
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

  const char* names[7] = { "in-core SIMT", "in-core TCU", "in-core TCU + DXA", "DTCU (no TMA)", "DTCU + DTCU_TMA", "TCU+DXA pipelined (3-stage)", "TCU+DXA pipelined (2-stage)" };
  std::vector<otype_t> out[7];
  Stats stats[7];
  int mode_errors[7] = {0};

  for (uint32_t m = 0; m < NUM_MODES; ++m) {
    if (!run_this(m)) continue;
    out[m].assign(M * N, 0);
    std::cout << "cgo27_motivation: ---------- Running mode " << m << " (" << names[m] << ") ----------" << std::endl;
    RT_CHECK(run_case(m, M, N, K, tcu_tileM, tcu_tileN, tcu_tileK, shape_n_size, hA, hB, hC, out[m], stats[m]));
  }

  // ---------- verify each mode against the CPU reference ----------
  std::cout << "cgo27_motivation: ---------- RESULT ----------" << std::endl;
  std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
  for (uint32_t m = 0; m < NUM_MODES; ++m) {
    if (!run_this(m) || stats[m].skipped) continue;
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
    if (stats[m].skipped) {
      // Still emit a [MOTI] line so the sweep scripts see the mode and can tell a
      // missing engine apart from a mode that simply was not requested.
      std::cout << "[MOTI] app=" << g_app << " size=" << (explicit_shape ? 0u : g_size_mult)
                << " M=" << M << " N=" << N << " K=" << K
                << " mode=" << m << " cycles=0 errors=0 skipped=1" << std::endl;
      std::cout << "[" << names[m] << "] skipped (engine not present in this build)" << std::endl;
      continue;
    }
    const Stats& s = stats[m];
    // Machine-parseable line for the sweep scripts (sweep_exp1.py / sweep_exp2.py).
    // `size=` stays an integer so their `size=(\d+)` regex keeps matching; it reports
    // 0 when the shape came from -M/-N/-K, where no single multiplier describes it.
    std::cout << "[MOTI] app=" << g_app << " size=" << (explicit_shape ? 0u : g_size_mult)
              << " M=" << M << " N=" << N << " K=" << K
              << " mode=" << m << " cycles=" << s.cycles
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
    if (m >= 3) {
      // instr_tcu is counted per FEDP, the same primitive the in-core TCU's
      // VX_CSR_MPM_INSTR_TCU counts, so mode 1/2/5/6's tcu= is directly comparable.
      std::cout << "    dtcu: instr_tcu=" << s.d_instr_tcu << " compute=" << s.d_compute << " next_k_load_stall=" << s.d_next_k_load_stall
                << " next_tile_load_stall=" << s.d_next_tile_load_stall
                << " prev_tile_store_stall=" << s.d_prev_tile_store_stall
                << " store_drain=" << s.d_store_drain << " desc_wait=" << s.d_desc_wait
                << " busy=" << s.d_busy << std::endl;
      std::cout << "    dtcu: tma_mem_wait=" << s.d_tma_mem_wait << " tma_buf_starve=" << s.d_tma_buf_starve
                << " tma_op_fill=" << s.d_tma_op_fill << " tma_acc_init=" << s.d_tma_acc_init
                << " tma_addrgen=" << s.d_tma_addrgen << " tma_store_issue_stall=" << s.d_tma_store_issue_stall
                << " smem_read_model=" << s.d_smem_read_model
                << " op_reqs=" << s.d_op_reqs << " out_reqs=" << s.d_out_reqs << std::endl;
    }
  }

  int total_errors = 0;
  for (uint32_t m = 0; m < NUM_MODES; ++m) total_errors += mode_errors[m];

  // Tripwire: blocking (mode 3) must never beat overlapped (mode 4). Only
  // meaningful when both DTCU modes actually ran (i.e. -m all). Equal cycles
  // are legitimate only when nothing can overlap (single output tile AND single K
  // tile); otherwise equality means DTENSOR_FLAG_NO_TMA was ignored (stale sim).
  const bool overlappable = (M / dtcu_tileM) * (N / dtcu_tileN) > 1 || (K / dtcu_tileK) > 1;
  if (run_this(3) && run_this(4) && !stats[3].skipped && !stats[4].skipped &&
      (stats[3].cycles < stats[4].cycles ||
       (overlappable && stats[3].cycles == stats[4].cycles))) {
    std::cerr << "WARNING: mode 3 (blocking) cycles=" << stats[3].cycles
              << " vs mode 4 (TMA) cycles=" << stats[4].cycles
              << " -- expected mode 3 slower; DTENSOR_FLAG_NO_TMA ignored (stale simulator?)"
              << std::endl;
    ++total_errors;
  }

  if (total_errors) { std::cout << "FAILED! total_errors=" << total_errors << std::endl; return total_errors; }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
