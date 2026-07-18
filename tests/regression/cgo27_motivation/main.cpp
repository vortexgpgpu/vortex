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

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
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

// Refer to sw/kernel/include/vx_dtensor.h::dtensor_desc_t
struct dtensor_desc_t {
  uint64_t ptrA, ptrB, ptrC, ptrD;
  uint32_t ldmA, ldmB, ldmC, ldmD;
  uint16_t M, N, K;
  uint8_t  fmt_s, fmt_d, flags, shape_n_size;
  uint16_t shape_policy;
  uint32_t reserved2;
};

// Descriptor flag bits (keep in sync with sw/kernel/include/vx_dtensor.h).
static constexpr uint8_t DTENSOR_FLAG_NO_TMA = 0x2; // blocking mode: no TMA overlap

struct Stats {
  uint64_t cycles = 0, instrs = 0;
  uint64_t loads = 0, stores = 0, stall_lsu = 0, stall_tcu = 0, instr_lsu = 0, instr_tcu = 0;
  uint64_t l2_reads = 0, l2_writes = 0, mem_reads = 0, mem_writes = 0;
  double   host_ms = 0.0;
  // DTCU engine counters, MPM class 9 (modes 3/4 only). wait_tma is the overlap
  // headline: ~0 with TMA on, the serialized k>=1 fetch time with TMA off.
  uint64_t d_op_reqs = 0, d_out_reqs = 0, d_compute = 0, d_wait_tma = 0, d_mem_wait = 0,
           d_wait_buf = 0, d_buf_write = 0, d_addrgen = 0, d_store_wait = 0,
           d_store_drain = 0, d_opread = 0, d_load_stall = 0, d_store_stall = 0;
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

  // mode 2 requires the DXA ISA extension.
  if (mode == 2) {
    uint64_t isa_flags = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
      std::cerr << "  (skipped: DXA ISA extension disabled)" << std::endl;
      vx_queue_release(queue); vx_device_release(device);
      return 0;
    }
  }

  kernel_arg_t karg{};
  karg.mode = mode; karg.M = M; karg.N = N; karg.K = K;

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
    desc.shape_n_size = shape_n_size; desc.shape_policy = 0; desc.reserved2 = 0;
    RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ, &desc_buf));
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
  if (mode == 2) {
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
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_launch_info_t li = {};
  li.struct_size = sizeof(li);
  li.kernel = kernel; li.args_host = &karg; li.args_size = sizeof(karg);
  if (mode == 0) {
    // SIMT: one thread per output element; warp = NUM_THREADS cols of one row.
    li.ndim = 2;
    li.grid_dim[0]  = N / NUM_THREADS; li.grid_dim[1]  = M;
    li.block_dim[0] = NUM_THREADS;     li.block_dim[1] = 1;
  } else if (mode == 1 || mode == 2) {
    // WMMA: one block (one warp) per output tile.
    li.ndim = 2;
    li.grid_dim[0]  = N / tcu_tileN; li.grid_dim[1]  = M / tcu_tileM;
    li.block_dim[0] = NUM_THREADS;   li.block_dim[1] = 1;
    if (mode == 2)
      li.lmem_size = (tcu_tileM * tcu_tileK + tcu_tileN * tcu_tileK) * sizeof(itype_t);
  } else {
    // DTCU: a single thread fires the whole-GEMM descriptor.
    li.ndim = 1;
    li.grid_dim[0] = 1; li.block_dim[0] = 1;
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, out.data(), D_buf, 0, out.size() * sizeof(otype_t), 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  auto t1 = std::chrono::high_resolution_clock::now();
  stats.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MCYCLE, 0, &stats.cycles));
  RT_CHECK(vx_mpm_query(device, 0, VX_CSR_MINSTRET, 0, &stats.instrs));
  {
    const uint32_t cls = VX_DCR_MPM_CLASS_CORE;
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_LOADS,     0, &stats.loads));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STORES,    0, &stats.stores));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_LSU, 0, &stats.stall_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_STALL_TCU, 0, &stats.stall_tcu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_LSU, 0, &stats.instr_lsu));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_INSTR_TCU, 0, &stats.instr_tcu));
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
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_WAIT_TMA,    0, &stats.d_wait_tma));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_MEM_WAIT,    0, &stats.d_mem_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_WAIT_BUF,    0, &stats.d_wait_buf));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_BUF_WRITE,   0, &stats.d_buf_write));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_ADDRGEN,     0, &stats.d_addrgen));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_STORE_WAIT,  0, &stats.d_store_wait));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_STORE_DRAIN, 0, &stats.d_store_drain));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_OPREAD,      0, &stats.d_opread));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_LOAD_STALL,  0, &stats.d_load_stall));
    RT_CHECK(vx_mpm_query(device, cls, VX_CSR_MPM_DTCU_STORE_STALL, 0, &stats.d_store_stall));
  }

  vx_event_release(read_ev); vx_event_release(launch_ev);
  vx_buffer_release(A_buf); vx_buffer_release(B_buf); vx_buffer_release(C_buf); vx_buffer_release(D_buf);
  if (desc_buf) vx_buffer_release(desc_buf);
  vx_kernel_release(kernel); vx_module_release(module_);
  vx_queue_release(queue); vx_device_release(device);
  return 0;
}

static uint32_t g_size_mult = SIZE_MULT;

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "s:h")) != -1) {
    switch (c) {
    case 's': g_size_mult = (uint32_t)atoi(optarg); break;
    case 'h':
      std::cout << "Usage: [-s size_mult] (GEMM = size_mult * DTCU native tile)\n";
      exit(0);
    default: exit(-1);
    }
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  const uint32_t tcu_i_ratio = 4 / sizeof(itype_t);
  const uint32_t tcu_tileM = cfg::tileM;
  const uint32_t tcu_tileN = cfg::tileN;
  const uint32_t tcu_tileK = cfg::tileK * tcu_i_ratio;

  const uint32_t dtcu_tileM = 64;
  const uint32_t dtcu_tileN = 32;
  const uint32_t dtcu_tileK = 8 * (4 / sizeof(itype_t));
  const uint8_t  shape_n_size = dtcu_tileN / 16;

  const uint32_t M = g_size_mult * dtcu_tileM;
  const uint32_t N = g_size_mult * dtcu_tileN;
  const uint32_t K = g_size_mult * dtcu_tileK;

  if ((M % tcu_tileM) || (N % tcu_tileN) || (K % tcu_tileK) || (N % NUM_THREADS)) {
    std::cerr << "cgo27_motivation: size unsupported M=" << M << " N=" << N << " K=" << K << std::endl;
    return -1;
  }

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
      hRef[i * N + j] = acc;
    }

  const char* names[5] = { "in-core SIMT", "in-core TCU", "in-core TCU + DXA", "DTCU (no TMA)", "DTCU + DTCU_TMA" };
  std::vector<otype_t> out[5];
  Stats stats[5];
  int mode_errors[5] = {0,0,0,0,0};

  for (uint32_t m = 0; m < 5; ++m) {
    out[m].assign(M * N, 0);
    std::cout << "cgo27_motivation: ---------- Running mode " << m << " (" << names[m] << ") ----------" << std::endl;
    RT_CHECK(run_case(m, M, N, K, tcu_tileM, tcu_tileN, tcu_tileK, shape_n_size, hA, hB, hC, out[m], stats[m]));
  }

  // ---------- verify each mode against the CPU reference ----------
  std::cout << "cgo27_motivation: ---------- RESULT ----------" << std::endl;
  std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;
  for (uint32_t m = 0; m < 5; ++m) {
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
  for (uint32_t m = 0; m < 5; ++m) {
    const Stats& s = stats[m];
    std::cout << "[" << names[m] << "]"
              << " cycles=" << s.cycles << " instrs=" << s.instrs
              << " IPC=" << (s.cycles ? double(s.instrs) / double(s.cycles) : 0.0)
              << " errors=" << mode_errors[m] << std::endl;
    std::cout << "    core: instr_lsu=" << s.instr_lsu << " instr_tcu=" << s.instr_tcu
              << " stall_lsu=" << s.stall_lsu << " stall_tcu=" << s.stall_tcu << std::endl;
    std::cout << "    mem:  l2_reads=" << s.l2_reads << " l2_writes=" << s.l2_writes
              << " mem_reads=" << s.mem_reads << " mem_writes=" << s.mem_writes << std::endl;
    if (m >= 3) {
      std::cout << "    dtcu: op_reqs=" << s.d_op_reqs << " out_reqs=" << s.d_out_reqs
                << " compute=" << s.d_compute << " wait_tma=" << s.d_wait_tma
                << " mem_wait=" << s.d_mem_wait << " wait_buf=" << s.d_wait_buf << std::endl;
      std::cout << "    dtcu: buf_write=" << s.d_buf_write << " addrgen=" << s.d_addrgen
                << " store_wait=" << s.d_store_wait << " store_drain=" << s.d_store_drain
                << " opread=" << s.d_opread << " next_tile_load_stall=" << s.d_load_stall
                << " curr_tile_store_stall=" << s.d_store_stall << std::endl;
    }
  }

  int total_errors = 0;
  for (uint32_t m = 0; m < 5; ++m) total_errors += mode_errors[m];

  // Tripwire: blocking (mode 3) must never beat overlapped (mode 4). Equal cycles
  // are legitimate only when nothing can overlap (single output tile AND single K
  // tile); otherwise equality means DTENSOR_FLAG_NO_TMA was ignored (stale sim).
  const bool overlappable = (M / dtcu_tileM) * (N / dtcu_tileN) > 1 || (K / dtcu_tileK) > 1;
  if (stats[3].cycles < stats[4].cycles ||
      (overlappable && stats[3].cycles == stats[4].cycles)) {
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
