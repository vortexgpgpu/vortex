// Comparison test for in-core TCU and DTCU
// Compares both in-core TCU / DTCU results / CPU reference
// Covers a larger GEMM with multiple output tiles and K-dimension accumulation 
// using fp16/bf16 input, fp32 output, and row-major C accumulator

#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

#include <VX_types.h>
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>

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

// ---- type conversion (sgemm_tcu 스타일) ----
template <typename T>
struct Convert;

template <>
struct Convert<vt::fp32> {
  using dtype = float;
  static inline dtype from_float(float f) { return f; }
  static inline float to_float(dtype x) { return x; }
};

template <>
struct Convert<vt::fp16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) {
    return rv_ftoh_s(bit_cast<uint32_t>(f), 0, nullptr);
  }
  static inline float to_float(dtype x) {
    uint32_t bits = rv_htof_s(x, 0, nullptr);
    return bit_cast<float>(bits);
  }
};

template <>
struct Convert<vt::bf16> {
  using dtype = uint16_t;
  static inline dtype from_float(float f) {
    return rv_ftob_s(bit_cast<uint32_t>(f), 0, nullptr);
  }
  static inline float to_float(dtype x) {
    uint32_t bits = rv_btof_s(x, 0, nullptr);
    return bit_cast<float>(bits);
  }
};

// Refer to kernel/include/vx_tensor.h::dtensor_desc_t
struct dtensor_desc_t {
  uint64_t ptrA;
  uint64_t ptrB;
  uint64_t ptrC;
  uint64_t ptrD;
  uint32_t ldmA;
  uint32_t ldmB;
  uint32_t ldmC;
  uint32_t ldmD;
  uint16_t M;
  uint16_t N;
  uint16_t K;
  uint8_t  fmt_s;
  uint8_t  fmt_d;
  uint8_t  flags;
  uint8_t  reserved0;
  uint16_t reserved1;
  uint32_t reserved2;
};

// Simple stats for comparison
struct Stats {
  uint64_t cycles = 0;
  uint64_t instrs = 0;
  uint64_t loads = 0;
  uint64_t stores = 0;
  uint64_t stall_lsu = 0;
  uint64_t stall_tcu = 0;
  uint64_t instr_lsu = 0;
  uint64_t instr_tcu = 0;
  uint64_t l2_reads = 0;
  uint64_t l2_writes = 0;
  uint64_t mem_reads = 0;
  uint64_t mem_writes = 0;
  double host_ms = 0.0;
};

static inline int ulp_diff(float a, float b) {
  if (std::isnan(a) && std::isnan(b))
    return 0;
  if (std::isinf(a) || std::isinf(b))
    return (a == b) ? 0 : 0x7fffffff;

  int ia, ib;
  std::memcpy(&ia, &a, sizeof(int));
  std::memcpy(&ib, &b, sizeof(int));
  if (ia < 0) ia = 0x80000000 - ia;
  if (ib < 0) ib = 0x80000000 - ib;
  return std::abs(ia - ib);
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  const uint32_t i_ratio = 4 / sizeof(itype_t);
  const uint32_t tileM = cfg::tileM;
  const uint32_t tileN = cfg::tileN;
  const uint32_t tileK = cfg::tileK * i_ratio;

  const uint32_t M = 2 * tileM;
  const uint32_t N = 2 * tileN;
  const uint32_t K = 2 * tileK;

  std::vector<itype_t> hA(M * K);
  std::vector<itype_t> hB(K * N);
  std::vector<otype_t> hC(M * N);
  std::vector<float>   hRef(M * N);

  // ---- Generate source data for A and B ----
  // Developed from generate_with_scale() in sgemm_tcu

  // A is row-major (M x K -> M rows, K cols)
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t k = 0; k < K; ++k) {
      float v = float((i * 13 + k * 7) % 11) - 5.0f;
      hA[i * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(v);
    }
  }

  // B is column-major (K x N -> K rows, N cols)
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t j = 0; j < N; ++j) {
      float v = float((k * 5 + j * 17) % 9) - 4.0f;
      hB[j * K + k] = (itype_t)Convert<vt::ITYPE>::from_float(v);
    }
  }

  // C is row-major 
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float v = float((i * 9 + j * 11) % 13) - 6.0f;
      hC[i * N + j] = (otype_t)Convert<vt::OTYPE>::from_float(v);
    }
  }

  // REFERENCE value to compare to
  // Created by CPU (D = A * B) which is adopted from sgemm_tcu's matmul_cpu()
  // matmult_cpu() is more complicated due to sub-byte formats and scaling factors -> NEED MORE WORK!!
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float acc = Convert<vt::OTYPE>::to_float(hC[i * N + j]);
      for (uint32_t k = 0; k < K; ++k) {
        float a = Convert<vt::ITYPE>::to_float(hA[i * K + k]);
        float b = Convert<vt::ITYPE>::to_float(hB[j * K + k]);
        acc += a * b;
      }
      hRef[i * N + j] = acc;
    }
  }

  // Statistics holders for each run
  std::vector<otype_t> out_tcu(M * N);
  std::vector<otype_t> out_dtcu(M * N);
  Stats stats_tcu{};
  Stats stats_dtcu{};

  // ---------------------------- Run In-core TCU ----------------------------
  {
    vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, D_buf = nullptr, args_buffer = nullptr;

    std::cout << "dtcu_compare: ---------------------- Running In-core TCU ----------------------" << std::endl;

    // ---- open device connection ----
    std::cout << "dtcu_compare: tcu - open device connection" << std::endl;
    vx_device_h device = nullptr;
    RT_CHECK(vx_dev_open(&device));

    // ---- upload program ----
    const char* kernel_file = "kernel.vxbin";
    vx_buffer_h krnl_buffer = nullptr;
    std::cout << "dtcu_compare: tcu - upload program" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    // ---- alloc device buffers ----
    kernel_arg_t karg{};
    karg.mode = 0;
    karg.grid_dim[0] = N / tileN;
    karg.grid_dim[1] = M / tileM;
    karg.block_dim[0] = NUM_THREADS;
    karg.block_dim[1] = 1;
    karg.M = M;
    karg.N = N;
    karg.K = K;

    // ---- alloc device memory (A, B, C, D) ----
    RT_CHECK(vx_mem_alloc(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
    RT_CHECK(vx_mem_address(A_buf, &karg.A_addr));

    RT_CHECK(vx_mem_alloc(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
    RT_CHECK(vx_mem_address(B_buf, &karg.B_addr));

    RT_CHECK(vx_mem_alloc(device, hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf));
    RT_CHECK(vx_mem_address(C_buf, &karg.C_addr));

    RT_CHECK(vx_mem_alloc(device, out_tcu.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
    RT_CHECK(vx_mem_address(D_buf, &karg.D_addr));

    RT_CHECK(vx_copy_to_dev(A_buf, hA.data(), 0, hA.size() * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(B_buf, hB.data(), 0, hB.size() * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(C_buf, hC.data(), 0, hC.size() * sizeof(otype_t)));

    std::vector<otype_t> zeros(M * N, 0);
    RT_CHECK(vx_copy_to_dev(D_buf, zeros.data(), 0, zeros.size() * sizeof(otype_t)));

    RT_CHECK(vx_upload_bytes(device, &karg, sizeof(kernel_arg_t), &args_buffer));
    RT_CHECK(vx_dcr_write(device, VX_DCR_BASE_MPM_CLASS, VX_DCR_MPM_CLASS_CORE));

    // ---- start device & start timer----
    auto t0 = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // ---- wait for completion ----
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto t1 = std::chrono::high_resolution_clock::now();

    //end timer & record
    stats_tcu.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

     // ---- download destination buffer ----
    std::cout << "dtcu_compare: tcu - download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(out_tcu.data(), D_buf, 0, out_tcu.size() * sizeof(otype_t)));

    // ---- query performance counters ----
    RT_CHECK(vx_mpm_query(device, VX_CSR_MCYCLE, 0, &stats_tcu.cycles));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MINSTRET, 0, &stats_tcu.instrs));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_LOADS, 0, &stats_tcu.loads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STORES, 0, &stats_tcu.stores));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STALL_LSU, 0, &stats_tcu.stall_lsu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STALL_TCU, 0, &stats_tcu.stall_tcu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_INSTR_LSU, 0, &stats_tcu.instr_lsu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_INSTR_TCU, 0, &stats_tcu.instr_tcu));

    RT_CHECK(vx_dcr_write(device, VX_DCR_BASE_MPM_CLASS, VX_DCR_MPM_CLASS_MEM));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_L2CACHE_READS, 0, &stats_tcu.l2_reads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_L2CACHE_WRITES, 0, &stats_tcu.l2_writes));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_MEM_READS, 0, &stats_tcu.mem_reads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_MEM_WRITES, 0, &stats_tcu.mem_writes));

    // ---- cleanup ----
    vx_mem_free(A_buf);
    vx_mem_free(B_buf);
    vx_mem_free(C_buf);
    vx_mem_free(D_buf);
    vx_mem_free(args_buffer);
    vx_mem_free(krnl_buffer);
    vx_dev_close(device);
  }

  // ---------------------------- Run DTCU ----------------------------
  {
    vx_buffer_h A_buf = nullptr, B_buf = nullptr, C_buf = nullptr, D_buf = nullptr, desc_buf = nullptr, args_buffer = nullptr;

    std::cout << "dtcu_compare: ---------------------- Running DTCU ----------------------" << std::endl;

    // ---- open device connection ----
    std::cout << "dtcu_compare: dtcu - open device connection" << std::endl;
    vx_device_h device = nullptr;
    RT_CHECK(vx_dev_open(&device));

    // ---- upload program ----
    const char* kernel_file = "kernel.vxbin";
    vx_buffer_h krnl_buffer = nullptr;
    std::cout << "dtcu_compare: dtcu- upload program" << std::endl;
    RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

    kernel_arg_t karg{};
    karg.mode = 1;
    karg.grid_dim[0] = 1;
    karg.grid_dim[1] = 1;
    karg.block_dim[0] = NUM_THREADS;
    karg.block_dim[1] = 1;
    karg.M = M;
    karg.N = N;
    karg.K = K;

    // ---- alloc device memory (A, B, C, D) ----
    RT_CHECK(vx_mem_alloc(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
    RT_CHECK(vx_mem_address(A_buf, &karg.A_addr));

    RT_CHECK(vx_mem_alloc(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
    RT_CHECK(vx_mem_address(B_buf, &karg.B_addr));

    RT_CHECK(vx_mem_alloc(device, hC.size() * sizeof(otype_t), VX_MEM_READ, &C_buf));
    RT_CHECK(vx_mem_address(C_buf, &karg.C_addr));

    RT_CHECK(vx_mem_alloc(device, out_dtcu.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
    RT_CHECK(vx_mem_address(D_buf, &karg.D_addr));

    dtensor_desc_t desc{};
    desc.ptrA = karg.A_addr;
    desc.ptrB = karg.B_addr;
    desc.ptrC = karg.C_addr;
    desc.ptrD = karg.D_addr;
    desc.ldmA = K;
    desc.ldmB = K;
    desc.ldmC = N;
    desc.ldmD = N;
    desc.M = M;
    desc.N = N;
    desc.K = K;
    desc.fmt_s = vt::ITYPE::id;
    desc.fmt_d = vt::OTYPE::id;
    desc.flags = 0x0;
    desc.reserved0 = 0;
    desc.reserved1 = 0;
    desc.reserved2 = 0;

    RT_CHECK(vx_mem_alloc(device, sizeof(dtensor_desc_t), VX_MEM_READ, &desc_buf));
    RT_CHECK(vx_mem_address(desc_buf, &karg.desc_addr));

    RT_CHECK(vx_copy_to_dev(A_buf, hA.data(), 0, hA.size() * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(B_buf, hB.data(), 0, hB.size() * sizeof(itype_t)));
    RT_CHECK(vx_copy_to_dev(C_buf, hC.data(), 0, hC.size() * sizeof(otype_t)));
    RT_CHECK(vx_copy_to_dev(desc_buf, &desc, 0, sizeof(dtensor_desc_t)));

    std::vector<otype_t> zeros(M * N, 0);
    RT_CHECK(vx_copy_to_dev(D_buf, zeros.data(), 0, zeros.size() * sizeof(otype_t)));

    RT_CHECK(vx_upload_bytes(device, &karg, sizeof(kernel_arg_t), &args_buffer));
    RT_CHECK(vx_dcr_write(device, VX_DCR_BASE_MPM_CLASS, VX_DCR_MPM_CLASS_CORE));

    // ---- start device & start timer----
    auto t0 = std::chrono::high_resolution_clock::now();
    RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

    // ---- wait for completion ----
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
    auto t1 = std::chrono::high_resolution_clock::now();

    //end timer & record
    stats_dtcu.host_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---- download destination buffer ----
    std::cout << "dtcu_compare: dtcu - download destination buffer" << std::endl;
    RT_CHECK(vx_copy_from_dev(out_dtcu.data(), D_buf, 0, out_dtcu.size() * sizeof(otype_t)));

    // ---- query performance counters ----
    RT_CHECK(vx_mpm_query(device, VX_CSR_MCYCLE, 0, &stats_dtcu.cycles));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MINSTRET, 0, &stats_dtcu.instrs));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_LOADS, 0, &stats_dtcu.loads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STORES, 0, &stats_dtcu.stores));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STALL_LSU, 0, &stats_dtcu.stall_lsu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_STALL_TCU, 0, &stats_dtcu.stall_tcu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_INSTR_LSU, 0, &stats_dtcu.instr_lsu));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_INSTR_TCU, 0, &stats_dtcu.instr_tcu));

    RT_CHECK(vx_dcr_write(device, VX_DCR_BASE_MPM_CLASS, VX_DCR_MPM_CLASS_MEM));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_L2CACHE_READS, 0, &stats_dtcu.l2_reads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_L2CACHE_WRITES, 0, &stats_dtcu.l2_writes));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_MEM_READS, 0, &stats_dtcu.mem_reads));
    RT_CHECK(vx_mpm_query(device, VX_CSR_MPM_MEM_WRITES, 0, &stats_dtcu.mem_writes));

    // ---- cleanup ----
    vx_mem_free(A_buf);
    vx_mem_free(B_buf);
    vx_mem_free(C_buf);
    vx_mem_free(D_buf);
    vx_mem_free(desc_buf);
    vx_mem_free(args_buffer);
    vx_mem_free(krnl_buffer);
    vx_dev_close(device);
  }

  // ---------------------------- Compare Results ----------------------------
  
  std::cout << "dtcu_compare: ---------------------- RESULT ----------------------" << std::endl;

  int errors_tcu = 0;
  int errors_dtcu = 0;
  int cross_errors = 0;

  // Check outputs against CPU reference and against each other
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float ref = hRef[i * N + j];
      float got_tcu = Convert<vt::OTYPE>::to_float(out_tcu[i * N + j]);
      float got_dtcu = Convert<vt::OTYPE>::to_float(out_dtcu[i * N + j]);

      if (ulp_diff(got_tcu, ref) > FLOAT_ULP) {
        if (errors_tcu < MAX_ERRORS) {
          std::cerr << "TCU mismatch D[" << i << "][" << j << "]: got=" << got_tcu << " exp=" << ref << "\n";
        }
        ++errors_tcu;
      }

      if (ulp_diff(got_dtcu, ref) > FLOAT_ULP) {
        if (errors_dtcu < MAX_ERRORS) {
          std::cerr << "DTCU mismatch D[" << i << "][" << j << "]: got=" << got_dtcu << " exp=" << ref << "\n";
        }
        ++errors_dtcu;
      }

      if (ulp_diff(got_tcu, got_dtcu) > FLOAT_ULP) {
        if (cross_errors < MAX_ERRORS) {
          std::cerr << "Cross mismatch D[" << i << "][" << j << "]: tcu=" << got_tcu << " dtcu=" << got_dtcu << "\n";
        }
        ++cross_errors;
      }
    }
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "M=" << M << " N=" << N << " K=" << K << std::endl;

  // In-core TCU stats
  std::cout << "\n[In-core TCU]" << std::endl;
  std::cout << "◯ host_ms=" << stats_tcu.host_ms
            << " cycles=" << stats_tcu.cycles
            << " instrs=" << stats_tcu.instrs
            << " IPC=" << (double(stats_tcu.instrs) / double(stats_tcu.cycles))
            << std::endl;
  
  std::cout << "◯ loads=" << stats_tcu.loads
            << " stores=" << stats_tcu.stores
            << " stall_lsu=" << stats_tcu.stall_lsu
            << " stall_tcu=" << stats_tcu.stall_tcu
            << " instr_lsu=" << stats_tcu.instr_lsu
            << " instr_tcu=" << stats_tcu.instr_tcu
            << std::endl;

  std::cout << "  l2_reads=" << stats_tcu.l2_reads
            << " l2_writes=" << stats_tcu.l2_writes
            << " mem_reads=" << stats_tcu.mem_reads
            << " mem_writes=" << stats_tcu.mem_writes
            << std::endl;

  // DTCU stats
  std::cout << "\n[DTCU]" << std::endl;
  std::cout << "  host_ms=" << stats_dtcu.host_ms
            << " cycles=" << stats_dtcu.cycles
            << " instrs=" << stats_dtcu.instrs
            << " IPC=" << (double(stats_dtcu.instrs) / double(stats_dtcu.cycles))
            << std::endl;

  std::cout << "  loads=" << stats_dtcu.loads
            << " stores=" << stats_dtcu.stores
            << " stall_lsu=" << stats_dtcu.stall_lsu
            << " stall_tcu=" << stats_dtcu.stall_tcu
            << " instr_lsu=" << stats_dtcu.instr_lsu
            << " instr_tcu=" << stats_dtcu.instr_tcu
            << std::endl;

  std::cout << "  l2_reads=" << stats_dtcu.l2_reads
            << " l2_writes=" << stats_dtcu.l2_writes
            << " mem_reads=" << stats_dtcu.mem_reads
            << " mem_writes=" << stats_dtcu.mem_writes
            << std::endl;

  // Ratio of DTCU over TCU
  std::cout << "\n[Ratio DTCU / TCU]" << std::endl;
  std::cout << "  host_ms=" << (stats_tcu.host_ms ? stats_dtcu.host_ms / stats_tcu.host_ms : 0.0)
            << " cycles=" << (stats_tcu.cycles ? double(stats_dtcu.cycles) / double(stats_tcu.cycles) : 0.0)
            << " l2_total=" << (double(stats_dtcu.l2_reads + stats_dtcu.l2_writes) / double(stats_tcu.l2_reads + stats_tcu.l2_writes))
            << " mem_total=" << (double(stats_dtcu.mem_reads + stats_dtcu.mem_writes) / double(stats_tcu.mem_reads + stats_tcu.mem_writes))
            << std::endl;

  if (errors_tcu || errors_dtcu || cross_errors) {
    std::cerr << "FAILED: errors_tcu=" << errors_tcu << " errors_dtcu=" << errors_dtcu << " cross_errors=" << cross_errors << std::endl;
    return errors_tcu + errors_dtcu + cross_errors;
  }

  std::cout << "PASSED" << std::endl;
  return 0;
}