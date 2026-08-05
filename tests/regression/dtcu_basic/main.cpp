// Simple test to validate DTCU functionality
// Compares DTCU result with CPU reference
// The test covers basic data movement and computation of DTCU over 1 tile (M=8, N=4, K=8) with fp16/bf16 input and fp32 output

#include "common.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include <dtcu_cfg.h>   // DTCU descriptor + native-tile geometry
#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>
#include <vortex.h>  // vx_dev_caps (the ISA-flag gate below)
#include <vortex2.h>

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

// dtensor_desc_t comes from <dtcu_cfg.h>; it used to be mirrored here, which meant a
// silent ABI drift whenever the real struct changed.

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

// Run one engine end to end: build operands sized for THAT engine's native tile, submit,
// verify against the CPU reference. Each engine gets a fresh device so their cache state
// and perf counters cannot bleed into each other.
//
// *skipped is set when the build does not advertise this engine; the caller must not
// then read a zero error count as a pass.
static int run_engine(int engine, const char* tag, bool* skipped, int* errors_out) {
  *skipped = false;
  *errors_out = 0;

  // ---- single DTCU tile test: exactly one native tile, one K tile. M and K are the
  // hardware-fixed tile dims; N is our pick out of the engine's legal tile-N range. ----
  // Geometry comes from the engine-parameterised traits: the default dtcu_config_t is
  // the CLUSTER one, so asking it about the socket engine would quietly answer for the
  // wrong hardware. `engine` is a runtime value here, hence the free functions.
  const uint32_t M = dtcu_tile_m_of(engine);
  const uint32_t K = dtcu_tile_k(sizeof(itype_t));
  // Cluster: 32, which is what this test has always used. Socket: TILE_N_MAX equals
  // TILE_N_GRAN, so there is exactly one legal value -- the shape freedom that variant
  // trades away for locality.
  const uint32_t N = (engine == DTCU_ENGINE_CLUSTER) ? 32u : dtcu_tile_n_max_of(engine);
  if (!dtcu_tile_n_valid_of(engine, N)) {
    std::cerr << tag << ": tile_n=" << N << " is not legal for this engine" << std::endl;
    return -1;
  }

  std::vector<itype_t> hA(M * K);
  std::vector<itype_t> hB(K * N);
  std::vector<otype_t> hD(M * N);
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

  // REFERENCE value to compare to
  // Created by CPU (D = A * B) which is adopted from sgemm_tcu's matmul_cpu()
  // matmult_cpu() is more complicated due to sub-byte formats and scaling factors -> NEED MORE WORK!!
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        float a = Convert<vt::ITYPE>::to_float(hA[i * K + k]);
        float b = Convert<vt::ITYPE>::to_float(hB[j * K + k]);
        acc += a * b;
      }
      hRef[i * N + j] = acc;
    }
  }

  // ---- open device + command queue (v3.0 async runtime) ----
  vx_device_h device = nullptr;
  RT_CHECK(vx_device_open(0, &device));

  // Gate on what the device actually advertises. A build can carry either variant, both,
  // or neither, and issuing a start for an absent engine is not a defined operation --
  // the SFU aborts on it rather than quietly running elsewhere.
  {
    const uint64_t need = (engine == DTCU_ENGINE_CLUSTER) ? VX_ISA_EXT_DTCU_CLUSTER
                                                          : VX_ISA_EXT_DTCU_SOCKET;
    uint64_t isa_flags = 0;
    RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
    if ((isa_flags & need) == 0) {
      std::cout << tag << ": skipped (engine not present in this build)" << std::endl;
      *skipped = true;
      vx_device_release(device);
      return 0;
    }
  }

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  vx_queue_h queue = nullptr;
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // ---- alloc device buffers (A, B, D) ----
  kernel_arg_t karg{};
  karg.M = M;
  karg.N = N;
  karg.K = K;
  karg.engine = (uint32_t)engine;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, D_buf = nullptr, desc_buf = nullptr;

  std::cout << tag << ": allocate device memory (M=" << M << " N=" << N << " K=" << K << ")" << std::endl;
  RT_CHECK(vx_buffer_create(device, hA.size() * sizeof(itype_t), VX_MEM_READ, &A_buf));
  RT_CHECK(vx_buffer_address(A_buf, &karg.A_addr));
  RT_CHECK(vx_buffer_create(device, hB.size() * sizeof(itype_t), VX_MEM_READ, &B_buf));
  RT_CHECK(vx_buffer_address(B_buf, &karg.B_addr));
  // READ_WRITE rather than WRITE: under ZERO_ACC the engine reads each D line once
  // before storing it, so the line is allocated in the cache the output must live in.
  RT_CHECK(vx_buffer_create(device, hD.size() * sizeof(otype_t), VX_MEM_READ_WRITE, &D_buf));
  RT_CHECK(vx_buffer_address(D_buf, &karg.D_addr));

  dtensor_desc_t desc{};
  desc.ptrA  = karg.A_addr;
  desc.ptrB  = karg.B_addr;
  desc.ptrC  = 0;
  desc.ptrD  = karg.D_addr;
  desc.ldmA  = K;   // A row-major
  desc.ldmB  = K;   // B col-major
  desc.ldmC  = 0;
  desc.ldmD  = N; 
  desc.fmt_s = vt::ITYPE::id;
  desc.fmt_d = vt::OTYPE::id;
  desc.flags = DTENSOR_FLAG_ZERO_ACC; // C=0 (no accumulate)
  desc.M     = M;
  desc.N     = N;
  desc.K     = K;
  desc.shape_n_size = dtcu_shape_n_size(N);
  desc.shape_policy  = 0;
  desc.done          = 0; // engine sets this once D is visible

  RT_CHECK(vx_buffer_create(device, sizeof(dtensor_desc_t), VX_MEM_READ_WRITE, &desc_buf));
  RT_CHECK(vx_buffer_address(desc_buf, &karg.desc_addr));

  // ---- upload A, B, descriptor (async enqueue) ----
  std::cout << tag << ": upload A/B/descriptor" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, A_buf, 0, hA.data(), hA.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buf, 0, hB.data(), hB.size() * sizeof(itype_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, desc_buf, 0, &desc, sizeof(dtensor_desc_t), 0, nullptr, nullptr));

  // ---- load kernel module ----
  const char* kernel_file = "kernel.vxbin";
  vx_module_h module_ = nullptr;
  vx_kernel_h kernel = nullptr;
  std::cout << tag << ": load kernel" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // ---- launch: a single thread fires the DTCU descriptor, then spins on poll ----
  std::cout << tag << ": launch kernel" << std::endl;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &karg;
  li.args_size    = sizeof(karg);
  li.ndim         = 1;
  li.grid_dim[0]  = 1;
  li.block_dim[0] = 1;
  vx_event_h launch_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));

  // ---- download D (ordered after the launch) + wait for completion ----
  std::cout << tag << ": download destination buffer" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, hD.data(), D_buf, 0, hD.size() * sizeof(otype_t), 1, &launch_ev, &read_ev));
  std::cout << tag << ": wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // ---- verify result ----
  std::cout << tag << ": verify result" << std::endl;
  int errors = 0;

  // Equivalent to matmul_cpu() from sgemm_tcu
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float got = Convert<vt::OTYPE>::to_float(hD[i * N + j]);
      float exp = hRef[i * N + j];

      int ulp = ulp_diff(got, exp);
      if (ulp > FLOAT_ULP) {
        if (errors < MAX_ERRORS) {
          std::cerr << tag << ": mismatch D[" << i << "][" << j << "]: got=" << got
                    << " exp=" << exp << " ulp=" << ulp << "\n";
        }
        ++errors;
      }
    }
  }
  *errors_out = errors;

  // ---- cleanup ----
  vx_buffer_release(A_buf);
  vx_buffer_release(B_buf);
  vx_buffer_release(D_buf);
  vx_buffer_release(desc_buf);
  vx_kernel_release(kernel);
  vx_module_release(module_);
  vx_queue_release(queue);
  vx_device_dump_perf(device, stdout); // print cycles/instrs/IPC (gated by VORTEX_PROFILING)
  vx_device_release(device);

  if (errors != 0)
    std::cout << tag << ": found " << std::dec << errors << " / " << (M * N) << " errors!" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  // Same GEMM shape family on both placements. They produce identical arithmetic from
  // identical descriptors; only the tile size and where D lands differ, so running both
  // is what distinguishes "the socket engine works" from "the socket opcode silently
  // ran on the cluster engine".
  struct { int engine; const char* tag; } kEngines[] = {
    { DTCU_ENGINE_CLUSTER, "dtcu_basic[cluster]" },
    { DTCU_ENGINE_SOCKET,  "dtcu_basic[socket]"  },
  };

  int total_errors = 0;
  int ran = 0;
  for (auto& e : kEngines) {
    bool skipped = false;
    int errors = 0;
    int rc = run_engine(e.engine, e.tag, &skipped, &errors);
    if (rc != 0)
      return rc;
    total_errors += errors;
    if (!skipped)
      ++ran;
  }

  // A build with no DTCU at all should not report success: this test exists to exercise
  // the engine, and skipping everything means it exercised nothing.
  if (ran == 0) {
    std::cout << "dtcu_basic: no DTCU engine present in this build" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  if (total_errors != 0) {
    std::cout << "Found " << std::dec << total_errors << " errors across "
              << ran << " engine(s)!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return total_errors;
  }

  std::cout << "dtcu_basic: " << ran << " engine(s) verified" << std::endl;
  std::cout << "PASSED!" << std::endl;
  return 0;
}