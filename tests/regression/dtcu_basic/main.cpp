#include "common.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

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
  uint32_t fmt_s;
  uint32_t fmt_d;
  uint32_t flags;
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

  // ---- single tile test ----
  const uint32_t M = 8;
  const uint32_t N = 4;
  const uint32_t K = 8;

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

  // ---- open device connection ----
  vx_device_h device = nullptr;
  RT_CHECK(vx_dev_open(&device));

  // ---- upload program ----
  const char* kernel_file = "kernel.vxbin";
  vx_buffer_h krnl_buffer = nullptr;
  std::cout << "dtcu_basic: upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // ---- alloc device buffers ----
  // Equivalent to set block size to warp size, but here we only use 1 tile
  kernel_arg_t karg{};
  karg.grid_dim[0]  = 1;
  karg.grid_dim[1]  = 1;
  karg.block_dim[0] = NUM_THREADS;
  karg.block_dim[1] = 1;
  karg.M = M;
  karg.N = N;
  karg.K = K;

  vx_buffer_h A_buf = nullptr, B_buf = nullptr, D_buf = nullptr, desc_buf = nullptr;

  // ---- alloc device memory (A, B, D) ----
  std::cout << "dtcu_basic: allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, hA.size() * sizeof(itype_t), VX_MEM_READ,  &A_buf));
  RT_CHECK(vx_mem_address(A_buf, &karg.A_addr));

  RT_CHECK(vx_mem_alloc(device, hB.size() * sizeof(itype_t), VX_MEM_READ,  &B_buf));
  RT_CHECK(vx_mem_address(B_buf, &karg.B_addr));

  RT_CHECK(vx_mem_alloc(device, hD.size() * sizeof(otype_t), VX_MEM_WRITE, &D_buf));
  RT_CHECK(vx_mem_address(D_buf, &karg.D_addr));

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
  desc.flags = 0x1; // C=0 (no accumulate)

  RT_CHECK(vx_mem_alloc(device, sizeof(dtensor_desc_t), VX_MEM_READ, &desc_buf));
  RT_CHECK(vx_mem_address(desc_buf, &karg.desc_addr));

  // ---- upload matrix A, B, descriptor buffer ----
  std::cout << "dtcu_basic: upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buf, hA.data(), 0, hA.size() * sizeof(itype_t)));
  std::cout << "dtcu_basic: upload matrix B buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buf, hB.data(), 0, hB.size() * sizeof(itype_t)));
  std::cout << "dtcu_basic: upload matrix descriptor buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(desc_buf, &desc, 0, sizeof(dtensor_desc_t)));

  // ---- upload kernel argument ----
  vx_buffer_h args_buffer = nullptr;
  std::cout << "dtcu_basic: upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &karg, sizeof(kernel_arg_t), &args_buffer));

  // ---- start device ----
  std::cout << "dtcu_basic: start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // ---- wait for completion ----
  std::cout << "dtcu_basic: wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // ---- download destination buffer ----
  std::cout << "dtcu_basic: download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(hD.data(), D_buf, 0, hD.size() * sizeof(otype_t)));

  // ---- verify result ----
  std::cout << "dtcu_basic: verify result" << std::endl;
  int errors = 0;

  // Equivalent to matmul_cpu() from sgemm_tcu
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = 0; j < N; ++j) {
      float got = Convert<vt::OTYPE>::to_float(hD[i * N + j]);
      float exp = hRef[i * N + j];

      int ulp = ulp_diff(got, exp);
      if (ulp > FLOAT_ULP) {
        if (errors < MAX_ERRORS) {
          std::cerr << "Mismatch D[" << i << "][" << j << "]: got=" << got
                    << " exp=" << exp << " ulp=" << ulp << "\n";
        }
        ++errors;
      }
    }
  }

  if (errors) {
    std::cerr << "FAILED with " << errors << " mismatches\n";
  } else {
    std::cout << "PASSED\n";
  }

  // ---- cleanup ----
  vx_mem_free(A_buf);
  vx_mem_free(B_buf);
  vx_mem_free(D_buf);
  vx_mem_free(desc_buf);
  vx_mem_free(args_buffer);
  vx_mem_free(krnl_buffer);
  vx_dev_close(device);

  if (errors != 0) {
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  return 0;
}