#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor.h>
#include <tensor_cfg.h>
#include <type_traits>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#ifndef FLOAT_ULP
#define FLOAT_ULP 6
#endif
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                         \
    if (0 == _ret)                                            \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

using namespace vortex;
namespace vt = tensor;

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static_assert(std::is_same_v<vt::ITYPE, vt::mxfp8>, "sgemm_tcu_mx requires ITYPE=mxfp8");
static_assert(std::is_same_v<vt::OTYPE, vt::fp32>, "sgemm_tcu_mx requires OTYPE=fp32");

///////////////////////////////////////////////////////////////////////////////

class ComparatorFP32 {
public:
  static bool compare(float a, float b, int index, int errors) {
    if (a == 0.0f && b == 0.0f) {
      return true;
    }
    float denom = std::max(std::abs(b), 1e-8f);
    float diff = std::abs((a - b) / denom);
    if (diff < 0.01f) {
      return true;
    }
    if (errors < MAX_ERRORS) {
      printf("*** error: [%d] expected=%f, actual=%f (rel=%f)\n", index, b, a, diff);
    }
    return false;
  }
};

static inline float dequant_mxfp8(uint8_t value, uint8_t scale) {
  return bit_cast<float>(rv_mxfp8tof_s(value, scale, 0, nullptr));
}

static void matmul_cpu_mxfp8_fp32(otype_t* C,
                                  const itype_t* A,
                                  const itype_t* B,
                                  const std::vector<uint8_t>& a_scales,
                                  const std::vector<uint8_t>& b_scales,
                                  uint32_t M,
                                  uint32_t N,
                                  uint32_t K) {
  uint32_t k_blocks = K / vt::mxfp8::ele_block;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        uint32_t kb = k / vt::mxfp8::ele_block;
        uint8_t sf_a = a_scales[m * k_blocks + kb];
        uint8_t sf_b = b_scales[kb * N + n];
        float a = dequant_mxfp8(A[m * K + k], sf_a);
        float b = dequant_mxfp8(B[k * N + n], sf_b);
        sum += a * b;
      }
      C[m * N + n] = sum;
    }
  }
}

static inline uint32_t pack4(const uint8_t* v) {
  return (static_cast<uint32_t>(v[0]) << 0)
       | (static_cast<uint32_t>(v[1]) << 8)
       | (static_cast<uint32_t>(v[2]) << 16)
       | (static_cast<uint32_t>(v[3]) << 24);
}

static void normalize_scales_for_wmma_metadata(std::vector<uint8_t>& a_scales,
                                               std::vector<uint8_t>& b_scales,
                                               uint32_t M,
                                               uint32_t N,
                                               uint32_t K) {
  if constexpr (cfg::tileM <= 8 && cfg::tileN <= 8) {
    (void)M;
    (void)N;
    (void)K;
    return;
  }

  uint32_t k_blocks = K / vt::mxfp8::ele_block;

  if constexpr (cfg::tileM > 8) {
    for (uint32_t tr = 0; tr < M; tr += cfg::tileM) {
      for (uint32_t r = 8; r < cfg::tileM && (tr + r) < M; ++r) {
        uint32_t src_r = r - 8;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          a_scales[(tr + r) * k_blocks + kb] = a_scales[(tr + src_r) * k_blocks + kb];
        }
      }
    }
  }

  if constexpr (cfg::tileN > 8) {
    for (uint32_t tc = 0; tc < N; tc += cfg::tileN) {
      for (uint32_t c = 8; c < cfg::tileN && (tc + c) < N; ++c) {
        uint32_t src_c = c - 8;
        for (uint32_t kb = 0; kb < k_blocks; ++kb) {
          b_scales[kb * N + (tc + c)] = b_scales[kb * N + (tc + src_c)];
        }
      }
    }
  }
}

static void pack_meta_mx_dse2(std::vector<uint32_t>& meta,
                              const std::vector<uint8_t>& a_scales,
                              const std::vector<uint8_t>& b_scales,
                              uint32_t M,
                              uint32_t N,
                              uint32_t K) {
  uint32_t k_blocks = K / vt::mxfp8::ele_block;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_tile_cols = N / cfg::tileN;
  uint32_t num_k_steps = K / cfg::tileK;
  constexpr uint32_t per_step_words = 4;

  meta.assign(num_tile_rows * num_tile_cols * num_k_steps * per_step_words, 0);

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t tc = 0; tc < num_tile_cols; ++tc) {
      for (uint32_t ks = 0; ks < num_k_steps; ++ks) {
        uint32_t k0 = ks * cfg::tileK;
        uint32_t kb = k0 / vt::mxfp8::ele_block;

        uint8_t a_pack[8] = {0};
        for (uint32_t i = 0; i < 8; ++i) {
          uint32_t row = tr * cfg::tileM + i;
          if (row < M) {
            a_pack[i] = a_scales[row * k_blocks + kb];
          }
        }

        uint8_t b_pack[8] = {0};
        for (uint32_t i = 0; i < 8; ++i) {
          uint32_t col = tc * cfg::tileN + i;
          if (col < N) {
            b_pack[i] = b_scales[kb * N + col];
          }
        }

        uint32_t base = (((tr * num_tile_cols + tc) * num_k_steps) + ks) * per_step_words;
        meta[base + 0] = pack4(&a_pack[0]);
        meta[base + 1] = pack4(&a_pack[4]);
        meta[base + 2] = pack4(&b_pack[0]);
        meta[base + 3] = pack4(&b_pack[4]);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = (cfg::tileK > vt::mxfp8::ele_block) ? cfg::tileK : vt::mxfp8::ele_block;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h meta_mx_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Sgemm TCU MX Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-h]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (c) {
    case 'm':
      xm = atoi(optarg);
      break;
    case 'n':
      xn = atoi(optarg);
      break;
    case 'k':
      xk = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (A_buffer) vx_mem_free(A_buffer);
    if (B_buffer) vx_mem_free(B_buffer);
    if (C_buffer) vx_mem_free(C_buffer);
    if (meta_mx_buffer) vx_mem_free(meta_mx_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_TCU) != 0;
  if (!has_ext) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM=" << cfg::tileM << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: N must be a multiple of tensor tileN=" << cfg::tileN << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: K must be a multiple of tensor tileK=" << cfg::tileK << "!" << std::endl;
    cleanup();
    return -1;
  }
  if ((K % vt::mxfp8::ele_block) != 0) {
    std::cout << "Error: K must be a multiple of mxfp8::ele_block=" << vt::mxfp8::ele_block << "!" << std::endl;
    cleanup();
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_tile_cols = N / cfg::tileN;
  uint32_t num_k_steps = K / cfg::tileK;
  size_t meta_mx_words = static_cast<size_t>(num_tile_rows) * num_tile_cols * num_k_steps * 4;
  uint32_t grid_dim[2] = {N / cfg::tileN, M / cfg::tileM};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "meta_mx words: " << meta_mx_words << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, meta_mx_words * sizeof(uint32_t), VX_MEM_READ, &meta_mx_buffer));
  RT_CHECK(vx_mem_address(meta_mx_buffer, &kernel_arg.meta_mx_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;
  std::cout << "meta_mx_addr=0x" << std::hex << kernel_arg.meta_mx_addr << std::endl;

  std::vector<float> h_A_dense(sizeA);
  std::vector<float> h_B_dense(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A_dense[i] = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B_dense[i] = 2.0f * (float(rand()) / RAND_MAX) - 1.0f;
  }

  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  std::vector<uint8_t> h_A_scales;
  std::vector<uint8_t> h_B_scales;

  if (!vt::quantize_mxfp8_a_rowmajor(h_A.data(), h_A_scales, h_A_dense.data(), M, K)) {
    std::cout << "Error: quantize_mxfp8_a_rowmajor failed" << std::endl;
    cleanup();
    return -1;
  }
  if (!vt::quantize_mxfp8_b_rowmajor(h_B.data(), h_B_scales, h_B_dense.data(), K, N)) {
    std::cout << "Error: quantize_mxfp8_b_rowmajor failed" << std::endl;
    cleanup();
    return -1;
  }

  normalize_scales_for_wmma_metadata(h_A_scales, h_B_scales, M, N, K);

  std::vector<uint32_t> h_meta_mx;
  pack_meta_mx_dse2(h_meta_mx, h_A_scales, h_B_scales, M, N, K);

  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));

  std::cout << "upload matrix B buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));

  std::cout << "upload matrix MX metadata buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(meta_mx_buffer, h_meta_mx.data(), 0, meta_mx_words * sizeof(uint32_t)));

  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu_mxfp8_fp32(h_ref.data(), h_A.data(), h_B.data(), h_A_scales, h_B_scales, M, N, K);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!ComparatorFP32::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
