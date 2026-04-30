#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>
#include <dxa.h>

#define FLOAT_ULP 10
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, fb.f, fa.f);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::fp16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoh_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint16_t a, uint16_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return static_cast<int8_t>(rand());
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return static_cast<int32_t>(rand());
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::tf32> {
public:
  static uint32_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftotf32_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint32_t a, uint32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

template <>
struct muladd_t<vt::fp16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::tf32, vt::fp32> {
  static float eval(uint32_t a, uint32_t b, float c) {
    auto fa = bit_cast<float>(rv_tf32tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_tf32tof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::tf32, vt::tf32> {
  static uint32_t eval(uint32_t a, uint32_t b, uint32_t c) {
    auto fa = bit_cast<float>(rv_tf32tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_tf32tof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_tf32tof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftotf32_s(bit_cast<uint32_t>(fd), 0, nullptr);
  }
};

///////////////////////////////////////////////////////////////////////////////

using cfg = vt::wgmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B,
                       uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum(0);
      for (uint32_t k = 0; k < K; ++k) {
        sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(A[m * K + k], B[k * N + n], sum);
      }
      C[m * N + n] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 64;
uint32_t xn = 64;
uint32_t xk = 64;
uint32_t warps = 4;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

static void show_usage() {
  std::cout << "Vortex SGEMM TCU WGMMA+DXA Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-w warps] [-h help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:w:h")) != -1) {
    switch (c) {
    case 'm': xm = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
    case 'w': warps = atoi(optarg); break;
    case 'h': show_usage(); exit(0); break;
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  // Check TCU extension
  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (!(isa_flags & VX_ISA_EXT_TCU)) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  // Check DXA extension
#ifdef ISA_EXT_DXA
  const uint64_t dxa_isa_bit = (1ull << (32 + ISA_EXT_DXA));
  if ((isa_flags & dxa_isa_bit) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled." << std::endl;
    cleanup();
    return -1;
  }
#endif

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device thread count (" << NT
              << ") must match NUM_THREADS=" << NUM_THREADS << std::endl;
    return -1;
  }

  uint64_t num_warps;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  if (warps > num_warps) {
    std::cout << "Error: requested warps (" << warps
              << ") exceeds device capacity (" << num_warps << ")" << std::endl;
    return -1;
  }

  uint64_t issue_width;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
  if (warps != issue_width) {
    std::cout << "Error: number of warps in TB (" << warps
              << ") must match device's ISSUE_WIDTH=" << issue_width << "!" << std::endl;
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  uint32_t cta_M = warps * cfg::xtileM;

  if ((M % cta_M) != 0) {
    std::cout << "Error: M (" << M << ") must be a multiple of cta_M=" << cta_M << std::endl;
    return -1;
  }
  if ((N % cfg::xtileN) != 0) {
    std::cout << "Error: N (" << N << ") must be a multiple of tileN=" << cfg::xtileN << std::endl;
    return -1;
  }
  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: K (" << K << ") must be a multiple of tileK=" << cfg::tileK << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  // Grid: one block per CTA output tile. Block: warps * NT threads.
  uint32_t grid_dim[2]  = {N / cfg::xtileN, M / cta_M};
  uint32_t block_dim[2] = {warps * (uint32_t)NT, 1};

  // SMEM: A tile [cta_M x tileK] + B tile [tileK x tileN]
  uint32_t smem_size = (cta_M * cfg::tileK + cfg::tileK * cfg::xtileN) * sizeof(itype_t);

  std::cout << "input type: " << vt::ITYPE::name << ", output type: " << vt::OTYPE::name << std::endl;
  std::cout << "WGMMA tile: M=" << cfg::xtileM << ", N=" << cfg::xtileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "CTA tile: M=" << cta_M << " (warps=" << warps << ")" << std::endl;
  std::cout << "grid: " << grid_dim[0] << "x" << grid_dim[1] << std::endl;
  std::cout << "block: " << block_dim[0] << "x" << block_dim[1] << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "smem: " << smem_size << " bytes" << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // Allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::dec << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::dec << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::dec << std::endl;

  // Generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<vt::ITYPE>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<vt::ITYPE>::generate();
  }

  std::cout << "upload source data" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));

  // Program DXA descriptors.
  // Descriptor A: fetches tileK columns x cta_M rows from A[row, k].
  //   dim0 = K-axis (tile0 = tileK), dim1 = M-axis (tile1 = cta_M)
  //   stride0_bytes = row stride of A = K * sizeof(itype_t)
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescA, kernel_arg.A_addr,
    /*size0=*/K, /*size1=*/M,
    /*stride0_bytes=*/K * sizeof(itype_t),
    /*tile0=*/cfg::tileK, /*tile1=*/cta_M,
    /*elem_bytes=*/sizeof(itype_t)));

  // Descriptor B: fetches tileN columns x tileK rows from B[k, col].
  //   dim0 = N-axis (tile0 = tileN), dim1 = K-axis (tile1 = tileK)
  //   stride0_bytes = row stride of B = N * sizeof(itype_t)
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescB, kernel_arg.B_addr,
    /*size0=*/N, /*size1=*/K,
    /*stride0_bytes=*/N * sizeof(itype_t),
    /*tile0=*/cfg::xtileN, /*tile1=*/cfg::tileK,
    /*elem_bytes=*/sizeof(itype_t)));

  // Upload program
  std::cout << "upload kernel" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // Start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, smem_size));

  // Wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Download result
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download result" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  // Verify
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " / " << sizeC << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
