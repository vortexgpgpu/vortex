#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include <tensor_cfg.h>
#include "float16.h"
#include "bfloat16.h"

namespace vt = vortex::tensor;

#define FLOAT_ULP 6

#define HALF_ULP 3

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

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct native_type_impl {};

template <>
struct native_type_impl<vt::fp32> {
  using type = float;
};

template <>
struct native_type_impl<vt::fp16> {
  using type = float16_t;
};

template <>
struct native_type_impl<vt::bf16> {
  using type = bfloat16_t;
};

template <>
struct native_type_impl<vt::int32> {
  using type = int32_t;
};

template <>
struct native_type_impl<vt::int16> {
  using type = int16_t;
};

template <>
struct native_type_impl<vt::int8> {
  using type = int8_t;
};

template <typename T>
using native_type_t = typename native_type_impl<T>::type;

using itype_t = native_type_t<vt::ITYPE>;
using otype_t = native_type_t<vt::OTYPE>;

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int8_t> {
public:
  static const char *type_str() {
    return "int8";
  }
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int a, int b, int index, int errors) {
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
class Comparator<int16_t> {
public:
  static const char *type_str() {
    return "int16";
  }
  static int16_t generate() {
    return (int16_t)rand();
  }
  static bool compare(int a, int b, int index, int errors) {
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
class Comparator<int32_t> {
public:
  static const char *type_str() {
    return "int32";
  }
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int a, int b, int index, int errors) {
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
class Comparator<float16_t> {
public:
  static const char *type_str() {
    return "float16";
  }
  static float16_t generate() {
    return static_cast<float16_t>(float(rand()) / RAND_MAX);
  }
  static bool compare(float16_t a, float16_t b, int index, int errors) {
    auto d = std::abs(a.bits - b.bits);
    if (d > HALF_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b.bits, a.bits);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<bfloat16_t> {
public:
  static const char *type_str() {
    return "bfloat16";
  }
  static bfloat16_t generate() {
    return static_cast<bfloat16_t>(float(rand()) / RAND_MAX);
  }
  static bool compare(bfloat16_t a, bfloat16_t b, int index, int errors) {
    auto d = std::abs(a.bits - b.bits);
    if (d > HALF_ULP) {
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b.bits, a.bits);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char *type_str() {
    return "float";
  }
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
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum(0);
      for (uint32_t k = 0; k < K; ++k) {
        sum = otype_t(A[m * K + k]) * otype_t(B[k * N + n]) + sum;
      }
      C[m * N + n] = sum;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t M = cfg::tileM;
uint32_t N = cfg::tileN;
uint32_t K = cfg::tileK;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sgemm TPU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n N] [-k: K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:i:o:h")) != -1) {
    switch (c) {
    case 'm':
      M = atoi(optarg);
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'k':
      K = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
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
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t NT;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != NUM_THREADS) {
    std::cout << "Error: device warp size (" << NT << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    return -1;
  }

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM!" << std::endl;
    return -1;
  }

  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileN!" << std::endl;
    return -1;
  }

  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileK!" << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "input data type: " << Comparator<itype_t>::type_str() << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << Comparator<otype_t>::type_str() << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  // set block size to warp size
  kernel_arg.grid_dim[0] = N / cfg::tileN;
  kernel_arg.grid_dim[1] = M / cfg::tileM;
  kernel_arg.block_dim[0] = NT; // warp sizeb
  kernel_arg.block_dim[1] = 1;

  // set matrix dimensions
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

  // generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<itype_t>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<itype_t>::generate();
  }

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
  }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download destination buffer
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<otype_t>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}