#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include "common.h"
#include <hfloats.h>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<int8_t> {
public:
  static const char* type_str() {
    return "int8";
  }
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "int8";
  }
  static int generate() {
    return (int)rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vortex::half_t> {
public:
  static const char* type_str() {
    return "f16";
  }
  static vortex::half_t generate() {
    return static_cast<vortex::half_t>(float(rand()) / RAND_MAX);
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static int generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(O_TYPE* C, const I_TYPE* A, const I_TYPE* B, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      O_TYPE sum(0);
      for (uint32_t k = 0; k < K; ++k) {
        sum += O_TYPE(A[m*K + k] * B[k*N + n]);
      }
      C[m*N + n] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t M = 32;
uint32_t N = 32;
uint32_t K = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-m: m] [-n N] [-k: K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
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
  if (NT < 4) {
    std::cout << "Error: warp size must be at least 4 threads!" << std::endl;
    return -1;
  }
  std::cout << "GPU warp size: " << NT << " threads" << std::endl;

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  uint32_t XlenB = VX_ISA_ARCH(isa_flags) / 8;
  std::cout << "GPU XLEN: " << 8 * XlenB << std::endl;

  // tile format ratio
  uint32_t o_ratio = XlenB / sizeof(O_TYPE);
  uint32_t i_ratio = XlenB / sizeof(I_TYPE);

  // determine tensor tile size
  uint32_t logNT = log2(NT);
  uint32_t tileM = 4 * (1 << (logNT / 2)) * o_ratio;
  uint32_t tileN = (logNT % 2 == 0) ? (tileM / 2) : tileM;
  uint32_t tileK = std::min(tileM, tileN) * i_ratio;

  std::cout << "GPU tensor tileM=" << tileM << ", tileN=" << tileM << ", tileK=" << tileK << std::endl;

  if ((M % tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM!" << std::endl;
    return -1;
  }

  if ((N % tileN) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileN!" << std::endl;
    return -1;
  }

  if ((K % tileK) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileK!" << std::endl;
    return -1;
  }

  kernel_arg.tileM = tileM;
  kernel_arg.tileN = tileN;
  kernel_arg.tileK = tileK;

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "input data type: " << Comparator<I_TYPE>::type_str() << " (" << sizeof(I_TYPE) << " bytes)" << std::endl;
  std::cout << "output data type: " << Comparator<O_TYPE>::type_str() << " (" << sizeof(O_TYPE) << " bytes)" << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  // set block size to warp size
  kernel_arg.grid_dim[0]  = N / tileN;
  kernel_arg.grid_dim[1]  = M / tileM;
  kernel_arg.block_dim[0] = NT; // warp size
  kernel_arg.block_dim[1] = 1;

  // set matrix dimensions
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(I_TYPE), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(I_TYPE), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(O_TYPE), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;

  // generate source data
  std::vector<I_TYPE> h_A(sizeA);
  std::vector<I_TYPE> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = Comparator<I_TYPE>::generate();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = Comparator<I_TYPE>::generate();
  }

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(I_TYPE)));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(I_TYPE)));
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
  std::vector<O_TYPE> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(O_TYPE)));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<O_TYPE> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<O_TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
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