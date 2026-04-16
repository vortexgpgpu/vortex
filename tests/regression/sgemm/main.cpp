#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include <type_traits>
#include <tensor_cfg.h>
#include "common.h"

namespace vt = vortex::tensor;

#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////
// SIMT type promotion: fp16 → fp32, int8/int4 → use OTYPE (int32)

// For SIMT, fp16 input is promoted to fp32 (no half-float ALU).
// Integer types use their native dtype for storage, cast to otype for compute.
using simt_itype = typename std::conditional<
    std::is_same<vt::ITYPE, vt::fp16>::value, vt::fp32, vt::ITYPE>::type;
using simt_otype = vt::OTYPE;

using itype_t = typename simt_itype::dtype;
using otype_t = typename simt_otype::dtype;

///////////////////////////////////////////////////////////////////////////////
// Data accessor for sub-byte types (int4)

template <typename T>
struct data_accessor_t {
  using Type = typename T::dtype;
  static Type read(const Type *ptr, uint32_t offset) {
    return ptr[offset];
  }
  static void write(Type *ptr, uint32_t offset, Type value) {
    ptr[offset] = value;
  }
};

template <>
struct data_accessor_t<vt::int4> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f);
  }
  static void write(uint8_t *ptr, uint32_t offset, int32_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[offset / 2] = new_value;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Comparator: generate + compare for each type

template <typename Type>
class Comparator {};

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > 6) { // ULP tolerance
      if (errors < MAX_ERRORS)
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS)
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS)
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, (uint8_t)b, (uint8_t)a);
      return false;
    }
    return true;
  }
};

template <>
class Comparator<vt::int4> {
public:
  static uint8_t generate() {
    return (uint8_t)rand(); // 2 nibbles packed per byte
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
    if (a != b) {
      if (errors < MAX_ERRORS)
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", index, b, a);
      return false;
    }
    return true;
  }
};

///////////////////////////////////////////////////////////////////////////////
// muladd: type-aware multiply-accumulate for CPU reference

template <typename S, typename D>
struct muladd_t {
  using stype = typename S::dtype;
  using dtype = typename D::dtype;
  static dtype eval(stype a, stype b, dtype c) {
    return static_cast<dtype>(a) * static_cast<dtype>(b) + c;
  }
};

template <>
struct muladd_t<vt::int4, vt::int32> {
  static int32_t eval(uint8_t a, uint8_t b, int32_t c) {
    int32_t a_val = a & 0xF;
    if (a & 0x8) a_val |= 0xFFFFFFF0;
    int32_t b_val = b & 0xF;
    if (b & 0x8) b_val |= 0xFFFFFFF0;
    return a_val * b_val + c;
  }
};

///////////////////////////////////////////////////////////////////////////////
// CPU reference matmul

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B,
                       uint32_t M, uint32_t N, uint32_t K) {
  // K is in element units (matching the kernel).
  // For int4, data_accessor_t handles nibble extraction from packed bytes.
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum(0);
      for (uint32_t k = 0; k < K; ++k) {
        auto a = data_accessor_t<simt_itype>::read(A, m * K + k);
        auto b = data_accessor_t<simt_itype>::read(B, k * N + n);
        sum = muladd_t<simt_itype, simt_otype>::eval(a, b, sum);
      }
      data_accessor_t<simt_otype>::write(C, m * N + n, sum);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex SIMT GEMM Test." << std::endl;
  std::cout << "Usage: [-m M] [-n N] [-k K] [-K kernel] [-h help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:K:h")) != -1) {
    switch (c) {
    case 'm': xm = atoi(optarg); break;
    case 'n': xn = atoi(optarg); break;
    case 'k': xk = atoi(optarg); break;
    case 'K': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(cycles_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::cout << "input type: " << simt_itype::name
            << " (ITYPE=" << vt::ITYPE::name << ")" << std::endl;
  std::cout << "output type: " << simt_otype::name << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  // For sub-byte types, storage size is smaller
  constexpr uint32_t ibytes = sizeof(itype_t);
  constexpr uint32_t obytes = sizeof(otype_t);
  constexpr uint32_t subbytes = (simt_itype::bits < 8) ? (8 / simt_itype::bits) : 0;

  // K in storage units (for int4: K/2 bytes per row)
  uint32_t K_storage = subbytes ? (K / subbytes) : K;

  size_t sizeA = M * K_storage;
  size_t sizeB = K_storage * N;  // B is K×N, stored row-major
  size_t sizeC = M * N;

  size_t bufA = sizeA * ibytes;
  size_t bufB = sizeB * ibytes;
  size_t bufC = sizeC * obytes;

  // Grid: 1 warp per block for clean cycle measurement
  uint32_t block_dim[2] = {NUM_THREADS, 1};
  uint32_t grid_dim[2]  = {N / block_dim[0], M / block_dim[1]};
  uint32_t num_blocks = grid_dim[0] * grid_dim[1];

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, bufA, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, bufB, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, bufC, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, num_blocks * 4 * sizeof(uint32_t), VX_MEM_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  std::cout << "num_blocks=" << std::dec << num_blocks << std::endl;

  // generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (size_t i = 0; i < sizeA; ++i)
    h_A[i] = Comparator<simt_itype>::generate();
  for (size_t i = 0; i < sizeB; ++i)
    h_B[i] = Comparator<simt_itype>::generate();

  // upload
  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, bufA));
  std::cout << "upload matrix B buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, bufB));

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

  // read back per-block (t0, t1) timestamps; report both per-CTA max diff
  // and global kernel latency = max(t1) - min(t0) across all CTAs.
  // NOTE: max/min reduction is only valid when NUM_CORES=1 (single mcycle CSR).
  {
    std::vector<uint32_t> h_cycles(num_blocks * 4);
    RT_CHECK(vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * 4 * sizeof(uint32_t)));
    uint64_t min_t0 = UINT64_MAX;
    uint64_t max_t1 = 0;
    uint32_t max_diff = 0;
    for (uint32_t i = 0; i < num_blocks; ++i) {
      uint64_t t0 = ((uint64_t)h_cycles[i*4+0] << 32) | h_cycles[i*4+1];
      uint64_t t1 = ((uint64_t)h_cycles[i*4+2] << 32) | h_cycles[i*4+3];
      if (t0 < min_t0) min_t0 = t0;
      if (t1 > max_t1) max_t1 = t1;
      uint32_t diff = (uint32_t)(t1 - t0);
      if (diff > max_diff) max_diff = diff;
    }
    printf("SIMT_CYCLES: max=%u (across %u blocks)\n", max_diff, num_blocks);
    printf("KERNEL_LATENCY: %lu\n", (unsigned long)(max_t1 - min_t0));
  }

  // download result
  std::cout << "download destination buffer" << std::endl;
  std::vector<otype_t> h_C(sizeC);
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, bufC));

  // verify
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC, 0);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);
    for (size_t i = 0; i < sizeC; ++i) {
      if (!Comparator<simt_otype>::compare(h_C[i], h_ref[i], i, errors))
        ++errors;
    }
  }

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
