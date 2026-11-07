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
#include <vortex2.h>

#define FLOAT_ULP 16
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

template <>
struct data_accessor_t<vt::uint4> {
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

template <typename Type>
class Comparator {};

// Input-only formats below (int8/uint8/int4/uint4/bf16/fp8/bf8/tf32): TMEM's
// accumulate type (Ot) is restricted to fp32/int32/tf32 (32-bit-wide, so
// these only ever need generate() for A/B; their compare() is copied
// alongside for parity with sgemm_tcu_wg/main.cpp but is not currently
// exercised as an Ot.

template <>
class Comparator<vt::int8> {
public:
  static int8_t generate() {
    return (int8_t)rand();
  }
  static bool compare(int8_t a, int8_t b, int index, int errors) {
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
class Comparator<vt::uint8> {
public:
  static uint8_t generate() {
    return (uint8_t)rand();
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
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
class Comparator<vt::int4> {
public:
  static uint8_t generate() {
    return (uint8_t)rand(); // store 2 nibbles in a byte
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
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
class Comparator<vt::uint4> {
public:
  static uint8_t generate() {
    return (uint8_t)rand(); // store 2 nibbles in a byte
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
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
class Comparator<vt::int32> {
public:
  static int32_t generate() {
    return (int32_t)rand();
  }
  static bool compare(int32_t a, int32_t b, int index, int errors) {
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
class Comparator<vt::bf16> {
public:
  static uint16_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftob_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
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
class Comparator<vt::fp8> {
public:
  static uint8_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoe4m3_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
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
class Comparator<vt::bf8> {
public:
  static uint8_t generate() {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftoe5m2_s(bit_cast<uint32_t>(fvalue), 0, nullptr);
  }
  static bool compare(uint8_t a, uint8_t b, int index, int errors) {
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

template <>
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    if constexpr (std::is_same<vt::ITYPE, vt::fp8>::value || std::is_same<vt::ITYPE, vt::bf8>::value) {
      if (a == 0.0f && b == 0.0f) {
        return true;
      }
      //relative error tolerance
      auto diff = std::abs((a - b)/b);
      if (diff < 0.01f) {
        return true;
      }
      if (errors < MAX_ERRORS) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    } else {
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
struct muladd_t<vt::bf16, vt::fp32> {
  static float eval(uint16_t a, uint16_t b, float c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::fp8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    auto fa = bit_cast<float>(rv_e4m3tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_e4m3tof_s(b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::bf8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    auto fa = bit_cast<float>(rv_e5m2tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_e5m2tof_s(b, 0, nullptr));
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
struct muladd_t<vt::int4, vt::int32> {
  static int32_t eval(uint8_t a, uint8_t b, int32_t c) {
    int32_t a_val = a & 0xF;
    if (a & 0x8) {
      a_val |= 0xFFFFFFF0; // sign extend
    }
    int32_t b_val = b & 0xF;
    if (b & 0x8) {
      b_val |= 0xFFFFFFF0; // sign extend
    }
    return a_val * b_val + c;
  }
};

template <>
struct muladd_t<vt::uint4, vt::int32> {
  static int32_t eval(uint8_t a, uint8_t b, int32_t c) {
    int32_t a_val = a & 0xF;
    int32_t b_val = b & 0xF;
    return a_val * b_val + c;
  }
};

template<typename T>
inline typename T::dtype generate_A_value() {
  return Comparator<T>::generate();
}

template<typename T>
inline typename T::dtype generate_B_value() {
  return Comparator<T>::generate();
}

///////////////////////////////////////////////////////////////////////////////

using umma_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, UMMA_NRC>;

static constexpr uint32_t per_warp_M = umma_cfg::xtileM;
static constexpr uint32_t per_warp_N = umma_cfg::xtileN;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      if constexpr (std::is_same<vt::OTYPE, vt::fp32>::value) {
        // fp32 output: the tensor core accumulates the K products in a wide
        // accumulator and rounds to fp32 once; a per-step-rounded reference
        // drifts by several ULP over K. Each product is exact in fp32, so a
        // double accumulation reproduces the single-rounding dot product.
        double acc = 0.0;
        for (uint32_t k = 0; k < KS; ++k) {
          auto a = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
          auto b = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
          acc += static_cast<double>(muladd_t<vt::ITYPE, vt::OTYPE>::eval(a, b, otype_t(0)));
        }
        data_accessor_t<vt::OTYPE>::write(C, m * N + n, static_cast<otype_t>(acc));
      } else {
        // int32/tf32 output: exact per-step accumulation (int32 has no
        // rounding loss; tf32 as an accumulate type is unexercised today
        // but this path is here for parity).
        otype_t sum(0);
        for (uint32_t k = 0; k < KS; ++k) {
          auto a = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
          auto b = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
          sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a, b, sum);
        }
        data_accessor_t<vt::OTYPE>::write(C, m * N + n, sum);
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 64;
uint32_t xn = 64;
uint32_t xk = 64;
// Warps per CTA (UMMA shares WGMMA's bbuf/lockstep, so its group size has
// the same constraint): derived at runtime from VX_CAPS_ISSUE_WIDTH.
uint32_t warps = 0;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Sgemm TCU UMMA/TMEM Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n: N] [-k: K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
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
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (A_buffer) {
      vx_buffer_release(A_buffer);
    }
    if (B_buffer) {
      vx_buffer_release(B_buffer);
    }
    if (C_buffer) {
      vx_buffer_release(C_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  bool has_ext = (isa_flags & VX_ISA_EXT_TCU) != 0;
  if (!has_ext) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t NT;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &NT));
  if (NT != VX_CFG_NUM_THREADS) {
    std::cout << "Error: device thread size (" << NT << ") must match VX_CFG_NUM_THREADS=" << VX_CFG_NUM_THREADS << "!" << std::endl;
    return -1;
  }

  uint64_t num_warps;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));

  uint64_t issue_width;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISSUE_WIDTH, &issue_width));
  warps = (uint32_t)issue_width;
  if (warps > num_warps) {
    std::cout << "Error: UMMA group size (" << warps
              << " = VX_CFG_ISSUE_WIDTH) exceeds device's per-core warp count ("
              << num_warps << ")" << std::endl;
    return -1;
  }

  // Up to (num_warps / warps) CTAs can be concurrently resident
  uint64_t tmem_cols;
  RT_CHECK(vx_device_query(device, VX_CAPS_TCU_TMEM_COLS, &tmem_cols));
  uint64_t max_concurrent_ctas = num_warps / warps;
  uint64_t max_concurrent_cols = uint64_t(UMMA_NRC + UMMA_ACC_OFFSET) * max_concurrent_ctas;
  if (max_concurrent_cols > tmem_cols) {
    std::cout << "Error: UMMA_NRC (" << UMMA_NRC << ") + UMMA_ACC_OFFSET (" << UMMA_ACC_OFFSET
              << ") x max concurrent CTAs ("
              << max_concurrent_ctas << " = " << num_warps << " warps / " << warps
              << " per CTA) = " << max_concurrent_cols
              << " columns, which exceeds the device's TMEM capacity ("
              << tmem_cols << " columns). Reduce UMMA_NRC, VX_CFG_NUM_WARPS,"
                 " or run with fewer concurrently-resident CTAs." << std::endl;
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  uint32_t cta_M = warps * per_warp_M;
  uint32_t a_size = cta_M * umma_cfg::tileK;
  uint32_t b_size = umma_cfg::tileK * umma_cfg::xtileN;

  if ((a_size % NT) != 0) {
    std::cerr << "Error: A_smem size (" << a_size
              << ") is not a perfect multiple of thread count (" << NT
              << "). Kernel will diverge or OOB!" << std::endl;
    return -1;
  }
  if ((b_size % NT) != 0) {
    std::cerr << "Error: B_smem size (" << b_size
              << ") is not a perfect multiple of thread count (" << NT
              << "). Kernel will diverge or OOB!" << std::endl;
    return -1;
  }

  auto round_up = [](uint32_t v, uint32_t m) { return ((v + m - 1) / m) * m; };
  if ((M % cta_M) != 0) {
    uint32_t M2 = round_up(M, cta_M);
    std::cout << "Note: M (" << M << ") rounded up to " << M2
              << " (multiple of cta_M=" << cta_M << ")" << std::endl;
    M = M2;
  }
  if ((N % per_warp_N) != 0) {
    uint32_t N2 = round_up(N, per_warp_N);
    std::cout << "Note: N (" << N << ") rounded up to " << N2
              << " (multiple of per_warp_N=" << per_warp_N << ")" << std::endl;
    N = N2;
  }
  if ((K % umma_cfg::tileK) != 0) {
    uint32_t K2 = round_up(K, umma_cfg::tileK);
    std::cout << "Note: K (" << K << ") rounded up to " << K2
              << " (multiple of tensor tileK=" << umma_cfg::tileK << ")" << std::endl;
    K = K2;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;
  uint32_t grid_dim[2]  = {N / per_warp_N, M / cta_M};
  uint32_t block_dim[2] = {warps * (uint32_t)NT, 1};

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "UMMA Core Dimension: M=" << umma_cfg::tcM << ", N=" << umma_cfg::tcN << ", K=" << umma_cfg::tileK << std::endl;
  std::cout << "UMMA Per-Warp Tile: M=" << per_warp_M << ", N=" << per_warp_N << ", K=" << umma_cfg::tileK << std::endl;
  std::cout << "UMMA CTA Tile: M=" << cta_M << ", N=" << per_warp_N << ", K=" << umma_cfg::tileK << std::endl;
  std::cout << "Grid dimension: " << grid_dim[0] << "x" << grid_dim[1] << std::endl;
  std::cout << "Block dimension: " << block_dim[0] << "x" << block_dim[1] << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::dec << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::dec << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::dec << std::endl;

  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = generate_A_value<vt::ITYPE>();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = generate_B_value<vt::ITYPE>();
  }

  {
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), sizeA * sizeof(itype_t), 0, nullptr, nullptr));
  }
  {
    std::cout << "upload matrix B buffer" << std::endl;
    RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, h_B.data(), sizeB * sizeof(itype_t), 0, nullptr, nullptr));
  }

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<otype_t> h_C(sizeC);

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  uint32_t smem_size = (cta_M * umma_cfg::tileK + umma_cfg::tileK * per_warp_N) * sizeof(itype_t);
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = smem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download destination buffer" << std::endl;
  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t), 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  vx_device_dump_perf(device, stdout);

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
