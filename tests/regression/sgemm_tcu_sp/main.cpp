#include "common.h"
#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <tensor.h>
#include <string.h>
#include <VX_config.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#ifndef FLOAT_ULP
#define FLOAT_ULP 6
#endif
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

static void convert_row_to_col_major_4bit(uint8_t *dst, uint32_t width, uint32_t height, const uint8_t *src) {
  // Calculate output size and stride
  uint32_t out_bytes = (width * height + 1) / 2;
  memset(dst, 0, out_bytes);
  uint32_t dst_stride = (height + 1) / 2; // Bytes per column in output

  // For each column in source (which becomes row in destination)
  for (uint32_t c = 0; c < width; ++c) {
    uint32_t base = c * dst_stride;

    // For each row in source (which becomes column in destination)
    for (uint32_t r = 0; r < height; r += 2) {
      // Calculate source indices (row-major)
      uint32_t idx_even = r * width + c;
      uint32_t idx_odd = (r + 1) * width + c;

      // Extract nibbles - consistent with data_accessor_t
      uint8_t b_even = src[idx_even / 2];
      uint8_t b_odd = (r + 1 < height) ? src[idx_odd / 2] : 0;

      uint8_t nib_even = (idx_even & 1) ? (b_even >> 4) : (b_even & 0x0F);
      uint8_t nib_odd = (r + 1 < height)
                            ? ((idx_odd & 1) ? (b_odd >> 4) : (b_odd & 0x0F))
                            : 0;

      // Pack into destination: even row in low nibble, odd row in high nibble
      dst[base + r / 2] = (nib_odd << 4) | nib_even;
    }
  }
}

template<typename T>
static void convert_row_to_col_major(T *dst, const T *src, uint32_t rows, uint32_t cols) {
  for (uint32_t r = 0; r < rows; ++r) {
    for (uint32_t c = 0; c < cols; ++c) {
      dst[c * rows + r] = src[r * cols + c];
    }
  }
}

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
    return odd ? (value8 >> 4) : (value8 & 0x0f); // to nibble
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
    return odd ? (value8 >> 4) : (value8 & 0x0f); // to nibble
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
struct muladd_t<vt::fp16, vt::fp16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_htof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_htof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_htof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoh_s(bit_cast<uint32_t>(fd), 0, nullptr);
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
struct muladd_t<vt::bf16, vt::bf16> {
  static uint16_t eval(uint16_t a, uint16_t b, uint16_t c) {
    auto fa = bit_cast<float>(rv_btof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_btof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_btof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftob_s(bit_cast<uint32_t>(fd), 0, nullptr);
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
struct muladd_t<vt::fp8, vt::fp8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    auto fa = bit_cast<float>(rv_e4m3tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_e4m3tof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_e4m3tof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoe4m3_s(bit_cast<uint32_t>(fd), 0, nullptr);
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
struct muladd_t<vt::bf8, vt::bf8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    auto fa = bit_cast<float>(rv_e5m2tof_s(a, 0, nullptr));
    auto fb = bit_cast<float>(rv_e5m2tof_s(b, 0, nullptr));
    auto fc = bit_cast<float>(rv_e5m2tof_s(c, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftoe5m2_s(bit_cast<uint32_t>(fd), 0, nullptr);
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

using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;


// Dense CPU reference matmul (pruned A has zeros at masked positions)
static void matmul_cpu(otype_t *C, const itype_t *A, const itype_t *B, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
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

// Pack per-group masks into VX_tcu_meta SRAM layout
// Output: h_meta vector indexed as [tile_row][k_tile][NT * meta_cols words]
static void pack_metadata(std::vector<uint32_t> &h_meta,
                           const std::vector<uint8_t> &masks,
                           uint32_t M, uint32_t K) {
  constexpr uint32_t I_RATIO = cfg::rtl_i_ratio;
  constexpr uint32_t TC_K = cfg::tcK;
  constexpr uint32_t TC_M = cfg::tcM;
  constexpr uint32_t meta_row_w = TC_K * 2 * I_RATIO;
  constexpr uint32_t mcols = cfg::meta_cols;
  constexpr uint32_t half_k_steps = cfg::k_steps / 2;
  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  constexpr uint32_t cols_per_load = (NUM_THREADS >= PD) ? (NUM_THREADS / PD) : 1;
  constexpr uint32_t banks_per_store = (NUM_THREADS < PD) ? NUM_THREADS : PD;
  constexpr uint32_t stores_per_col = (PD + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t num_meta_loads = (PD * mcols + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;

  uint32_t subbytes = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
  uint32_t tileK_elem = subbytes ? (cfg::tileK * subbytes) : cfg::tileK;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  uint32_t num_groups_per_row = KS / 4;
  uint32_t elts_per_sparse_step = tileK_elem / half_k_steps;

  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;

  h_meta.assign(num_tile_rows * num_k_tiles * per_k_tile_words, 0);

  for (uint32_t tr = 0; tr < num_tile_rows; ++tr) {
    for (uint32_t kt = 0; kt < num_k_tiles; ++kt) {
      uint32_t section_base = (tr * num_k_tiles + kt) * per_k_tile_words;

      for (uint32_t sm = 0; sm < cfg::m_steps; ++sm) {
        for (uint32_t sk = 0; sk < half_k_steps; ++sk) {
          uint32_t sram_row = sm * half_k_steps + sk;

          for (uint32_t i = 0; i < TC_M; ++i) {
            uint32_t physical_row = tr * cfg::tileM + sm * TC_M + i;
            uint32_t k_elem_start = kt * tileK_elem + sk * elts_per_sparse_step;
            uint32_t groups_in_step = elts_per_sparse_step / 4;

            // Iterate over individual elements in this sparse step.
              // Using a flat element loop handles both full groups (I_RATIO >= 2)
              // and partial groups (I_RATIO = 1, tf32) where elts_per_sparse_step < 4.
              for (uint32_t e = 0; e < elts_per_sparse_step; ++e) {
                uint32_t global_elt = k_elem_start + e;
                uint32_t global_group = global_elt / 4;
                uint32_t pos_in_group = global_elt % 4;
                uint8_t mask = masks[physical_row * num_groups_per_row + global_group];

                if (mask & (1u << pos_in_group)) {
                  // Map element position within step to meta_row bit position
                  uint32_t k_reg = e / (2 * I_RATIO);
                  uint32_t pos_in_k = e % (2 * I_RATIO);
                  uint32_t meta_bit;
                  if (pos_in_k < I_RATIO) {
                    meta_bit = k_reg * I_RATIO + pos_in_k;
                  } else {
                    meta_bit = (TC_K + k_reg) * I_RATIO + (pos_in_k - I_RATIO);
                  }
                  uint32_t block_bit = i * meta_row_w + meta_bit;
                  uint32_t word_idx = block_bit / 32;
                  uint32_t bit_idx = block_bit % 32;
                  uint32_t store_in_col = sram_row / banks_per_store;
                  uint32_t thread_in_store = sram_row % banks_per_store;
                  uint32_t flat_store = word_idx * stores_per_col + store_in_col;
                  uint32_t load_idx = flat_store / cols_per_load;
                  uint32_t store_in_load = flat_store % cols_per_load;
                  uint32_t meta_idx = load_idx * NUM_THREADS + store_in_load * banks_per_store + thread_in_store;
                  h_meta[section_base + meta_idx] |= (1u << bit_idx);
                }
              }
          }
        }
      }
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
vx_buffer_h meta_buffer = nullptr;
vx_buffer_h cycles_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

bool zero_a_test = false; // T5: zero out A matrix for FEDP/metadata diagnostic

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sgemm TCU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n N] [-k: K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:i:o:hsZ")) != -1) {
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
    case 'Z':
      zero_a_test = true;
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
    vx_mem_free(meta_buffer);
    vx_mem_free(cycles_buffer);
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
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;

  if ((M % cfg::tileM) != 0) {
    std::cout << "Error: M must be a multiple of tensor tileM=" << cfg::tileM << "!" << std::endl;
    return -1;
  }

  if ((N % cfg::tileN) != 0) {
    std::cout << "Error: N must be a multiple of tensor tileN=" << cfg::tileN << "!" << std::endl;
    return -1;
  }

  if ((K % cfg::tileK) != 0) {
    std::cout << "Error: K must be a multiple of tensor tileK=" << cfg::tileK << "!" << std::endl;
    return -1;
  }

  size_t sizeA_full = M * K;
  size_t sizeA = (M * K) / 2;
  size_t sizeB = K * N;
  size_t sizeC = M * N;

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;

  uint32_t grid_dim[2]  = {N / cfg::tileN, M / cfg::tileM};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

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

  // allocate metadata buffer per (tile_row, k_tile)
  constexpr uint32_t meta_cols = cfg::meta_cols;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;
  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_THREADS - 1) / NUM_THREADS;
  uint32_t meta_buf_entries = num_tile_rows * num_k_tiles * (num_meta_loads * NUM_THREADS);
  RT_CHECK(vx_mem_alloc(device, meta_buf_entries * sizeof(uint32_t), VX_MEM_READ, &meta_buffer));
  RT_CHECK(vx_mem_address(meta_buffer, &kernel_arg.meta_sp_addr));

  uint32_t num_blocks = grid_dim[0] * grid_dim[1];
  uint64_t num_mma_sync_instrs = uint64_t(num_blocks) * num_k_tiles;
  RT_CHECK(vx_mem_alloc(device, num_blocks * sizeof(uint32_t), VX_MEM_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.cycles_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;
  std::cout << "meta_sp_addr=0x" << std::hex << kernel_arg.meta_sp_addr << std::endl;

  // generate source data
  // Generate full matrix A (M × K), prune in-place, then compress to M × K/2
  std::vector<itype_t> h_A_full(sizeA_full);
  for (uint32_t i = 0; i < sizeA_full; ++i) {
    h_A_full[i] = generate_A_value<vt::ITYPE>();
  }
  if (!vt::prune_2to4_matrix<vt::ITYPE>(h_A_full.data(), M, K)) {
    std::cerr << "prune_2to4_matrix failed" << std::endl;
    return -1;
  }
  std::vector<itype_t> h_A(sizeA);
  std::vector<uint8_t> masks;
  if (!vt::compress_2to4_matrix<vt::ITYPE>(h_A.data(), h_A_full.data(), masks, M, K)) {
    std::cerr << "compress_2to4_matrix failed" << std::endl;
    return -1;
  }

  // T5: Zero-A test — zero out compressed A to test FEDP/metadata isolation
  if (zero_a_test) {
    printf("*** ZERO-A TEST MODE: zeroing compressed A matrix ***\n");
    memset(h_A.data(), 0, sizeA * sizeof(itype_t));
    memset(h_A_full.data(), 0, sizeA_full * sizeof(itype_t));
  }

  std::vector<itype_t> h_B(sizeB);
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = generate_B_value<vt::ITYPE>();
  }

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    if constexpr (std::is_same<vt::ITYPE, vt::int4>::value ||
                  std::is_same<vt::ITYPE, vt::uint4>::value) {
      // sub-byte: existing 4-bit col-major conversion
      std::vector<uint8_t> h_B_col(sizeB);
      convert_row_to_col_major_4bit(h_B_col.data(), N, 2 * K, (uint8_t*)h_B.data());
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB));
    } else {
      // byte+ types: convert to col-major
      std::vector<itype_t> h_B_col(sizeB);
      convert_row_to_col_major(h_B_col.data(), h_B.data(), K, N);
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB * sizeof(itype_t)));
    }
  }

  // upload metadata buffer (per-group metadata bytes from compression)
  {
    std::cout << "upload metadata buffer" << std::endl;
    std::vector<uint32_t> h_meta;
    pack_metadata(h_meta, masks, M, K);
    RT_CHECK(vx_copy_to_dev(meta_buffer, h_meta.data(), 0, meta_buf_entries * sizeof(uint32_t)));
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
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, 0));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // download and report cycle counts
  {
    std::vector<uint32_t> h_cycles(num_blocks);
    RT_CHECK(vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t)));
    uint64_t cycles_sum = 0;
    uint32_t cycles_max = 0;
    for (auto cycles : h_cycles) {
      cycles_sum += cycles;
      cycles_max = std::max(cycles_max, cycles);
    }
    std::cout << std::dec;
    std::cout << "mma_sync cycles max: " << cycles_max << std::endl;
    std::cout << "mma_sync cycles total: " << cycles_sum << std::endl;
    std::cout << "mma_sync cycles average per mma_sync instr: "
              << (num_mma_sync_instrs ? (double(cycles_sum) / num_mma_sync_instrs) : 0.0)
              << std::endl;
  }

  // download destination buffer
  std::vector<otype_t> h_C(sizeC);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));

  // === DEBUG: dump masks, metadata, compressed A for row 0 ===
  {
    uint32_t subbytes_d = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
    uint32_t KS_d = subbytes_d ? (K * subbytes_d) : K;
    uint32_t num_groups_d = KS_d / 4;
    std::cout << "=== DEBUG: ITYPE::bits=" << vt::ITYPE::bits
              << " I_RATIO=" << cfg::rtl_i_ratio
              << " TC_K=" << cfg::tcK << " TC_M=" << cfg::tcM
              << " meta_cols=" << cfg::meta_cols
              << " tileK=" << cfg::tileK
              << " k_steps=" << cfg::k_steps
              << " half_k_steps=" << cfg::k_steps/2
              << std::endl;

    // Print masks for row 0
    std::cout << "Masks row 0:";
    for (uint32_t g = 0; g < num_groups_d && g < 8; ++g) {
      printf(" g%u=0x%x", g, masks[0 * num_groups_d + g]);
    }
    std::cout << std::endl;

    // Print compressed A for row 0 (first 8 elements)
    uint32_t stride_comp_d = KS_d / 2;
    std::cout << "Compressed A row 0 (hex):";
    for (uint32_t k = 0; k < stride_comp_d && k < 16; ++k) {
      auto val = data_accessor_t<vt::ITYPE>::read(h_A.data(), 0 * stride_comp_d + k);
      printf(" 0x%x", (unsigned)val);
    }
    std::cout << std::endl;

    // Recompute and print metadata words
    std::vector<uint32_t> h_meta_dbg;
    pack_metadata(h_meta_dbg, masks, M, K);
    constexpr uint32_t mcols_d = cfg::meta_cols;
    constexpr uint32_t PD_d = cfg::m_steps * (cfg::k_steps / 2);
    constexpr uint32_t num_meta_loads_d = (PD_d * mcols_d + NUM_THREADS - 1) / NUM_THREADS;
    uint32_t per_k_words_d = num_meta_loads_d * NUM_THREADS;
    std::cout << "Metadata words (tile_row=0, k_tile=0):";
    for (uint32_t w = 0; w < per_k_words_d; ++w) {
      printf(" [%u]=0x%08x", w, h_meta_dbg[w]);
    }
    std::cout << std::endl;

    // Decode metadata bits for sram_row 0 (sm=0, sk=0)
    // Each sram_row has mcols_d words = mcols_d*32 bits
    // TC_M rows, each META_ROW_WIDTH bits
    constexpr uint32_t meta_row_w_d = cfg::tcK * 2 * cfg::rtl_i_ratio;
    std::cout << "  sram_row0 decoded (TC_M=" << cfg::tcM << " rows, " << meta_row_w_d << " bits each):" << std::endl;
    uint32_t sram0_word = h_meta_dbg[0];
    for (uint32_t i = 0; i < cfg::tcM; ++i) {
      uint32_t row_bits = (sram0_word >> (i * meta_row_w_d)) & ((1u << meta_row_w_d) - 1);
      printf("    TC_M row %u: bits=0x%x (binary:", i, row_bits);
      for (int b = meta_row_w_d-1; b >= 0; --b) printf("%d", (row_bits >> b) & 1);
      printf(")\n");
    }

    // Show what pruned A looks like for row 0 (full K)
    std::cout << "Pruned A row 0 (full, first 16 hex):";
    for (uint32_t k = 0; k < KS_d && k < 16; ++k) {
      auto val = data_accessor_t<vt::ITYPE>::read(h_A_full.data(), 0 * KS_d + k);
      printf(" 0x%x", (unsigned)val);
    }
    std::cout << std::endl;
  }
  // === END DEBUG ===

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeC);
    matmul_cpu(h_ref.data(), h_A_full.data(), h_B.data(), M, N, K);

    // Sparse reference: manually compute using compressed A + mask-selected B
    // This mimics exactly what the hardware should do
    uint32_t subbytes_v = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
    uint32_t KS_v = subbytes_v ? (K * subbytes_v) : K;
    uint32_t stride_comp_v = KS_v / 2;
    uint32_t num_groups_v = KS_v / 4;
    std::vector<otype_t> h_sparse_ref(sizeC);
    for (uint32_t m = 0; m < M; ++m) {
      for (uint32_t n = 0; n < N; ++n) {
        otype_t sum(0);
        uint32_t comp_idx = 0;
        for (uint32_t g = 0; g < num_groups_v; ++g) {
          uint8_t mask = masks[m * num_groups_v + g];
          // Extract first set and last set positions (matching VX_tcu_sel)
          int first_set = -1, last_set = -1;
          for (int p = 0; p < 4; ++p) {
            if (mask & (1 << p)) {
              if (first_set < 0) first_set = p;
              last_set = p;
            }
          }
          uint32_t k_base = g * 4;
          // compressed A stores in ascending order: first_set then last_set
          auto a_first = data_accessor_t<vt::ITYPE>::read(h_A.data(), m * stride_comp_v + comp_idx);
          auto a_last  = data_accessor_t<vt::ITYPE>::read(h_A.data(), m * stride_comp_v + comp_idx + 1);
          auto b_first = data_accessor_t<vt::ITYPE>::read(h_B.data(), (k_base + first_set) * N + n);
          auto b_last  = data_accessor_t<vt::ITYPE>::read(h_B.data(), (k_base + last_set) * N + n);
          sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a_first, b_first, sum);
          sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a_last, b_last, sum);
          comp_idx += 2;
        }
        data_accessor_t<vt::OTYPE>::write(h_sparse_ref.data(), m * N + n, sum);
      }
    }

    // Compare sparse ref with dense ref (should match)
    int sparse_ref_errors = 0;
    for (uint32_t i = 0; i < sizeC; ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_sparse_ref[i], h_ref[i], i, sparse_ref_errors)) {
        if (sparse_ref_errors <= 5) {
          printf("  sparse_ref[%u]=%f vs cpu_ref[%u]=%f\n", i,
            static_cast<float>(h_sparse_ref[i]), i,
            static_cast<float>(h_ref[i]));
        }
        ++sparse_ref_errors;
      }
    }
    if (sparse_ref_errors > 0) {
      printf("WARNING: sparse_ref vs cpu_ref: %d / %zu mismatches!\n", sparse_ref_errors, sizeC);
    } else {
      printf("sparse_ref vs cpu_ref: ALL MATCH\n");
    }

    // Compare GPU output with sparse ref
    int gpu_vs_sparse = 0;
    for (uint32_t i = 0; i < sizeC; ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_C[i], h_sparse_ref[i], i, gpu_vs_sparse)) {
        if (gpu_vs_sparse <= 5) {
          printf("  gpu[%u]=%f vs sparse_ref[%u]=%f\n", i,
            static_cast<float>(h_C[i]), i,
            static_cast<float>(h_sparse_ref[i]));
        }
        ++gpu_vs_sparse;
      }
    }
    if (gpu_vs_sparse > 0) {
      printf("GPU vs sparse_ref: %d / %zu mismatches\n", gpu_vs_sparse, sizeC);
    } else {
      printf("GPU vs sparse_ref: ALL MATCH\n");
    }

    // Print first few entries for manual inspection
    printf("First 8 entries: cpu_ref / sparse_ref / gpu\n");
    for (uint32_t i = 0; i < 8 && i < sizeC; ++i) {
      printf("  [%u] %f / %f / %f\n", i,
        static_cast<float>(h_ref[i]),
        static_cast<float>(h_sparse_ref[i]),
        static_cast<float>(h_C[i]));
    }

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }

    // === DETAILED ERROR ANALYSIS (FPGA root-cause investigation) ===
    {
      printf("\n=== DETAILED ERROR ANALYSIS ===\n");
      printf("Config: tileM=%u tileN=%u tileK=%u tcM=%u tcN=%u tcK=%u\n",
          cfg::tileM, cfg::tileN, cfg::tileK, cfg::tcM, cfg::tcN, cfg::tcK);
      printf("Steps: m_steps=%u n_steps=%u k_steps=%u\n",
          cfg::m_steps, cfg::n_steps, cfg::k_steps);
      printf("Matrix: M=%u N=%u K=%u, Grid=%ux%u\n",
          M, N, K, M/cfg::tileM, N/cfg::tileN);
#ifdef TCU_SPARSE_ENABLE
      printf("Sparse: meta_cols=%u per_warp_depth=%u\n",
          cfg::meta_cols, cfg::per_warp_depth);
#endif

      struct ErrInfo {
        uint32_t idx, m, n;
        uint32_t block_row, block_col;
        uint32_t in_block_m, in_block_n;
        uint32_t step_m, step_n;
        uint32_t sub_m, sub_n;
        otype_t expected, actual;
      };
      std::vector<ErrInfo> errs;

      for (uint32_t i = 0; i < sizeC; ++i) {
        // Use Comparator for type-appropriate comparison but suppress printing
        bool match = Comparator<vt::OTYPE>::compare(h_C[i], h_ref[i], i, INT_MAX);
        if (!match) {
          ErrInfo e;
          e.idx = i;
          e.m = i / N;
          e.n = i % N;
          e.block_row = e.m / cfg::tileM;
          e.block_col = e.n / cfg::tileN;
          e.in_block_m = e.m % cfg::tileM;
          e.in_block_n = e.n % cfg::tileN;
          e.step_m = e.in_block_m / cfg::tcM;
          e.step_n = e.in_block_n / cfg::tcN;
          e.sub_m = e.in_block_m % cfg::tcM;
          e.sub_n = e.in_block_n % cfg::tcN;
          e.expected = h_ref[i];
          e.actual = h_C[i];
          errs.push_back(e);
        }
      }

      printf("\nTotal errors: %zu / %u\n\n", errs.size(), (unsigned)sizeC);

      // Print all errors with position mapping (up to 200)
      for (size_t i = 0; i < errs.size() && i < 200; ++i) {
        auto &e = errs[i];
        double diff = static_cast<double>(e.actual) - static_cast<double>(e.expected);
        printf("ERR[%zu]: pos=(%u,%u) blk=(%u,%u) step=(%u,%u) sub=(%u,%u) exp=%g act=%g diff=%+g\n",
            i, e.m, e.n, e.block_row, e.block_col,
            e.step_m, e.step_n, e.sub_m, e.sub_n,
            static_cast<double>(e.expected), static_cast<double>(e.actual), diff);
      }

      if (!errs.empty()) {
        // Block distribution
        uint32_t grid_y = M / cfg::tileM;
        uint32_t grid_x = N / cfg::tileN;
        printf("\nBlock distribution (grid %ux%u):\n", grid_y, grid_x);
        for (uint32_t br = 0; br < grid_y; ++br) {
          for (uint32_t bc = 0; bc < grid_x; ++bc) {
            int cnt = 0;
            for (auto &e : errs) {
              if (e.block_row == br && e.block_col == bc) cnt++;
            }
            if (cnt > 0) printf("  blk(%u,%u): %d errors\n", br, bc, cnt);
          }
        }

        // Sub-tile position distribution
        printf("\nSub-tile position (tcM=%u x tcN=%u):\n", cfg::tcM, cfg::tcN);
        for (uint32_t sm = 0; sm < cfg::tcM; ++sm) {
          for (uint32_t sn = 0; sn < cfg::tcN; ++sn) {
            int cnt = 0;
            for (auto &e : errs) {
              if (e.sub_m == sm && e.sub_n == sn) cnt++;
            }
            if (cnt > 0) printf("  sub(%u,%u): %d errors\n", sm, sn, cnt);
          }
        }

        // Step distribution
        printf("\nStep distribution (m_steps=%u x n_steps=%u):\n",
            cfg::m_steps, cfg::n_steps);
        for (uint32_t sm = 0; sm < cfg::m_steps; ++sm) {
          for (uint32_t sn = 0; sn < cfg::n_steps; ++sn) {
            int cnt = 0;
            for (auto &e : errs) {
              if (e.step_m == sm && e.step_n == sn) cnt++;
            }
            if (cnt > 0) printf("  step(%u,%u): %d errors\n", sm, sn, cnt);
          }
        }

        // Error magnitude analysis (integer types)
        if constexpr (std::is_integral_v<otype_t>) {
          int64_t min_diff = INT64_MAX, max_diff = INT64_MIN;
          int64_t sum_abs = 0;
          bool all_positive = true, all_negative = true;
          for (auto &e : errs) {
            int64_t d = static_cast<int64_t>(e.actual) - static_cast<int64_t>(e.expected);
            if (d < min_diff) min_diff = d;
            if (d > max_diff) max_diff = d;
            sum_abs += std::abs(d);
            if (d > 0) all_negative = false;
            if (d < 0) all_positive = false;
          }
          printf("\nError magnitude: min=%+ld max=%+ld mean_abs=%ld\n",
              (long)min_diff, (long)max_diff,
              (long)(sum_abs / (int64_t)errs.size()));
          printf("Direction: %s\n",
              all_positive ? "ALL POSITIVE" :
              all_negative ? "ALL NEGATIVE" : "MIXED");
        }

        // Per-block error detail: for each block with errors, show which K-tiles contribute
        printf("\nPer-block detail:\n");
        for (uint32_t br = 0; br < grid_y; ++br) {
          for (uint32_t bc = 0; bc < grid_x; ++bc) {
            std::vector<ErrInfo*> blk_errs;
            for (auto &e : errs) {
              if (e.block_row == br && e.block_col == bc) blk_errs.push_back(&e);
            }
            if (blk_errs.empty()) continue;
            uint32_t block_id = br * grid_x + bc;
            printf("  blk(%u,%u) id=%u: %zu errors\n", br, bc, block_id, blk_errs.size());
            for (auto *ep : blk_errs) {
              double diff = static_cast<double>(ep->actual) - static_cast<double>(ep->expected);
              printf("    pos=(%u,%u) step=(%u,%u) sub=(%u,%u) diff=%+g\n",
                  ep->m, ep->n, ep->step_m, ep->step_n, ep->sub_m, ep->sub_n, diff);
            }
          }
        }
      }

      // Raw hex dump of first 16 entries for manual inspection
      printf("\nFirst 16 entries (raw values):\n");
      for (uint32_t i = 0; i < 16 && i < sizeC; ++i) {
        if constexpr (std::is_integral_v<otype_t>) {
          printf("  [%u] ref=0x%08x gpu=0x%08x %s\n",
              i, static_cast<unsigned>(h_ref[i]), static_cast<unsigned>(h_C[i]),
              (h_ref[i] == h_C[i]) ? "OK" : "MISMATCH");
        } else {
          union { float f; uint32_t u; } ref_u, gpu_u;
          ref_u.f = h_ref[i]; gpu_u.f = h_C[i];
          printf("  [%u] ref=0x%08x (%f) gpu=0x%08x (%f) %s\n",
              i, ref_u.u, ref_u.f, gpu_u.u, gpu_u.f,
              (ref_u.u == gpu_u.u) ? "EXACT" : "DIFF");
        }
      }
    }
  }

  // cleanup
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
