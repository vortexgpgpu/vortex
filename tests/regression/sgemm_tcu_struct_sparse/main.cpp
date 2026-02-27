#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <VX_config.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 6
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

template <>
struct data_accessor_t<vt::nvfp4> {
  static uint8_t read(const uint8_t *ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f); // extract nibble
  }
  static void write(uint8_t *ptr, uint32_t offset, uint8_t value) {
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
class Comparator<vt::mxint8> {
public:
  static int8_t generate() {
    return (int8_t)(rand() % 256 - 128);
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

// TODO: temp arbitrarily hardcoded scale factors
constexpr uint8_t SCALE_FACTOR_E8M0_A = 129;  // val = 4, bias = 127
constexpr uint8_t SCALE_FACTOR_E8M0_B = 131;  // val = 16
constexpr uint8_t SCALE_FACTOR_E4M3_A = 0x41; // val = 2.25, bias = 7
constexpr uint8_t SCALE_FACTOR_E4M3_B = 0x33; // val = 0.6875

template <>
class Comparator<vt::mxfp8> {
public:
  static uint8_t generate() {
    return generate_with_scale(SCALE_FACTOR_E8M0_A);
  }
  
  static uint8_t generate_with_scale(uint8_t scale_factor) {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftomxfp8_s(bit_cast<uint32_t>(fvalue), scale_factor, 0, nullptr);
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
class Comparator<vt::nvfp4> {
public:
  static uint8_t generate() {
    return generate_with_scale(SCALE_FACTOR_E4M3_A);
  }
  
  static uint8_t generate_with_scale(uint8_t scale_factor) {
    auto fvalue = float(rand()) / RAND_MAX;
    return rv_ftonvfp4_s(bit_cast<uint32_t>(fvalue), scale_factor, 0, nullptr);
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
class Comparator<vt::fp32> {
public:
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    if constexpr (std::is_same<vt::ITYPE, vt::fp8>::value || std::is_same<vt::ITYPE, vt::bf8>::value ||
                  std::is_same<vt::ITYPE, vt::mxfp8>::value || std::is_same<vt::ITYPE, vt::nvfp4>::value) {
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
struct muladd_t<vt::mxfp8, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E8M0_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E8M0_B;
    auto fa = bit_cast<float>(rv_mxfp8tof_s(a, sf_a, 0, nullptr));
    auto fb = bit_cast<float>(rv_mxfp8tof_s(b, sf_b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::mxfp8, vt::mxfp8> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    constexpr uint8_t sf = SCALE_FACTOR_E8M0_A;
    auto fa = bit_cast<float>(rv_mxfp8tof_s(a, sf, 0, nullptr));
    auto fb = bit_cast<float>(rv_mxfp8tof_s(b, sf, 0, nullptr));
    auto fc = bit_cast<float>(rv_mxfp8tof_s(c, sf, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftomxfp8_s(bit_cast<uint32_t>(fd), sf, 0, nullptr);
  }
};

template <>
struct muladd_t<vt::nvfp4, vt::fp32> {
  static float eval(uint8_t a, uint8_t b, float c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E4M3_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E4M3_B;
    auto fa = bit_cast<float>(rv_nvfp4tof_s(a, sf_a, 0, nullptr));
    auto fb = bit_cast<float>(rv_nvfp4tof_s(b, sf_b, 0, nullptr));
    return fa * fb + c;
  }
};

template <>
struct muladd_t<vt::nvfp4, vt::nvfp4> {
  static uint8_t eval(uint8_t a, uint8_t b, uint8_t c) {
    constexpr uint8_t sf = SCALE_FACTOR_E4M3_A;
    auto fa = bit_cast<float>(rv_nvfp4tof_s(a, sf, 0, nullptr));
    auto fb = bit_cast<float>(rv_nvfp4tof_s(b, sf, 0, nullptr));
    auto fc = bit_cast<float>(rv_nvfp4tof_s(c, sf, 0, nullptr));
    auto fd = fa * fb + fc;
    return rv_ftonvfp4_s(bit_cast<uint32_t>(fd), sf, 0, nullptr);
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

template <>
struct muladd_t<vt::mxint8, vt::int32> {
  static int32_t eval(int8_t a, int8_t b, int32_t c) {
    constexpr uint8_t sf_a = SCALE_FACTOR_E8M0_A;
    constexpr uint8_t sf_b = SCALE_FACTOR_E8M0_B;
    int32_t scale_exp_a = (int32_t)sf_a - 133;
    float scale_factor_a = std::ldexp(1.0f, scale_exp_a);
    int32_t scale_exp_b = (int32_t)sf_b - 133;
    float scale_factor_b = std::ldexp(1.0f, scale_exp_b);
    float product = (float)a * scale_factor_a * (float)b * scale_factor_b;
    return (int32_t)product + c;
  }
};

template<typename T>
inline typename T::dtype generate_A_value() {
  if constexpr (std::is_same_v<T, vt::mxfp8>) {
    return Comparator<T>::generate_with_scale(SCALE_FACTOR_E8M0_A);
  } else if constexpr (std::is_same_v<T, vt::nvfp4>) {
    return Comparator<T>::generate_with_scale(SCALE_FACTOR_E4M3_A);
  } else {
    return Comparator<T>::generate();
  }
}

template<typename T>
inline typename T::dtype generate_B_value() {
  if constexpr (std::is_same_v<T, vt::mxfp8>) {
    return Comparator<T>::generate_with_scale(SCALE_FACTOR_E8M0_B);
  } else if constexpr (std::is_same_v<T, vt::nvfp4>) {
    return Comparator<T>::generate_with_scale(SCALE_FACTOR_E4M3_B);
  } else {
    return Comparator<T>::generate();
  }
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

// Get magnitude of element at given offset in A matrix (for pruning comparison)
static float get_element_magnitude(const itype_t *A, uint32_t offset) {
  auto val = data_accessor_t<vt::ITYPE>::read(A, offset);
  if constexpr (std::is_same_v<vt::ITYPE, vt::int8> || std::is_same_v<vt::ITYPE, vt::mxint8>) {
    return std::abs(static_cast<float>(static_cast<int8_t>(val)));
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::uint8>) {
    return static_cast<float>(val);
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::int4>) {
    int32_t sval = val & 0xF;
    if (sval & 0x8) sval |= ~0xF;
    return std::abs(static_cast<float>(sval));
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::uint4>) {
    return static_cast<float>(val & 0xF);
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16>) {
    return std::abs(bit_cast<float>(rv_htof_s(val, 0, nullptr)));
  } else if constexpr (std::is_same_v<vt::ITYPE, vt::bf16>) {
    return std::abs(bit_cast<float>(rv_btof_s(val, 0, nullptr)));
  } else {
    return std::abs(static_cast<float>(val));
  }
}

// Prune matrix A with real 2:4 structured sparsity (top-2 by magnitude per group of 4)
// Zeros pruned elements in-place and stores per-group 4-bit masks
static void prune_2to4(itype_t *A, std::vector<uint8_t> &masks, uint32_t M, uint32_t K) {
  uint32_t subbytes = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  uint32_t num_groups = KS / 4;
  masks.resize(M * num_groups);

  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t k_start = g * 4;

      // Get magnitudes
      float mags[4];
      for (int p = 0; p < 4; ++p) {
        mags[p] = get_element_magnitude(A, m * KS + k_start + p);
      }

      // Find indices of top-2 by magnitude (ties broken by lower index)
      int top[2] = {0, 1};
      if (mags[1] > mags[0]) { top[0] = 1; top[1] = 0; }
      for (int p = 2; p < 4; ++p) {
        if (mags[p] > mags[top[0]]) {
          top[1] = top[0];
          top[0] = p;
        } else if (mags[p] > mags[top[1]]) {
          top[1] = p;
        }
      }

      // Build mask and zero pruned elements
      uint8_t mask = (1 << top[0]) | (1 << top[1]);
      masks[m * num_groups + g] = mask;
      for (int p = 0; p < 4; ++p) {
        if (!(mask & (1 << p))) {
          data_accessor_t<vt::ITYPE>::write(A, m * KS + k_start + p, 0);
        }
      }
    }
  }
}

// Compress pruned A (M x K) to M x K/2 using per-group masks
static void compress_2to4(itype_t *compressed, const itype_t *pruned_A,
                           const std::vector<uint8_t> &masks, uint32_t M, uint32_t K) {
  uint32_t subbytes = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  uint32_t stride_comp = KS / 2;
  uint32_t num_groups = KS / 4;

  for (uint32_t m = 0; m < M; ++m) {
    uint32_t a_out = 0;
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t k_start = g * 4;
      uint8_t mask = masks[m * num_groups + g];
      for (uint32_t k2 = 0; k2 < 4; ++k2) {
        if (mask & (1 << k2)) {
          auto val = data_accessor_t<vt::ITYPE>::read(pruned_A, m * KS + k_start + k2);
          data_accessor_t<vt::ITYPE>::write(compressed, m * stride_comp + a_out, val);
          a_out++;
        }
      }
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

  uint32_t subbytes = (vt::ITYPE::bits < 8) ? (8 / vt::ITYPE::bits) : 0;
  uint32_t tileK_elem = subbytes ? (cfg::tileK * subbytes) : cfg::tileK;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  uint32_t num_groups_per_row = KS / 4;
  uint32_t elts_per_sparse_step = tileK_elem / half_k_steps;

  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;
  uint32_t per_k_tile_words = PD * mcols;

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

            for (uint32_t g = 0; g < groups_in_step; ++g) {
              uint32_t global_group = (k_elem_start / 4) + g;
              uint8_t mask = masks[physical_row * num_groups_per_row + global_group];

              for (int p = 0; p < 4; ++p) {
                if (mask & (1 << p)) {
                  // Map element position to meta_row bit position
                  uint32_t elt = g * 4 + p;
                  uint32_t k_reg = elt / (2 * I_RATIO);
                  uint32_t pos_in_k = elt % (2 * I_RATIO);
                  uint32_t meta_bit;
                  if (pos_in_k < I_RATIO) {
                    meta_bit = k_reg * I_RATIO + pos_in_k;
                  } else {
                    meta_bit = (TC_K + k_reg) * I_RATIO + (pos_in_k - I_RATIO);
                  }
                  uint32_t block_bit = i * meta_row_w + meta_bit;
                  uint32_t word_idx = block_bit / 32;
                  uint32_t bit_idx = block_bit % 32;
                  h_meta[section_base + sram_row + word_idx * PD] |= (1u << bit_idx);
                }
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

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sgemm TCU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n N] [-k: K] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:i:o:hs")) != -1) {
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

  // allocate metadata buffer per (tile_row, k_tile)
  constexpr uint32_t meta_cols = cfg::meta_cols;
  uint32_t num_tile_rows = M / cfg::tileM;
  uint32_t num_k_tiles = K / cfg::tileK;
  constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
  uint32_t meta_buf_entries = num_tile_rows * num_k_tiles * PD * meta_cols;
  RT_CHECK(vx_mem_alloc(device, meta_buf_entries * sizeof(uint32_t), VX_MEM_READ, &meta_buffer));
  RT_CHECK(vx_mem_address(meta_buffer, &kernel_arg.meta_addr));

  uint32_t num_blocks = kernel_arg.grid_dim[0] * kernel_arg.grid_dim[1];
  RT_CHECK(vx_mem_alloc(device, num_blocks * sizeof(uint32_t), VX_MEM_WRITE, &cycles_buffer));
  RT_CHECK(vx_mem_address(cycles_buffer, &kernel_arg.tcu_cycles_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;
  std::cout << "meta_addr=0x" << std::hex << kernel_arg.meta_addr << std::endl;

  // generate source data
  // Generate full matrix A (M × K), prune in-place, then compress to M × K/2
  std::vector<itype_t> h_A_full(sizeA_full);
  for (uint32_t i = 0; i < sizeA_full; ++i) {
    h_A_full[i] = generate_A_value<vt::ITYPE>();
  }
  std::vector<uint8_t> masks;
  prune_2to4(h_A_full.data(), masks, M, K);
  std::vector<itype_t> h_A(sizeA);
  compress_2to4(h_A.data(), h_A_full.data(), masks, M, K);

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
                  std::is_same<vt::ITYPE, vt::uint4>::value ||
                  std::is_same<vt::ITYPE, vt::nvfp4>::value) {
      // sub-byte matrix B must be in col-major format
      // we convert the 4-bit row-major to col-major here
      std::vector<uint8_t> h_B_col(sizeB);
      convert_row_to_col_major_4bit(h_B_col.data(), N, 2 * K, (uint8_t*)h_B.data());
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB));
    } else {
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));
    }
  }

  // upload metadata buffer (real masks from pruning)
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

  // download TCU K-loop cycle counts
  {
    std::vector<uint32_t> h_cycles(num_blocks);
    RT_CHECK(vx_copy_from_dev(h_cycles.data(), cycles_buffer, 0, num_blocks * sizeof(uint32_t)));
    uint32_t max_cyc = 0;
    for (uint32_t i = 0; i < num_blocks; ++i) {
      if (h_cycles[i] > max_cyc) max_cyc = h_cycles[i];
    }
    printf("TCU_CYCLES: max=%u (across %u blocks)\n", max_cyc, num_blocks);
  }

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
    uint32_t per_k_words_d = PD_d * mcols_d;
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
      printf("WARNING: sparse_ref vs cpu_ref: %d / %u mismatches!\n", sparse_ref_errors, sizeC);
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
      printf("GPU vs sparse_ref: %d / %u mismatches\n", gpu_vs_sparse, sizeC);
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