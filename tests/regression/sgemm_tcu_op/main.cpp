
#include "common.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <rvfloats.h>
#include <sstream>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>
#include <dxa.h>
#include <cstring>

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

static std::vector<itype_t> pack_A_colmajor_tiled32(const std::vector<itype_t>& A,
                                                    uint32_t M,
                                                    uint32_t K,
                                                    uint32_t tile_M = 32,
                                                    uint32_t tile_K = cfg::tileK) {
  std::vector<itype_t> packed(A.size());
  uint32_t tile_id = 0;

  for (uint32_t tile_row = 0; tile_row < M; tile_row += tile_M) {
    for (uint32_t tile_col = 0; tile_col < K; tile_col += tile_K, ++tile_id) {
      const uint32_t tile_base = tile_id * tile_M * tile_K;
      for (uint32_t idx = 0; idx < tile_M * tile_K; ++idx) {
        const uint32_t col = idx / tile_M;
        const uint32_t row = idx % tile_M;
        packed[tile_base + idx] = A[(tile_row + row) * K + tile_col + col];
      }
    }
  }

  return packed;
}

static std::vector<itype_t> pack_B_rowmajor_tiled32(const std::vector<itype_t>& B,
                                                    uint32_t K,
                                                    uint32_t N,
                                                    uint32_t tile_K = cfg::tileK,
                                                    uint32_t tile_N = 32) {
  std::vector<itype_t> packed(B.size());
  uint32_t tile_id = 0;

  for (uint32_t tile_col = 0; tile_col < N; tile_col += tile_N) {
    for (uint32_t tile_row = 0; tile_row < K; tile_row += tile_K, ++tile_id) {
      const uint32_t tile_base = tile_id * tile_K * tile_N;
      for (uint32_t idx = 0; idx < tile_K * tile_N; ++idx) {
        const uint32_t row = idx / tile_N;
        const uint32_t col = idx % tile_N;
        packed[tile_base + idx] = B[(tile_row + row) * N + tile_col + col];
      }
    }
  }

  return packed;
}

static std::vector<otype_t> pack_C_blocked_tiled32(const std::vector<otype_t>& C,
                                                   uint32_t M,
                                                   uint32_t N) {
  constexpr uint32_t TILE_M = 32;
  constexpr uint32_t TILE_N = 32;
  constexpr uint32_t BLOCK_M = 4;
  constexpr uint32_t BLOCK_N = 8;
  const uint32_t blk_cols = TILE_N / BLOCK_N;
  const uint32_t elems_per_tile = TILE_M * TILE_N;
  std::vector<otype_t> packed(C.size());
  uint32_t tile_id = 0;

  for (uint32_t tile_row = 0; tile_row < M; tile_row += TILE_M) {
    for (uint32_t tile_col = 0; tile_col < N; tile_col += TILE_N, ++tile_id) {
      const uint32_t tile_base = tile_id * elems_per_tile;
      for (uint32_t dest_idx = 0; dest_idx < elems_per_tile; ++dest_idx) {
        const uint32_t dest_block = dest_idx / (BLOCK_M * BLOCK_N);
        const uint32_t elem_in_block = dest_idx % (BLOCK_M * BLOCK_N);
        const uint32_t block_row = dest_block / blk_cols;
        const uint32_t block_col = dest_block % blk_cols;
        const uint32_t elem_row = elem_in_block / BLOCK_N;
        const uint32_t elem_col = elem_in_block % BLOCK_N;
        const uint32_t src_row = tile_row + block_row * BLOCK_M + elem_row;
        const uint32_t src_col = tile_col + block_col * BLOCK_N + elem_col;

        packed[tile_base + dest_idx] = C[src_row * N + src_col];
      }
    }
  }

  return packed;
}

static std::vector<otype_t> unpack_D_tiled32_rowmajor(const std::vector<otype_t>& D_tiled,
                                                      uint32_t M,
                                                      uint32_t N) {
  constexpr uint32_t TILE_M = 32;
  constexpr uint32_t TILE_N = 32;
  const uint32_t elems_per_tile = TILE_M * TILE_N;
  std::vector<otype_t> unpacked(D_tiled.size());
  uint32_t tile_id = 0;

  for (uint32_t tile_row = 0; tile_row < M; tile_row += TILE_M) {
    for (uint32_t tile_col = 0; tile_col < N; tile_col += TILE_N, ++tile_id) {
      const uint32_t tile_base = tile_id * elems_per_tile;
      for (uint32_t idx = 0; idx < elems_per_tile; ++idx) {
        const uint32_t row = idx / TILE_N;
        const uint32_t col = idx % TILE_N;
        unpacked[(tile_row + row) * N + tile_col + col] = D_tiled[tile_base + idx];
      }
    }
  }

  return unpacked;
}


static void matmul_cpu(otype_t *D, const itype_t *A, const itype_t *B, otype_t *C, uint32_t M, uint32_t N, uint32_t K) {
  uint32_t subbytes = 8 / vt::ITYPE::bits;
  uint32_t KS = subbytes ? (K * subbytes) : K;
  for (uint32_t m = 0; m < M; ++m) {
    for (uint32_t n = 0; n < N; ++n) {
      otype_t sum = data_accessor_t<vt::OTYPE>::read(C, m * N + n);
      for (uint32_t k = 0; k < KS; ++k) {
        auto a = data_accessor_t<vt::ITYPE>::read(A, m * KS + k);
        auto b = data_accessor_t<vt::ITYPE>::read(B, k * N + n);
        sum = muladd_t<vt::ITYPE, vt::OTYPE>::eval(a, b, sum);
      }
      data_accessor_t<vt::OTYPE>::write(D, m * N + n, sum);
    }
  }
}


///////////////////////////////////////////////////////////////////////////////

const char *kernel_file = "kernel.vxbin";

uint32_t xm = 32;
uint32_t xn = 32;
uint32_t xk = 32;

int sparsity = 0;
float a_sparsity = 0.0;
float b_sparsity = 0.0;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h D_buffer = nullptr;
vx_buffer_h A_bitmap_buffer = nullptr;
vx_buffer_h B_bitmap_buffer = nullptr;
vx_buffer_h A_nz_buffer = nullptr;
vx_buffer_h B_nz_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};
static constexpr uint32_t kDescA = 0;
static constexpr uint32_t kDescB = 1;
static constexpr uint32_t kDescC = 2;
static constexpr uint32_t kDescABitmap = 3;
static constexpr uint32_t kDescBBitmap = 4;

std::string last_build_options;

static void show_usage() {
  std::cout << "Vortex Sgemm TCU Test." << std::endl;
  std::cout << "Usage: [-m: m] [-n: N] [-k: K] [-s: Sparsity_mode] [-a: A_sparsity (0.0 - 1.0)] [-b: B_sparsity (0.0 - 1.0)] [-h: help]" << std::endl;
  std::cout << "  -s  Sparsity Modes: 0: Dense x Dense   1: Dense x Sparse   2: Sparse x Sparse" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:i:o:s:a:b:h")) != -1) {
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
    case 's':
      sparsity = atoi(optarg);
      std::cout << "Sparse mode enabled (-s) with sparsity level: " << (int)sparsity << std::endl;
      break;
    case 'a':
      a_sparsity = atof(optarg);
      std::cout << "A sparsity level set to: " << a_sparsity << std::endl;
      break;
    case 'b':
      b_sparsity = atof(optarg);
      std::cout << "B sparsity level set to: " << b_sparsity << std::endl;
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

template <typename T>
static void apply_pruning(std::vector<T>& matrix,
                          uint32_t rows,
                          uint32_t cols,
                          float sparsity_level,
                          char mode) {
  auto clamp_prob = [](float p) -> float {
    if (p < 0.0f) return 0.0f;
    if (p > 1.0f) return 1.0f;
    return p;
  };

  const float prune_prob = clamp_prob(sparsity_level);
  if (mode == 'u') {
    for (uint32_t idx = 0; idx < rows * cols; ++idx) {
      const float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      if (r < prune_prob) {
        data_accessor_t<vt::ITYPE>::write(matrix.data(), idx, 0);
      }
    }
    return;
  }

  if (mode == 'c') {
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t col = 0; col < cols; ++col) {
        if (((row + col) & 1u) != 0) {
          data_accessor_t<vt::ITYPE>::write(matrix.data(), row * cols + col, 0);
        }
      }
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(D_buffer);
    vx_mem_free(A_bitmap_buffer);
    vx_mem_free(B_bitmap_buffer);
    vx_mem_free(A_nz_buffer);
    vx_mem_free(B_nz_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

static void print_2d_input_matrix(const std::vector<itype_t>& A, int M, int N)
{
  size_t zeros = std::count(A.begin(), A.end(), 0);
  std::cout << "Actual sparsity: " << (zeros * 100.0 / (M * N)) << "%" << std::endl;
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      auto v = A[i * N + j];
      if constexpr (std::is_same_v<vt::ITYPE, vt::fp32>) {
        printf("0x%08x ", bit_cast<uint32_t>(v));
      } else if constexpr (std::is_same_v<vt::ITYPE, vt::fp16> || std::is_same_v<vt::ITYPE, vt::bf16>) {
        printf("0x%04x ", (uint32_t)v);
      } else if constexpr (std::is_same_v<vt::ITYPE, vt::int8> || std::is_same_v<vt::ITYPE, vt::uint8>) {
        printf("0x%02x ", (uint32_t)(uint8_t)v);
      } else if constexpr (std::is_same_v<vt::ITYPE, vt::int4> || std::is_same_v<vt::ITYPE, vt::uint4>) {
        printf("0x%01x ", (uint32_t)(uint8_t)v);
      } else if constexpr (std::is_same_v<vt::ITYPE, vt::fp8>) {
        printf("0x%02x ", (uint32_t)(uint8_t)v);
      } else {
        printf("%d ", v);
      }
    }
    printf("\n");
  }
}

static void print_2d_output_matrix(std::vector<otype_t> C, int M, int N, std::vector<otype_t> Bias)
{
  if (C != Bias)
  {
    size_t zeros = 0;
    for (size_t i = 0; i < C.size(); ++i) {
        if (C[i] - Bias[i] == 0)
            ++zeros;
    }

    std::cout << "Mult sparsity: " << (zeros * 100.0 / (M * N)) << "%" << std::endl;
  }

  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      auto v = C[i * N + j];
      if constexpr (std::is_same_v<vt::OTYPE, vt::fp32>) {
        printf("0x%08x ", bit_cast<uint32_t>(v));
      } else if constexpr (std::is_same_v<vt::OTYPE, vt::int32>) {
        printf("0x%08x ", (uint32_t)v);
      } else {
        printf("%d ", (uint32_t)v);
      }
    }
    printf("\n");
  }
}

static std::vector<uint32_t> build_A_tile_offsets(const std::vector<itype_t>& matrix,
                                                  uint32_t M,
                                                  uint32_t K,
                                                  uint32_t tile_M = 32,
                                                  uint32_t tile_K = cfg::tileK) {
  const uint32_t tile_rows = M / tile_M;
  const uint32_t tile_cols = K / tile_K;
  std::vector<uint32_t> metadata(tile_rows * tile_cols, 0);
  uint32_t compressed_offset = 0;

  // A compressed payload is tile-row-major. Inside each tile, values stay column-major.
  for (uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (uint32_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
      for (uint32_t col = 0; col < tile_K; ++col) {
        for (uint32_t row = 0; row < tile_M; ++row) {
          const uint32_t idx = (tile_row * tile_M + row) * K
                             + (tile_col * tile_K + col);
          if (data_accessor_t<vt::ITYPE>::read(matrix.data(), idx) != 0) {
            ++compressed_offset;
          }
        }
      }
      metadata[tile_row * tile_cols + tile_col] = compressed_offset;
    }
  }

  return metadata;
}

static std::vector<uint32_t> build_B_tile_offsets(const std::vector<itype_t>& matrix,
                                                  uint32_t K,
                                                  uint32_t N,
                                                  uint32_t tile_K = cfg::tileK,
                                                  uint32_t tile_N = 32) {
  const uint32_t tile_rows = K / tile_K;
  const uint32_t tile_cols = N / tile_N;
  std::vector<uint32_t> metadata(tile_rows * tile_cols, 0);
  uint32_t compressed_offset = 0;

  // B compressed payload is tile-column-major. Inside each tile, values stay row-major.
  for (uint32_t tile_col = 0; tile_col < tile_cols; ++tile_col) {
    for (uint32_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
      for (uint32_t row = 0; row < tile_K; ++row) {
        for (uint32_t col = 0; col < tile_N; ++col) {
          const uint32_t idx = (tile_row * tile_K + row) * N
                             + (tile_col * tile_N + col);
          if (data_accessor_t<vt::ITYPE>::read(matrix.data(), idx) != 0) {
            ++compressed_offset;
          }
        }
      }
      metadata[tile_col * tile_rows + tile_row] = compressed_offset;
    }
  }

  return metadata;
}

static void print_tile_offset_matrix(const char* name,
                                     const std::vector<uint32_t>& metadata,
                                     uint32_t rows,
                                     uint32_t cols,
                                     bool col_major_storage) {
  std::cout << name << " tile offsets (" << rows << "x" << cols << "):" << std::endl;
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < cols; ++col) {
      const uint32_t idx = col_major_storage ? (col * rows + row)
                                             : (row * cols + col);
      printf("%4u ", metadata[idx]);
    }
    printf("\n");
  }
}

// Bitmap helpers: pack bits LSB-first into bytes
struct bitmap_builder_t {
  std::vector<uint8_t> data;
  uint32_t bit_count = 0;

  void push(bool set) {
    if ((bit_count & 7u) == 0) {
      data.push_back(0);
    }
    if (set) {
      data.back() |= static_cast<uint8_t>(1u << (bit_count & 7u));
    }
    ++bit_count;
  }
};

static std::vector<uint8_t> build_bitmap_A_colmajor_tiled32(const std::vector<itype_t>& A,
                                                            uint32_t M, uint32_t K) {
  bitmap_builder_t builder;
  builder.data.reserve((M * K + 7) / 8);

  for (uint32_t row_block = 0; row_block < M; row_block += 32) {
    for (uint32_t col = 0; col < K; ++col) {
      for (uint32_t r = 0; r < 32 && (row_block + r) < M; ++r) {
        uint32_t row = row_block + r;
        uint32_t idx = row * K + col;
        auto val = data_accessor_t<vt::ITYPE>::read(A.data(), idx);
        builder.push(val != 0);
      }
    }
  }
  return builder.data;
}

static std::vector<uint8_t> build_bitmap_B_rowmajor_tiled32N(const std::vector<itype_t>& B,
                                                             uint32_t K, uint32_t N) {
  bitmap_builder_t builder;
  builder.data.reserve((K * N + 7) / 8);
  for (uint32_t col_block = 0; col_block < N; col_block += 32) {
    for (uint32_t row = 0; row < K; ++row) {
      for (uint32_t c = 0; c < 32 && (col_block + c) < N; ++c) {
        uint32_t col = col_block + c;
        uint32_t idx = row * N + col;
        auto val = data_accessor_t<vt::ITYPE>::read(B.data(), idx);
        builder.push(val != 0);
      }
    }
  }
  return builder.data;
}

static void trace_bitmap(const char* name, const std::vector<uint8_t>& bitmap) {
  std::cout << name << " bitmap bytes=" << bitmap.size() << std::endl;
  constexpr size_t kPerLine = 16;
  for (size_t i = 0; i < bitmap.size(); i += kPerLine) {
    std::ostringstream oss;
    oss << name << " bitmap[" << std::dec << i << "]: ";
    for (size_t j = 0; j < kPerLine && (i + j) < bitmap.size(); ++j) {
      if (j != 0) oss << ' ';
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<unsigned>(bitmap[i + j]);
    }
    std::cout << oss.str() << std::endl;
  }
}

static bool validate_A_sparse_tile_lmem_budget(const std::vector<itype_t>& A,
                                               uint32_t M,
                                               uint32_t K,
                                               uint32_t tile_M = 32,
                                               uint32_t tile_K = cfg::tileK) {
  const uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype_t);
  const uint32_t reserved_regs_per_tile = tile_M * tile_K / i_ratio;
  const uint32_t reserved_bytes_per_tile = reserved_regs_per_tile * sizeof(uint32_t);
  const uint32_t bitmap_bytes_per_tile = tile_K * sizeof(uint32_t);
  bool ok = true;

  for (uint32_t tile_row = 0; tile_row < M; tile_row += tile_M) {
    for (uint32_t tile_col = 0; tile_col < K; tile_col += tile_K) {
      uint32_t nonzeros = 0;
      for (uint32_t col = 0; col < tile_K; ++col) {
        for (uint32_t row = 0; row < tile_M; ++row) {
          const uint32_t idx = (tile_row + row) * K + (tile_col + col);
          if (data_accessor_t<vt::ITYPE>::read(A.data(), idx) != 0) {
            ++nonzeros;
          }
        }
      }

      const uint32_t compressed_bytes = nonzeros * sizeof(itype_t);
      const uint32_t tile_total_bytes = compressed_bytes + bitmap_bytes_per_tile;
      if (tile_total_bytes > reserved_bytes_per_tile) {
        std::cerr << "A sparse tile LMEM overflow: tile_row=" << (tile_row / tile_M)
                  << ", tile_col=" << (tile_col / tile_K)
                  << ", compressed_bytes=" << compressed_bytes
                  << ", bitmap_bytes=" << bitmap_bytes_per_tile
                  << ", total_bytes=" << tile_total_bytes
                  << ", reserved_bytes=" << reserved_bytes_per_tile
                  << " (tile_M * tile_K / i_ratio = " << reserved_regs_per_tile
                  << " regs)" << std::endl;
        ok = false;
      }
    }
  }

  return ok;
}

static bool validate_B_sparse_tile_lmem_budget(const std::vector<itype_t>& B,
                                               uint32_t K,
                                               uint32_t N,
                                               uint32_t tile_K = cfg::tileK,
                                               uint32_t tile_N = 32) {
  const uint32_t i_ratio = sizeof(uint32_t) / sizeof(itype_t);
  const uint32_t reserved_regs_per_tile = tile_K * tile_N / i_ratio;
  const uint32_t reserved_bytes_per_tile = reserved_regs_per_tile * sizeof(uint32_t);
  const uint32_t bitmap_bytes_per_tile = tile_K * sizeof(uint32_t);
  bool ok = true;

  for (uint32_t tile_col = 0; tile_col < N; tile_col += tile_N) {
    for (uint32_t tile_row = 0; tile_row < K; tile_row += tile_K) {
      uint32_t nonzeros = 0;
      for (uint32_t row = 0; row < tile_K; ++row) {
        for (uint32_t col = 0; col < tile_N; ++col) {
          const uint32_t idx = (tile_row + row) * N + (tile_col + col);
          if (data_accessor_t<vt::ITYPE>::read(B.data(), idx) != 0) {
            ++nonzeros;
          }
        }
      }

      const uint32_t compressed_bytes = nonzeros * sizeof(itype_t);
      const uint32_t tile_total_bytes = compressed_bytes + bitmap_bytes_per_tile;
      if (tile_total_bytes > reserved_bytes_per_tile) {
        std::cerr << "B sparse tile LMEM overflow: tile_row=" << (tile_row / tile_K)
                  << ", tile_col=" << (tile_col / tile_N)
                  << ", compressed_bytes=" << compressed_bytes
                  << ", bitmap_bytes=" << bitmap_bytes_per_tile
                  << ", total_bytes=" << tile_total_bytes
                  << ", reserved_bytes=" << reserved_bytes_per_tile
                  << " (tile_K * tile_N / i_ratio = " << reserved_regs_per_tile
                  << " regs)" << std::endl;
        ok = false;
      }
    }
  }

  return ok;
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

#ifdef TCU_DISABLE_S1
  if (sparsity != 0 && sparsity != 2) {
    std:: cout << "Error: Sparsity mode: " << sparsity << " is not supported!\n";
    return 1;
  }
#else
  if (sparsity != 0 && sparsity != 1 && sparsity != 2) {
    std:: cout << "Error: Sparsity mode: " << sparsity << " is not supported!\n";
    return 1;
  }
#endif

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;
  const uint32_t input_tile_k = cfg::tileK;

  if ((M % 32) != 0) {
    std::cout << "Error: M must be a multiple of 32!" << std::endl;
    return -1;
  }

  if ((N % 32) != 0) {
    std::cout << "Error: N must be a multiple of 32!" << std::endl;
    return -1;
  }

  if ((K % input_tile_k) != 0) {
    std::cout << "Error: K must be a multiple of WMMA tileK (" << input_tile_k << ")!" << std::endl;
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;
  size_t sizeD = M * N;

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Core Dimension: M=" << cfg::tcM << ", N=" << cfg::tcN << ", K=" << cfg::tcK << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  uint32_t grid_dim[2]  = {1, 1};
  uint32_t block_dim[2] = {(uint32_t)NT, 1};

  // set matrix dimensions
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  // Set sparsity mode
  kernel_arg.sparsity = sparsity;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, sizeD * sizeof(otype_t), VX_MEM_WRITE, &D_buffer));
  RT_CHECK(vx_mem_address(D_buffer, &kernel_arg.D_addr));

  std::cout << "A_addr=0x" << std::hex << kernel_arg.A_addr << std::endl;
  std::cout << "B_addr=0x" << std::hex << kernel_arg.B_addr << std::endl;
  std::cout << "C_addr=0x" << std::hex << kernel_arg.C_addr << std::endl;
  std::cout << "D_addr=0x" << std::hex << kernel_arg.D_addr << std::endl;

  // generate source data
  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  std::vector<otype_t> h_C(sizeC);
  std::vector<itype_t> h_A_packed;
  std::vector<itype_t> h_B_packed;
  std::vector<otype_t> h_C_packed;
  std::vector<uint32_t> h_A_nz;
  std::vector<uint32_t> h_B_nz;
  uint32_t h_A_bitmap_words = 0;
  uint32_t h_B_bitmap_words = 0;
  // std::vector<otype_t> h_D(sizeD);

  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = generate_A_value<vt::ITYPE>();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = generate_B_value<vt::ITYPE>();
  }

#ifdef TCU_DISABLE_S1
  std::cout << "TCU s1 Sparsity mode is disabled!\n";
#endif

  if (sparsity >= 1) {
    apply_pruning(h_A, M, K, a_sparsity, 'u');
    apply_pruning(h_B, K, N, b_sparsity, 'u');
    std::cout << "Applied pruning with mode=c"
              << ", a_sparsity=" << a_sparsity
              << ", b_sparsity=" << b_sparsity << std::endl;
  }
  // h_A[0] = 0;
  // h_B[0] = 0;

  const uint32_t a_nz_rows = M / 32;
  const uint32_t a_nz_cols = K / input_tile_k;
  const uint32_t b_nz_rows = K / input_tile_k;
  const uint32_t b_nz_cols = N / 32;

  if (sparsity == 2) {
    h_A_nz = build_A_tile_offsets(h_A, M, K, 32, input_tile_k);
  }
  if (sparsity >= 1) {
    h_B_nz = build_B_tile_offsets(h_B, K, N, input_tile_k, 32);
  }

  if (sparsity == 2
   && !validate_A_sparse_tile_lmem_budget(h_A, M, K, 32, input_tile_k)) {
    cleanup();
    return -1;
  }
  if (sparsity >= 1
   && !validate_B_sparse_tile_lmem_budget(h_B, K, N, input_tile_k, 32)) {
    cleanup();
    return -1;
  }

  /* Sparsity levels: 
     0: Both A, B matrices are dense
     1: A is dense (uncompressed), B is sparse (compressed + bitmap)
     2: Both A and B are sparse (compressed + bitmap) */ 

  if (sparsity == 2) {
    std::vector<uint8_t> h_A_bitmap = build_bitmap_A_colmajor_tiled32(h_A, M, K);
    h_A_bitmap_words = h_A_bitmap.size() / sizeof(uint32_t);
    std::cout << "A bitmap bytes: " << h_A_bitmap.size() << " (bits=" << (M * K) << ")" << std::endl;
    trace_bitmap("A", h_A_bitmap);

    std::cout << "upload A bitmap buffer" << std::endl;
    RT_CHECK(vx_mem_alloc(device, h_A_bitmap.size(), VX_MEM_READ, &A_bitmap_buffer));
    RT_CHECK(vx_mem_address(A_bitmap_buffer, &kernel_arg.A_bitmap_addr));
    RT_CHECK(vx_copy_to_dev(A_bitmap_buffer, h_A_bitmap.data(), 0, h_A_bitmap.size()));

    std::cout << "upload A non-zeros buffer" << std::endl;
    RT_CHECK(vx_mem_alloc(device, h_A_nz.size() * sizeof(uint32_t), VX_MEM_READ, &A_nz_buffer));
    RT_CHECK(vx_mem_address(A_nz_buffer, &kernel_arg.A_nz_addr));
    RT_CHECK(vx_copy_to_dev(A_nz_buffer, h_A_nz.data(), 0, h_A_nz.size() * sizeof(uint32_t)));
  }
  if (sparsity >= 1) {
    std::vector<uint8_t> h_B_bitmap = build_bitmap_B_rowmajor_tiled32N(h_B, K, N);
    h_B_bitmap_words = h_B_bitmap.size() / sizeof(uint32_t);
    std::cout << "B bitmap bytes: " << h_B_bitmap.size() << " (bits=" << (K * N) << ")" << std::endl;
    trace_bitmap("B", h_B_bitmap);

    std::cout << "upload B bitmap buffer" << std::endl;
    RT_CHECK(vx_mem_alloc(device, h_B_bitmap.size(), VX_MEM_READ, &B_bitmap_buffer));
    RT_CHECK(vx_mem_address(B_bitmap_buffer, &kernel_arg.B_bitmap_addr));
    RT_CHECK(vx_copy_to_dev(B_bitmap_buffer, h_B_bitmap.data(), 0, h_B_bitmap.size()));

    std::cout << "upload B metadata buffer" << std::endl;
    RT_CHECK(vx_mem_alloc(device, h_B_nz.size() * sizeof(uint32_t), VX_MEM_READ, &B_nz_buffer));
    RT_CHECK(vx_mem_address(B_nz_buffer, &kernel_arg.B_nz_addr));
    RT_CHECK(vx_copy_to_dev(B_nz_buffer, h_B_nz.data(), 0, h_B_nz.size() * sizeof(uint32_t)));
  } 
  if (sparsity == 0) {
    kernel_arg.A_bitmap_addr = 0;
    kernel_arg.B_bitmap_addr = 0;
    kernel_arg.A_nz_addr = 0;
    kernel_arg.B_nz_addr = 0;
  } else if (sparsity == 1) {
    kernel_arg.A_nz_addr = 0;
  }
  
  // Fill C with a constant stride of 0x8000 in bit-pattern space
  static_assert(sizeof(otype_t) == sizeof(uint32_t), "C pattern fill assumes 32-bit output type");
  for (uint32_t i = 0; i < sizeC; ++i) {
    uint32_t bits = 0x43800000 + (i * 0x8000); // start at 256.0f, step in 0x8000 increments
    otype_t tmp;
    std::memcpy(&tmp, &bits, sizeof(tmp));  // bitwise copy, no conversion
    h_C[i] = tmp;
  }

  std::cout << "Matrix A:" << std::endl;
  print_2d_input_matrix(h_A, M, K);

  std::cout << "Matrix B:" << std::endl;
  print_2d_input_matrix(h_B, K, N);

  if (sparsity == 2) {
    print_tile_offset_matrix("A", h_A_nz, a_nz_rows, a_nz_cols, false);
  }
  if (sparsity >= 1) {
    print_tile_offset_matrix("B", h_B_nz, b_nz_rows, b_nz_cols, true);
  }

  std::cout << "Matrix C:" << std::endl;
  print_2d_output_matrix(h_C, M, N, h_C);
  if (sparsity != 2) {
    h_A_packed = pack_A_colmajor_tiled32(h_A, M, K, 32, input_tile_k);
  }
  if (sparsity == 0) {
    h_B_packed = pack_B_rowmajor_tiled32(h_B, K, N, input_tile_k, 32);
  }
  h_C_packed = pack_C_blocked_tiled32(h_C, M, N);


  std::vector<itype_t> h_A_compressed;
  std::vector<itype_t> h_B_compressed;

  if (sparsity == 2) {
    h_A_compressed.reserve(M * K);
    for (uint32_t tile_row = 0; tile_row < M; tile_row += 32) {
      for (uint32_t tile_col = 0; tile_col < K; tile_col += 32) {
        for (uint32_t col = 0; col < 32; ++col) {
          for (uint32_t row = 0; row < 32; ++row) {
            uint32_t idx = (tile_row + row) * K + (tile_col + col);
            auto val = data_accessor_t<vt::ITYPE>::read(h_A.data(), idx);
            if (val != 0) {
              h_A_compressed.push_back(static_cast<itype_t>(val));
            }
          }
        }
      }
    }

    std::cout << "Compressed A (column-major, non-zero only), count="
              << h_A_compressed.size() << std::endl;
    for (size_t i = 0; i < h_A_compressed.size(); ++i) {
      printf("0x%04x ", static_cast<uint32_t>(h_A_compressed[i]));
    }
    printf("\n");
  }
  if (sparsity >= 1) {
    h_B_compressed.reserve(K * N);
    for (uint32_t tile_col = 0; tile_col < N; tile_col += 32) {
      for (uint32_t tile_row = 0; tile_row < K; tile_row += 32) {
        for (uint32_t row = 0; row < 32; ++row) {
          for (uint32_t col = 0; col < 32; ++col) {
            uint32_t idx = (tile_row + row) * N + (tile_col + col);
            auto val = data_accessor_t<vt::ITYPE>::read(h_B.data(), idx);
            if (val != 0) {
              h_B_compressed.push_back(static_cast<itype_t>(val));
            }
          }
        }
      }
    }

    std::cout << "Compressed B (row-major, non-zero only), count="
              << h_B_compressed.size() << std::endl;
    for (size_t i = 0; i < h_B_compressed.size(); ++i) {
      printf("0x%04x ", static_cast<uint32_t>(h_B_compressed[i]));
    }
    printf("\n");
  }

  kernel_arg.A_compressed_blocks = (sparsity == 2) ? (h_A_compressed.size() * sizeof(itype_t) + 4*32 - 1) / (4*32) : 0; // number of 128B blocks for compressed A (column-major, non-zero only)
  kernel_arg.B_compressed_blocks = (sparsity >= 1) ? (h_B_compressed.size() * sizeof(itype_t) + 4*32 - 1) / (4*32) : 0; // number of 128B blocks for compressed B (row-major, non-zero only)
  std::cout << "Compressed A blocks: " << kernel_arg.A_compressed_blocks << std::endl;
  std::cout << "Compressed B blocks: " << kernel_arg.B_compressed_blocks << std::endl;

  // upload matrix A buffer
  {
    std::cout << "upload matrix A buffer" << std::endl;
    if (sparsity == 2) {
      RT_CHECK(vx_copy_to_dev(A_buffer, h_A_compressed.data(), 0, h_A_compressed.size() * sizeof(itype_t)));
    }
    else {
      RT_CHECK(vx_copy_to_dev(A_buffer, h_A_packed.data(), 0, sizeA * sizeof(itype_t)));
    }
  }

  // upload matrix B buffer
  {
    std::cout << "upload matrix B buffer" << std::endl;
    if constexpr (std::is_same<vt::ITYPE, vt::int4>::value || std::is_same<vt::ITYPE, vt::uint4>::value) {
      // sub-byte matrix B must be in col-major format
      // we convert the 4-bit row-major to col-major here
      std::vector<uint8_t> h_B_col(sizeB);
      convert_row_to_col_major_4bit(h_B_col.data(), N, 2 * K, (uint8_t*)h_B.data());
      RT_CHECK(vx_copy_to_dev(B_buffer, h_B_col.data(), 0, sizeB));
    } else {
      if (sparsity >= 1) {
        RT_CHECK(vx_copy_to_dev(B_buffer, h_B_compressed.data(), 0, h_B_compressed.size() * sizeof(itype_t)));
      } 
      else {
        RT_CHECK(vx_copy_to_dev(B_buffer, h_B_packed.data(), 0, sizeB * sizeof(itype_t)));
      }
    }
  }

  {
    std::cout << "upload matrix C buffer" << std::endl;
    RT_CHECK(vx_copy_to_dev(C_buffer, h_C_packed.data(), 0, sizeC * sizeof(otype_t)));
  }

  {
    constexpr uint32_t tile_M = 32;
    constexpr uint32_t tile_N = 32;
    const uint32_t tile_c_elems = tile_M * tile_N;
    const uint32_t total_c_tiles = (M / tile_M) * (N / tile_N);

    RT_CHECK(vx_dxa_program_desc_2d(
        device, kDescC, kernel_arg.C_addr,
        tile_c_elems, total_c_tiles,
        tile_c_elems * sizeof(otype_t),
        tile_c_elems, 1,
        sizeof(otype_t)));
  }

	  {
	    constexpr uint32_t tile_M = 32;
	    constexpr uint32_t tile_N = 32;
	    const uint32_t tile_a_elems = tile_M * input_tile_k;
	    const uint32_t total_a_tiles = (M / tile_M) * (K / input_tile_k);
    const uint32_t tile_b_elems = input_tile_k * tile_N;
    const uint32_t total_b_tiles = (N / tile_N) * (K / input_tile_k);

    if (sparsity == 2) {
      RT_CHECK(vx_dxa_program_desc_1d(
          device, kDescA, kernel_arg.A_addr,
          h_A_compressed.size(),
          tile_a_elems,
          sizeof(itype_t)));
    } else {
      RT_CHECK(vx_dxa_program_desc_2d(
          device, kDescA, kernel_arg.A_addr,
          tile_a_elems, total_a_tiles,
          tile_a_elems * sizeof(itype_t),
          tile_a_elems, 1,
          sizeof(itype_t)));
    }

	    if (sparsity >= 1) {
	      RT_CHECK(vx_dxa_program_desc_1d(
	          device, kDescB, kernel_arg.B_addr,
	          h_B_compressed.size(),
	          tile_b_elems,
	          sizeof(itype_t)));
	    } else {
	      RT_CHECK(vx_dxa_program_desc_2d(
	          device, kDescB, kernel_arg.B_addr,
	          tile_b_elems, total_b_tiles,
	          tile_b_elems * sizeof(itype_t),
	          tile_b_elems, 1,
	          sizeof(itype_t)));
	    }

	    if (sparsity == 2) {
	      RT_CHECK(vx_dxa_program_desc_1d(
	          device, kDescABitmap, kernel_arg.A_bitmap_addr,
	          h_A_bitmap_words,
	          input_tile_k,
	          sizeof(uint32_t)));
	    }

	    if (sparsity >= 1) {
	      RT_CHECK(vx_dxa_program_desc_1d(
	          device, kDescBBitmap, kernel_arg.B_bitmap_addr,
	          h_B_bitmap_words,
	          input_tile_k,
	          sizeof(uint32_t)));
	    }
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

  // download destination buffer
  std::vector<otype_t> h_D_tiled(sizeD);
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_D_tiled.data(), D_buffer, 0, sizeD * sizeof(otype_t)));
  auto h_D = unpack_D_tiled32_rowmajor(h_D_tiled, M, N);

  std::cout << "Matrix D:" << std::endl;
  print_2d_output_matrix(h_D, M, N, h_C);

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<otype_t> h_ref(sizeD);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), h_C.data(), M, N, K);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<vt::OTYPE>::compare(h_D[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "ERROR: Found " << std::dec << errors << " / " << sizeD << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
