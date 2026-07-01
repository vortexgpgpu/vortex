#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

static constexpr uint32_t kQueryRows = 8;
static constexpr uint32_t kSequenceLength = 32;
static constexpr uint32_t kHeadDimension = 16;
static constexpr uint32_t kQueryStart = 24;
static constexpr uint32_t kKeyGroupSize = 16;
static constexpr uint32_t kLocalMemoryBytes = 1536;

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t output_addr;
  uint32_t causal;
} kernel_arg_t;

static inline uint32_t float_as_bits(float value) {
  union {
    float f;
    uint32_t u;
  } bits = {value};
  return bits.u;
}

static inline float bits_as_float(uint32_t value) {
  union {
    uint32_t u;
    float f;
  } bits = {value};
  return bits.f;
}

static inline uint16_t float_to_half(float value) {
  const uint32_t bits = float_as_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t exponent = (bits >> 23) & 0xffu;
  uint32_t mantissa = bits & 0x7fffffu;

  if (exponent == 0xffu) {
    return static_cast<uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
  }

  int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x800000u;
    const uint32_t shift = static_cast<uint32_t>(14 - half_exponent);
    uint32_t half_mantissa = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
      ++half_mantissa;
    }
    return static_cast<uint16_t>(sign | half_mantissa);
  }

  if (half_exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  mantissa += 0xfffu + ((mantissa >> 13) & 1u);
  uint32_t result = sign + (static_cast<uint32_t>(half_exponent) << 10)
                  + (mantissa >> 13);
  if ((result & 0x7c00u) == 0x7c00u) {
    result = sign | 0x7c00u;
  }
  return static_cast<uint16_t>(result);
}

static inline float half_to_float(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return bits_as_float(sign);
    }
    exponent = 1;
    while ((mantissa & 0x0400u) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 0x03ffu;
    exponent += 127 - 15;
  } else if (exponent == 31) {
    exponent = 255;
  } else {
    exponent += 127 - 15;
  }

  return bits_as_float(sign | (exponent << 23) | (mantissa << 13));
}

#endif
