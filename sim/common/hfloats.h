// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <stdint.h>
#include <cmath>
#include <cstring>

// A minimal IEEE 754 half-precision (16-bit) float implementation
// Provides conversion to/from 32-bit float and basic arithmetic operators.

namespace vortex {

struct half_t {
  uint16_t bits;

  half_t() = default;
  // Construct from float
  half_t(float f) { bits = float_to_half(f); }
  // Convert to float
  operator float() const { return half_to_float(bits); }

  // Arithmetic operators
  friend half_t operator+(half_t a, half_t b) { return half_t((float)a + (float)b); }
  friend half_t operator-(half_t a, half_t b) { return half_t((float)a - (float)b); }
  friend half_t operator*(half_t a, half_t b) { return half_t((float)a * (float)b); }
  friend half_t operator/(half_t a, half_t b) { return half_t((float)a / (float)b); }

private:
  static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t mant = x & 0x007FFFFF;
    uint32_t exp = x & 0x7F800000;

    if (exp >= 0x47800000) {
      // Inf or NaN
      if (mant && exp == 0x7F800000) {
        // NaN: preserve some payload
        return static_cast<uint16_t>(sign | 0x0200);
      }
      // Infinity
      return static_cast<uint16_t>(sign | 0x7C00);
    }
    if (exp <= 0x38000000) {
      // Subnormal or zero
      if (exp < 0x33000000) {
        // Too small: underflows to zero
        return static_cast<uint16_t>(sign);
      }
      // Subnormal
      mant |= 0x00800000;
      int shift = 113 - (exp >> 23);
      mant = (mant >> shift) + ((mant >> (shift - 1)) & 1);
      return static_cast<uint16_t>(sign | (mant & 0x03FF));
    }
    // Normalized number
    uint16_t h_exp = static_cast<uint16_t>(((exp - 0x38000000) >> 13) & 0x7C00);
    uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
    return static_cast<uint16_t>(sign | h_exp | h_mant);
  }

  static float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = h & 0x7C00;
    uint32_t mant = h & 0x03FF;
    uint32_t f;

    if (exp == 0x7C00) {
      // Inf or NaN
      f = sign | 0x7F800000 | (mant << 13);
    } else if (exp != 0) {
      // Normalized
      uint32_t e = ((exp >> 10) + 112) << 23;
      f = sign | e | (mant << 13);
    } else if (mant != 0) {
      // Subnormal
      mant <<= 1;
      int e = -1;
      while (!(mant & 0x0400)) {
        mant <<= 1;
        e--;
      }
      mant &= 0x03FF;
      uint32_t f_e = static_cast<uint32_t>(e + 1 + 127) << 23;
      f = sign | f_e | (mant << 13);
    } else {
      // Zero
      f = sign;
    }
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
  }
};

} // namespace vortex