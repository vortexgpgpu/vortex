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

#include <rvfloats.h>
#include <stdint.h>

// A minimal bfloat16 (16-bit) float implementation
// Provides conversion to/from 32-bit float and basic arithmetic operators.

class bfloat16_t {
public:

  uint16_t bits;

  bfloat16_t() = default;

  // Construct from float
  bfloat16_t(float f) { bits = float_to_bfloat16(f); }

  // Convert to float
  operator float() const { return bfloat16_to_float(bits); }

  // Arithmetic operators
  friend bfloat16_t operator+(bfloat16_t a, bfloat16_t b) { return bfloat16_t((float)a + (float)b); }
  friend bfloat16_t operator-(bfloat16_t a, bfloat16_t b) { return bfloat16_t((float)a - (float)b); }
  friend bfloat16_t operator*(bfloat16_t a, bfloat16_t b) { return bfloat16_t((float)a * (float)b); }
  friend bfloat16_t operator/(bfloat16_t a, bfloat16_t b) { return bfloat16_t((float)a / (float)b); }

private:

  union fp32_u32_t {
    float    f;
    uint32_t u;
  };

  static uint16_t float_to_bfloat16(float f) {
    fp32_u32_t pun;
    pun.f = f;
    return rv_ftob_s(pun.u, 0, nullptr);
  }

  static float bfloat16_to_float(uint16_t h) {
    fp32_u32_t pun;
    pun.u = rv_btof_s(h, 0, nullptr);
    return pun.f;
  }
};