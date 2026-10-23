// Copyright © 2019-2026
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

#include <cstdint>

// RISC-V NaN-boxing: a narrow floating-point value held in a wider FP register
// has every register bit above it set to one; a value that is not boxed reads
// as the narrow format's canonical NaN.
namespace vortex::nanbox {

constexpr uint64_t low_mask(uint32_t bits) {
  return (bits >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bits) - 1);
}

constexpr uint64_t box(uint64_t value, uint32_t bits, uint32_t reg_bits = 64) {
  return ((value & low_mask(bits)) | ~low_mask(bits)) & low_mask(reg_bits);
}

constexpr bool is_boxed(uint64_t value, uint32_t bits, uint32_t reg_bits = 64) {
  uint64_t upper = low_mask(reg_bits) & ~low_mask(bits);
  return (value & upper) == upper;
}

constexpr uint64_t unbox(uint64_t value, uint32_t bits, uint64_t nan, uint32_t reg_bits = 64) {
  return is_boxed(value, bits, reg_bits) ? (value & low_mask(bits)) : nan;
}

constexpr uint64_t box32(uint32_t value) {
  return box(value, 32);
}

constexpr uint32_t unbox32(uint64_t value, uint32_t nan) {
  return uint32_t(unbox(value, 32, nan));
}

constexpr uint64_t box16(uint16_t value) {
  return box(value, 16);
}

constexpr uint16_t unbox16(uint64_t value, uint16_t nan) {
  return uint16_t(unbox(value, 16, nan));
}

}
