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
#include <cstdlib>
#include "types.h"

namespace vortex {

// Pure RMW kernel for the RVA op set. Operates on word-level
// holders; `width` selects W (4 bytes) or D (8 bytes). Returns
// {new_word, ret_word}: `new_word` is what gets merged back into
// the line on a store-bearing op; `ret_word` is what rd should
// observe (sign-extended by the LSU response formatter).
//
// LR returns {old, old} (no store); SC's return is decided by the
// caller (using the reservation table) and does not flow through
// here.
struct AmoComputeResult {
  uint64_t new_word;
  uint64_t ret_word;
};

inline AmoComputeResult amo_compute(AmoType op, uint8_t width,
                                    uint64_t old_word, uint64_t rhs) {
  // Mask both inputs to width-sized values; sign-extension at the
  // word boundary is needed for signed comparisons (MIN/MAX).
  const bool is_w = (width == 2);
  auto mask_w = [](uint64_t v) { return v & 0xFFFFFFFFull; };
  auto sext_w = [](uint64_t v) -> int64_t {
    return (int64_t)(int32_t)(uint32_t)v;
  };

  uint64_t a = is_w ? mask_w(old_word) : old_word;
  uint64_t b = is_w ? mask_w(rhs)      : rhs;
  int64_t  ai = is_w ? sext_w(old_word) : (int64_t)old_word;
  int64_t  bi = is_w ? sext_w(rhs)      : (int64_t)rhs;

  uint64_t newv = a;
  switch (op) {
  case AmoType::LR:      /* no store */ newv = a; break;
  case AmoType::SC:      newv = b;                break;
  case AmoType::AMOSWAP: newv = b;                break;
  case AmoType::AMOADD:  newv = a + b;            break;
  case AmoType::AMOAND:  newv = a & b;            break;
  case AmoType::AMOOR:   newv = a | b;            break;
  case AmoType::AMOXOR:  newv = a ^ b;            break;
  case AmoType::AMOMIN:  newv = (uint64_t)((ai < bi) ? ai : bi); break;
  case AmoType::AMOMAX:  newv = (uint64_t)((ai > bi) ? ai : bi); break;
  case AmoType::AMOMINU: newv = (a < b) ? a : b;  break;
  case AmoType::AMOMAXU: newv = (a > b) ? a : b;  break;
  default:
    std::abort();
  }
  if (is_w) newv = mask_w(newv);

  // Return value: original loaded word (LSU sext at width when writing rd).
  uint64_t retv = is_w ? mask_w(old_word) : old_word;
  return {newv, retv};
}

// Pack/unpack the active word inside a `mem_block_t` at byte_off,
// width 2 (4 bytes) or 3 (8 bytes).
inline uint64_t amo_load_word(const uint8_t* line, uint32_t byte_off, uint8_t width) {
  uint64_t v = 0;
  uint32_t n = 1u << width; // 4 or 8
  for (uint32_t i = 0; i < n; ++i) {
    v |= (uint64_t)line[byte_off + i] << (8 * i);
  }
  return v;
}

inline void amo_store_word(uint8_t* line, uint32_t byte_off, uint8_t width, uint64_t v) {
  uint32_t n = 1u << width;
  for (uint32_t i = 0; i < n; ++i) {
    line[byte_off + i] = (uint8_t)((v >> (8 * i)) & 0xff);
  }
}

inline uint64_t amo_byteen(uint32_t byte_off, uint8_t width) {
  uint32_t n = 1u << width;
  return ((1ull << n) - 1ull) << byte_off;
}

} // namespace vortex
