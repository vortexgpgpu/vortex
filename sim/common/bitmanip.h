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

#include <cstdint>
#include <assert.h>

constexpr uint32_t count_leading_zeros(uint32_t value) {
  return value ? __builtin_clz(value) : 32;
}

constexpr uint32_t count_leading_zeros(uint64_t value) {
  return value ? __builtin_clzll(value) : 64;
}

constexpr uint32_t count_trailing_zeros(uint32_t value) {
  return value ? __builtin_ctz(value) : 32;
}

constexpr uint32_t count_trailing_zeros(uint64_t value) {
  return value ? __builtin_ctzll(value) : 64;
}

constexpr bool ispow2(uint32_t value) {
  return value && !(value & (value - 1));
}

constexpr bool ispow2(uint64_t value) {
  return value && !(value & (value - 1));
}

constexpr uint32_t log2ceil(uint32_t value) {
  return 32 - count_leading_zeros(value - 1);
}

constexpr uint32_t log2ceil(uint64_t value) {
  return 64 - count_leading_zeros(value - 1);
}

inline unsigned log2up(uint32_t value) {
  return std::max<uint32_t>(1, log2ceil(value));
}

inline unsigned log2up(uint64_t value) {
  return std::max<uint32_t>(1, log2ceil(value));
}

constexpr unsigned log2floor(uint32_t value) {
  return 31 - count_leading_zeros(value);
}

constexpr unsigned log2floor(uint64_t value) {
  return 63 - count_leading_zeros(value);
}

constexpr unsigned ceil2(uint32_t value) {
  return 32 - count_leading_zeros(value);
}

constexpr unsigned ceil2(uint64_t value) {
  return 64 - count_leading_zeros(value);
}

inline uint64_t bit_clr(uint64_t bits, uint32_t index) {
    assert(index <= 63);
    return bits & ~(1ull << index);
}

inline uint64_t bit_set(uint64_t bits, uint32_t index) {
    assert(index <= 63);
    return bits | (1ull << index);
}

inline bool bit_get(uint64_t bits, uint32_t index) {
    assert(index <= 63);
    return (bits >> index) & 0x1;
}

inline uint64_t bit_clrw(uint64_t bits, uint32_t start, uint32_t end) {
    assert(end >= start);
    assert(end <= 63);
    uint32_t shift = 63 - end;
    uint64_t mask = (0xffffffffffffffff << (shift + start)) >> shift;
    return bits & ~mask;
}

inline uint64_t bit_setw(uint64_t bits, uint32_t start, uint32_t end, uint64_t value) {
    assert(end >= start);
    assert(end <= 63);
    uint32_t shift = 63 - end;
    uint64_t dirty = (value << (shift + start)) >> shift;
    return bit_clrw(bits, start, end) | dirty;
}

inline uint64_t bit_getw(uint64_t bits, uint32_t start, uint32_t end) {
    assert(end >= start);
    assert(end <= 63);
    uint32_t shift = 63 - end;
    return (bits << shift) >> (shift + start);
}

template <typename T = uint32_t>
T sext(const T& word, uint32_t width) {
  assert(width > 1);
  assert(width <= (sizeof(T) * 8));
  if (width == (sizeof(T) * 8))
    return word;
  T mask((static_cast<T>(1) << width) - 1);
  return ((word >> (width - 1)) & 0x1) ? (word | ~mask) : (word & mask);
}

template <typename T = uint32_t>
T zext(const T& word, uint32_t width) {
  assert(width > 1);
  assert(width <= (sizeof(T) * 8));
  if (width == (sizeof(T) * 8))
    return word;
  T mask((static_cast<T>(1) << width) - 1);
  return word & mask;
}
