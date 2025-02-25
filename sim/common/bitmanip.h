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

template <typename T>
constexpr uint32_t count_leading_zeros(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  if constexpr (sizeof(T) > 4) {
    return value ? __builtin_clzll(value) - (64 - sizeof(T) * 8) : sizeof(T) * 8;
  } else {
    return value ? __builtin_clz(value) - (32 - sizeof(T) * 8) : sizeof(T) * 8;
  }
}

template <typename T>
constexpr uint32_t count_trailing_zeros(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  if constexpr (sizeof(T) > 4) {
    return value ? __builtin_ctzll(value) : (sizeof(T) * 8);
  } else {
    return value ? __builtin_ctz(value) : (sizeof(T) * 8);
  }
}

template <typename T>
constexpr bool ispow2(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  return value && !(value & (value - 1));
}

template <typename T>
constexpr uint32_t log2ceil(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
    return (sizeof(T) * 8) - count_leading_zeros<T>(value - 1);
}

template <typename T>
inline unsigned log2up(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  return std::max<uint32_t>(1, log2ceil(value));
}

template <typename T>
constexpr unsigned log2floor(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  return (sizeof(T) * 8 - 1) - count_leading_zeros<T>(value);
}

template <typename T>
constexpr unsigned ceil2(T value) {
  static_assert(std::is_integral<T>::value, "invalid data type");
  return (sizeof(T) * 8) - count_leading_zeros<T>(value);
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

inline uint64_t bit_reverse(uint64_t bits) {
  bits = ((bits & 0xAAAAAAAAAAAAAAAA) >>  1) | ((bits & 0x5555555555555555) <<  1);
  bits = ((bits & 0xCCCCCCCCCCCCCCCC) >>  2) | ((bits & 0x3333333333333333) <<  2);
  bits = ((bits & 0xF0F0F0F0F0F0F0F0) >>  4) | ((bits & 0x0F0F0F0F0F0F0F0F) <<  4);
  bits = ((bits & 0xFF00FF00FF00FF00) >>  8) | ((bits & 0x00FF00FF00FF00FF) <<  8);
  bits = ((bits & 0xFFFF0000FFFF0000) >> 16) | ((bits & 0x0000FFFF0000FFFF) << 16);
  bits = (bits >> 32) | (bits << 32);
  return bits;
}

inline uint64_t bit_reverse(uint64_t bits, uint32_t width) {
  assert(width <= 64);
  uint64_t reversed(0);
  for (uint32_t i = 0; i < width; ++i) {
    if (bits & (1ULL << i)) {
      reversed |= (1ULL << (width - 1 - i));
    }
  }
  return reversed;
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
