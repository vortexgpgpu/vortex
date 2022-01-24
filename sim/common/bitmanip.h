#pragma once

#include <cstdint>
#include <algorithm>
#include <assert.h>
#include "xlen.h"

constexpr uint32_t count_leading_zeros(uint32_t value) {
  return value ? __builtin_clz(value) : 32;
}

constexpr uint32_t count_trailing_zeros(uint32_t value) {
  return value ? __builtin_ctz(value) : 32;
}

constexpr bool ispow2(uint32_t value) {
  return value && !(value & (value - 1));
}

constexpr uint32_t log2ceil(uint32_t value) {
  return 32 - count_leading_zeros(value - 1);
}

inline unsigned log2up(uint32_t value) {
  return std::max<uint32_t>(1, log2ceil(value));
}

constexpr unsigned log2floor(uint32_t value) {
  return 31 - count_leading_zeros(value);
}

constexpr unsigned ceil2(uint32_t value) {
  return 32 - count_leading_zeros(value);
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

// Apply integer sign extension
inline uintx_t sext(uintx_t word, uint32_t width) {
  assert(width > 1);
  assert(width <= 32);
  uintx_t unity = 1;
  uintx_t mask = (unity << width) - 1;
  return ((word >> (width - 1)) & 0x1) ? (word | ~mask) : word;
}

inline uint64_t sext64(uint64_t word, uint64_t width) {
  assert(width > 1);
  assert(width <= 64);
  uint64_t unity = 1;
  uint64_t mask = (unity << width) - 1;
  return ((word >> (width - 1)) & 0x1) ? (word | ~mask) : word;
}

inline uintm_t sext_mul(uintm_t word, uint32_t width) {
  assert(width > 1);
  assert(width <= 32);
  uintm_t unity = 1;
  uintm_t mask = (unity << width) - 1;
  return ((word >> (width - 1)) & 0x1) ? (word | ~mask) : word;
}

inline uintf_t nan_box(uint32_t word) {
  uintf_t mask = uintf_t(0xffffffff00000000);
  return word | mask;
}