#pragma once

#include <cstdint>
#include <assert.h>

template <typename... Args>
void unused(Args&&...) {}

#define __unused(...) unused(__VA_ARGS__)

constexpr bool ispow2(uint64_t value) {
  return value && !(value & (value - 1));
}

constexpr unsigned log2ceil(uint32_t value) {
  return 32 - __builtin_clz(value - 1);
}

inline uint64_t align_size(uint64_t size, uint64_t alignment) {        
    assert(0 == (alignment & (alignment - 1)));
    return (size + alignment - 1) & ~(alignment - 1);
}

// 64-bit sign extension
inline uint64_t signExt(uint64_t w, uint64_t bit, uint64_t mask) {
  if (w >> (bit - 1))
    w |= ~mask;
  return w;
}

// 128-bit sign extension
inline __uint128_t signExt128(__uint128_t w, uint32_t bit, __uint128_t mask) {
  if (w >> (bit - 1))
    w |= ~mask;
  return w;
}

// return file extension
const char* fileExtension(const char* filepath);