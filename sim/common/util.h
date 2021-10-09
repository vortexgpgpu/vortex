#pragma once

#include <cstdint>
#include <assert.h>

namespace vortex {

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

// Apply integer sign extension
uint32_t signExt(uint32_t w, uint32_t bit, uint32_t mask);

// Convert a floating point number to IEEE-754 32-bit representation
uint32_t floatToBin(float in_value);

// check floating-point number in binary format is NaN
uint8_t fpBinIsNan(uint32_t din);

// check floating-point number in binary format is zero
uint8_t fpBinIsZero(uint32_t din);

// check floating-point number in binary format is infinity
uint8_t fpBinIsInf(uint32_t din);

// return file extension
const char* fileExtension(const char* filepath);

}