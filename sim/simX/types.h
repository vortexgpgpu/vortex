#pragma once

#include <stdint.h>
#include <bitset>
#include <VX_config.h>

namespace vortex {

typedef uint8_t  Byte;
// simx64
typedef uint64_t Word;
typedef int64_t  WordI;

// simx64
typedef uint32_t HalfWord;
typedef int32_t HalfWordI;

// simx64
typedef uint64_t Addr;
typedef uint64_t Size;

typedef std::bitset<32> RegMask;

typedef std::bitset<32> ThreadMask;

typedef std::bitset<32> WarpMask;

}