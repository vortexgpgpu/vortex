#pragma once

#include <vector>
#include "types.h"

namespace vortex {

template <typename... Args>
void unused(Args&&...) {}

#define __unused(...) unused(__VA_ARGS__)

constexpr bool ispow2(uint32_t value) {
  return value && !(value & (value - 1));
}

constexpr unsigned log2ceil(uint32_t value) {
  return 32 - __builtin_clz(value - 1);
}

Word signExt(Word w, Size bit, Word mask);

Word bytesToWord(const Byte *b, Size wordSize);
void wordToBytes(Byte *b, Word w, Size wordSize);
Word flagsToWord(bool r, bool w, bool x);
void wordToFlags(bool &r, bool &w, bool &x, Word f);

Byte readByte(const std::vector<Byte> &b, Size &n);
Word readWord(const std::vector<Byte> &b, Size &n, Size wordSize);
void writeByte(std::vector<Byte> &p, Size &n, Byte b);
void writeWord(std::vector<Byte> &p, Size &n, Size wordSize, Word w);

}