#pragma once

#include <vector>
#include "types.h"

namespace vortex {

template <typename... Args>
void unused(Args&&...) {}

#define __unused(...) unused(__VA_ARGS__)

Word signExt(Word w, Size bit, Word mask);

Word_u bytesToWord(const Byte *b, Size wordSize);
void wordToBytes(Byte *b, Word_u w, Size wordSize);
Word_u flagsToWord(bool r, bool w, bool x);
void wordToFlags(bool &r, bool &w, bool &x, Word_u f);

Byte readByte(const std::vector<Byte> &b, Size &n);
Word_u readWord(const std::vector<Byte> &b, Size &n, Size wordSize);
void writeByte(std::vector<Byte> &p, Size &n, Byte b);
void writeWord(std::vector<Byte> &p, Size &n, Size wordSize, Word w);

}