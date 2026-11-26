#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef LIMBS
#define LIMBS 24
#endif

typedef struct {
  uint32_t num_items;
  uint32_t num_iters;
  uint32_t n0inv;       // -modulus^-1 mod 2^32
  uint32_t padding;
  uint64_t mod_addr;    // [LIMBS] modulus, shared
  uint64_t x_addr;      // [LIMBS][num_items] operand x; holds the result
  uint64_t y_addr;      // [LIMBS][num_items] operand y
} kernel_arg_t;

#if defined(__clang__)
#define BIG_UNROLL _Pragma("clang loop unroll(full)")
#else
#define BIG_UNROLL _Pragma("GCC unroll 64")
#endif

// Montgomery product x = x * y / 2^(32 LIMBS) mod n (CIOS), for x, y < n. x, n
// and the accumulator t are indexed by constants only, so they stay in
// registers; y supplies one limb per outer step through get_y(i).
template <typename GetY>
static inline __attribute__((always_inline)) void mont_mul(uint32_t* x, GetY get_y,
                                                            const uint32_t* n, uint32_t n0inv) {
  uint32_t t[LIMBS + 2];
  BIG_UNROLL
  for (int j = 0; j < LIMBS + 2; ++j)
    t[j] = 0;
  for (int i = 0; i < LIMBS; ++i) {
    uint32_t yi = get_y(i);
    uint64_t c = 0;
    BIG_UNROLL
    for (int j = 0; j < LIMBS; ++j) {
      uint64_t s = uint64_t(t[j]) + uint64_t(x[j]) * yi + c;
      t[j] = uint32_t(s);
      c = s >> 32;
    }
    uint64_t s = uint64_t(t[LIMBS]) + c;
    t[LIMBS] = uint32_t(s);
    t[LIMBS + 1] = uint32_t(s >> 32);
    uint32_t m = t[0] * n0inv;
    s = uint64_t(t[0]) + uint64_t(m) * n[0];
    c = s >> 32;
    BIG_UNROLL
    for (int j = 1; j < LIMBS; ++j) {
      s = uint64_t(t[j]) + uint64_t(m) * n[j] + c;
      t[j - 1] = uint32_t(s);
      c = s >> 32;
    }
    s = uint64_t(t[LIMBS]) + c;
    t[LIMBS - 1] = uint32_t(s);
    t[LIMBS] = t[LIMBS + 1] + uint32_t(s >> 32);
  }
  // x = t >= n ? t - n : t
  uint32_t d[LIMBS];
  uint32_t borrow = 0;
  BIG_UNROLL
  for (int j = 0; j < LIMBS; ++j) {
    uint64_t s = uint64_t(t[j]) - n[j] - borrow;
    d[j] = uint32_t(s);
    borrow = uint32_t(s >> 32) & 1;
  }
  bool ge = t[LIMBS] != 0 || borrow == 0;
  BIG_UNROLL
  for (int j = 0; j < LIMBS; ++j)
    x[j] = ge ? d[j] : t[j];
}

#endif
