#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_states;
  uint32_t num_perms;
  uint64_t state_addr;  // [25][num_states] lanes; lane l of state s at l * num_states + s
} kernel_arg_t;

#if defined(__clang__)
#define KECCAK_UNROLL _Pragma("clang loop unroll(full)")
#else
#define KECCAK_UNROLL _Pragma("GCC unroll 25")
#endif

// rho rotation of lane x + 5y
static constexpr int KECCAK_RHO[25] = {
   0,  1, 62, 28, 27,
  36, 44,  6, 55, 20,
   3, 10, 43, 25, 39,
  41, 45, 15, 21,  8,
  18,  2, 61, 56, 14,
};

static constexpr uint64_t KECCAK_RC[24] = {
  0x0000000000000001ull, 0x0000000000008082ull, 0x800000000000808Aull, 0x8000000080008000ull,
  0x000000000000808Bull, 0x0000000080000001ull, 0x8000000080008081ull, 0x8000000000008009ull,
  0x000000000000008Aull, 0x0000000000000088ull, 0x0000000080008009ull, 0x000000008000000Aull,
  0x000000008000808Bull, 0x800000000000008Bull, 0x8000000000008089ull, 0x8000000000008003ull,
  0x8000000000008002ull, 0x8000000000000080ull, 0x000000000000800Aull, 0x800000008000000Aull,
  0x8000000080008081ull, 0x8000000000008080ull, 0x0000000080000001ull, 0x8000000080008008ull,
};

static inline __attribute__((always_inline)) uint64_t keccak_rol(uint64_t x, int n) {
  return n ? (x << n) | (x >> (64 - n)) : x;
}

// Keccak-f[1600]. Every lane index is a constant after unrolling, so the 25
// lanes (50 words on RV32) stay in registers across the 24 rounds.
static inline __attribute__((always_inline)) void keccak_f(uint64_t* A) {
  for (int round = 0; round < 24; ++round) {
    uint64_t C[5], D[5], B[25];
    KECCAK_UNROLL
    for (int x = 0; x < 5; ++x)
      C[x] = A[x] ^ A[x + 5] ^ A[x + 10] ^ A[x + 15] ^ A[x + 20];
    KECCAK_UNROLL
    for (int x = 0; x < 5; ++x)
      D[x] = C[(x + 4) % 5] ^ keccak_rol(C[(x + 1) % 5], 1);
    KECCAK_UNROLL
    for (int i = 0; i < 25; ++i)
      A[i] ^= D[i % 5];
    KECCAK_UNROLL
    for (int i = 0; i < 25; ++i) {
      int x = i % 5, y = i / 5;
      B[y + 5 * ((2 * x + 3 * y) % 5)] = keccak_rol(A[i], KECCAK_RHO[i]);
    }
    KECCAK_UNROLL
    for (int i = 0; i < 25; ++i) {
      int x = i % 5, y = i / 5;
      A[i] = B[i] ^ (~B[(x + 1) % 5 + 5 * y] & B[(x + 2) % 5 + 5 * y]);
    }
    A[0] ^= KECCAK_RC[round];
  }
}

#endif
