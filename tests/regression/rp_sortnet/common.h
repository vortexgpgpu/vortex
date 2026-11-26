#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef KEYS
#define KEYS 64
#endif

typedef struct {
  uint32_t num_rows;
  uint32_t padding;
  uint64_t keys_addr;   // [KEYS][num_rows] keys; key k of row r at k * num_rows + r; sorted in place
} kernel_arg_t;

#if defined(__clang__)
#define SORT_UNROLL _Pragma("clang loop unroll(full)")
#else
#define SORT_UNROLL _Pragma("GCC unroll 64")
#endif

// Bitonic sorting network over KEYS keys. The stage, step and key loops have
// constant bounds, so after unrolling every compare-exchange names two fixed
// keys and the keys stay in registers.
template <int K, int J>
static inline __attribute__((always_inline)) void bitonic_step(uint32_t* a) {
  SORT_UNROLL
  for (int i = 0; i < KEYS; ++i) {
    int l = i ^ J;
    if (l > i) {
      uint32_t x = a[i], y = a[l];
      bool swap = ((i & K) == 0) ? (x > y) : (x < y);
      a[i] = swap ? y : x;
      a[l] = swap ? x : y;
    }
  }
  if constexpr (J > 1)
    bitonic_step<K, J / 2>(a);
}

template <int K>
static inline __attribute__((always_inline)) void bitonic_stage(uint32_t* a) {
  bitonic_step<K, K / 2>(a);
  if constexpr (K < KEYS)
    bitonic_stage<K * 2>(a);
}

static inline __attribute__((always_inline)) void bitonic_sort(uint32_t* a) {
  bitonic_stage<2>(a);
}

#endif
