#pragma once

// CTA-wide building blocks shared by the device kernels that order work across
// a CTA's threads. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <stdint.h>

namespace gfx_cta {

// The CTA's exclusive scan of one value per thread, through 2 * T words of
// scratch; *total receives the CTA's sum. Every thread of the CTA calls it.
static inline uint32_t exclusive(uint32_t* scratch, uint32_t tid, uint32_t T, uint32_t v,
                                 uint32_t* total) {
  uint32_t* a = scratch;
  uint32_t* b = scratch + T;
  a[tid] = v;
  __syncthreads();
  for (uint32_t d = 1; d < T; d <<= 1) {
    b[tid] = a[tid] + (tid >= d ? a[tid - d] : 0u);
    __syncthreads();
    uint32_t* t = a; a = b; b = t;
  }
  const uint32_t incl = a[tid];
  *total = a[T - 1];
  __syncthreads();
  return incl - v;
}

// Add n to 64-bit counter `slot` of a query counter block, from one thread.
static inline void query_add(uint64_t block, uint32_t slot, uint64_t n) {
  auto w = reinterpret_cast<uint32_t*>(block) + 2u * slot;
  const uint32_t lo = (uint32_t)n;
  const uint32_t old = __atomic_fetch_add(&w[0], lo, __ATOMIC_RELAXED);
  const uint32_t carry = old + lo < old ? 1u : 0u;
  const uint32_t hi = (uint32_t)(n >> 32) + carry;
  if (hi) {
    __atomic_fetch_add(&w[1], hi, __ATOMIC_RELAXED);
  }
}

} // namespace gfx_cta
