#include <vx_spawn2.h>
#include "common.h"

// Warp-uniform loads around single-lane stores. Per iteration k, lane
// s = k % W (so every lane, and with it every dcache channel, takes a turn):
//
//   scattered loads  - every lane a different line, so the channels drain
//                      unevenly before the accesses below
//   RAW: lane s stores its word, then the whole warp loads it (one address):
//        lane s must see its own store
//   WAR: the whole warp loads a word, then lane s stores it: lane s's load
//        must not see its own later store
//   read-only loads  - lanes in groups of 1, 2, 4, 8 or 16 share a word of
//                      one line: every lane must get its own word
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto raw     = reinterpret_cast<volatile uint32_t*>(arg->raw_addr);
  auto war     = reinterpret_cast<volatile uint32_t*>(arg->war_addr);
  auto ro      = reinterpret_cast<volatile uint32_t*>(arg->ro_addr);
  auto scatter = reinterpret_cast<volatile uint32_t*>(arg->scatter_addr);
  auto err     = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto sink    = reinterpret_cast<uint32_t*>(arg->sink_addr);
  const uint32_t W    = arg->warp_size;
  const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % W;
  const uint32_t warp = tid / W;
  uint32_t errors = 0;
  uint32_t acc = 0;
  for (uint32_t k = 0; k < arg->num_iters; ++k) {
    uint32_t s = k % W;
    uint32_t slot = warp * arg->num_iters + k;

    acc += scatter[(lane * RO_LINE + slot * 5) % arg->scatter_words];

    uint32_t ri = slot * RAW_STRIDE;
    if (lane == s) {
      raw[ri] = UNIFORM_MARK + ri;
    }
    uint32_t rv = raw[ri];
    if (lane == s && rv != UNIFORM_MARK + ri) {
      ++errors;
    }

    uint32_t wi = slot * WAR_STRIDE;
    uint32_t wv = war[wi];
    if (lane == s) {
      war[wi] = UNIFORM_MARK + wi;
    }
    if (lane == s && wv != wi * 7 + 1) {
      ++errors;
    }

    uint32_t shift = k % 5;
    uint32_t oi = slot * RO_LINE + ((lane >> shift) & (RO_LINE - 1));
    if (ro[oi] != oi * 3 + 2) {
      ++errors;
    }
  }
  err[tid] = errors;
  sink[tid] = acc;
}
