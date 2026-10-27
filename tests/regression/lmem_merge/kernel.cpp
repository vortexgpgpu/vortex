#include <vx_spawn2.h>
#include "common.h"

// Local-memory reads that share words, around single-lane stores. Per
// iteration k, lane s = k % W takes a turn:
//
//   shared words - lanes in groups of 1, 2, 4, ... W read one word; then 2
//                  and 4 distinct words per warp, in one bank (32 words
//                  apart) and in different banks: every lane must get its
//                  own word; a third of the lanes read one word (partial
//                  thread mask); every warp reads the same word (lanes of
//                  different LSU blocks)
//   RAW         - lane s stores its word, then the whole warp reads it:
//                  lane s must see its own store
//   WAR          - the whole warp reads a word, then lane s stores it: lane
//                  s's read must not see its own later store
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto err  = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr);
  const uint32_t W     = arg->warp_size;
  const uint32_t lid   = threadIdx.x;
  const uint32_t bsize = blockDim.x;
  const uint32_t tid   = blockIdx.x * bsize + lid;
  const uint32_t lane  = lid % W;
  const uint32_t warp  = lid / W;
  const uint32_t words = arg->lmem_words;

  for (uint32_t i = lid; i < words; i += bsize) {
    lmem[i] = i * 3 + 2;
  }
  __syncthreads();

  uint32_t errors = 0;
  for (uint32_t k = 0; k < arg->num_iters; ++k) {
    uint32_t s = k % W;
    uint32_t slot = warp * arg->num_iters + k;

    uint32_t base = (slot * 7) % (RO_WORDS / 2);
    uint32_t shared_idx[5] = {
      base + (lane >> (k % 6)) % (RO_WORDS / 2),
      base + (lane & 1) * 32,
      base + (lane & 3) * 16,
      base + (lane & 1),
      base + (lane & 3),
    };
    for (uint32_t p = 0; p < 5; ++p) {
      uint32_t idx = shared_idx[p] % RO_WORDS;
      if (lmem[idx] != idx * 3 + 2) {
        ++errors;
      }
    }
    if (lane % 3 == k % 3) {
      uint32_t idx = (base + 5) % RO_WORDS;
      if (lmem[idx] != idx * 3 + 2) {
        ++errors;
      }
    }
    {
      uint32_t idx = (k * 5) % RO_WORDS;
      if (lmem[idx] != idx * 3 + 2) {
        ++errors;
      }
    }

    uint32_t ri = arg->raw_base + slot;
    if (lane == s) {
      lmem[ri] = LMEM_MARK + ri;
    }
    uint32_t rv = lmem[ri];
    if (lane == s && rv != LMEM_MARK + ri) {
      ++errors;
    }

    uint32_t wi = arg->war_base + slot;
    uint32_t wv = lmem[wi];
    if (lane == s) {
      lmem[wi] = LMEM_MARK + wi;
    }
    if (lane == s && wv != wi * 3 + 2) {
      ++errors;
    }
  }
  err[tid] = errors;

  __syncthreads();
  for (uint32_t i = lid; i < words; i += bsize) {
    dump[blockIdx.x * words + i] = lmem[i];
  }
}
