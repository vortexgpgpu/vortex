#include <vx_spawn2.h>
#include "common.h"

// Local-memory reads that share words, around single-lane stores and
// atomics. Per iteration k, lane s = k % W takes a turn:
//
//   shared words - lanes in groups of 1, 2, 4, ... W read one word; then 2
//                  and 4 distinct words per warp, in one bank (32 words
//                  apart) and in different banks: every lane must get its
//                  own word; a third of the lanes read one word (partial
//                  thread mask); every warp reads the same word (lanes of
//                  different LSU blocks)
//   conflicts    - every lane reads its own word of one bank, so the request
//                  replays over W passes
//   sub-words    - every lane loads a byte (signed and unsigned) and a
//                  half-word of one shared word, at its own offset
//   RAW          - lane s stores its word, then the whole warp reads it:
//                  lane s must see its own store
//   WAR          - the whole warp reads a word, then lane s stores it: lane
//                  s's read must not see its own later store
//   AMO          - the whole warp reads a word, every lane adds 1 to it
//                  atomically, then the whole warp reads it again: each lane's
//                  first read must not see its own add, and its last must
//   sub-word     - every lane stores one byte, four lanes to a word, then
//   stores         reads its word back: it must hold all four lanes' bytes
//   same-word    - every lane stores its own value to one word, then the
//   stores         whole warp reads it: the highest lane's store is the last
//   cross-warp   - one lane of warp k % NW stores iteration k's word while
//                  the other warps, on other LSU blocks, read it: each read
//                  returns the old word or the stored one
//   counter      - every thread adds 1 to one word of the block atomically
//                  each iteration: no add is lost
// After the iterations, every thread increments one more word of the block
// through a load-reserved/store-conditional loop: no increment is lost. The
// host can leave this phase out (its retries make the instruction count
// depend on timing).
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
  const uint32_t NW    = bsize / W;
  const uint32_t words = arg->lmem_words;

  for (uint32_t i = lid; i < words; i += bsize) {
    lmem[i] = ro_word(i);
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
      if (lmem[idx] != ro_word(idx)) {
        ++errors;
      }
    }
    if (lane % 3 == k % 3) {
      uint32_t idx = (base + 5) % RO_WORDS;
      if (lmem[idx] != ro_word(idx)) {
        ++errors;
      }
    }
    {
      uint32_t idx = (k * 5) % RO_WORDS;
      if (lmem[idx] != ro_word(idx)) {
        ++errors;
      }
    }

    {
      uint32_t idx = arg->cf_base + lane * W + s;
      if (lmem[idx] != ro_word(idx)) {
        ++errors;
      }
    }

    {
      uint32_t idx = (base + 9) % RO_WORDS;
      uint32_t v   = ro_word(idx);
      uint32_t bo  = lane & 3;
      uint32_t ho  = lane & 1;
      auto lmem_u8  = reinterpret_cast<volatile uint8_t*>(lmem);
      auto lmem_s8  = reinterpret_cast<volatile int8_t*>(lmem);
      auto lmem_u16 = reinterpret_cast<volatile uint16_t*>(lmem);
      if (lmem_u8[idx * 4 + bo] != ((v >> (8 * bo)) & 0xff)) {
        ++errors;
      }
      if (lmem_s8[idx * 4 + bo] != (int8_t)(v >> (8 * bo))) {
        ++errors;
      }
      if (lmem_u16[idx * 2 + ho] != ((v >> (16 * ho)) & 0xffff)) {
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
    if (lane == s && wv != ro_word(wi)) {
      ++errors;
    }

  #if VX_CFG_EXT_A_ENABLED
    uint32_t ai = arg->amo_base + slot;
    uint32_t a0 = lmem[ai];
    uint32_t old = __atomic_fetch_add(const_cast<uint32_t*>(&lmem[ai]), 1u, __ATOMIC_RELAXED);
    uint32_t a1 = lmem[ai];
    // Other lanes' adds may land on either side of this lane's reads; this
    // lane's own add must not reach its earlier read and must reach its later
    // one, and returns a value no other add has passed over.
    if (old - ro_word(ai) >= W || (old - a0) >= W || (a1 - old - 1) >= W) {
      ++errors;
    }
  #endif

    {
      uint32_t si = arg->sub_base + slot * SUB_WORDS(W) + lane / 4;
      auto lmem_u8 = reinterpret_cast<volatile uint8_t*>(lmem);
      lmem_u8[si * 4 + (lane & 3)] = sub_byte(slot, lane);
      uint32_t v = lmem[si];
      uint32_t expected = ro_word(si);
      for (uint32_t b = 0; b < 4; ++b) {
        uint32_t l = (lane & ~3u) + b;
        if (l < W) {
          expected = (expected & ~(0xffu << (8 * b))) | ((uint32_t)sub_byte(slot, l) << (8 * b));
        }
      }
      if (v != expected) {
        ++errors;
      }
    }

    {
      uint32_t si = arg->same_base + slot;
      lmem[si] = same_word(slot, lane);
      if (lmem[si] != same_word(slot, W - 1)) {
        ++errors;
      }
    }

    {
      uint32_t xi = arg->xwarp_base + k;
      if (warp == k % NW) {
        if (lane == s) {
          lmem[xi] = xwarp_word(k);
        }
      } else {
        uint32_t v = lmem[xi];
        if (v != ro_word(xi) && v != xwarp_word(k)) {
          ++errors;
        }
      }
    }

  #if VX_CFG_EXT_A_ENABLED
    {
      uint32_t ci  = arg->count_word;
      uint32_t old = __atomic_fetch_add(const_cast<uint32_t*>(&lmem[ci]), 1u, __ATOMIC_RELAXED);
      if (old - ro_word(ci) >= bsize * arg->num_iters) {
        ++errors;
      }
    }
  #endif
  }

#if VX_CFG_EXT_A_ENABLED
  if (arg->lrsc) {
    // The retry is written out, not left to the compiler's compare-and-swap
    // expansion, whose inner retry branch is not reconverged per lane. It is
    // bounded, so a store-conditional that never succeeds leaves the count
    // short instead of hanging the kernel.
    auto word = const_cast<uint32_t*>(&lmem[arg->lrsc_word]);
    uint32_t old;
    uint32_t sc_fail;
    uint32_t guard = 0;
    do {
      asm volatile ("lr.w %0, (%1)" : "=r"(old) : "r"(word) : "memory");
      asm volatile ("sc.w %0, %2, (%1)" : "=r"(sc_fail) : "r"(word), "r"(old + 1) : "memory");
      ++guard;
    } while (sc_fail && guard < LRSC_MAX_RETRIES);
    if (sc_fail || old - ro_word(arg->lrsc_word) >= bsize) {
      ++errors;
    }
  }
#endif
  err[tid] = errors;

  __syncthreads();
  for (uint32_t i = lid; i < words; i += bsize) {
    dump[blockIdx.x * words + i] = lmem[i];
  }
}
