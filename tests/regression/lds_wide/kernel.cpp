#include <vx_spawn2.h>
#include "common.h"

// vx_flds.d / vx_flds.q (custom-3), written as .insn so the encodings are
// tested independently of the compiler. Both are I-type like fld/flq:
// funct3 = 3 / 4, imm[11:0] = signed byte offset; fd is an aligned pair /
// quad, the address 8 / 16-byte aligned.
#if (__riscv_xlen == 32) && defined(__riscv_f)
#define HAS_FLDS 1
#endif

// A wide load and the reads of its registers sit in one asm block, so the
// compiler cannot reuse them in between. fd + 3 (fd + 1) is read first: it is
// the last phase's register.
#define VX_FLDS_Q(r0, r1, r2, r3, off, p, w) \
  asm volatile (".insn i 0x7b, 4, " r0 ", %4, %5\n\t" \
                "fmv.x.w %3, " r3 "\n\t" \
                "fmv.x.w %0, " r0 "\n\t" \
                "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %2, " r2 \
                : "=&r"((w)[0]), "=&r"((w)[1]), "=&r"((w)[2]), "=&r"((w)[3]) \
                : "r"(p), "i"(off) \
                : r0, r1, r2, r3, "memory")

#define VX_FLDS_D(r0, r1, off, p, w) \
  asm volatile (".insn i 0x7b, 3, " r0 ", %2, %3\n\t" \
                "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %0, " r0 \
                : "=&r"((w)[0]), "=&r"((w)[1]) \
                : "r"(p), "i"(off) \
                : r0, r1, "memory")

#ifdef HAS_FLDS
static inline uint32_t ro_misses(const uint32_t* w, uint32_t n, uint32_t first) {
  uint32_t misses = 0;
  for (uint32_t k = 0; k < n; ++k) {
    if (w[k] != ro_word(first + k)) {
      ++misses;
    }
  }
  return misses;
}
#else
static inline uint32_t lmem_misses(volatile const uint32_t* lmem, uint32_t n, uint32_t first) {
  uint32_t misses = 0;
  for (uint32_t k = 0; k < n; ++k) {
    if (lmem[first + k] != ro_word(first + k)) {
      ++misses;
    }
  }
  return misses;
}
#endif

// Each thread checks the words its wide loads return against the values the
// host and the generic path define:
//   - a warp-uniform group (broadcast), a 16-byte lane stride (every lane on
//     the same banks), a lane permutation, half the lanes under a divergent
//     branch; .d and .q through the window address and the bare offset (local
//     memory decodes only the low bits)
//   - fd = f0 and f28; two wide loads back to back
//   - scalar writes to fd + 2 just before and just after
//   - a wide load, an FP op on fd + 2 that waits for it, and a wide load of
//     the same group again
//   - the offset limits, -2048 and 2032 (.q) or 2040 (.d), through the window
//     address and the bare offset
//   - a generic store, then a wide load of the same words
// Without wide loads, generic loads read the same words.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto err  = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr);
  const uint32_t lid   = threadIdx.x;
  const uint32_t bsize = blockDim.x;
  const uint32_t tid   = blockIdx.x * bsize + lid;
  const uint32_t words = arg->lmem_words;

  if (tid == 0) {
  #ifdef HAS_FLDS
    *reinterpret_cast<uint32_t*>(arg->wide_addr) = 1;
  #else
    *reinterpret_cast<uint32_t*>(arg->wide_addr) = 0;
  #endif
  }

  for (uint32_t i = lid; i < RO_WORDS; i += bsize) {
    lmem[i] = ro_word(i);
  }
  __syncthreads();

  uint32_t errors = 0;

  // uniform, lane stride and permuted groups; gs + 7 and gq + 3 stay in the
  // read-only words for any block size, and gq * 4 exceeds the largest offset
  const uint32_t gu = 384;
  const uint32_t gs = 256 + (lid % 128) * 4;
  const uint32_t gp = 512 + ((lid * 7 + 3) % 64) * 4;
  const uint32_t go = 512;
  const uint32_t gq = 512 + (lid % 128) * 4;
  const uint32_t wi = RO_WORDS + lid * WIDE_WORDS;

#ifdef HAS_FLDS
  {
    const uintptr_t window = reinterpret_cast<uintptr_t>(lmem);
    const uintptr_t bare   = window - VX_MEM_LMEM_BASE_ADDR;
    uint32_t w[8];

    VX_FLDS_Q("f8", "f9", "f10", "f11", 0, window + gu * 4, w);
    errors += ro_misses(w, 4, gu);
    VX_FLDS_D("f8", "f9", 8, bare + gu * 4, w);
    errors += ro_misses(w, 2, gu + 2);

    VX_FLDS_Q("f12", "f13", "f14", "f15", 0, window + gs * 4, w);
    errors += ro_misses(w, 4, gs);
    VX_FLDS_D("f14", "f15", 0, window + gs * 4, w);
    errors += ro_misses(w, 2, gs);

    VX_FLDS_Q("f16", "f17", "f18", "f19", 0, bare + gp * 4, w);
    errors += ro_misses(w, 4, gp);
    VX_FLDS_D("f16", "f17", 8, window + gp * 4, w);
    errors += ro_misses(w, 2, gp + 2);

    if (lid & 1) {
      VX_FLDS_Q("f20", "f21", "f22", "f23", 0, window + gs * 4, w);
      errors += ro_misses(w, 4, gs);
      VX_FLDS_D("f22", "f23", 8, window + gp * 4, w);
      errors += ro_misses(w, 2, gp + 2);
    }

    VX_FLDS_Q("f0", "f1", "f2", "f3", 0, window + gp * 4, w);
    errors += ro_misses(w, 4, gp);
    VX_FLDS_Q("f28", "f29", "f30", "f31", 0, bare + gs * 4, w);
    errors += ro_misses(w, 4, gs);
    VX_FLDS_D("f30", "f31", 0, window + gu * 4, w);
    errors += ro_misses(w, 2, gu);

    asm volatile (".insn i 0x7b, 4, f8, %8, 0\n\t"
                  ".insn i 0x7b, 4, f12, %8, 16\n\t"
                  "fmv.x.w %0, f8\n\t"
                  "fmv.x.w %1, f9\n\t"
                  "fmv.x.w %2, f10\n\t"
                  "fmv.x.w %3, f11\n\t"
                  "fmv.x.w %4, f12\n\t"
                  "fmv.x.w %5, f13\n\t"
                  "fmv.x.w %6, f14\n\t"
                  "fmv.x.w %7, f15"
                  : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3]),
                    "=&r"(w[4]), "=&r"(w[5]), "=&r"(w[6]), "=&r"(w[7])
                  : "r"(window + gs * 4)
                  : "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "memory");
    errors += ro_misses(w, 4, gs);
    errors += ro_misses(w + 4, 4, gs + 4);

    const uint32_t marker = 0x7F00D000u | lid;
    asm volatile ("fmv.w.x f10, %1\n\t"
                  ".insn i 0x7b, 4, f8, %2, 0\n\t"
                  "fmv.x.w %0, f10"
                  : "=&r"(w[2])
                  : "r"(marker), "r"(window + gp * 4)
                  : "f8", "f9", "f10", "f11", "memory");
    if (w[2] != ro_word(gp + 2)) {
      ++errors;
    }
    asm volatile (".insn i 0x7b, 4, f8, %4, 0\n\t"
                  "fmv.w.x f10, %5\n\t"
                  "fmv.x.w %0, f8\n\t"
                  "fmv.x.w %1, f9\n\t"
                  "fmv.x.w %2, f10\n\t"
                  "fmv.x.w %3, f11"
                  : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                  : "r"(window + gp * 4), "r"(marker)
                  : "f8", "f9", "f10", "f11", "memory");
    if (w[0] != ro_word(gp) || w[1] != ro_word(gp + 1) || w[2] != marker || w[3] != ro_word(gp + 3)) {
      ++errors;
    }
    asm volatile (".insn i 0x7b, 4, f8, %4, 0\n\t"
                  "fadd.s f10, f10, f10\n\t"
                  ".insn i 0x7b, 4, f8, %4, 16\n\t"
                  "fmv.x.w %0, f8\n\t"
                  "fmv.x.w %1, f9\n\t"
                  "fmv.x.w %2, f10\n\t"
                  "fmv.x.w %3, f11"
                  : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                  : "r"(window + gs * 4)
                  : "f8", "f9", "f10", "f11", "memory");
    errors += ro_misses(w, 4, gs + 4);

    VX_FLDS_Q("f4", "f5", "f6", "f7", -2048, window + go * 4 + 2048, w);
    errors += ro_misses(w, 4, go);
    VX_FLDS_Q("f4", "f5", "f6", "f7", 2032, window + go * 4 - 2032, w);
    errors += ro_misses(w, 4, go);
    VX_FLDS_Q("f24", "f25", "f26", "f27", -2048, bare + gs * 4 + 2048, w);
    errors += ro_misses(w, 4, gs);
    VX_FLDS_Q("f24", "f25", "f26", "f27", 2032, bare + gq * 4 - 2032, w);
    errors += ro_misses(w, 4, gq);
    VX_FLDS_D("f26", "f27", -2048, window + gp * 4 + 2048, w);
    errors += ro_misses(w, 2, gp);
    VX_FLDS_D("f26", "f27", 2040, bare + gq * 4 - 2040, w);
    errors += ro_misses(w, 2, gq);

    for (uint32_t k = 0; k < WIDE_WORDS; ++k) {
      lmem[wi + k] = wide_word(tid, k);
    }
    VX_FLDS_Q("f16", "f17", "f18", "f19", 0, window + wi * 4, w);
    VX_FLDS_D("f20", "f21", 8, bare + wi * 4, w + 4);
    for (uint32_t k = 0; k < WIDE_WORDS; ++k) {
      if (w[k] != wide_word(tid, k)) {
        ++errors;
      }
    }
    if (w[4] != wide_word(tid, 2) || w[5] != wide_word(tid, 3)) {
      ++errors;
    }
  }
#else
  errors += lmem_misses(lmem, 4, gu);
  errors += lmem_misses(lmem, 8, gs);
  errors += lmem_misses(lmem, 4, gp);
  errors += lmem_misses(lmem, 4, go);
  errors += lmem_misses(lmem, 4, gq);
  for (uint32_t k = 0; k < WIDE_WORDS; ++k) {
    lmem[wi + k] = wide_word(tid, k);
  }
  for (uint32_t k = 0; k < WIDE_WORDS; ++k) {
    if (lmem[wi + k] != wide_word(tid, k)) {
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
