#include <vx_spawn2.h>
#include "common.h"

// vx_flw.d/.q (custom-3 funct3 3/4, I-type like fld/flq) and vx_fsw.d/.q
// (funct3 5/6, S-type like fsd/fsq), written as .insn so the encodings are
// tested independently of the compiler. fd/fs2 is an aligned pair or quad and
// imm a multiple of 8 or 16; each word needs only 4-byte alignment.
#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
// A wide load and the reads of its registers sit in one asm block, so the
// compiler cannot reuse them in between. The last uop's register is read first.
#define VX_FLW_Q(r0, r1, r2, r3, off, p, w) \
  asm volatile (".insn i 0x7b, 4, " r0 ", %4, %5\n\t" \
                "fmv.x.w %3, " r3 "\n\t" \
                "fmv.x.w %0, " r0 "\n\t" \
                "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %2, " r2 \
                : "=&r"((w)[0]), "=&r"((w)[1]), "=&r"((w)[2]), "=&r"((w)[3]) \
                : "r"(p), "i"(off) \
                : r0, r1, r2, r3, "memory")

#define VX_FLW_D(r0, r1, off, p, w) \
  asm volatile (".insn i 0x7b, 3, " r0 ", %2, %3\n\t" \
                "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %0, " r0 \
                : "=&r"((w)[0]), "=&r"((w)[1]) \
                : "r"(p), "i"(off) \
                : r0, r1, "memory")

#define VX_FSW_Q(r0, r1, r2, r3, off, p, w) \
  asm volatile ("fmv.w.x " r0 ", %0\n\t" \
                "fmv.w.x " r1 ", %1\n\t" \
                "fmv.w.x " r2 ", %2\n\t" \
                "fmv.w.x " r3 ", %3\n\t" \
                ".insn s 0x7b, 6, " r0 ", %5(%4)" \
                : \
                : "r"((w)[0]), "r"((w)[1]), "r"((w)[2]), "r"((w)[3]), "r"(p), "i"(off) \
                : r0, r1, r2, r3, "memory")

#define VX_FSW_D(r0, r1, off, p, w) \
  asm volatile ("fmv.w.x " r0 ", %0\n\t" \
                "fmv.w.x " r1 ", %1\n\t" \
                ".insn s 0x7b, 5, " r0 ", %3(%2)" \
                : \
                : "r"((w)[0]), "r"((w)[1]), "r"(p), "i"(off) \
                : r0, r1, "memory")
#else
// Without the wide forms, generic accesses move the same words.
#define VX_GENERIC_LD(n, off, p, w) \
  do { \
    auto _a = reinterpret_cast<volatile const uint32_t*>((p) + (off)); \
    for (uint32_t _k = 0; _k < (n); ++_k) { \
      (w)[_k] = _a[_k]; \
    } \
  } while (0)
#define VX_GENERIC_ST(n, off, p, w) \
  do { \
    auto _a = reinterpret_cast<volatile uint32_t*>((p) + (off)); \
    for (uint32_t _k = 0; _k < (n); ++_k) { \
      _a[_k] = (w)[_k]; \
    } \
  } while (0)
#define VX_FLW_Q(r0, r1, r2, r3, off, p, w) VX_GENERIC_LD(4, off, p, w)
#define VX_FLW_D(r0, r1, off, p, w)         VX_GENERIC_LD(2, off, p, w)
#define VX_FSW_Q(r0, r1, r2, r3, off, p, w) VX_GENERIC_ST(4, off, p, w)
#define VX_FSW_D(r0, r1, off, p, w)         VX_GENERIC_ST(2, off, p, w)
#endif

static inline uint32_t ro_misses(const uint32_t* w, uint32_t n, uint32_t first, uint32_t salt) {
  uint32_t misses = 0;
  for (uint32_t k = 0; k < n; ++k) {
    if (w[k] != (ro_word(first + k) ^ salt)) {
      ++misses;
    }
  }
  return misses;
}

// Wide loads of the read-only words at ro (local or global), checked against
// the words the host defines:
//   - a warp-uniform group (broadcast), a 16-byte lane stride (every lane on
//     the same banks), a lane permutation, half the lanes under a divergent
//     branch, and addresses 4- but not 8- or 16-byte aligned
//   - fd = f0 and f28; two wide loads back to back
//   - scalar writes to fd + 2 just before and just after a wide load
//   - a wide load, an FP op on fd + 2 that waits for it, and a wide load of
//     the same group again
//   - the offset limits, -2048 and 2032 (.q) or 2040 (.d)
static inline uint32_t check_loads(uintptr_t ro, uint32_t salt, uint32_t lid) {
  // gs + 8, gp + 5 and gq + 3 stay in the read-only words for any block
  // size, and gq * 4 exceeds the largest offset
  const uint32_t gu = 384;
  const uint32_t gs = 256 + (lid % 128) * 4;
  const uint32_t gp = 512 + ((lid * 7 + 3) % 64) * 4;
  const uint32_t go = 512;
  const uint32_t gq = 512 + (lid % 128) * 4;
  uint32_t errors = 0;
  uint32_t w[8];

  VX_FLW_Q("f8", "f9", "f10", "f11", 0, ro + gu * 4, w);
  errors += ro_misses(w, 4, gu, salt);
  VX_FLW_D("f8", "f9", 8, ro + gu * 4, w);
  errors += ro_misses(w, 2, gu + 2, salt);

  VX_FLW_Q("f12", "f13", "f14", "f15", 0, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs, salt);
  VX_FLW_D("f14", "f15", 0, ro + gs * 4, w);
  errors += ro_misses(w, 2, gs, salt);
  VX_FLW_Q("f12", "f13", "f14", "f15", 16, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs + 4, salt);

  VX_FLW_Q("f16", "f17", "f18", "f19", 0, ro + gp * 4, w);
  errors += ro_misses(w, 4, gp, salt);
  VX_FLW_D("f16", "f17", 8, ro + gp * 4, w);
  errors += ro_misses(w, 2, gp + 2, salt);

  if (lid & 1) {
    VX_FLW_Q("f20", "f21", "f22", "f23", 0, ro + gs * 4, w);
    errors += ro_misses(w, 4, gs, salt);
    VX_FLW_D("f22", "f23", 8, ro + gp * 4, w);
    errors += ro_misses(w, 2, gp + 2, salt);
  }

  VX_FLW_Q("f20", "f21", "f22", "f23", 16, ro + (gs + 1) * 4, w);
  errors += ro_misses(w, 4, gs + 5, salt);
  VX_FLW_Q("f24", "f25", "f26", "f27", 0, ro + (gp + 2) * 4, w);
  errors += ro_misses(w, 4, gp + 2, salt);
  VX_FLW_D("f26", "f27", 8, ro + (gq - 1) * 4, w);
  errors += ro_misses(w, 2, gq + 1, salt);

  VX_FLW_Q("f0", "f1", "f2", "f3", 0, ro + gp * 4, w);
  errors += ro_misses(w, 4, gp, salt);
  VX_FLW_Q("f28", "f29", "f30", "f31", 0, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs, salt);
  VX_FLW_D("f30", "f31", 0, ro + gu * 4, w);
  errors += ro_misses(w, 2, gu, salt);

  VX_FLW_Q("f4", "f5", "f6", "f7", -2048, ro + go * 4 + 2048, w);
  errors += ro_misses(w, 4, go, salt);
  VX_FLW_Q("f4", "f5", "f6", "f7", 2032, ro + go * 4 - 2032, w);
  errors += ro_misses(w, 4, go, salt);
  VX_FLW_Q("f24", "f25", "f26", "f27", -2048, ro + (gs + 1) * 4 + 2048, w);
  errors += ro_misses(w, 4, gs + 1, salt);
  VX_FLW_Q("f24", "f25", "f26", "f27", 2032, ro + gq * 4 - 2032, w);
  errors += ro_misses(w, 4, gq, salt);
  VX_FLW_D("f26", "f27", -2048, ro + gp * 4 + 2048, w);
  errors += ro_misses(w, 2, gp, salt);
  VX_FLW_D("f26", "f27", 2040, ro + (gq + 1) * 4 - 2040, w);
  errors += ro_misses(w, 2, gq + 1, salt);

#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
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
                : "r"(ro + gs * 4)
                : "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "memory");
  errors += ro_misses(w, 8, gs, salt);

  const uint32_t marker = 0x7F00D000u | lid;
  asm volatile ("fmv.w.x f10, %1\n\t"
                ".insn i 0x7b, 4, f8, %2, 0\n\t"
                "fmv.x.w %0, f10"
                : "=&r"(w[2])
                : "r"(marker), "r"(ro + gp * 4)
                : "f8", "f9", "f10", "f11", "memory");
  errors += (w[2] != (ro_word(gp + 2) ^ salt));
  asm volatile (".insn i 0x7b, 4, f8, %4, 0\n\t"
                "fmv.w.x f10, %5\n\t"
                "fmv.x.w %0, f8\n\t"
                "fmv.x.w %1, f9\n\t"
                "fmv.x.w %2, f10\n\t"
                "fmv.x.w %3, f11"
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                : "r"(ro + gp * 4), "r"(marker)
                : "f8", "f9", "f10", "f11", "memory");
  errors += (w[0] != (ro_word(gp) ^ salt)) + (w[1] != (ro_word(gp + 1) ^ salt))
          + (w[2] != marker) + (w[3] != (ro_word(gp + 3) ^ salt));
  asm volatile (".insn i 0x7b, 4, f8, %4, 0\n\t"
                "fadd.s f10, f10, f10\n\t"
                ".insn i 0x7b, 4, f8, %4, 16\n\t"
                "fmv.x.w %0, f8\n\t"
                "fmv.x.w %1, f9\n\t"
                "fmv.x.w %2, f10\n\t"
                "fmv.x.w %3, f11"
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                : "r"(ro + gs * 4)
                : "f8", "f9", "f10", "f11", "memory");
  errors += ro_misses(w, 4, gs + 4, salt);
#endif
  return errors;
}

// Wide stores into the thread's slot (local or global), then wide loads of
// the same words:
//   - .q at the slot, then on odd lanes (divergent) .q again over it at the
//     -2048 offset limit
//   - .q at a 4- but not 16-byte aligned word, then a scalar store into it
//   - .d at the 2040 offset limit and at a 4- but not 8-byte aligned word
//   - scalar stores between the wide ones
static inline uint32_t check_stores(uintptr_t slot, uint32_t tid, uint32_t lid) {
  auto s = reinterpret_cast<volatile uint32_t*>(slot);
  uint32_t v[16];
  for (uint32_t k = 0; k < 16; ++k) {
    v[k] = wide_word(tid, k);
  }
  uint32_t a[4];
  for (uint32_t k = 0; k < 4; ++k) {
    a[k] = alt_word(tid, k);
  }

  VX_FSW_Q("f8", "f9", "f10", "f11", 0, slot, v);
  if (lid & 1) {
    VX_FSW_Q("f12", "f13", "f14", "f15", -2048, slot + 2048, a);
  }
  s[4] = v[4];
  VX_FSW_Q("f16", "f17", "f18", "f19", 16, slot + 4, v + 5);
  s[7] = alt_word(tid, 7);
  s[9] = v[9];
  VX_FSW_D("f20", "f21", 2040, slot + 40 - 2040, v + 10);
  s[12] = v[12];
  VX_FSW_D("f22", "f23", 0, slot + 52, v + 13);
  s[15] = v[15];

  uint32_t errors = 0;
  uint32_t w[4];
  VX_FLW_Q("f24", "f25", "f26", "f27", 0, slot, w);
  for (uint32_t k = 0; k < 4; ++k) {
    errors += (w[k] != slot_word(tid, lid, k));
  }
  VX_FLW_Q("f0", "f1", "f2", "f3", 0, slot + 20, w);
  for (uint32_t k = 0; k < 4; ++k) {
    errors += (w[k] != slot_word(tid, lid, 5 + k));
  }
  VX_FLW_D("f4", "f5", 8, slot + 32, w);
  for (uint32_t k = 0; k < 2; ++k) {
    errors += (w[k] != slot_word(tid, lid, 10 + k));
  }
  VX_FLW_D("f6", "f7", 0, slot + 52, w);
  for (uint32_t k = 0; k < 2; ++k) {
    errors += (w[k] != slot_word(tid, lid, 13 + k));
  }
  VX_FLW_Q("f28", "f29", "f30", "f31", 32, slot + 16, w);
  for (uint32_t k = 0; k < 4; ++k) {
    errors += (w[k] != slot_word(tid, lid, 12 + k));
  }
  return errors;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto err  = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr);
  const uint32_t lid   = threadIdx.x;
  const uint32_t bsize = blockDim.x;
  const uint32_t tid   = blockIdx.x * bsize + lid;
  const uint32_t words = arg->lmem_words;

  if (tid == 0) {
  #ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    *reinterpret_cast<uint32_t*>(arg->wide_addr) = 1;
  #else
    *reinterpret_cast<uint32_t*>(arg->wide_addr) = 0;
  #endif
  }

  for (uint32_t i = lid; i < RO_WORDS; i += bsize) {
    lmem[i] = ro_word(i);
  }
  __syncthreads();

  const uintptr_t lmem_ro = reinterpret_cast<uintptr_t>(lmem);
  const uintptr_t lmem_slot = lmem_ro + (RO_WORDS + lid * SLOT_WORDS) * 4;
  const uintptr_t glob_slot = static_cast<uintptr_t>(arg->dst_addr) + tid * SLOT_WORDS * 4;

  uint32_t errors = 0;
  errors += check_loads(lmem_ro, 0, lid);
  errors += check_loads(static_cast<uintptr_t>(arg->src_addr), GLOBAL_SALT, lid);
  errors += check_stores(lmem_slot, tid, lid);
  errors += check_stores(glob_slot, tid, lid);
  err[tid] = errors;

  __syncthreads();
  for (uint32_t i = lid; i < words; i += bsize) {
    dump[blockIdx.x * words + i] = lmem[i];
  }
}
