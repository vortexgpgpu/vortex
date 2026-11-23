#include <vx_spawn2.h>
#include "common.h"

// FR(n) names the test's register n. Under Zfinx the groups live in x
// registers, so each aligned group of four maps onto one the kernel may
// clobber (never x0-x4); a test block never uses two groups that collide.
#ifdef VX_CFG_EXT_ZFINX_ENABLE
#define FR(n) FR_##n
#define FR_0  "x20"
#define FR_1  "x21"
#define FR_2  "x22"
#define FR_3  "x23"
#define FR_4  "x24"
#define FR_5  "x25"
#define FR_6  "x26"
#define FR_7  "x27"
#define FR_8  "x12"
#define FR_9  "x13"
#define FR_10 "x14"
#define FR_11 "x15"
#define FR_12 "x16"
#define FR_13 "x17"
#define FR_14 "x18"
#define FR_15 "x19"
#define FR_16 "x20"
#define FR_17 "x21"
#define FR_18 "x22"
#define FR_19 "x23"
#define FR_20 "x24"
#define FR_21 "x25"
#define FR_22 "x26"
#define FR_23 "x27"
#define FR_24 "x28"
#define FR_25 "x29"
#define FR_26 "x30"
#define FR_27 "x31"
#define FR_28 "x28"
#define FR_29 "x29"
#define FR_30 "x30"
#define FR_31 "x31"
#define MV_X_F "mv"
#define MV_F_X "mv"
#else
#define FR(n) "f" #n
#define MV_X_F "fmv.x.w"
#define MV_F_X "fmv.w.x"
#endif

// vx_flw.d/.q (custom-3 funct3 3/4, I-type like fld/flq) and vx_fsw.d/.q
// (funct3 5/6, S-type like fsd/fsq), written as .insn so the encodings are
// tested independently of the compiler. fd/fs2 is an aligned pair or quad and
// imm a multiple of 8 or 16; each word needs only 4-byte alignment.
#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
// A wide load and the reads of its registers sit in one asm block, so the
// compiler cannot reuse them in between. The last uop's register is read first.
#define VX_FLW_Q(r0, r1, r2, r3, off, p, w) \
  asm volatile (".insn i 0x7b, 4, " r0 ", %4, %5\n\t" \
                MV_X_F " %3, " r3 "\n\t" \
                MV_X_F " %0, " r0 "\n\t" \
                MV_X_F " %1, " r1 "\n\t" \
                MV_X_F " %2, " r2 \
                : "=&r"((w)[0]), "=&r"((w)[1]), "=&r"((w)[2]), "=&r"((w)[3]) \
                : "r"(p), "i"(off) \
                : r0, r1, r2, r3, "memory")

#define VX_FLW_D(r0, r1, off, p, w) \
  asm volatile (".insn i 0x7b, 3, " r0 ", %2, %3\n\t" \
                MV_X_F " %1, " r1 "\n\t" \
                MV_X_F " %0, " r0 \
                : "=&r"((w)[0]), "=&r"((w)[1]) \
                : "r"(p), "i"(off) \
                : r0, r1, "memory")

#define VX_FSW_Q(r0, r1, r2, r3, off, p, w) \
  asm volatile (MV_F_X " " r0 ", %0\n\t" \
                MV_F_X " " r1 ", %1\n\t" \
                MV_F_X " " r2 ", %2\n\t" \
                MV_F_X " " r3 ", %3\n\t" \
                ".insn s 0x7b, 6, " r0 ", %5(%4)" \
                : \
                : "r"((w)[0]), "r"((w)[1]), "r"((w)[2]), "r"((w)[3]), "r"(p), "i"(off) \
                : r0, r1, r2, r3, "memory")

#define VX_FSW_D(r0, r1, off, p, w) \
  asm volatile (MV_F_X " " r0 ", %0\n\t" \
                MV_F_X " " r1 ", %1\n\t" \
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
//   - fd = FR(0) and FR(28); two wide loads back to back
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

  VX_FLW_Q(FR(8), FR(9), FR(10), FR(11), 0, ro + gu * 4, w);
  errors += ro_misses(w, 4, gu, salt);
  VX_FLW_D(FR(8), FR(9), 8, ro + gu * 4, w);
  errors += ro_misses(w, 2, gu + 2, salt);

  VX_FLW_Q(FR(12), FR(13), FR(14), FR(15), 0, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs, salt);
  VX_FLW_D(FR(14), FR(15), 0, ro + gs * 4, w);
  errors += ro_misses(w, 2, gs, salt);
  VX_FLW_Q(FR(12), FR(13), FR(14), FR(15), 16, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs + 4, salt);

  VX_FLW_Q(FR(16), FR(17), FR(18), FR(19), 0, ro + gp * 4, w);
  errors += ro_misses(w, 4, gp, salt);
  VX_FLW_D(FR(16), FR(17), 8, ro + gp * 4, w);
  errors += ro_misses(w, 2, gp + 2, salt);

  if (lid & 1) {
    VX_FLW_Q(FR(20), FR(21), FR(22), FR(23), 0, ro + gs * 4, w);
    errors += ro_misses(w, 4, gs, salt);
    VX_FLW_D(FR(22), FR(23), 8, ro + gp * 4, w);
    errors += ro_misses(w, 2, gp + 2, salt);
  }

  VX_FLW_Q(FR(20), FR(21), FR(22), FR(23), 16, ro + (gs + 1) * 4, w);
  errors += ro_misses(w, 4, gs + 5, salt);
  VX_FLW_Q(FR(24), FR(25), FR(26), FR(27), 0, ro + (gp + 2) * 4, w);
  errors += ro_misses(w, 4, gp + 2, salt);
  VX_FLW_D(FR(26), FR(27), 8, ro + (gq - 1) * 4, w);
  errors += ro_misses(w, 2, gq + 1, salt);

  VX_FLW_Q(FR(0), FR(1), FR(2), FR(3), 0, ro + gp * 4, w);
  errors += ro_misses(w, 4, gp, salt);
  VX_FLW_Q(FR(28), FR(29), FR(30), FR(31), 0, ro + gs * 4, w);
  errors += ro_misses(w, 4, gs, salt);
  VX_FLW_D(FR(30), FR(31), 0, ro + gu * 4, w);
  errors += ro_misses(w, 2, gu, salt);

  VX_FLW_Q(FR(4), FR(5), FR(6), FR(7), -2048, ro + go * 4 + 2048, w);
  errors += ro_misses(w, 4, go, salt);
  VX_FLW_Q(FR(4), FR(5), FR(6), FR(7), 2032, ro + go * 4 - 2032, w);
  errors += ro_misses(w, 4, go, salt);
  VX_FLW_Q(FR(24), FR(25), FR(26), FR(27), -2048, ro + (gs + 1) * 4 + 2048, w);
  errors += ro_misses(w, 4, gs + 1, salt);
  VX_FLW_Q(FR(24), FR(25), FR(26), FR(27), 2032, ro + gq * 4 - 2032, w);
  errors += ro_misses(w, 4, gq, salt);
  VX_FLW_D(FR(26), FR(27), -2048, ro + gp * 4 + 2048, w);
  errors += ro_misses(w, 2, gp, salt);
  VX_FLW_D(FR(26), FR(27), 2040, ro + (gq + 1) * 4 - 2040, w);
  errors += ro_misses(w, 2, gq + 1, salt);

#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
  asm volatile (".insn i 0x7b, 4, " FR(8) ", %8, 0\n\t"
                ".insn i 0x7b, 4, " FR(12) ", %8, 16\n\t"
                MV_X_F " %0, " FR(8) "\n\t"
                MV_X_F " %1, " FR(9) "\n\t"
                MV_X_F " %2, " FR(10) "\n\t"
                MV_X_F " %3, " FR(11) "\n\t"
                MV_X_F " %4, " FR(12) "\n\t"
                MV_X_F " %5, " FR(13) "\n\t"
                MV_X_F " %6, " FR(14) "\n\t"
                MV_X_F " %7, " FR(15)
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3]),
                  "=&r"(w[4]), "=&r"(w[5]), "=&r"(w[6]), "=&r"(w[7])
                : "r"(ro + gs * 4)
                : FR(8), FR(9), FR(10), FR(11), FR(12), FR(13), FR(14), FR(15), "memory");
  errors += ro_misses(w, 8, gs, salt);

  const uint32_t marker = 0x7F00D000u | lid;
  asm volatile (MV_F_X " " FR(10) ", %1\n\t"
                ".insn i 0x7b, 4, " FR(8) ", %2, 0\n\t"
                MV_X_F " %0, " FR(10)
                : "=&r"(w[2])
                : "r"(marker), "r"(ro + gp * 4)
                : FR(8), FR(9), FR(10), FR(11), "memory");
  errors += (w[2] != (ro_word(gp + 2) ^ salt));
  asm volatile (".insn i 0x7b, 4, " FR(8) ", %4, 0\n\t"
                MV_F_X " " FR(10) ", %5\n\t"
                MV_X_F " %0, " FR(8) "\n\t"
                MV_X_F " %1, " FR(9) "\n\t"
                MV_X_F " %2, " FR(10) "\n\t"
                MV_X_F " %3, " FR(11)
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                : "r"(ro + gp * 4), "r"(marker)
                : FR(8), FR(9), FR(10), FR(11), "memory");
  errors += (w[0] != (ro_word(gp) ^ salt)) + (w[1] != (ro_word(gp + 1) ^ salt))
          + (w[2] != marker) + (w[3] != (ro_word(gp + 3) ^ salt));
  asm volatile (".insn i 0x7b, 4, " FR(8) ", %4, 0\n\t"
                "fadd.s " FR(10) ", " FR(10) ", " FR(10) "\n\t"
                ".insn i 0x7b, 4, " FR(8) ", %4, 16\n\t"
                MV_X_F " %0, " FR(8) "\n\t"
                MV_X_F " %1, " FR(9) "\n\t"
                MV_X_F " %2, " FR(10) "\n\t"
                MV_X_F " %3, " FR(11)
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3])
                : "r"(ro + gs * 4)
                : FR(8), FR(9), FR(10), FR(11), "memory");
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

  VX_FSW_Q(FR(8), FR(9), FR(10), FR(11), 0, slot, v);
  if (lid & 1) {
    VX_FSW_Q(FR(12), FR(13), FR(14), FR(15), -2048, slot + 2048, a);
  }
  s[4] = v[4];
  VX_FSW_Q(FR(16), FR(17), FR(18), FR(19), 16, slot + 4, v + 5);
  s[7] = alt_word(tid, 7);
  s[9] = v[9];
  VX_FSW_D(FR(20), FR(21), 2040, slot + 40 - 2040, v + 10);
  s[12] = v[12];
  VX_FSW_D(FR(22), FR(23), 0, slot + 52, v + 13);
  s[15] = v[15];

  uint32_t errors = 0;
  uint32_t w[4];
  VX_FLW_Q(FR(24), FR(25), FR(26), FR(27), 0, slot, w);
  for (uint32_t k = 0; k < 4; ++k) {
    errors += (w[k] != slot_word(tid, lid, k));
  }
  VX_FLW_Q(FR(0), FR(1), FR(2), FR(3), 0, slot + 20, w);
  for (uint32_t k = 0; k < 4; ++k) {
    errors += (w[k] != slot_word(tid, lid, 5 + k));
  }
  VX_FLW_D(FR(4), FR(5), 8, slot + 32, w);
  for (uint32_t k = 0; k < 2; ++k) {
    errors += (w[k] != slot_word(tid, lid, 10 + k));
  }
  VX_FLW_D(FR(6), FR(7), 0, slot + 52, w);
  for (uint32_t k = 0; k < 2; ++k) {
    errors += (w[k] != slot_word(tid, lid, 13 + k));
  }
  VX_FLW_Q(FR(28), FR(29), FR(30), FR(31), 32, slot + 16, w);
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
