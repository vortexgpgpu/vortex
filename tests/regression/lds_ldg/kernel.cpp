#include <vx_spawn2.h>
#include "common.h"

// vx_lds/vx_sts (custom-3) and vx_ldg/vx_stg (custom-0), written as .insn so
// the encodings are tested independently of the compiler.
//   vx_lds:  I-type, funct3 = LOAD width, imm[11] = FP, imm[10:0] = offset
//   vx_sts:  S-type, funct3 = 7, imm[11] = FP, imm[10:9] = size, imm[8:0] = offset
//   vx_ldg:  R-type, funct7 = 0b0001_F_HH, funct3 = LOAD width, rs2 = uimm5 offset
//   vx_stg:  R-type, funct7 = 0b0010_F_HH, funct3 = size, rd = uimm5 offset

namespace {

template <int F3, int OFF>
inline uint32_t lds(uintptr_t p) {
  uint32_t v;
  asm volatile (".insn i 0x7b, %2, %0, %1, %3" : "=r"(v) : "r"(p), "i"(F3), "i"(OFF & 0x7ff) : "memory");
  return v;
}

template <int OFF>
inline uint32_t flds_w(uintptr_t p) {
  float v;
  asm volatile (".insn i 0x7b, 2, %0, %1, %2" : "=f"(v) : "r"(p), "i"((OFF & 0x7ff) - 2048) : "memory");
  uint32_t bits;
  __builtin_memcpy(&bits, &v, sizeof(bits));
  return bits;
}

constexpr int sts_imm(int fp, int size, int off) {
  return ((fp << 11) | (size << 9) | (off & 0x1ff)) - (fp ? 4096 : 0);
}

template <int SIZE, int OFF>
inline void sts(uintptr_t p, uint32_t v) {
  asm volatile (".insn s 0x7b, 7, %0, %2(%1)" :: "r"(v), "r"(p), "i"(sts_imm(0, SIZE, OFF)) : "memory");
}

template <int OFF>
inline void fsts_w(uintptr_t p, uint32_t bits) {
  float v;
  __builtin_memcpy(&v, &bits, sizeof(v));
  asm volatile (".insn s 0x7b, 7, %0, %2(%1)" :: "f"(v), "r"(p), "i"(sts_imm(1, 2, OFF)) : "memory");
}

} // namespace

#define VX_LDG(f3, f7, imm, p) ({ \
  uint32_t _v; \
  asm volatile (".insn r 0x0b, " #f3 ", " #f7 ", %0, %1, x" #imm : "=r"(_v) : "r"(p) : "memory"); \
  _v; })

#define VX_FLDG_W(f7, imm, p) ({ \
  float _f; \
  asm volatile (".insn r 0x0b, 2, " #f7 ", %0, %1, x" #imm : "=f"(_f) : "r"(p) : "memory"); \
  uint32_t _v; \
  __builtin_memcpy(&_v, &_f, sizeof(_v)); \
  _v; })

#define VX_STG(f3, f7, imm, p, v) \
  asm volatile (".insn r 0x0b, " #f3 ", " #f7 ", x" #imm ", %0, %1" :: "r"(p), "r"(v) : "memory")

#define VX_FSTG_W(f7, imm, p, bits) do { \
  float _f; \
  uint32_t _b = (bits); \
  __builtin_memcpy(&_f, &_b, sizeof(_f)); \
  asm volatile (".insn r 0x0b, 2, " #f7 ", x" #imm ", %0, %1" :: "r"(p), "f"(_f) : "memory"); \
} while (0)

static inline uint32_t sext8(uint32_t v)  { return (uint32_t)(int32_t)(int8_t)v; }
static inline uint32_t sext16(uint32_t v) { return (uint32_t)(int32_t)(int16_t)v; }

// Each thread checks, against the values the host and the generic path
// define:
//   vx_lds   - words at the offset limits, through the window address and
//              through the bare offset (local memory decodes only the low
//              bits); bytes and half-words, signed and unsigned; an FP word
//   vx_sts   - words at the offset limits, half-words through the bare
//              offset, bytes, an FP word; each read back by a generic load
//   ordering - a generic store, then a vx_lds of the same word
//   vx_ldg   - every width from offset 0 to the 31-byte limit, the FP form,
//              every hint
//   vx_stg   - words, half-words, bytes up to the 31-byte limit, the FP form
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  auto in   = reinterpret_cast<const uint32_t*>(arg->in_addr);
  auto out  = reinterpret_cast<uint32_t*>(arg->out_addr);
  auto err  = reinterpret_cast<uint32_t*>(arg->err_addr);
  auto dump = reinterpret_cast<uint32_t*>(arg->dump_addr);
  const uint32_t lid   = threadIdx.x;
  const uint32_t bsize = blockDim.x;
  const uint32_t tid   = blockIdx.x * bsize + lid;
  const uint32_t words = arg->lmem_words;

  for (uint32_t i = lid; i < RO_WORDS; i += bsize) {
    lmem[i] = ro_word(i);
  }
  __syncthreads();

  uint32_t errors = 0;
  const uintptr_t window = reinterpret_cast<uintptr_t>(lmem);
  const uintptr_t bare   = window - VX_MEM_LMEM_BASE_ADDR;

  // reads reach qi - 256 .. qi + 255 words, inside the read-only words
  const uint32_t qi = 384 + (lid * 7) % 256;
  const uintptr_t q  = window + qi * 4;
  const uintptr_t qb = bare + qi * 4;
  if (lds<2, 0>(q) != ro_word(qi)) {
    ++errors;
  }
  if (lds<2, 1020>(q) != ro_word(qi + 255)) {
    ++errors;
  }
  if (lds<2, -1024>(q) != ro_word(qi - 256)) {
    ++errors;
  }
  if (lds<2, 0>(qb) != ro_word(qi)) {
    ++errors;
  }
  if (lds<2, 1020>(qb) != ro_word(qi + 255)) {
    ++errors;
  }
  if (lds<2, -1024>(qb) != ro_word(qi - 256)) {
    ++errors;
  }

  {
    const uint32_t v  = ro_word(qi);
    const uint32_t bo = lid & 3;
    const uint32_t ho = lid & 1;
    if (lds<0, 0>(q + bo) != sext8(v >> (8 * bo))) {
      ++errors;
    }
    if (lds<4, 0>(q + bo) != ((v >> (8 * bo)) & 0xff)) {
      ++errors;
    }
    if (lds<1, 0>(q + 2 * ho) != sext16(v >> (16 * ho))) {
      ++errors;
    }
    if (lds<5, 0>(q + 2 * ho) != ((v >> (16 * ho)) & 0xffff)) {
      ++errors;
    }
    if (lds<4, 3>(q) != (v >> 24)) {
      ++errors;
    }
    if (lds<0, -1>(q) != sext8(ro_word(qi - 1) >> 24)) {
      ++errors;
    }
    if (lds<1, -2>(qb) != sext16(ro_word(qi - 1) >> 16)) {
      ++errors;
    }
    if (flds_w<4>(q) != ro_word(qi + 1)) {
      ++errors;
    }
    if (flds_w<-1024>(qb) != ro_word(qi - 256)) {
      ++errors;
    }
  }

  {
    const uint32_t si = RO_WORDS + lid * SLOT_WORDS;
    const uintptr_t s  = window + si * 4;
    const uintptr_t sb = bare + si * 4;
    sts<2, -256>(s + 256, slot_word(tid, 0));
    sts<2, 252>(s + 4 - 252, slot_word(tid, 1));
    sts<1, 0>(sb + 8, slot_word(tid, 2));
    sts<1, 2>(sb + 8, slot_word(tid, 2) >> 16);
    fsts_w<12>(s, slot_word(tid, 3));
    sts<0, 16>(s, slot_word(tid, 4));
    sts<0, 17>(s, slot_word(tid, 4) >> 8);
    sts<0, -238>(s + 256, slot_word(tid, 4) >> 16);
    sts<0, 19>(sb, slot_word(tid, 4) >> 24);
    for (uint32_t k = 0; k < SLOT_WORDS; ++k) {
      if (lmem[si + k] != slot_word(tid, k)) {
        ++errors;
      }
    }
    lmem[si] = ~slot_word(tid, 0);
    if (lds<2, 0>(s) != ~slot_word(tid, 0)) {
      ++errors;
    }
    lmem[si] = slot_word(tid, 0);
  }

  {
    const uintptr_t g = reinterpret_cast<uintptr_t>(in + tid);
    if (VX_LDG(2, 0x08, 0, g) != in_word(tid)) {
      ++errors;
    }
    if (VX_LDG(2, 0x09, 28, g) != in_word(tid + 7)) {
      ++errors;
    }
    if (VX_LDG(0, 0x0a, 1, g) != sext8(in_word(tid) >> 8)) {
      ++errors;
    }
    if (VX_LDG(4, 0x09, 31, g) != (in_word(tid + 7) >> 24)) {
      ++errors;
    }
    if (VX_LDG(1, 0x08, 30, g) != sext16(in_word(tid + 7) >> 16)) {
      ++errors;
    }
    if (VX_LDG(5, 0x08, 0, g) != (in_word(tid) & 0xffff)) {
      ++errors;
    }
    if (VX_FLDG_W(0x0c, 12, g) != in_word(tid + 3)) {
      ++errors;
    }
    if (VX_FLDG_W(0x0e, 28, g) != in_word(tid + 7)) {
      ++errors;
    }

    const uintptr_t o = reinterpret_cast<uintptr_t>(out + tid * SLOT_WORDS);
    VX_STG(2, 0x10, 0, o, out_word(tid, 0));
    VX_STG(2, 0x11, 28, o + 4 - 28, out_word(tid, 1));
    VX_STG(1, 0x12, 8, o, out_word(tid, 2));
    VX_STG(1, 0x10, 10, o, out_word(tid, 2) >> 16);
    VX_FSTG_W(0x14, 12, o, out_word(tid, 3));
    VX_STG(0, 0x10, 16, o, out_word(tid, 4));
    VX_STG(0, 0x12, 31, o + 17 - 31, out_word(tid, 4) >> 8);
    VX_STG(0, 0x10, 18, o, out_word(tid, 4) >> 16);
    VX_STG(0, 0x10, 19, o, out_word(tid, 4) >> 24);
  }

  err[tid] = errors;

  __syncthreads();
  for (uint32_t i = lid; i < words; i += bsize) {
    dump[blockIdx.x * words + i] = lmem[i];
  }
}
