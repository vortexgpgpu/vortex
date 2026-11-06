#include <vx_spawn2.h>
#include <vx_tensor.h>
#include "common.h"

namespace vt = vortex::tensor;
using cfg = vt::wmma_config_t<VX_CFG_NUM_THREADS>;

// vx_ldm fd, imm(rs1) and vx_stm fs2, imm(rs1) (custom-3 funct3 0/2, I- and
// S-type), written as .insn so the encodings are tested independently of the
// compiler. imm[4:0] is the fragment shape, imm[11:5] a byte offset; rs1 is
// the first of J registers, a multiple of J, holding one address per row group.
enum : uint32_t {
  SHAPE_A    = 0x00,
  SHAPE_A_SP = 0x01,
  SHAPE_B    = 0x06,
  SHAPE_B_SP = 0x07, // symmetric sparse only
  SHAPE_C    = 0x08,
  SHAPE_WG_C = 0x18
};

// A fragment of a shape: J row groups of C words, CS bytes apart in a row.
struct geom_t {
  uint32_t J, C, CS;
};

constexpr geom_t geom(uint32_t shape) {
  switch (shape) {
  case SHAPE_A:    return {cfg::m_steps, cfg::k_steps, cfg::tcK * 4};
  case SHAPE_A_SP: return {cfg::m_steps, cfg::k_steps / 2, cfg::tcK * 4};
  case SHAPE_B:    return {cfg::b_sub_steps, cfg::k_steps, cfg::tcK * 4};
  case SHAPE_B_SP: return {cfg::NRB, 1, 4};
  case SHAPE_C:    return {cfg::m_steps, cfg::n_steps, cfg::tcN * 4};
  default:         return {2, 4, cfg::tcN * 4};
  }
}

constexpr uint32_t lg2(uint32_t x) {
  return (x < 2) ? 0 : 1 + lg2(x / 2);
}

constexpr uint32_t bit(uint32_t x, uint32_t i) {
  return (x >> i) & 1;
}

// the top bit of a step index of lg bits
constexpr uint32_t hi(uint32_t x, uint32_t lg) {
  return bit(x, (lg > 0) ? (lg - 1) : 0);
}

// The register the WMMA/WGMMA expansion reads element c of row group j of a
// fragment from: the TCU's bank-conflict-free operand permutations, written
// out independently of the header, SimX and the RTL.
constexpr uint32_t frag_reg(uint32_t shape, uint32_t j, uint32_t c) {
  constexpr uint32_t lgM = lg2(cfg::m_steps), lgN = lg2(cfg::n_steps), lgK = lg2(cfg::k_steps);
  constexpr bool two_sb = (cfg::b_sub_blocks > 1);
  switch (shape) {
  case SHAPE_A: // A={m[0],~m[hi],k} or {k[0],~m,m^k[hi]}
    return two_sb ? ((bit(c, 0) << 2) | ((1 - bit(j, 0)) << 1) | (bit(j, 0) ^ hi(c, lgK)))
                  : ((bit(j, 0) << 2) | ((1 - hi(j, lgM)) << 1) | bit(c, 0));
  case SHAPE_B: { // B={n^k,~k} or {k[0],k[hi]^n[hi],~n[hi]}, n the row group's first step
    uint32_t n = j * cfg::b_sub_blocks;
    return two_sb ? ((bit(c, 0) << 2) | ((hi(c, lgK) ^ hi(n, lgN)) << 1) | (1 - hi(n, lgN)))
                  : (((bit(n, 0) ^ bit(c, 0)) << 1) | (1 - bit(c, 0)));
  }
  case SHAPE_C: // C={m[0],~m[hi],~(m[hi]^n)} or {n[0],~m,n[hi]}
    return two_sb ? ((bit(c, 0) << 2) | ((1 - bit(j, 0)) << 1) | hi(c, lgN))
                  : ((bit(j, 0) << 2) | ((1 - hi(j, lgM)) << 1) | (1 - (hi(j, lgM) ^ bit(c, 0))));
  case SHAPE_WG_C: // n-major
    return (c << 1) | j;
  default: // sparse A and B: element order
    return j * geom(shape).C + c;
  }
}

template <uint32_t SHAPE>
constexpr bool frag_reg_is_permutation() {
  constexpr geom_t g = geom(SHAPE);
  uint32_t seen = 0;
  for (uint32_t j = 0; j < g.J; ++j) {
    for (uint32_t c = 0; c < g.C; ++c) {
      uint32_t r = frag_reg(SHAPE, j, c);
      if (r >= g.J * g.C || (seen & (1u << r))) {
        return false;
      }
      seen |= 1u << r;
    }
  }
  return true;
}

constexpr uint32_t pow2ceil(uint32_t x) {
  uint32_t p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

// One asm block: vx_ldm from the source group, read every register of the
// fragment, vx_stm the same registers to the destination group.
#define FRAG_MOVE8(imm, r0, r1, r2, r3, r4, r5, r6, r7) \
  asm volatile (".insn i 0x7b, 0, " r0 ", %[gs], %[im]\n\t" \
                "fmv.x.w %0, " r0 "\n\t" "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %2, " r2 "\n\t" "fmv.x.w %3, " r3 "\n\t" \
                "fmv.x.w %4, " r4 "\n\t" "fmv.x.w %5, " r5 "\n\t" \
                "fmv.x.w %6, " r6 "\n\t" "fmv.x.w %7, " r7 "\n\t" \
                ".insn s 0x7b, 2, " r0 ", %[im](%[gd])" \
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3]), \
                  "=&r"(w[4]), "=&r"(w[5]), "=&r"(w[6]), "=&r"(w[7]) \
                : [gs]"r"(s0), "r"(s1), "r"(s2), "r"(s3), \
                  [gd]"r"(d0), "r"(d1), "r"(d2), "r"(d3), [im]"i"(imm) \
                : r0, r1, r2, r3, r4, r5, r6, r7, "memory")

#define FRAG_MOVE4(imm, r0, r1, r2, r3) \
  asm volatile (".insn i 0x7b, 0, " r0 ", %[gs], %[im]\n\t" \
                "fmv.x.w %0, " r0 "\n\t" "fmv.x.w %1, " r1 "\n\t" \
                "fmv.x.w %2, " r2 "\n\t" "fmv.x.w %3, " r3 "\n\t" \
                ".insn s 0x7b, 2, " r0 ", %[im](%[gd])" \
                : "=&r"(w[0]), "=&r"(w[1]), "=&r"(w[2]), "=&r"(w[3]) \
                : [gs]"r"(s0), "r"(s1), "r"(s2), "r"(s3), \
                  [gd]"r"(d0), "r"(d1), "r"(d2), "r"(d3), [im]"i"(imm) \
                : r0, r1, r2, r3, "memory")

// Moves the fragment of SHAPE at byte offset OFF, with fd = FD, between the
// thread's source and destination regions, whose row groups are row bytes
// apart from byte S on; returns the number of wrong words.
template <uint32_t SHAPE, uint32_t FD, int32_t OFF>
uint32_t check(volatile uint32_t* src, volatile uint32_t* dst, uint32_t region_words, uint32_t row) {
  constexpr geom_t g = geom(SHAPE);
  constexpr uint32_t NR = g.J * g.C;
  constexpr uint32_t S = pow2ceil((g.C * g.CS > 32) ? (g.C * g.CS) : 32);
  static_assert((g.J == 2 || g.J == 4) && (NR == 4 || NR == 8), "unexpected fragment geometry");
  static_assert(frag_reg_is_permutation<SHAPE>(), "fragment register map is not a permutation");

  uintptr_t gs[4] = {0, 0, 0, 0};
  uintptr_t gd[4] = {0, 0, 0, 0};
  for (uint32_t j = 0; j < g.J; ++j) {
    gs[j] = reinterpret_cast<uintptr_t>(src) + S + j * row;
    gd[j] = reinterpret_cast<uintptr_t>(dst) + S + j * row;
  }
  register uintptr_t s0 __asm__("x28") = gs[0];
  register uintptr_t s1 __asm__("x29") = gs[1];
  register uintptr_t s2 __asm__("x30") = gs[2];
  register uintptr_t s3 __asm__("x31") = gs[3];
  register uintptr_t d0 __asm__("x12") = gd[0];
  register uintptr_t d1 __asm__("x13") = gd[1];
  register uintptr_t d2 __asm__("x14") = gd[2];
  register uintptr_t d3 __asm__("x15") = gd[3];

  constexpr int32_t imm = OFF | int32_t(SHAPE);
  uint32_t w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if constexpr (NR == 8 && FD == 0) {
    FRAG_MOVE8(imm, "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7");
  } else if constexpr (NR == 8 && FD == 10) {
    FRAG_MOVE8(imm, "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
  } else if constexpr (NR == 8 && FD == 13) {
    FRAG_MOVE8(imm, "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20");
  } else if constexpr (NR == 8 && FD == 24) {
    FRAG_MOVE8(imm, "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31");
  } else if constexpr (NR == 4 && FD == 10) {
    FRAG_MOVE4(imm, "f10", "f11", "f12", "f13");
  } else {
    static_assert(NR == 4 && FD == 28, "unsupported fragment registers");
    FRAG_MOVE4(imm, "f28", "f29", "f30", "f31");
  }

  // the footprint, as word indices into the regions
  uint32_t foot[8];
  uint32_t errors = 0;
  for (uint32_t j = 0, e = 0; j < g.J; ++j) {
    for (uint32_t c = 0; c < g.C; ++c, ++e) {
      foot[e] = (S + j * row + OFF + c * g.CS) / 4;
      // each register holds the element the TCU reads from it
      errors += (w[frag_reg(SHAPE, j, c)] != src[foot[e]]);
    }
  }
  // the store wrote the footprint; clearing it lets the final scans of the
  // destination regions catch a word written anywhere else
  for (uint32_t e = 0; e < NR; ++e) {
    errors += (foot[e] >= region_words) || (dst[foot[e]] != src[foot[e]]);
  }
  for (uint32_t e = 0; e < NR; ++e) {
    dst[foot[e]] = 0;
  }
  return errors;
}

// Every offset form on one shape and register base: zero, positive, negative.
template <uint32_t SHAPE, uint32_t FD>
uint32_t check_offsets(volatile uint32_t* src, volatile uint32_t* dst, uint32_t region_words, uint32_t row) {
  constexpr geom_t g = geom(SHAPE);
  constexpr int32_t S = int32_t(pow2ceil((g.C * g.CS > 32) ? (g.C * g.CS) : 32));
  uint32_t errors = 0;
  errors += check<SHAPE, FD, 0>(src, dst, region_words, row);
  errors += check<SHAPE, FD, S>(src, dst, region_words, row);
  errors += check<SHAPE, FD, 2 * S>(src, dst, region_words, row);
  errors += check<SHAPE, FD, -S>(src, dst, region_words, row);
  return errors;
}

constexpr uint32_t kFdB = (cfg::NRB == 8) ? 24 : 28;

// The same move through the vx_ldm/vx_stm intrinsics, with the register bases
// and 4-register groups (QS for the load, QD for the store) the fragment
// helpers of vx_tensor.h do not use.
template <uint32_t SHAPE, uint32_t FD, uint32_t QS, uint32_t QD>
uint32_t check_intrinsics(volatile uint32_t* src, volatile uint32_t* dst, uint32_t region_words, uint32_t row) {
  constexpr geom_t g = geom(SHAPE);
  constexpr uint32_t NR = g.J * g.C;
  constexpr uint32_t S = pow2ceil((g.C * g.CS > 32) ? (g.C * g.CS) : 32);
  uintptr_t gs[g.J];
  uintptr_t gd[g.J];
  for (uint32_t j = 0; j < g.J; ++j) {
    gs[j] = reinterpret_cast<uintptr_t>(src) + S + j * row;
    gd[j] = reinterpret_cast<uintptr_t>(dst) + S + j * row;
  }
  float v[NR];
  vx_ldm<FD, NR, g.J, QS, int32_t(SHAPE)>(v, gs);
  vx_stm<FD, NR, g.J, QD, int32_t(SHAPE)>(v, gd);

  uint32_t foot[NR];
  uint32_t errors = 0;
  for (uint32_t j = 0, e = 0; j < g.J; ++j) {
    for (uint32_t c = 0; c < g.C; ++c, ++e) {
      foot[e] = (S + j * row + c * g.CS) / 4;
      uint32_t w;
      __builtin_memcpy(&w, &v[frag_reg(SHAPE, j, c)], sizeof(w));
      errors += (w != src[foot[e]]);
    }
  }
  for (uint32_t e = 0; e < NR; ++e) {
    errors += (foot[e] >= region_words) || (dst[foot[e]] != src[foot[e]]);
  }
  for (uint32_t e = 0; e < NR; ++e) {
    dst[foot[e]] = 0;
  }
  return errors;
}

uint32_t check_shapes(volatile uint32_t* src, volatile uint32_t* dst, uint32_t region_words, uint32_t row) {
  uint32_t errors = 0;
  errors += check_offsets<SHAPE_A, 10>(src, dst, region_words, row);
  errors += check_offsets<SHAPE_A, 13>(src, dst, region_words, row);
  errors += check_offsets<SHAPE_B, kFdB>(src, dst, region_words, row);
  errors += check_offsets<SHAPE_C, 0>(src, dst, region_words, row);
  errors += check_offsets<SHAPE_A_SP, 10>(src, dst, region_words, row);
  if constexpr (cfg::sym_sparse) {
    errors += check_offsets<SHAPE_B_SP, kFdB>(src, dst, region_words, row);
  }
  errors += check_offsets<SHAPE_WG_C, 0>(src, dst, region_words, row);
  errors += check_offsets<SHAPE_WG_C, 24>(src, dst, region_words, row);
  errors += check_intrinsics<SHAPE_A, 3, 16, 20>(src, dst, region_words, row);
  errors += check_intrinsics<SHAPE_A_SP, 21, 24, 16>(src, dst, region_words, row);
  errors += check_intrinsics<SHAPE_B, 17, 20, 24>(src, dst, region_words, row);
  errors += check_intrinsics<SHAPE_C, 9, 24, 12>(src, dst, region_words, row);
  errors += check_intrinsics<SHAPE_WG_C, 22, 28, 16>(src, dst, region_words, row);
  return errors;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto lmem = reinterpret_cast<volatile uint32_t*>(__local_mem());
  const uint32_t lid = threadIdx.x;
  const uint32_t tid = blockIdx.x * blockDim.x + lid;

  if (tid == 0) {
    uint32_t shapes = (1u << SHAPE_A) | (1u << SHAPE_A_SP) | (1u << SHAPE_B) | (1u << SHAPE_C)
                    | (1u << SHAPE_WG_C) | (cfg::sym_sparse ? (1u << SHAPE_B_SP) : 0);
    *reinterpret_cast<uint32_t*>(arg->shape_addr) = shapes;
  }

  // local regions: this thread's source, then its destination
  auto lsrc = lmem + lid * 2 * LOCAL_REGION_WORDS;
  auto ldst = lsrc + LOCAL_REGION_WORDS;
  for (uint32_t i = 0; i < LOCAL_REGION_WORDS; ++i) {
    lsrc[i] = src_word(tid, i, LOCAL_SALT);
    ldst[i] = 0;
  }
  auto gsrc = reinterpret_cast<volatile uint32_t*>(arg->src_addr) + tid * GLOBAL_REGION_WORDS;
  auto gdst = reinterpret_cast<volatile uint32_t*>(arg->dst_addr) + tid * GLOBAL_REGION_WORDS;

  // row strides that are not powers of two; the global one exceeds 256 bytes
  uint32_t errors = 0;
  errors += check_shapes(lsrc, ldst, LOCAL_REGION_WORDS, 68);
  errors += check_shapes(gsrc, gdst, GLOBAL_REGION_WORDS, 68);
  errors += check_shapes(gsrc, gdst, GLOBAL_REGION_WORDS, 260);
  // the host scans the global destination regions
  for (uint32_t i = 0; i < LOCAL_REGION_WORDS; ++i) {
    errors += (ldst[i] != 0);
  }
  reinterpret_cast<uint32_t*>(arg->err_addr)[tid] = errors;
}
