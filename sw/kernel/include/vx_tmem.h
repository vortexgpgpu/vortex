// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __VX_TMEM_H__
#define __VX_TMEM_H__

#include <stdint.h>
#include <vx_intrinsics.h>

// Tensor memory (CUSTOM0, funct7 = 5) and tcgen05-style async tensor ops
// (funct7 = 6).
//
// TMEM is one per-core array of lanes x 32-bit columns, addressed
// {lane [31:16], column [15:0]}. A lane is an M row of an MMA: TCU block b
// computes lanes [b*M, (b+1)*M), M being the WGMMA tile M. A column is an N
// element.

namespace vortex {
namespace tmem {

// Instruction descriptor of a tc.mma slot: formats, n_frag N fragments and
// k_tiles K-tiles per op.
// sparse: A is 2:4 compressed and the op's metadata column is the first TMEM
// column of its metadata. One K-tile's metadata is one column per 32 bits: a
// lane holds, from bit 0, the VX_tcu_sp_mux rows of the K-tile's lower then
// upper TC_K words.
// a_tmem: the op's A operand is A's first TMEM column instead of a stage
// offset; a lane is an M row, and a K-tile is fedpK columns of K-packed words.
inline constexpr uint32_t make_idesc(uint32_t fmt_s, uint32_t fmt_d, uint32_t n_frag, uint32_t k_tiles,
                                     bool sparse = false, bool a_tmem = false) {
  return (fmt_s & 0x1f) | ((fmt_d & 0x1f) << 5) | ((sparse ? 1u : 0u) << 10) | ((a_tmem ? 1u : 0u) << 12)
       | ((n_frag & 0xff) << 13) | ((k_tiles & 0xff) << 24);
}

// Leading bytes of A and B; ldb = 0 selects block-major B (dense only).
inline constexpr uint32_t make_ld(uint32_t lda, uint32_t ldb) {
  return (lda & 0xffff) | (ldb << 16);
}

// Shared-memory byte offset of a pointer, as the descriptors expect.
inline uint32_t smem_offset(const void* ptr) {
  size_t lmem_base = csr_read_nv(VX_CSR_LOCAL_MEM_BASE);
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) - lmem_base);
}

// tmem.ld/st register layouts (funct3 bit 2).
enum : uint32_t {
  LANE_PER_THREAD = 0,
  LANE_PER_REG    = 4,
};

inline constexpr uint32_t make_taddr(uint32_t lane, uint32_t col) {
  return (lane << 16) | col;
}

// The smallest allocation holding `cols` columns.
inline constexpr uint32_t alloc_cols(uint32_t cols) {
  uint32_t n = 32;
  while (n < cols) {
    n <<= 1;
  }
  return n;
}

// tcgen05.alloc: the first of `ncols` TMEM columns (a power of two from 32 to
// VX_CFG_TCU_TMEM_COLS), retried until another CTA frees enough. One warp of
// the CTA allocates and shares the column; the CTA frees it with dealloc().
inline uint32_t alloc(uint32_t ncols) {
  uint32_t col;
  do {
    __asm__ volatile (".insn r %1, 6, 5, %0, %2, x0"
      : "=r"(col) : "i"(RISCV_CUSTOM0), "r"(ncols) : "memory");
  } while (col == ~0u);
  return col;
}

// tcgen05.dealloc: waits for the CTA's ops on the columns, then frees them.
inline void dealloc(uint32_t col, uint32_t ncols) {
  __asm__ volatile (".insn r %0, 7, 5, x0, %1, %2"
    :: "i"(RISCV_CUSTOM0), "r"(col), "r"(ncols) : "memory");
}

// Load the CTA's tc.mma descriptor slot Slot (0..3) with an instruction
// descriptor and leading bytes, for the CTA's later tc.mma ops on the slot. A
// tc.mma from another warp needs a barrier after it.
template <uint32_t Slot>
inline void set_idesc(uint32_t idesc, uint32_t ld) {
  static_assert(Slot < 4, "four descriptor slots");
  __asm__ volatile (".insn r %0, 0, 6, x%1, %2, %3"
    :: "i"(RISCV_CUSTOM0), "i"(Slot), "r"(idesc), "r"(ld) : "memory");
}

// tcgen05.mma: D (column taddr) += A . B with descriptor slot Slot, A and B at
// shared-memory byte offsets (or A at a TMEM column with a_tmem), issued
// asynchronously. tc.* ops take their scalars from thread 0, which must be
// active. acc_init zeroes D first; meta is the first metadata column of a
// sparse op. Ops of a CTA complete in order; tc.commit tracks them.
template <uint32_t Slot>
inline void mma(uint32_t taddr, uint32_t a, uint32_t b, bool acc_init = false, uint32_t meta = 0) {
  static_assert(Slot < 4, "four descriptor slots");
  uint32_t w = (taddr & 0xffff) | ((meta & 0x7fff) << 16) | (acc_init ? 0x80000000u : 0u);
  __asm__ volatile (".insn r %0, %1, 6, %2, %3, %4"
    :: "i"(RISCV_CUSTOM0), "i"(4 + Slot), "r"(b), "r"(w), "r"(a) : "memory");
}

// tcgen05.commit: arrive on barrier `bar`, as one of its `count` participants,
// once the CTA's earlier tc.mma and tc.cp ops have completed.
inline void commit(uint32_t bar, uint32_t count) {
  __asm__ volatile (".insn r %0, 1, 6, x0, %1, %2"
    :: "i"(RISCV_CUSTOM0), "r"(bar), "r"(count) : "memory");
}

// tcgen05.cp: copy `cols` columns of every TMEM lane from shared memory, lane l
// from `src + l * stride` bytes, in order with the CTA's other ops.
// src and stride are NT-word aligned; cols is a multiple of NT.
inline void cp(uint32_t col, const void* src, uint32_t stride, uint32_t cols) {
  uint32_t w = (col & 0xffff) | (cols << 16);
  __asm__ volatile (".insn r %0, 2, 6, %1, %2, %3"
    :: "i"(RISCV_CUSTOM0), "r"(stride), "r"(w), "r"(smem_offset(src)) : "memory");
}

// TMEM -> registers f0..f(NR-1). taddr's lane is a multiple of M.
// LANE_PER_THREAD: thread t gets lane + t % M, register r column
//   + r * (NT / M) + t / M.
// LANE_PER_REG: register r gets lane + r % M, thread t column + (r / M) * NT + t,
//   so each register is NT contiguous columns of one row.
template <uint32_t Shape = LANE_PER_THREAD, typename Frag>
inline __attribute__((always_inline)) void load(Frag& frag, uint32_t taddr) {
  constexpr uint32_t F3 = 0 | Shape;
  static_assert(Frag::NR == 8 || Frag::NR == 16 || Frag::NR == 32, "unsupported fragment size");
  if constexpr (Frag::NR == 8) {
    register float fd0 __asm__("f0");
    register float fd1 __asm__("f1");
    register float fd2 __asm__("f2");
    register float fd3 __asm__("f3");
    register float fd4 __asm__("f4");
    register float fd5 __asm__("f5");
    register float fd6 __asm__("f6");
    register float fd7 __asm__("f7");
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x7"
      : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3),
      "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr)
      : "memory");
    frag.data[0] = fd0;
    frag.data[1] = fd1;
    frag.data[2] = fd2;
    frag.data[3] = fd3;
    frag.data[4] = fd4;
    frag.data[5] = fd5;
    frag.data[6] = fd6;
    frag.data[7] = fd7;
  } else if constexpr (Frag::NR == 16) {
    register float fd0 __asm__("f0");
    register float fd1 __asm__("f1");
    register float fd2 __asm__("f2");
    register float fd3 __asm__("f3");
    register float fd4 __asm__("f4");
    register float fd5 __asm__("f5");
    register float fd6 __asm__("f6");
    register float fd7 __asm__("f7");
    register float fd8 __asm__("f8");
    register float fd9 __asm__("f9");
    register float fd10 __asm__("f10");
    register float fd11 __asm__("f11");
    register float fd12 __asm__("f12");
    register float fd13 __asm__("f13");
    register float fd14 __asm__("f14");
    register float fd15 __asm__("f15");
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x15"
      : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3),
      "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7),
      "=f"(fd8), "=f"(fd9), "=f"(fd10), "=f"(fd11),
      "=f"(fd12), "=f"(fd13), "=f"(fd14), "=f"(fd15)
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr)
      : "memory");
    frag.data[0] = fd0;
    frag.data[1] = fd1;
    frag.data[2] = fd2;
    frag.data[3] = fd3;
    frag.data[4] = fd4;
    frag.data[5] = fd5;
    frag.data[6] = fd6;
    frag.data[7] = fd7;
    frag.data[8] = fd8;
    frag.data[9] = fd9;
    frag.data[10] = fd10;
    frag.data[11] = fd11;
    frag.data[12] = fd12;
    frag.data[13] = fd13;
    frag.data[14] = fd14;
    frag.data[15] = fd15;
  } else if constexpr (Frag::NR == 32) {
    register float fd0 __asm__("f0");
    register float fd1 __asm__("f1");
    register float fd2 __asm__("f2");
    register float fd3 __asm__("f3");
    register float fd4 __asm__("f4");
    register float fd5 __asm__("f5");
    register float fd6 __asm__("f6");
    register float fd7 __asm__("f7");
    register float fd8 __asm__("f8");
    register float fd9 __asm__("f9");
    register float fd10 __asm__("f10");
    register float fd11 __asm__("f11");
    register float fd12 __asm__("f12");
    register float fd13 __asm__("f13");
    register float fd14 __asm__("f14");
    register float fd15 __asm__("f15");
    register float fd16 __asm__("f16");
    register float fd17 __asm__("f17");
    register float fd18 __asm__("f18");
    register float fd19 __asm__("f19");
    register float fd20 __asm__("f20");
    register float fd21 __asm__("f21");
    register float fd22 __asm__("f22");
    register float fd23 __asm__("f23");
    register float fd24 __asm__("f24");
    register float fd25 __asm__("f25");
    register float fd26 __asm__("f26");
    register float fd27 __asm__("f27");
    register float fd28 __asm__("f28");
    register float fd29 __asm__("f29");
    register float fd30 __asm__("f30");
    register float fd31 __asm__("f31");
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x31"
      : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3),
      "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7),
      "=f"(fd8), "=f"(fd9), "=f"(fd10), "=f"(fd11),
      "=f"(fd12), "=f"(fd13), "=f"(fd14), "=f"(fd15),
      "=f"(fd16), "=f"(fd17), "=f"(fd18), "=f"(fd19),
      "=f"(fd20), "=f"(fd21), "=f"(fd22), "=f"(fd23),
      "=f"(fd24), "=f"(fd25), "=f"(fd26), "=f"(fd27),
      "=f"(fd28), "=f"(fd29), "=f"(fd30), "=f"(fd31)
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr)
      : "memory");
    frag.data[0] = fd0;
    frag.data[1] = fd1;
    frag.data[2] = fd2;
    frag.data[3] = fd3;
    frag.data[4] = fd4;
    frag.data[5] = fd5;
    frag.data[6] = fd6;
    frag.data[7] = fd7;
    frag.data[8] = fd8;
    frag.data[9] = fd9;
    frag.data[10] = fd10;
    frag.data[11] = fd11;
    frag.data[12] = fd12;
    frag.data[13] = fd13;
    frag.data[14] = fd14;
    frag.data[15] = fd15;
    frag.data[16] = fd16;
    frag.data[17] = fd17;
    frag.data[18] = fd18;
    frag.data[19] = fd19;
    frag.data[20] = fd20;
    frag.data[21] = fd21;
    frag.data[22] = fd22;
    frag.data[23] = fd23;
    frag.data[24] = fd24;
    frag.data[25] = fd25;
    frag.data[26] = fd26;
    frag.data[27] = fd27;
    frag.data[28] = fd28;
    frag.data[29] = fd29;
    frag.data[30] = fd30;
    frag.data[31] = fd31;
  }
}

// Registers f0..f(NR-1) -> TMEM, in the layout of load().
template <uint32_t Shape = LANE_PER_THREAD, typename Frag>
inline __attribute__((always_inline)) void store(uint32_t taddr, const Frag& frag) {
  constexpr uint32_t F3 = 1 | Shape;
  static_assert(Frag::NR == 8 || Frag::NR == 16 || Frag::NR == 32, "unsupported fragment size");
  if constexpr (Frag::NR == 8) {
    register float fs0 __asm__("f0") = frag.data[0];
    register float fs1 __asm__("f1") = frag.data[1];
    register float fs2 __asm__("f2") = frag.data[2];
    register float fs3 __asm__("f3") = frag.data[3];
    register float fs4 __asm__("f4") = frag.data[4];
    register float fs5 __asm__("f5") = frag.data[5];
    register float fs6 __asm__("f6") = frag.data[6];
    register float fs7 __asm__("f7") = frag.data[7];
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x7"
      :
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr),
        "f"(fs0), "f"(fs1), "f"(fs2), "f"(fs3),
        "f"(fs4), "f"(fs5), "f"(fs6), "f"(fs7)
      : "memory");
  } else if constexpr (Frag::NR == 16) {
    register float fs0 __asm__("f0") = frag.data[0];
    register float fs1 __asm__("f1") = frag.data[1];
    register float fs2 __asm__("f2") = frag.data[2];
    register float fs3 __asm__("f3") = frag.data[3];
    register float fs4 __asm__("f4") = frag.data[4];
    register float fs5 __asm__("f5") = frag.data[5];
    register float fs6 __asm__("f6") = frag.data[6];
    register float fs7 __asm__("f7") = frag.data[7];
    register float fs8 __asm__("f8") = frag.data[8];
    register float fs9 __asm__("f9") = frag.data[9];
    register float fs10 __asm__("f10") = frag.data[10];
    register float fs11 __asm__("f11") = frag.data[11];
    register float fs12 __asm__("f12") = frag.data[12];
    register float fs13 __asm__("f13") = frag.data[13];
    register float fs14 __asm__("f14") = frag.data[14];
    register float fs15 __asm__("f15") = frag.data[15];
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x15"
      :
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr),
        "f"(fs0), "f"(fs1), "f"(fs2), "f"(fs3),
        "f"(fs4), "f"(fs5), "f"(fs6), "f"(fs7),
        "f"(fs8), "f"(fs9), "f"(fs10), "f"(fs11),
        "f"(fs12), "f"(fs13), "f"(fs14), "f"(fs15)
      : "memory");
  } else if constexpr (Frag::NR == 32) {
    register float fs0 __asm__("f0") = frag.data[0];
    register float fs1 __asm__("f1") = frag.data[1];
    register float fs2 __asm__("f2") = frag.data[2];
    register float fs3 __asm__("f3") = frag.data[3];
    register float fs4 __asm__("f4") = frag.data[4];
    register float fs5 __asm__("f5") = frag.data[5];
    register float fs6 __asm__("f6") = frag.data[6];
    register float fs7 __asm__("f7") = frag.data[7];
    register float fs8 __asm__("f8") = frag.data[8];
    register float fs9 __asm__("f9") = frag.data[9];
    register float fs10 __asm__("f10") = frag.data[10];
    register float fs11 __asm__("f11") = frag.data[11];
    register float fs12 __asm__("f12") = frag.data[12];
    register float fs13 __asm__("f13") = frag.data[13];
    register float fs14 __asm__("f14") = frag.data[14];
    register float fs15 __asm__("f15") = frag.data[15];
    register float fs16 __asm__("f16") = frag.data[16];
    register float fs17 __asm__("f17") = frag.data[17];
    register float fs18 __asm__("f18") = frag.data[18];
    register float fs19 __asm__("f19") = frag.data[19];
    register float fs20 __asm__("f20") = frag.data[20];
    register float fs21 __asm__("f21") = frag.data[21];
    register float fs22 __asm__("f22") = frag.data[22];
    register float fs23 __asm__("f23") = frag.data[23];
    register float fs24 __asm__("f24") = frag.data[24];
    register float fs25 __asm__("f25") = frag.data[25];
    register float fs26 __asm__("f26") = frag.data[26];
    register float fs27 __asm__("f27") = frag.data[27];
    register float fs28 __asm__("f28") = frag.data[28];
    register float fs29 __asm__("f29") = frag.data[29];
    register float fs30 __asm__("f30") = frag.data[30];
    register float fs31 __asm__("f31") = frag.data[31];
    __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x31"
      :
      : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr),
        "f"(fs0), "f"(fs1), "f"(fs2), "f"(fs3),
        "f"(fs4), "f"(fs5), "f"(fs6), "f"(fs7),
        "f"(fs8), "f"(fs9), "f"(fs10), "f"(fs11),
        "f"(fs12), "f"(fs13), "f"(fs14), "f"(fs15),
        "f"(fs16), "f"(fs17), "f"(fs18), "f"(fs19),
        "f"(fs20), "f"(fs21), "f"(fs22), "f"(fs23),
        "f"(fs24), "f"(fs25), "f"(fs26), "f"(fs27),
        "f"(fs28), "f"(fs29), "f"(fs30), "f"(fs31)
      : "memory");
  }
}

// tcgen05.wait::st: stores are posted, so a warp waits for its own to land
// before a barrier hands their columns to a job.
inline void wait_st() {
  vx_wsync();
}

} // namespace tmem
} // namespace vortex

#endif // __VX_TMEM_H__
