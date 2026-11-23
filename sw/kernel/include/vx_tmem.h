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
#include <tensor_cfg.h>
#include <vx_tensor.h>

// Tensor memory (CUSTOM0, funct7 = 5) and tcgen05-style async tensor ops
// (funct7 = 6).
//
// TMEM is one per-core array of lanes x 32-bit columns, addressed
// {lane [31:16], column [15:0]}. A lane is an M row of an MMA: TCU block b
// computes lanes [b*M, (b+1)*M), M being the WGMMA tile M. A column is an N
// element.

namespace vortex {
namespace tmem {

// Columns of a fragment row (the WGMMA tile's TC_N): TMEM is stored
// fragment-major, so D, metadata and A-from-TMEM columns and tmem.ld/st column
// addresses are multiples of it.
inline constexpr uint32_t frag_cols() {
  uint32_t lg = 0;
  while ((1u << (lg + 1)) <= VX_CFG_NUM_THREADS) {
    ++lg;
  }
  return 1u << (lg / 2);
}

// TMEM lanes of a TCU block: the WGMMA tile M.
inline constexpr uint32_t block_lanes() {
  uint32_t lg = 0;
  while ((1u << (lg + 1)) <= VX_CFG_NUM_THREADS) {
    ++lg;
  }
  return 2u << ((lg + 1) / 2);
}

// The calling warp's first TMEM lane. A warp's tmem.ld/st reach only the
// lanes of the TCU block it issues to (warp w -> block w % ISSUE_WIDTH); a
// CTA launched with a TMEM budget (vx_launch_info_t::tmem_cols) has warp rank
// r on block r % ISSUE_WIDTH.
inline uint32_t warp_lane() {
  return (vx_warp_id() % VX_CFG_NUM_TCU_BLOCKS) * block_lanes();
}

// tc.mma instruction descriptor: one K step of D (n_frag fragments of TC_N
// columns from column d_col) = A . B (+ D unless acc_init), IType inputs
// (fp32 out, int32 for integer inputs).
// sparse: A is 2:4 compressed and side_col is the first TMEM column of the
// step's metadata: a lane holds, from bit 0, the VX_tcu_sp_mux rows of the
// step's lower then upper TC_K words.
// MX IType: side_col is the first TMEM column of the step's scales,
// tensor::mx_max_fedp_sf(fedpK) per A row and B column, laid out as in
// docs/designs/tensor_memory.md (A: lane = row, byte s at column side + s/4;
// B: a dense byte stream over M step 1 of the blocks from group side/TC_N + 1).
struct idesc_t {
  uint32_t value;
};

template <typename IType>
inline constexpr idesc_t make_idesc(uint32_t n_frag, uint32_t d_col, bool acc_init,
                                    bool sparse = false, uint32_t side_col = 0) {
  return {(IType::id & 0x1f) | ((sparse ? 1u : 0u) << 5)
        | ((acc_init ? 1u : 0u) << 7) | (((n_frag - 1) & 0x7f) << 8)
        | (((d_col / frag_cols()) & 0xff) << 15) | ((side_col & 0x1ff) << 23)};
}

// A's TMEM column, for an MMA that reads A from TMEM: a lane is an M row, the
// step's fedpK K-packed words.
struct tmem_col {
  uint32_t value;
};

// Shared-memory byte offset of a pointer, as the descriptors expect.
inline uint32_t smem_offset(const void* ptr) {
  size_t lmem_base = csr_read_nv(VX_CSR_LOCAL_MEM_BASE);
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) - lmem_base);
}

// Shared-memory matrix descriptor (tensor::smem_matrix_desc) from a byte
// offset and the leading bytes (row stride; nonzero: A is row-major, B K-major). The
// offset form keeps the LMEM base read out of a K loop; vx_make_smem_desc
// builds one from a pointer.
inline constexpr tensor::smem_matrix_desc smem_desc(uint32_t offset, uint32_t leading_bytes) {
  return {tensor::smem_desc_pack(offset, leading_bytes)};
}

// tmem.ld/st register layouts (funct3 bit 2).
enum : uint32_t {
  LANE_PER_THREAD = 0,
  FRAGMENT        = 4,
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
// VX_CFG_TCU_TMEM_COLS), owned by the CTA; waits until they are free. A CTA
// allocates within the budget it was launched with, so it never waits on a
// CTA that is not running. One warp allocates and shares the column; the
// columns are freed by dealloc() or when the CTA exits. The column is returned
// in thread 0 only (the other threads' values are undefined), so thread 0
// stores it for the CTA, as tcgen05.alloc writes it to shared memory.
inline uint32_t alloc(uint32_t ncols) {
  uint32_t col;
  __asm__ volatile (".insn r %1, 6, 5, %0, %2, x0"
    : "=r"(col) : "i"(RISCV_CUSTOM0), "r"(ncols) : "memory");
  return col;
}

// tcgen05.dealloc: frees the CTA's columns. Their MMAs must have completed
// (tc.commit and a barrier wait).
inline void dealloc(uint32_t col, uint32_t ncols) {
  __asm__ volatile (".insn r %0, 7, 5, x0, %1, %2"
    :: "i"(RISCV_CUSTOM0), "r"(col), "r"(ncols) : "memory");
}

// tcgen05.mma: one K step D (+)= A . B, issued asynchronously by a single
// thread (thread 0, which must be active). A is in shared memory or, with a
// tmem_col, in TMEM; B is in shared memory. A core's tc.mma, tc.cp and
// tc.commit run in issue order.
inline void mma(idesc_t idesc, tensor::smem_matrix_desc a, tensor::smem_matrix_desc b) {
  __asm__ volatile (".insn r %0, 4, 6, %1, %2, %3"
    :: "i"(RISCV_CUSTOM0), "r"(b.value), "r"(idesc.value), "r"(a.value) : "memory");
}

inline void mma(idesc_t idesc, tmem_col a, tensor::smem_matrix_desc b) {
  __asm__ volatile (".insn r %0, 4, 6, %1, %2, %3"
    :: "i"(RISCV_CUSTOM0), "r"(b.value), "r"(idesc.value | (1u << 6)), "r"(a.value) : "memory");
}

// tcgen05.commit: arrive on barrier `bar`, as one of its `count` participants,
// once the core's earlier tc.mma and tc.cp ops have completed.
inline void commit(uint32_t bar, uint32_t count) {
  __asm__ volatile (".insn r %0, 1, 6, x0, %1, %2"
    :: "i"(RISCV_CUSTOM0), "r"(bar), "r"(count) : "memory");
}

// tcgen05.cp: copy columns [col, col + cols) of every TMEM lane from `src`
// in shared memory, pre-packed in TMEM's storage order: column group by
// column group (frag_cols() columns), TCU block by block, M step by M step,
// NT words each, word i * frag_cols() + j holding lane m * TC_M + i of the
// block, column j of the group. col and cols are multiples of frag_cols();
// src and the copy are whole LMEM bank rows. In order with tc.mma.
inline void cp(uint32_t col, const void* src, uint32_t cols) {
  uint32_t w = (col & 0xffff) | (cols << 16);
  __asm__ volatile (".insn r %0, 2, 6, x0, %1, %2"
    :: "i"(RISCV_CUSTOM0), "r"(w), "r"(smem_offset(src)) : "memory");
}

// Register lists for transfers of NR = 8, 16 or 32 FP registers: a
// transfer names f0..f(NR-1) as its operands (the uops step through them), so
// each register is pinned. VX_FREG_REST_<NR>(M) expands M(1) .. M(NR-1).
#define VX_FREG_REST_8(M)  M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define VX_FREG_REST_16(M) VX_FREG_REST_8(M) M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15)
#define VX_FREG_REST_32(M) VX_FREG_REST_16(M) M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31)

#define VX_TMEM_LD_PIN(i)  register float fr##i __asm__("f" #i);
#define VX_TMEM_LD_OUT(i)  , "=f"(fr##i)
#define VX_TMEM_LD_COPY(i) frag.data[i] = fr##i;
#define VX_TMEM_ST_PIN(i)  register float fr##i __asm__("f" #i) = frag.data[i];
#define VX_TMEM_ST_IN(i)   , "f"(fr##i)

// One transfer of NR registers; rs2 = x(NR-1) encodes the count.
#define VX_TMEM_LOAD(NR, LAST)                                              \
  VX_TMEM_LD_PIN(0) VX_FREG_REST_##NR(VX_TMEM_LD_PIN)                       \
  __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x" #LAST      \
    : "=f"(fr0) VX_FREG_REST_##NR(VX_TMEM_LD_OUT)                           \
    : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr)              \
    : "memory");                                                            \
  VX_TMEM_LD_COPY(0) VX_FREG_REST_##NR(VX_TMEM_LD_COPY)

#define VX_TMEM_STORE(NR, LAST)                                             \
  VX_TMEM_ST_PIN(0) VX_FREG_REST_##NR(VX_TMEM_ST_PIN)                       \
  __asm__ volatile (".insn r %[insn], %[f3], 5, x0, %[taddr], x" #LAST      \
    :                                                                       \
    : [insn]"i"(RISCV_CUSTOM0), [f3]"i"(F3), [taddr]"r"(taddr),             \
      "f"(fr0) VX_FREG_REST_##NR(VX_TMEM_ST_IN)                             \
    : "memory");

// TMEM -> registers f0..f(NR-1). taddr's lane is warp_lane() and its column a
// multiple of frag_cols().
// LANE_PER_THREAD: thread t gets lane + t % M, register r column
//   + r * (NT / M) + t / M.
// FRAGMENT: register r is the C/D fragment of M step r % 2 and columns
//   + (r / 2) * TC_N: thread t gets lane + (r % 2) * TC_M + t / TC_N, column
//   + (r / 2) * TC_N + t % TC_N, the layout of the WGMMA accumulator registers
//   (so the WGMMA context's store_matrix_sync stores it).
template <uint32_t Shape = LANE_PER_THREAD, typename Frag>
inline __attribute__((always_inline)) void load(Frag& frag, uint32_t taddr) {
  constexpr uint32_t F3 = 0 | Shape;
  static_assert(Frag::NR == 8 || Frag::NR == 16 || Frag::NR == 32, "unsupported fragment size");
  if constexpr (Frag::NR == 8) {
    VX_TMEM_LOAD(8, 7)
  } else if constexpr (Frag::NR == 16) {
    VX_TMEM_LOAD(16, 15)
  } else {
    VX_TMEM_LOAD(32, 31)
  }
}

// Registers f0..f(NR-1) -> TMEM, in the layout of load().
template <uint32_t Shape = LANE_PER_THREAD, typename Frag>
inline __attribute__((always_inline)) void store(uint32_t taddr, const Frag& frag) {
  constexpr uint32_t F3 = 1 | Shape;
  static_assert(Frag::NR == 8 || Frag::NR == 16 || Frag::NR == 32, "unsupported fragment size");
  if constexpr (Frag::NR == 8) {
    VX_TMEM_STORE(8, 7)
  } else if constexpr (Frag::NR == 16) {
    VX_TMEM_STORE(16, 15)
  } else {
    VX_TMEM_STORE(32, 31)
  }
}

#undef VX_TMEM_LOAD
#undef VX_TMEM_STORE
#undef VX_TMEM_LD_PIN
#undef VX_TMEM_LD_OUT
#undef VX_TMEM_LD_COPY
#undef VX_TMEM_ST_PIN
#undef VX_TMEM_ST_IN

// tcgen05.wait::st: stores are posted, so a warp waits for its own to land
// before a barrier hands their columns to a job.
inline void wait_st() {
  vx_wsync();
}

} // namespace tmem
} // namespace vortex

#endif // __VX_TMEM_H__
