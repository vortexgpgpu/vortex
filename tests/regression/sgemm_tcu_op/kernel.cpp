
#include "common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_tensor.h>

#include <VX_config.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <stdint.h>

#ifndef SGEMM_CONST_M
#define SGEMM_CONST_M 32
#endif

#ifndef SGEMM_CONST_N
#define SGEMM_CONST_N 32
#endif

#ifndef SGEMM_CONST_K
#define SGEMM_CONST_K 32
#endif

#ifndef SGEMM_CONST_SPARSITY
#define SGEMM_CONST_SPARSITY 0
#endif

#define MARKER 0x12345678
#define BLUE_MARKER 0x12345677
#define PRE_TCU_MARKER 0x12345679

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
static constexpr uint32_t LMEM_OVERFLOW_MARKER = 0x4c4d454d;
static constexpr uint32_t kDescA = 0;
static constexpr uint32_t kDescB = 1;
static constexpr uint32_t kDescC = 2;

#undef __local_mem
#define __local_mem(size) \
  (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))

static inline uint64_t rdcycle() {
#if __riscv_xlen == 64
    uint64_t value;
    asm volatile ("csrr %0, 0xB00" : "=r"(value));
    return value;
#else
    uint32_t hi0, lo, hi1;
    asm volatile ("csrr %0, 0xB80" : "=r"(hi0));
    asm volatile ("csrr %0, 0xB00" : "=r"(lo));
    asm volatile ("csrr %0, 0xB80" : "=r"(hi1));
    if (hi0 != hi1) {
        asm volatile ("csrr %0, 0xB00" : "=r"(lo));
    }
    return ((uint64_t)hi1 << 32) | lo;
#endif
}

template <typename T>
static inline void copy_tile_to_lmem(T* dst,
                                     const T* src,
                                     uint32_t count,
                                     uint32_t gtid,
                                     uint32_t gstride) {
  for (uint32_t idx = gtid; idx < count; idx += gstride) {
    dst[idx] = src[idx];
  }
}

static inline uint32_t div_up(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

static constexpr uint32_t div_up_constexpr(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  if (vx_thread_id() == 0) {(reinterpret_cast<uint32_t *>(arg->D_addr))[0] = MARKER;}

  auto pA = reinterpret_cast<uint32_t *>(arg->A_addr);
  auto pB = reinterpret_cast<uint32_t *>(arg->B_addr);
  auto pC = reinterpret_cast<uint32_t *>(arg->C_addr);
  auto pD = reinterpret_cast<uint32_t *>(arg->D_addr);
  auto pA_bitmap = reinterpret_cast<const uint32_t *>(arg->A_bitmap_addr);
  auto pB_bitmap = reinterpret_cast<const uint32_t *>(arg->B_bitmap_addr);
  auto pA_nz = reinterpret_cast<const uint32_t *>(arg->A_nz_addr);
  auto pB_nz = reinterpret_cast<const uint32_t *>(arg->B_nz_addr);

  const uint32_t A_compressed_blocks = arg->A_compressed_blocks;
  const uint32_t B_compressed_blocks = arg->B_compressed_blocks;
  static constexpr uint32_t M = SGEMM_CONST_M;
  static constexpr uint32_t N = SGEMM_CONST_N;
  static constexpr uint32_t K = SGEMM_CONST_K;
  const uint8_t sparsity = arg->sparsity;
  static constexpr uint8_t kConstSparsity = SGEMM_CONST_SPARSITY;
  static constexpr bool kDense = (kConstSparsity == 0);
  static constexpr bool kSparseA = (kConstSparsity == 2);
  static constexpr bool kSparseB = (kConstSparsity >= 1);

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  static constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(ctx::input_t);
  static constexpr uint32_t o_ratio = sizeof(uint32_t) / sizeof(ctx::output_t);

  static constexpr uint32_t tile_M = 32;
  static constexpr uint32_t tile_N = 32;
  static constexpr uint32_t tile_K = 16 * i_ratio;
  const uint32_t tiles_n = (N / tile_N);
  const uint32_t tiles_m = (M / tile_M);
  const uint32_t tiles_k = (K / tile_K);
  const uint32_t total_tiles = tiles_n * tiles_m;
  const uint32_t warps_per_group = (__warps_per_group == 0) ? 1 : __warps_per_group;
  const uint32_t local_warp = (warps_per_group > 1) ? (vx_warp_id() % warps_per_group) : 0;
  const uint32_t block_tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  const uint32_t warp_tile_id = block_tile_id * warps_per_group + local_warp;
  const bool active_warp = (warp_tile_id < total_tiles);
  vortex::barrier c_tile_bar(local_warp, 1);
  vortex::barrier ab_tile_bar(2 + local_warp, 1);
  vortex::barrier tcu_bar(4 + local_warp, 1);

  const uint32_t tile_row = (tiles_n == 0) ? 0 : (warp_tile_id / tiles_n) * tile_M;

  static_assert (LMEM_ENABLED);
  const uint32_t input_bytes = sizeof(ctx::input_t);
  static constexpr uint32_t tileC_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t tileD_regs = tile_M * tile_N / o_ratio;
  const uint32_t lmem_capacity_bytes = (1u << LMEM_LOG_SIZE);
  const uint32_t half_lmem_bytes = lmem_capacity_bytes / 2;
  static constexpr uint32_t dense_half0_first_k = tile_K;
  static constexpr uint32_t dense_half0_reuse_k = 2 * tile_K;
  static constexpr uint32_t dense_half1_k = 2 * tile_K;
  static constexpr uint32_t dense_half0_first_a_regs = tile_M * dense_half0_first_k / i_ratio;
  static constexpr uint32_t dense_half0_first_b_regs = dense_half0_first_k * tile_N / i_ratio;
  static constexpr uint32_t dense_half0_reuse_a_regs = tile_M * dense_half0_reuse_k / i_ratio;
  static constexpr uint32_t dense_half0_reuse_b_regs = dense_half0_reuse_k * tile_N / i_ratio;
  static constexpr uint32_t dense_half1_a_regs = tile_M * dense_half1_k / i_ratio;
  static constexpr uint32_t dense_half1_b_regs = dense_half1_k * tile_N / i_ratio;
  const uint32_t dense_half0_first_bytes = (tileC_regs + dense_half0_first_a_regs + dense_half0_first_b_regs) * sizeof(uint32_t);
  const uint32_t dense_half0_reuse_bytes = (dense_half0_reuse_a_regs + dense_half0_reuse_b_regs) * sizeof(uint32_t);
  const uint32_t dense_half1_bytes = (dense_half1_a_regs + dense_half1_b_regs) * sizeof(uint32_t);
  static constexpr uint32_t sparse_dense_a_regs = tile_M * tile_K / i_ratio;
  const uint32_t sparse_comp_regs = div_up(tile_M * tile_K * input_bytes, sizeof(uint32_t));
  const uint32_t tileA_regs = (sparsity == 0) ? dense_half0_first_a_regs
                           : (sparsity == 2) ? sparse_comp_regs
                                             : sparse_dense_a_regs;
  const uint32_t tileB_regs = (sparsity == 0) ? dense_half0_first_b_regs
                           : sparse_comp_regs;
  const uint32_t bitmap_span = (sparsity == 0) ? 0 : tile_K;
  const uint32_t bitmap_sep_regs = (bitmap_span <= 16)
                                   ? 16
                                   : (((bitmap_span + 31) / 32) * 32 + 16);
  const uint32_t bitmap_skew_regs = (sparsity == 2) ? (bitmap_sep_regs - bitmap_span) : 0;
  const uint32_t bitmap_regs = (sparsity == 2) ? (2 * bitmap_span) :
                               (sparsity == 1) ?      bitmap_span  :
                               0;
  const uint32_t sparse_regs_per_warp = tileA_regs + tileB_regs + tileC_regs + bitmap_regs + bitmap_skew_regs;
  const uint32_t regs_per_warp = (sparsity == 0) ? (lmem_capacity_bytes / sizeof(uint32_t))
                                                 : sparse_regs_per_warp;
  const uint32_t lmem_needed_bytes = regs_per_warp * warps_per_group * sizeof(uint32_t);
  if (sparsity == 0 && (dense_half0_first_bytes > half_lmem_bytes
                     || dense_half0_reuse_bytes > half_lmem_bytes
                     || dense_half1_bytes > half_lmem_bytes)) {
    if (vx_thread_id() == 0) {
      pD[0] = LMEM_OVERFLOW_MARKER;
    }
    return;
  }
  if (lmem_needed_bytes > lmem_capacity_bytes) {
    if (vx_thread_id() == 0) {
      pD[0] = LMEM_OVERFLOW_MARKER;
    }
    return;
  }
  uint32_t* local_base = reinterpret_cast<uint32_t *>(__local_mem(lmem_needed_bytes));
  uint32_t* warp_base = local_base + local_warp * regs_per_warp;
  uint32_t* dense_half0 = warp_base;
  uint32_t* dense_half1 = warp_base + (half_lmem_bytes / sizeof(uint32_t));
  uint32_t* A_lmem = warp_base;
  uint32_t* B_lmem = A_lmem + tileA_regs;
  uint32_t* C_lmem = (sparsity == 0) ? dense_half0
                                     : (B_lmem + tileB_regs);
  uint32_t* bitmap_lmem = C_lmem + tileC_regs;
  uint32_t* A_bitmap_lmem = (sparsity == 2) ? bitmap_lmem     : nullptr;
  uint32_t* B_bitmap_lmem = (sparsity == 2) ? bitmap_lmem + bitmap_span + bitmap_skew_regs :
                            (sparsity == 1) ? bitmap_lmem     :
                            nullptr;
  uint32_t* D_lmem = C_lmem;
  auto pA_elems = reinterpret_cast<ctx::input_t *>(pA);
  auto pB_elems = reinterpret_cast<ctx::input_t *>(pB);
  auto A_lmem_elems = reinterpret_cast<ctx::input_t *>(A_lmem);
  auto B_lmem_elems = reinterpret_cast<ctx::input_t *>(B_lmem);
  const uint32_t gtid = vx_thread_id();
  const uint32_t gstride = vx_num_threads();
  const bool lane0 = (gtid == 0);
  const bool is_dxa_quad = (gtid < 4);

  if (active_warp && (vx_warp_id() == 0) && lane0) {C_lmem[0] = MARKER;}

  for (uint32_t tile_row_idx = tile_row / tile_M; tile_row_idx < tiles_m; ++tile_row_idx) {
    const uint32_t a_tile_base = tile_row_idx * tile_M * K;

    for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {
      const uint32_t b_tile_base = tile_col_idx * K * tile_N;

      const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr)
                                 + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
      uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);
      const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(C_lmem);
      const uint32_t* pC_tile = pC + tile_id * tileC_regs;

      if (active_warp && lane0) {mma_D[0] = MARKER;}
      if (active_warp && is_dxa_quad) {
        vx_dxa_issue_2d_wg(kDescC, c_tile_bar.id(), C_lmem, 0, tile_id);
      }
      c_tile_bar.arrive_and_wait();
      if (active_warp && lane0) {mma_D[0] = MARKER;}

      if constexpr (kDense) {
        static constexpr uint32_t kDenseLaunches =
            1 + ((K > tile_K) ? div_up_constexpr(K - tile_K, 2 * tile_K) : 0);
#pragma unroll
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
          const uint32_t k_offset = (dense_iter == 0) ? 0 : (tile_K + (dense_iter - 1) * 2 * tile_K);
          if (k_offset >= K) {
            break;
          }
          const uint32_t* mma_A_bitmap = reinterpret_cast<const uint32_t *>(A_bitmap_lmem);
          const uint32_t* mma_B_bitmap = reinterpret_cast<const uint32_t *>(B_bitmap_lmem);
          const uint32_t k_tile_idx = k_offset / tile_K;
          const bool use_half0 = ((dense_iter & 1u) == 0);
          const bool first_dense_launch = (dense_iter == 0);
          uint32_t* chunk_A = A_lmem;
          uint32_t* chunk_B = B_lmem;
          auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          uint32_t curr_k = tile_K;
          uint32_t a_elems = tile_M * curr_k;
          uint32_t b_elems = curr_k * tile_N;
          uint32_t a_blocks = A_compressed_blocks;
          uint32_t b_blocks = B_compressed_blocks;
          const ctx::input_t* pA_chunk = pA_elems + a_tile_base + k_offset * tile_M;
          const ctx::input_t* pB_chunk = pB_elems + b_tile_base + k_offset * tile_N;
          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          const uint32_t dense_k_cap = use_half0
                                     ? (first_dense_launch ? dense_half0_first_k : dense_half0_reuse_k)
                                     : dense_half1_k;
          const uint32_t k_remaining = K - k_offset;
          curr_k = (k_remaining < dense_k_cap) ? k_remaining : dense_k_cap;
          a_elems = tile_M * curr_k;
          b_elems = curr_k * tile_N;
          chunk_A = use_half0 ? (first_dense_launch ? (dense_half0 + tileC_regs) : dense_half0) : dense_half1;
          chunk_B = use_half0
                  ? (chunk_A + (first_dense_launch ? dense_half0_first_a_regs : dense_half0_reuse_a_regs))
                  : (chunk_A + dense_half1_a_regs);
          chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          auto chunk_A_elems_hi = chunk_A_elems + (tile_M * tile_K);
          auto chunk_B_elems_hi = chunk_B_elems + (tile_K * tile_N);
          const bool has_second_k_tile = (curr_k > tile_K);

          const uint32_t flags_chunk = (((k_offset == 0) ? 1u : 0u) << 1) | (((k_offset + curr_k) == K) ? 1u : 0u);

          if (active_warp && lane0) {mma_D[0] = MARKER;}
          if (active_warp && is_dxa_quad) {
            vx_dxa_issue_2d_wg(kDescA, ab_tile_bar.id(), chunk_A, 0, tile_row_idx * tiles_k + k_tile_idx);
            if (has_second_k_tile) {
              vx_dxa_issue_2d_wg(kDescA, ab_tile_bar.id(), chunk_A_elems_hi, 0,
                                 tile_row_idx * tiles_k + k_tile_idx + 1);
            }
            vx_dxa_issue_2d_wg(kDescB, ab_tile_bar.id(), chunk_B, 0, tile_col_idx * tiles_k + k_tile_idx);
            if (has_second_k_tile) {
              vx_dxa_issue_2d_wg(kDescB, ab_tile_bar.id(), chunk_B_elems_hi, 0,
                                 tile_col_idx * tiles_k + k_tile_idx + 1);
            }
          }
          ab_tile_bar.arrive_and_wait();
          if (active_warp && lane0) {mma_D[0] = MARKER;}

          if (gtid == 0) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_A);
            rs2_val = a_blocks;
          } else if (gtid == 1) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_B);
            rs2_val = b_blocks;
          } else if (gtid == 2) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_C);
            rs2_val = curr_k;
          } else if (gtid == 3) {
            rs1_val = mma_D_addr;
            rs2_val = vt::ITYPE::id;
          } else if (gtid == 4) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_A_bitmap);
            rs2_val = vt::OTYPE::id;
          } else if (gtid == 5) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_B_bitmap);
            rs2_val = static_cast<uint32_t>(kConstSparsity);
          } else if (gtid == 6) {
            rs2_val = tcu_bar.id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }
          if (active_warp && lane0) {mma_D[0] = PRE_TCU_MARKER;}
          ctx::mma_op(rs1_val, rs2_val);
          tcu_bar.arrive_and_wait();
          if (active_warp && (vx_warp_id() == 0) && lane0) {C_lmem[0] = MARKER;}
        }
      } else {
        static constexpr uint32_t kConstTilesK = K / tile_K;
#pragma unroll
        for (uint32_t k_tile_idx = 0; k_tile_idx < kConstTilesK; ++k_tile_idx) {
          const uint32_t k_offset = k_tile_idx * tile_K;
          const uint32_t* mma_A_bitmap = reinterpret_cast<const uint32_t *>(A_bitmap_lmem);
          const uint32_t* mma_B_bitmap = reinterpret_cast<const uint32_t *>(B_bitmap_lmem);
          uint32_t* chunk_A = A_lmem;
          uint32_t* chunk_B = B_lmem;
          auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          uint32_t curr_k = tile_K;
          uint32_t a_elems = tile_M * curr_k;
          uint32_t b_elems = curr_k * tile_N;
          uint32_t a_blocks = A_compressed_blocks;
          uint32_t b_blocks = B_compressed_blocks;
          const ctx::input_t* pA_chunk = pA_elems + a_tile_base + k_offset * tile_M;
          const ctx::input_t* pB_chunk = pB_elems + b_tile_base + k_offset * tile_N;
          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          if constexpr (kSparseA) {
            const uint32_t a_meta_idx = tile_row_idx * tiles_k + k_tile_idx;
            const uint32_t a_end = pA_nz[a_meta_idx];
            const uint32_t a_start = (a_meta_idx == 0) ? 0 : pA_nz[a_meta_idx - 1];
            a_elems = a_end - a_start;
            a_blocks = div_up(a_elems * input_bytes, 128);
            pA_chunk = pA_elems + a_start;

            const uint32_t* pA_bitmap_chunk = pA_bitmap + tile_row_idx * K + k_offset;
            if (active_warp && (vx_warp_id() == 0) && lane0) {mma_D[0] = BLUE_MARKER;}
            copy_tile_to_lmem(A_bitmap_lmem, pA_bitmap_chunk, curr_k, gtid, gstride);
            if (active_warp && (vx_warp_id() == 0) && lane0) {mma_D[0] = BLUE_MARKER;}
          }

          if constexpr (kSparseB) {
            const uint32_t b_meta_idx = tile_col_idx * tiles_k + k_tile_idx;
            const uint32_t b_end = pB_nz[b_meta_idx];
            const uint32_t b_start = (b_meta_idx == 0) ? 0 : pB_nz[b_meta_idx - 1];
            b_elems = b_end - b_start;
            b_blocks = div_up(b_elems * input_bytes, 128);
            pB_chunk = pB_elems + b_start;

            const uint32_t* pB_bitmap_chunk = pB_bitmap + tile_col_idx * K + k_offset;
            if (active_warp && (vx_warp_id() == 0) && lane0) {mma_D[0] = BLUE_MARKER;}
            copy_tile_to_lmem(B_bitmap_lmem, pB_bitmap_chunk, curr_k, gtid, gstride);
            if (active_warp && (vx_warp_id() == 0) && lane0) {mma_D[0] = BLUE_MARKER;}
          }

          const uint32_t flags_chunk = (((k_offset == 0) ? 1u : 0u) << 1) | (((k_offset + curr_k) == K) ? 1u : 0u);

          if (active_warp && lane0) {mma_D[0] = MARKER;}
          copy_tile_to_lmem(chunk_A_elems, pA_chunk, a_elems, gtid, gstride);
          if (active_warp && lane0) {mma_D[0] = MARKER;}
          copy_tile_to_lmem(chunk_B_elems, pB_chunk, b_elems, gtid, gstride);
          if (active_warp && lane0) {mma_D[0] = MARKER;}

          if (gtid == 0) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_A);
            rs2_val = a_blocks;
          } else if (gtid == 1) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_B);
            rs2_val = b_blocks;
          } else if (gtid == 2) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_C);
            rs2_val = curr_k;
          } else if (gtid == 3) {
            rs1_val = mma_D_addr;
            rs2_val = vt::ITYPE::id;
          } else if (gtid == 4) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_A_bitmap);
            rs2_val = vt::OTYPE::id;
          } else if (gtid == 5) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_B_bitmap);
            rs2_val = static_cast<uint32_t>(kConstSparsity);
          } else if (gtid == 6) {
            rs2_val = tcu_bar.id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }
          if (active_warp && lane0) {mma_D[0] = PRE_TCU_MARKER;}
          ctx::mma_op(rs1_val, rs2_val);
          tcu_bar.arrive_and_wait();
          if (active_warp && (vx_warp_id() == 0) && lane0) {C_lmem[0] = MARKER;}
        }
      }
    }
  }

  if (active_warp && (vx_warp_id() == 0) && lane0) {A_lmem[0] = MARKER;}
}
