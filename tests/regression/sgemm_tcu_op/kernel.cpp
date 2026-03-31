
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
#define GREEN_MARKER 0x12345676
#define PRE_TCU_MARKER 0x12345679

#ifndef SGEMM_TRACE_MARKERS
#define SGEMM_TRACE_MARKERS 1
#endif

#ifndef SGEMM_TRACE_STAGE_PLAN_MARKERS
#define SGEMM_TRACE_STAGE_PLAN_MARKERS 0
#endif

#if SGEMM_TRACE_MARKERS
#define TRACE_STORE(ptr, value) (*(ptr) = (value))
#else
#define TRACE_STORE(ptr, value) ((void)0)
#endif

#if SGEMM_TRACE_STAGE_PLAN_MARKERS
#define TRACE_STAGE_PLAN_STORE(ptr, value) TRACE_STORE(ptr, value)
#else
#define TRACE_STAGE_PLAN_STORE(ptr, value) ((void)0)
#endif

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;
static constexpr uint32_t LMEM_OVERFLOW_MARKER = 0x4c4d454d;
static constexpr uint32_t kDescA = 0;
static constexpr uint32_t kDescB = 1;
static constexpr uint32_t kDescC = 2;
static constexpr uint32_t kDescABitmap = 3;
static constexpr uint32_t kDescBBitmap = 4;

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

static inline uint32_t bytes_to_regs(uint32_t bytes) {
  return div_up(bytes, sizeof(uint32_t));
}

static inline uint32_t bytes_to_blocks_128(uint32_t bytes) {
  return div_up(bytes, 128);
}

static inline uint32_t meta_tile_start(const uint32_t* offsets, uint32_t idx) {
  return (idx == 0) ? 0 : offsets[idx - 1];
}

static inline uint32_t meta_tile_end(const uint32_t* offsets, uint32_t idx) {
  return offsets[idx];
}

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
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
  vortex::barrier load_bar[2] = {
      vortex::barrier(local_warp, 1),
      vortex::barrier(warps_per_group + local_warp, 1),
  };
  vortex::barrier tcu_bar[2] = {
      vortex::barrier(2 * warps_per_group + local_warp, 1),
      vortex::barrier(3 * warps_per_group + local_warp, 1),
  };

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
  static constexpr uint32_t sparse_dense_a_elems = tile_M * tile_K;
  static constexpr uint32_t sparse_bitmap_regs_per_tile = tile_K;
  static constexpr uint32_t sparse_dense_b_regs = tile_K * tile_N / i_ratio;
  static constexpr uint32_t sparse_dense_b_elems = tile_K * tile_N;
  const uint32_t sparse_comp_regs = div_up(tile_M * tile_K * input_bytes, sizeof(uint32_t));
  const uint32_t sparse_first_stage_k = tile_K;
  const uint32_t sparse_reuse_stage_k = 2 * tile_K;
  const uint32_t sparse_first_stage_tiles = 1;
  const uint32_t sparse_reuse_stage_tiles = 2;
  const uint32_t sparse_first_stage_a_regs = sparse_dense_a_regs;
  const uint32_t sparse_first_stage_b_regs = sparse_dense_b_regs;
  const uint32_t sparse_reuse_stage_a_regs = 2 * sparse_dense_a_regs;
  const uint32_t sparse_reuse_stage_b_regs = 2 * sparse_dense_b_regs;
  const uint32_t sparse_first_bitmap_sep_regs = (sparse_first_stage_k <= 16)
                                              ? 16
                                              : (((sparse_first_stage_k + 31) / 32) * 32 + 16);
  const uint32_t sparse_reuse_bitmap_sep_regs = (sparse_reuse_stage_k <= 16)
                                              ? 16
                                              : (((sparse_reuse_stage_k + 31) / 32) * 32 + 16);
  const uint32_t sparse_first_bitmap_skew_regs = (kSparseA && kSparseB)
                                               ? (sparse_first_bitmap_sep_regs - sparse_first_stage_k)
                                               : 0;
  const uint32_t sparse_reuse_bitmap_skew_regs = (kSparseA && kSparseB)
                                               ? (sparse_reuse_bitmap_sep_regs - sparse_reuse_stage_k)
                                               : 0;
  const uint32_t sparse_first_bitmap_regs = (kSparseA ? (sparse_first_stage_tiles * sparse_bitmap_regs_per_tile) : 0)
                                          + (kSparseB ? (sparse_first_stage_tiles * sparse_bitmap_regs_per_tile) : 0);
  const uint32_t sparse_reuse_bitmap_regs = (kSparseA ? (sparse_reuse_stage_tiles * sparse_bitmap_regs_per_tile) : 0)
                                          + (kSparseB ? (sparse_reuse_stage_tiles * sparse_bitmap_regs_per_tile) : 0);
  const uint32_t sparse_first_stage_payload_regs = sparse_first_stage_a_regs
                                                 + sparse_first_stage_b_regs
                                                 + sparse_first_bitmap_regs
                                                 + sparse_first_bitmap_skew_regs;
  const uint32_t sparse_reuse_stage_payload_regs = sparse_reuse_stage_a_regs
                                                 + sparse_reuse_stage_b_regs
                                                 + sparse_reuse_bitmap_regs
                                                 + sparse_reuse_bitmap_skew_regs;
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
  const uint32_t sparse_first_stage_regs_per_warp = tileC_regs + sparse_first_stage_payload_regs;
  const uint32_t sparse_reuse_stage_regs_per_warp = sparse_reuse_stage_payload_regs;
  const uint32_t sparse_regs_per_warp = (sparse_first_stage_regs_per_warp > sparse_reuse_stage_regs_per_warp)
                                      ? sparse_first_stage_regs_per_warp
                                      : sparse_reuse_stage_regs_per_warp;
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

  if (lane0) {TRACE_STORE(C_lmem, MARKER);}

  for (uint32_t tile_row_idx = tile_row / tile_M; tile_row_idx < tiles_m; ++tile_row_idx) {
    const uint32_t a_tile_base = tile_row_idx * tile_M * K;

    for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {
      const uint32_t b_tile_base = tile_col_idx * K * tile_N;
      const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
      const uint32_t tile_col_tiles_base = tile_col_idx * tiles_k;

      const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr)
                                 + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
      uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);
      const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(C_lmem);
      const uint32_t* pC_tile = pC + tile_id * tileC_regs;

      if constexpr (kDense) {
        bool have_pending_mma = false;
        bool have_inflight_mma = false;
        uint32_t pending_stage = 0;
        uint32_t inflight_stage = 0;
        uintptr_t pending_rs1_val = 0;
        uintptr_t pending_rs2_val = 0;

        if (is_dxa_quad) {
          vx_dxa_issue_2d_wg(kDescC, load_bar[0].id(), C_lmem, 0, tile_id);
        }
        // load_bar[0].arrive_and_wait();

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
          const uint32_t stage = dense_iter & 1u;
          const bool use_half0 = (stage == 0);
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

          if (is_dxa_quad) {
            vx_dxa_issue_2d_wg(kDescA, load_bar[stage].id(), chunk_A, 0, tile_row_idx * tiles_k + k_tile_idx);
            if (has_second_k_tile) {
              vx_dxa_issue_2d_wg(kDescA, load_bar[stage].id(), chunk_A_elems_hi, 0,
                                 tile_row_idx * tiles_k + k_tile_idx + 1);
            }
            vx_dxa_issue_2d_wg(kDescB, load_bar[stage].id(), chunk_B, 0, tile_col_idx * tiles_k + k_tile_idx);
            if (has_second_k_tile) {
              vx_dxa_issue_2d_wg(kDescB, load_bar[stage].id(), chunk_B_elems_hi, 0,
                                 tile_col_idx * tiles_k + k_tile_idx + 1);
            }
          }

          if (have_pending_mma) {
            if (have_inflight_mma) {
              tcu_bar[inflight_stage].arrive_and_wait();
            }
            load_bar[pending_stage].arrive_and_wait();
            ctx::mma_op(pending_rs1_val, pending_rs2_val);
            inflight_stage = pending_stage;
            have_inflight_mma = true;
            // if (lane0) {C_lmem[0] = MARKER;}
          }

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
            rs2_val = tcu_bar[stage].id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }

          // if (lane0) {mma_D[0] = PRE_TCU_MARKER;}
          pending_stage = stage;
          pending_rs1_val = rs1_val;
          pending_rs2_val = rs2_val;
          have_pending_mma = true;
        }

        if (have_pending_mma) {
          if (have_inflight_mma) {
            tcu_bar[inflight_stage].arrive_and_wait();
          }
          load_bar[pending_stage].arrive_and_wait();
          ctx::mma_op(pending_rs1_val, pending_rs2_val);
          tcu_bar[pending_stage].arrive_and_wait();
          // if (lane0) {C_lmem[0] = MARKER;}
        }
      } 
// @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      else {
        if (lane0) {TRACE_STORE(mma_D, MARKER);}
        if (is_dxa_quad) {
          vx_dxa_issue_2d_wg(kDescC, load_bar[0].id(), C_lmem, 0, tile_id);
        }

        static constexpr uint32_t kConstTilesK = K / tile_K;
        if (lane0) {TRACE_STORE(mma_D, MARKER);}
        
        for (uint32_t k_tile_idx = 0; k_tile_idx < kConstTilesK; ) {
          if (lane0) {TRACE_STORE(mma_D, MARKER);}
          const uint32_t k_offset = k_tile_idx * tile_K;
          const bool stage_has_c = (k_tile_idx == 0);
          const uint32_t stage_dense_a_regs_limit = stage_has_c
                                                  ? sparse_first_stage_a_regs
                                                  : sparse_reuse_stage_a_regs;
          const uint32_t stage_dense_b_regs_limit = stage_has_c
                                                  ? sparse_first_stage_b_regs
                                                  : sparse_reuse_stage_b_regs;
          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;
          uint32_t packed_tiles = 0;
          uint32_t curr_k = 0;
          uint32_t a_elems = 0;
          uint32_t b_elems = 0;
          uint32_t a_start = 0;
          uint32_t b_start = 0;
          TRACE_STAGE_PLAN_STORE(mma_D, GREEN_MARKER);
#pragma unroll
          for (uint32_t candidate_tiles = 1; candidate_tiles <= kConstTilesK; ++candidate_tiles) {
            const uint32_t curr_tile_idx = k_tile_idx + candidate_tiles - 1;
            if (curr_tile_idx >= kConstTilesK) {
              break;
            }

            const uint32_t next_k = candidate_tiles * tile_K;
            uint32_t next_a_elems = a_elems;
            uint32_t next_b_elems = b_elems;

            if constexpr (kSparseA) {
              const uint32_t a_meta_idx = tile_row_tiles_base + curr_tile_idx;
              const uint32_t tile_a_start = meta_tile_start(pA_nz, a_meta_idx);
              const uint32_t tile_a_end = meta_tile_end(pA_nz, a_meta_idx);
              if (candidate_tiles == 1) {
                a_start = tile_a_start;
              }
              next_a_elems = tile_a_end - a_start;
            } else {
              next_a_elems = tile_M * next_k;
            }

            if constexpr (kSparseB) {
              const uint32_t b_meta_idx = tile_col_tiles_base + curr_tile_idx;
              const uint32_t tile_b_start = meta_tile_start(pB_nz, b_meta_idx);
              const uint32_t tile_b_end = meta_tile_end(pB_nz, b_meta_idx);
              if (candidate_tiles == 1) {
                b_start = tile_b_start;
              }
              next_b_elems = tile_b_end - b_start;
            } else {
              next_b_elems = tile_N * next_k;
            }

            const uint32_t next_a_regs = bytes_to_regs(next_a_elems * input_bytes);
            const uint32_t next_b_regs = bytes_to_regs(next_b_elems * input_bytes);
            const uint32_t next_bitmap_sep_regs = (next_k <= 16)
                                                 ? 16
                                                 : (((next_k + 31) / 32) * 32 + 16);
            const uint32_t next_bitmap_skew_regs = (kSparseA && kSparseB)
                                                 ? (next_bitmap_sep_regs - next_k)
                                                 : 0;
            const uint32_t next_bitmap_regs = (kSparseA ? (candidate_tiles * sparse_bitmap_regs_per_tile) : 0)
                                            + (kSparseB ? (candidate_tiles * sparse_bitmap_regs_per_tile) : 0);
            const uint32_t next_total_regs = (stage_has_c ? tileC_regs : 0)
                                           + next_a_regs + next_b_regs
                                           + next_bitmap_regs + next_bitmap_skew_regs;
            if (next_a_regs > stage_dense_a_regs_limit
             || next_b_regs > stage_dense_b_regs_limit
             || next_total_regs > regs_per_warp) {
              break;
            }

            packed_tiles = candidate_tiles;
            curr_k = next_k;
            a_elems = next_a_elems;
            b_elems = next_b_elems;
          }

          if (packed_tiles == 0) {
            packed_tiles = 1;
            curr_k = tile_K;
            if constexpr (kSparseA) {
              const uint32_t a_meta_idx = tile_row_tiles_base + k_tile_idx;
              a_start = meta_tile_start(pA_nz, a_meta_idx);
              a_elems = meta_tile_end(pA_nz, a_meta_idx) - a_start;
            } else {
              a_elems = tile_M * tile_K;
            }
            if constexpr (kSparseB) {
              const uint32_t b_meta_idx = tile_col_tiles_base + k_tile_idx;
              b_start = meta_tile_start(pB_nz, b_meta_idx);
              b_elems = meta_tile_end(pB_nz, b_meta_idx) - b_start;
            } else {
              b_elems = tile_K * tile_N;
            }
          }
          TRACE_STAGE_PLAN_STORE(mma_D, GREEN_MARKER);

          uint32_t a_blocks = A_compressed_blocks;
          uint32_t b_blocks = B_compressed_blocks;
          const uint32_t bitmap_sep_regs = (curr_k <= 16)
                                         ? 16
                                         : (((curr_k + 31) / 32) * 32 + 16);
          const uint32_t bitmap_skew_regs_curr = (kSparseA && kSparseB)
                                               ? (bitmap_sep_regs - curr_k)
                                               : 0;
          uint32_t* chunk_A = stage_has_c ? (C_lmem + tileC_regs) : A_lmem;
          if constexpr (kSparseA) {
            a_blocks = bytes_to_blocks_128(a_elems * input_bytes);
          } else {
            a_blocks = bytes_to_blocks_128(a_elems * input_bytes);
          }
          uint32_t* chunk_B = chunk_A + stage_dense_a_regs_limit;
          if constexpr (kSparseB) {
            b_blocks = bytes_to_blocks_128(b_elems * input_bytes);
          } else {
            b_blocks = bytes_to_blocks_128(b_elems * input_bytes);
          }
          uint32_t* bitmap_base = chunk_B + stage_dense_b_regs_limit;
          uint32_t* mma_A_bitmap_lmem = kSparseA ? bitmap_base : nullptr;
          uint32_t* mma_B_bitmap_lmem = kSparseA ? (bitmap_base + curr_k + bitmap_skew_regs_curr)
                                                 : bitmap_base;
          const uint32_t* mma_A_bitmap = reinterpret_cast<const uint32_t *>(mma_A_bitmap_lmem);
          const uint32_t* mma_B_bitmap = reinterpret_cast<const uint32_t *>(mma_B_bitmap_lmem);
          auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          auto chunk_A_elems_hi = chunk_A_elems + sparse_dense_a_elems;
          auto chunk_B_elems_hi = chunk_B_elems + sparse_dense_b_elems;
          const uint32_t first_a_tile_idx = tile_row_tiles_base + k_tile_idx;
          const uint32_t first_b_tile_idx = tile_col_tiles_base + k_tile_idx;

          if ((kSparseA || kSparseB) && lane0) {TRACE_STORE(mma_D, BLUE_MARKER);}
          if (is_dxa_quad) {
            if constexpr (kSparseA) {
              const uint32_t a_bitmap_start = tile_row_idx * K + k_offset;
              vx_dxa_retile_1d_wg(kDescABitmap, curr_k);
              vx_dxa_issue_1d_wg(kDescABitmap, load_bar[0].id(), mma_A_bitmap_lmem, a_bitmap_start);
            }
            if constexpr (kSparseB) {
              const uint32_t b_bitmap_start = tile_col_idx * K + k_offset;
              vx_dxa_retile_1d_wg(kDescBBitmap, curr_k);
              vx_dxa_issue_1d_wg(kDescBBitmap, load_bar[0].id(), mma_B_bitmap_lmem, b_bitmap_start);
            }
            if constexpr (kSparseA) {
              vx_dxa_issue_1d_wg(kDescA, load_bar[0].id(), chunk_A, a_start);
              if (!stage_has_c) {
                vx_dxa_issue_1d_wg(kDescA, load_bar[0].id(), chunk_A_elems_hi,
                                   a_start + sparse_dense_a_elems);
              }
            } else {
              vx_dxa_issue_2d_wg(kDescA, load_bar[0].id(), chunk_A, 0, first_a_tile_idx);
              if (!stage_has_c) {
                vx_dxa_issue_2d_wg(kDescA, load_bar[0].id(), chunk_A_elems_hi, 0,
                                   first_a_tile_idx + 1);
              }
            }
            if constexpr (kSparseB) {
              vx_dxa_issue_1d_wg(kDescB, load_bar[0].id(), chunk_B, b_start);
              if (!stage_has_c) {
                vx_dxa_issue_1d_wg(kDescB, load_bar[0].id(), chunk_B_elems_hi,
                                   b_start + sparse_dense_b_elems);
              }
            } else {
              vx_dxa_issue_2d_wg(kDescB, load_bar[0].id(), chunk_B, 0, first_b_tile_idx);
              if (!stage_has_c) {
                vx_dxa_issue_2d_wg(kDescB, load_bar[0].id(), chunk_B_elems_hi, 0,
                                   first_b_tile_idx + 1);
              }
            }
          }
          if ((kSparseA || kSparseB) && lane0) {TRACE_STORE(mma_D, BLUE_MARKER);}

          const uint32_t flags_chunk = (((k_offset == 0) ? 1u : 0u) << 1) | (((k_offset + curr_k) == K) ? 1u : 0u);

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
            rs2_val = tcu_bar[0].id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }

          if (lane0) {TRACE_STORE(mma_D, MARKER);}
          load_bar[0].arrive_and_wait();
          if (lane0) {TRACE_STORE(mma_D, PRE_TCU_MARKER);}
          tcu_bar[0].arrive_and_wait();
          ctx::mma_op(rs1_val, rs2_val);
          if (lane0) {TRACE_STORE(C_lmem, MARKER);}

          k_tile_idx += packed_tiles;
        }

        tcu_bar[0].arrive_and_wait();
      }
    }
  }

  if (lane0) {TRACE_STORE(A_lmem, MARKER);}
}
