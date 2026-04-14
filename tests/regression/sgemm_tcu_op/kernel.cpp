
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
// #define LMEM_OVERFLOW_MARKER = 0x4c4d454d;

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

static inline uint32_t meta_tile_offset(const uint32_t* offsets, uint32_t idx) {
  return (idx == 0) ? 0 : offsets[idx - 1];
}

// static inline uint32_t meta_tile_end(const uint32_t* offsets, uint32_t idx) {
//   return offsets[idx];
// }

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
  static constexpr uint32_t lg_i_ratio = __builtin_ctz(i_ratio);
  static constexpr uint32_t o_ratio = sizeof(uint32_t) / sizeof(ctx::output_t);

  static constexpr uint32_t tile_M = 32;
  static constexpr uint32_t tile_N = 32;
  static constexpr uint32_t tile_K = 16 * i_ratio;
  static constexpr uint32_t tiles_n = (N / tile_N);
  static constexpr uint32_t tiles_m = (M / tile_M);
  static constexpr uint32_t tiles_k = (K / tile_K);
  static constexpr uint32_t total_tiles = tiles_n * tiles_m;
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
  static constexpr uint32_t input_bytes = sizeof(ctx::input_t);
  static constexpr uint32_t lg_input_bytes = __builtin_ctz(input_bytes);
  static constexpr uint32_t tileC_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t tileD_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t lmem_capacity_bytes = (1u << LMEM_LOG_SIZE);
  static constexpr uint32_t half_lmem_bytes = lmem_capacity_bytes >> 1;
  static constexpr uint32_t dense_first_k = tile_K;
  static constexpr uint32_t dense_reuse_k = 2 * tile_K;
  static constexpr uint32_t dense_first_a_regs = tile_M * dense_first_k / i_ratio;
  static constexpr uint32_t dense_first_b_regs = dense_first_k * tile_N / i_ratio;
  static constexpr uint32_t dense_reuse_a_regs = tile_M * dense_reuse_k / i_ratio;
  static constexpr uint32_t dense_reuse_b_regs = dense_reuse_k * tile_N / i_ratio;
  static constexpr uint32_t dense_first_bytes = (tileC_regs + dense_first_a_regs + dense_first_b_regs) * sizeof(uint32_t);
  static constexpr uint32_t dense_reuse_bytes = (dense_reuse_a_regs + dense_reuse_b_regs) * sizeof(uint32_t);

  static constexpr uint32_t sparse_dense_a_regs = tile_M * tile_K / i_ratio;
  static constexpr uint32_t sparse_dense_a_elems = tile_M * tile_K;
  static constexpr uint32_t sparse_bitmap_regs_per_tile = tile_K;
  static constexpr uint32_t sparse_dense_b_regs = tile_K * tile_N / i_ratio;
  static constexpr uint32_t sparse_dense_b_elems = tile_K * tile_N;
  static constexpr uint32_t sparse_comp_regs = div_up_constexpr(tile_M * tile_K * input_bytes, sizeof(uint32_t));
  static constexpr uint32_t sparse_first_stage_k = tile_K;
  static constexpr uint32_t sparse_reuse_stage_k = 2 * tile_K;
  static constexpr uint32_t sparse_first_stage_tiles = 1;
  static constexpr uint32_t sparse_reuse_stage_tiles = 2;
  static constexpr uint32_t sparse_first_stage_a_regs = sparse_dense_a_regs;
  static constexpr uint32_t sparse_first_stage_b_regs = sparse_dense_b_regs;
  static constexpr uint32_t sparse_reuse_stage_a_regs = 2 * sparse_dense_a_regs;
  static constexpr uint32_t sparse_reuse_stage_b_regs = 2 * sparse_dense_b_regs;
  static constexpr uint32_t sparse_first_bitmap_sep_regs = (sparse_first_stage_k <= 16)
                                              ? 16
                                              : (((sparse_first_stage_k + 31) / 32) * 32 + 16);
  static constexpr uint32_t sparse_reuse_bitmap_sep_regs = (sparse_reuse_stage_k <= 16)
                                              ? 16
                                              : (((sparse_reuse_stage_k + 31) / 32) * 32 + 16);
  static constexpr uint32_t sparse_first_bitmap_skew_regs = (kSparseA && kSparseB)
                                               ? (sparse_first_bitmap_sep_regs - sparse_first_stage_k)
                                               : 0;
  static constexpr uint32_t sparse_reuse_bitmap_skew_regs = (kSparseA && kSparseB)
                                               ? (sparse_reuse_bitmap_sep_regs - sparse_reuse_stage_k)
                                               : 0;
  static constexpr uint32_t sparse_first_bitmap_regs = (kSparseA ? (sparse_first_stage_tiles * sparse_bitmap_regs_per_tile) : 0)
                                          + (kSparseB ? (sparse_first_stage_tiles * sparse_bitmap_regs_per_tile) : 0);
  static constexpr uint32_t sparse_reuse_bitmap_regs = (kSparseA ? (sparse_reuse_stage_tiles * sparse_bitmap_regs_per_tile) : 0)
                                          + (kSparseB ? (sparse_reuse_stage_tiles * sparse_bitmap_regs_per_tile) : 0);
  static constexpr uint32_t sparse_first_stage_payload_regs = sparse_first_stage_a_regs
                                                 + sparse_first_stage_b_regs
                                                 + sparse_first_bitmap_regs
                                                 + sparse_first_bitmap_skew_regs;
  static constexpr uint32_t sparse_reuse_stage_payload_regs = sparse_reuse_stage_a_regs
                                                 + sparse_reuse_stage_b_regs
                                                 + sparse_reuse_bitmap_regs
                                                 + sparse_reuse_bitmap_skew_regs;
  const uint32_t tileA_regs = (sparsity == 0) ? dense_first_a_regs
                            : (sparsity == 2) ? sparse_comp_regs
                                              : sparse_dense_a_regs;
  const uint32_t tileB_regs = (sparsity == 0) ? dense_first_b_regs
                                              : sparse_comp_regs;

  const uint32_t bitmap_span_1_tile  = (sparsity == 0) ? 0 : tile_K;
  const uint32_t bitmap_span_2_tiles = (sparsity == 0) ? 0 : 2 * tile_K;
  const uint32_t bitmap_sep_regs_1_tile = (bitmap_span_1_tile <= 16)
                                        ? 16
                                        : (((bitmap_span_1_tile + 31) / 32) * 32 + 16);
  const uint32_t bitmap_sep_regs_2_tiles = (bitmap_span_2_tiles <= 16)
                                         ? 16
                                         : (((bitmap_span_2_tiles + 31) / 32) * 32 + 16);
  const uint32_t bitmap_skew_regs_1_tile  = (sparsity == 2) ? (bitmap_sep_regs_1_tile  - bitmap_span_1_tile)  : 0;
  const uint32_t bitmap_skew_regs_2_tiles = (sparsity == 2) ? (bitmap_sep_regs_2_tiles - bitmap_span_2_tiles) : 0;

  // const uint32_t bitmap_regs = (sparsity == 2) ? (2 * bitmap_span_1_tile) :
  //                              (sparsity == 1) ?      bitmap_span_1_tile  :
  //                              0;
  const uint32_t sparse_first_stage_regs_per_warp = tileC_regs + sparse_first_stage_payload_regs;
  const uint32_t sparse_reuse_stage_regs_per_warp = sparse_reuse_stage_payload_regs;
  const uint32_t sparse_regs_per_warp = (sparse_first_stage_regs_per_warp > sparse_reuse_stage_regs_per_warp)
                                      ? sparse_first_stage_regs_per_warp
                                      : sparse_reuse_stage_regs_per_warp;
  const uint32_t regs_per_warp = (sparsity == 0) ? (lmem_capacity_bytes / sizeof(uint32_t))
                                                 : sparse_regs_per_warp;

  static constexpr uint32_t LMEM_OVERFLOW_MARKER = 0x4c4d454d;
  if (sparsity == 0 && (dense_first_bytes > half_lmem_bytes
                     || dense_reuse_bytes > half_lmem_bytes)) {
    if (vx_thread_id() == 0) {
      pD[0] = LMEM_OVERFLOW_MARKER;
    }
    return;
  }
  
  uint32_t* lmem_base = reinterpret_cast<uint32_t *>(__local_mem(lmem_capacity_bytes));
  uint32_t* half0_base = lmem_base;
  uint32_t* half1_base = half0_base + (half_lmem_bytes / sizeof(uint32_t));
  static constexpr uint32_t half_lmem_regs = half_lmem_bytes / sizeof(uint32_t);
  uint32_t* half0_C_base = half0_base + half_lmem_regs - tileC_regs;
  uint32_t* half1_C_base = half1_base + half_lmem_regs - tileC_regs;
  uint32_t* A_lmem = lmem_base;
  uint32_t* B_lmem = A_lmem + tileA_regs;
  uint32_t* C_lmem = half0_C_base;
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

  bool have_pending_mma = false;
  bool have_inflight_mma = false;
  uint32_t pending_stage = 0;
  uint32_t inflight_stage = 0;
  uintptr_t pending_rs1_val = 0;
  uintptr_t pending_rs2_val = 0;
  uint32_t stage = 0;
  auto launch_pending_mma = [&]() {
    if (have_inflight_mma) {
      tcu_bar[inflight_stage].arrive_and_wait();
      have_inflight_mma = false;
    }
    load_bar[pending_stage].arrive_and_wait();
    ctx::mma_op(pending_rs1_val, pending_rs2_val);
    inflight_stage = pending_stage;
    have_inflight_mma = true;
    have_pending_mma = false;
  };


  if constexpr (kDense) {
    
#pragma unroll
    for (uint32_t tile_row_idx = tile_row / tile_M; tile_row_idx < tiles_m; ++tile_row_idx) {
      const uint32_t a_tile_base = tile_row_idx * tile_M * K;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        const uint32_t b_tile_base = tile_col_idx * K * tile_N;
        const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
        const uint32_t tile_col_tiles_base = tile_col_idx * tiles_k;

        const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
        const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
        uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);


        const uint32_t* pC_tile = pC + tile_id * tileC_regs;

        // Cross-tile pipelining can leave previous-tile work owning this half.
        if (have_pending_mma && pending_stage == stage) {
          launch_pending_mma();
        }
        // if (have_inflight_mma && inflight_stage == stage) {
        //   // tcu_bar[inflight_stage].arrive_and_wait();
        //   have_inflight_mma = false;
        // }

        static constexpr uint32_t kDenseLaunches = 1 + ((K > tile_K) ? div_up_constexpr(K - tile_K, 2 * tile_K) : 0);
#pragma unroll
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
          const uint32_t k_offset = (dense_iter == 0) ? 0 : (tile_K + (dense_iter - 1) * 2 * tile_K);
          // const uint32_t* mma_A_bitmap = reinterpret_cast<const uint32_t *>(A_bitmap_lmem);
          // const uint32_t* mma_B_bitmap = reinterpret_cast<const uint32_t *>(B_bitmap_lmem);
          const uint32_t k_tile_idx = k_offset / tile_K;
          // const uint32_t stage = dense_iter & 1u;
          const bool use_half0 = (stage == 0);
          const bool first_dense_launch = (dense_iter == 0);
          const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(use_half0 ? half0_C_base : half1_C_base);
          uint32_t* chunk_A = A_lmem;
          uint32_t* chunk_B = B_lmem;
          // auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          // auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          uint32_t curr_k = tile_K;
          // uint32_t a_elems = tile_M * tile_K;
          // uint32_t b_elems = tile_K * tile_N;
          // uint32_t a_blocks = A_compressed_blocks;
          // uint32_t b_blocks = B_compressed_blocks;
          const ctx::input_t* pA_chunk = pA_elems + a_tile_base + k_offset * tile_M;
          const ctx::input_t* pB_chunk = pB_elems + b_tile_base + k_offset * tile_N;
          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          const uint32_t dense_k_cap = first_dense_launch ? dense_first_k : dense_reuse_k;
          const uint32_t k_remaining = K - k_offset;
          curr_k = (k_remaining < dense_k_cap) ? k_remaining : dense_k_cap;
          uint32_t a_elems = tile_M * curr_k;
          uint32_t b_elems = curr_k * tile_N;
          chunk_A = (use_half0 ? half0_base : half1_base);
          chunk_B = chunk_A + (first_dense_launch ? dense_first_a_regs : dense_reuse_a_regs);
          auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);

          const uint32_t flags_chunk = (((k_offset == 0) ? 1u : 0u) << 1) | (((k_offset + curr_k) == K) ? 1u : 0u);

          if (is_dxa_quad) {
            if (first_dense_launch) {
              vx_dxa_issue_2d_wg(kDescC, load_bar[stage].id(), mma_C, 0, tile_id);
            }
            // tcu_bar[stage].arrive_and_wait();
            vx_dxa_retile_2d_wg(kDescA, a_elems, 1);
            vx_dxa_issue_2d_wg(kDescA, load_bar[stage].id(), chunk_A, 0, tile_row_idx * tiles_k + k_tile_idx);
            vx_dxa_retile_2d_wg(kDescB, b_elems, 1);
            vx_dxa_issue_2d_wg(kDescB, load_bar[stage].id(), chunk_B, 0, tile_col_idx * tiles_k + k_tile_idx);
          }

          if (have_pending_mma) {
            launch_pending_mma();
          }

          if (gtid == 0) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_A);
            // rs2_val = a_blocks;
          } else if (gtid == 1) {
            rs1_val = reinterpret_cast<uintptr_t>(chunk_B);
            // rs2_val = b_blocks;
          } else if (gtid == 2) {
            rs1_val = reinterpret_cast<uintptr_t>(mma_C);
            rs2_val = curr_k;
          } else if (gtid == 3) {
            rs1_val = mma_D_addr;
            rs2_val = vt::ITYPE::id;
          } else if (gtid == 4) {
            // rs1_val = reinterpret_cast<uintptr_t>(mma_A_bitmap);
            rs2_val = vt::OTYPE::id;
          } else if (gtid == 5) {
            // rs1_val = reinterpret_cast<uintptr_t>(mma_B_bitmap);
            rs2_val = static_cast<uint32_t>(kConstSparsity);
          } else if (gtid == 6) {
            rs2_val = tcu_bar[stage].id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }

          pending_stage = stage;
          pending_rs1_val = rs1_val;
          pending_rs2_val = rs2_val;
          have_pending_mma = true;

          stage = (stage == 0) ? 1 : 0;
        }
      }
    }

    if (have_pending_mma) {
      launch_pending_mma();
    }
    if (have_inflight_mma) {
      tcu_bar[inflight_stage].arrive_and_wait();
      have_inflight_mma = false;
      // if (lane0) {C_lmem[0] = MARKER;}
    }
  } 
  
  
  
  
  
  
  
  
  
  else {
   
#pragma unroll
    for (uint32_t tile_row_idx = tile_row / tile_M; tile_row_idx < tiles_m; ++tile_row_idx) {
      const uint32_t a_tile_base = tile_row_idx * tile_M * K;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        const uint32_t b_tile_base = tile_col_idx * K * tile_N;
        const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
        const uint32_t tile_col_tiles_base = tile_col_idx * tiles_k;

        const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
        const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
        uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);


        const uint32_t* pC_tile = pC + tile_id * tileC_regs;

        // Cross-tile pipelining can leave previous-tile work owning this half.
        if (have_pending_mma && pending_stage == stage) {
          launch_pending_mma();
        }
        // if (have_inflight_mma && inflight_stage == stage) {
        //   // tcu_bar[inflight_stage].arrive_and_wait();
        //   have_inflight_mma = false;
        // }

        static constexpr uint32_t kDenseLaunches = 1 + ((K > tile_K) ? div_up_constexpr(K - tile_K, 2 * tile_K) : 0);
#pragma unroll
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
          const uint32_t k_offset = (dense_iter == 0) ? 0 : (tile_K + (dense_iter - 1) * 2 * tile_K);
          // const uint32_t* mma_A_bitmap = reinterpret_cast<const uint32_t *>(A_bitmap_lmem);
          // const uint32_t* mma_B_bitmap = reinterpret_cast<const uint32_t *>(B_bitmap_lmem);
          const uint32_t k_tile_idx = k_offset / tile_K;
          // const uint32_t stage = dense_iter & 1u;
          const bool use_half0 = (stage == 0);
          const bool first_dense_launch = (dense_iter == 0);
          const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(use_half0 ? half0_C_base : half1_C_base);
          uint32_t* chunk_A = A_lmem;
          uint32_t* chunk_B = B_lmem;
          // auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          // auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);
          uint32_t curr_k = tile_K;
          // uint32_t a_elems = tile_M * tile_K;
          // uint32_t b_elems = tile_K * tile_N;
          // uint32_t a_blocks = A_compressed_blocks;
          // uint32_t b_blocks = B_compressed_blocks;
          const ctx::input_t* pA_chunk = pA_elems + a_tile_base + k_offset * tile_M;
          const ctx::input_t* pB_chunk = pB_elems + b_tile_base + k_offset * tile_N;
          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          const uint32_t dense_k_cap = first_dense_launch ? dense_first_k : dense_reuse_k;
          const uint32_t k_remaining = K - k_offset;
          curr_k = (k_remaining < dense_k_cap) ? k_remaining : dense_k_cap;
          // uint32_t 
          // uint32_t b_elems = curr_k * tile_N;
          auto chunk_A_elems = reinterpret_cast<ctx::input_t*>(chunk_A);
          auto chunk_B_elems = reinterpret_cast<ctx::input_t*>(chunk_B);

          const uint32_t flags_chunk = (((k_offset == 0) ? 1u : 0u) << 1) | (((k_offset + curr_k) == K) ? 1u : 0u);

          // NEW BEGIN
          /* Extraction of compressed elements */
          uint32_t a_offset;
          uint32_t a_elems; 
          uint32_t a_blocks;
          if constexpr (kSparseA) {
            const uint32_t a_meta_idx = tile_row_idx * kDenseLaunches + dense_iter;
            a_offset = meta_tile_offset(pA_nz, a_meta_idx); // Where the current DXA transaction starts
            a_elems  = meta_tile_offset(pA_nz, a_meta_idx+1) - a_offset; // How many elements the current DXA transaction carrries
            a_blocks = div_up((a_elems << lg_input_bytes), 128);
          }
          else {
            a_elems = tile_M * curr_k;
            a_blocks = 0;
          }
          const uint32_t b_meta_idx = tile_col_idx * kDenseLaunches + dense_iter;
          uint32_t b_offset  = meta_tile_offset(pB_nz, b_meta_idx); // Where the current DXA transaction starts
          uint32_t b_elems   = meta_tile_offset(pB_nz, b_meta_idx+1) - b_offset; // How many elements the current DXA transaction carrries
          uint32_t b_blocks = div_up((b_elems << lg_input_bytes), 128);
          
          /* Formation of pointers */
          uint32_t* A_bitmap_lmem;
          uint32_t* B_bitmap_lmem;
          if constexpr (kSparseA) {
            A_bitmap_lmem = (use_half0 ? half0_base : half1_base);
            B_bitmap_lmem = A_bitmap_lmem + (first_dense_launch ? (tile_K + bitmap_skew_regs_1_tile) : (2 * tile_K + bitmap_skew_regs_2_tiles));
          } else {
            A_bitmap_lmem = nullptr;
            B_bitmap_lmem = (use_half0 ? half0_base : half1_base) + (first_dense_launch ? tile_K : 2 * tile_K);
          }
          
          // if constexpr (kSparseA) {
          chunk_A = B_bitmap_lmem + (first_dense_launch ? tile_K : 2 * tile_K);
          // } else {
          //   chunk_A = (use_half0 ? half0_base : half1_base);
          // }

          if constexpr (kSparseA) {
            chunk_B = chunk_A + div_up((a_elems << lg_input_bytes), 4); // (first_dense_launch ? dense_first_a_regs : dense_reuse_a_regs);
          } else {
            chunk_B = chunk_A + (first_dense_launch ? dense_first_a_regs : dense_reuse_a_regs);
          }

          const uint32_t b_payload_regs = bytes_to_regs(b_elems << lg_input_bytes);
          uint32_t* stage_base = use_half0 ? half0_base : half1_base;
          uint32_t* chunk_B_end = chunk_B + b_payload_regs;
          uint32_t* stage_limit = first_dense_launch ? const_cast<uint32_t*>(mma_C)
                                                     : (stage_base + half_lmem_regs);
          if (chunk_B_end > stage_limit) {
            if (lane0) {
              pD[0] = LMEM_OVERFLOW_MARKER;
            }
            return;
          }

          /* Start DXA transactions */
          if (is_dxa_quad) {
            if (first_dense_launch) {
              vx_dxa_issue_2d_wg(kDescC, load_bar[stage].id(), mma_C, 0, tile_id);
            }
            if constexpr (kSparseA) {
              const uint32_t a_bitmap_start = tile_row_idx * K + k_offset;
              vx_dxa_retile_1d_wg(kDescABitmap, curr_k);
              vx_dxa_issue_1d_wg(kDescABitmap, load_bar[stage].id(), A_bitmap_lmem, a_bitmap_start);
              vx_dxa_retile_1d_wg(kDescA, a_elems);
              vx_dxa_issue_1d_wg(kDescA, load_bar[stage].id(), chunk_A, a_offset);
            } else {
              vx_dxa_retile_2d_wg(kDescA, a_elems, 1);
              vx_dxa_issue_2d_wg(kDescA, load_bar[stage].id(), chunk_A, 0, tile_row_tiles_base + k_tile_idx);
            }

            const uint32_t b_bitmap_start = tile_col_idx * K + k_offset;
            vx_dxa_retile_1d_wg(kDescBBitmap, curr_k);
            vx_dxa_issue_1d_wg(kDescBBitmap, load_bar[stage].id(), B_bitmap_lmem, b_bitmap_start);
            vx_dxa_retile_1d_wg(kDescB, b_elems);
            vx_dxa_issue_1d_wg(kDescB, load_bar[stage].id(), chunk_B, b_offset);
          }

          // NEW END
          
          // if (is_dxa_quad) {
          //   if (first_dense_launch) {
          //     vx_dxa_issue_2d_wg(kDescC, load_bar[stage].id(), mma_C, 0, tile_id);
          //   }
          //   // tcu_bar[stage].arrive_and_wait();
          //   vx_dxa_retile_2d_wg(kDescA, a_elems, 1);
          //   vx_dxa_issue_2d_wg(kDescA, load_bar[stage].id(), chunk_A, 0, tile_row_idx * tiles_k + k_tile_idx);
          //   vx_dxa_retile_2d_wg(kDescB, b_elems, 1);
          //   vx_dxa_issue_2d_wg(kDescB, load_bar[stage].id(), chunk_B, 0, tile_col_idx * tiles_k + k_tile_idx);
          // }

          if (have_pending_mma) {
            launch_pending_mma();
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
            rs1_val = reinterpret_cast<uintptr_t>(A_bitmap_lmem);
            rs2_val = vt::OTYPE::id;
          } else if (gtid == 5) {
            rs1_val = reinterpret_cast<uintptr_t>(B_bitmap_lmem);
            rs2_val = static_cast<uint32_t>(kConstSparsity);
          } else if (gtid == 6) {
            rs2_val = tcu_bar[stage].id();
          } else if (gtid == 7) {
            rs2_val = flags_chunk;
          }

          pending_stage = stage;
          pending_rs1_val = rs1_val;
          pending_rs2_val = rs2_val;
          have_pending_mma = true;

          stage = (stage == 0) ? 1 : 0;
        }
      }
    }

    if (have_pending_mma) {
      launch_pending_mma();
    }
    if (have_inflight_mma) {
      tcu_bar[inflight_stage].arrive_and_wait();
      have_inflight_mma = false;
      // if (lane0) {C_lmem[0] = MARKER;}
    }
  }
} 
