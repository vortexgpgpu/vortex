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
static constexpr uint32_t kDescA = 0;
static constexpr uint32_t kDescB = 1;
static constexpr uint32_t kDescC = 2;
static constexpr uint32_t kDescABitmap = 3;
static constexpr uint32_t kDescBBitmap = 4;

#undef __local_mem
#define __local_mem(size) \
  (void*)(csr_read(VX_CSR_CTA_LMEM_ADDR))

static constexpr uint32_t clog2_constexpr(uint32_t value) {
  return (value <= 1) ? 0 : 1 + clog2_constexpr((value + 1) >> 1);
}

template <uint32_t Divisor>
static constexpr uint32_t div_up(uint32_t value) {
  static_assert(Divisor != 0, "Divisor must be non-zero");
  static_assert((Divisor & (Divisor - 1)) == 0, "Divisor must be a power of two");
  return (value + Divisor - 1) >> clog2_constexpr(Divisor);
}

static constexpr uint32_t div_up_constexpr(uint32_t value, uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}



extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) 
{
  // TODO: ADD CYCLE - INSTRUCTION COUNTING

  auto pA = reinterpret_cast<uint32_t *>(arg->A_addr);
  auto pB = reinterpret_cast<uint32_t *>(arg->B_addr);
  auto pC = reinterpret_cast<uint32_t *>(arg->C_addr);
  auto pD = reinterpret_cast<uint32_t *>(arg->D_addr);
  auto pA_bitmap = reinterpret_cast<const uint32_t *>(arg->A_bitmap_addr);
  auto pB_bitmap = reinterpret_cast<const uint32_t *>(arg->B_bitmap_addr);
  // auto pA_nz = reinterpret_cast<const uint32_t *>(arg->A_nz_addr);
  // auto pB_nz = reinterpret_cast<const uint32_t *>(arg->B_nz_addr);

  // const uint32_t A_compressed_blocks = arg->A_compressed_blocks;
  // const uint32_t B_compressed_blocks = arg->B_compressed_blocks;
  const uint32_t max_a_blocks = arg->max_a_blocks;
  const uint32_t max_b_blocks = arg->max_b_blocks;
  static constexpr uint32_t M = SGEMM_CONST_M;
  static constexpr uint32_t N = SGEMM_CONST_N;
  static constexpr uint32_t K = SGEMM_CONST_K;
  const uint8_t sparsity = arg->sparsity;
  static constexpr uint8_t kConstSparsity = SGEMM_CONST_SPARSITY;
  static constexpr bool kDense = (kConstSparsity == 0);
  static constexpr bool kSparseA = (kConstSparsity == 2);
  static constexpr bool kSparseB = (kConstSparsity >= 1);

  // ctx::fragment_a   fragA;
  // ctx::fragment_b   fragB;
  // ctx::fragment_acc fragC;

  static constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(ctx::input_t);
  // static constexpr uint32_t lg_i_ratio = __builtin_ctz(i_ratio);
  static constexpr uint32_t o_ratio = sizeof(uint32_t) / sizeof(ctx::output_t);

  static constexpr uint32_t tile_M = 32;
  static constexpr uint32_t tile_N = 32;
  static constexpr uint32_t tile_K = 16 * i_ratio * 2;
  static constexpr uint32_t tiles_n = (N / tile_N);
  static constexpr uint32_t tiles_m = (M / tile_M);
  static constexpr uint32_t tiles_k = (K / tile_K);
  static constexpr uint32_t total_tiles = tiles_n * tiles_m;
  const uint32_t block_tile_id = blockIdx.y * gridDim.x + blockIdx.x;

  vortex::barrier load_bar[2] = { vortex::barrier(0, 1), vortex::barrier(1, 1) };
  vortex::barrier tcu_bar[2]  = { vortex::barrier(2, 1), vortex::barrier(3, 1) };

  static_assert (LMEM_ENABLED);
  // static constexpr uint32_t input_bytes = sizeof(ctx::input_t);
  // static constexpr uint32_t lg_input_bytes = __builtin_ctz(input_bytes);
  static constexpr uint32_t tileC_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t tileD_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t lmem_capacity_bytes = (1u << LMEM_LOG_SIZE);
  static constexpr uint32_t half_lmem_bytes = lmem_capacity_bytes >> 1;
  static constexpr uint32_t dense_a_tile_regs = tile_M * tile_K / i_ratio;
  static constexpr uint32_t dense_b_tile_regs = tile_K * tile_N / i_ratio;
  // static constexpr uint32_t tiles_bytes = (dense_a_tile_regs + dense_b_tile_regs) * sizeof(uint32_t);


  // static constexpr uint32_t LMEM_OVERFLOW_MARKER = 0x4c4d454d;
  // if (sparsity == 0 && (tiles_bytes > half_lmem_bytes)) 
  // {
  //   if (vx_thread_id() == 0) {
  //     pD[0] = LMEM_OVERFLOW_MARKER;
  //   }
  //   return;
  // }
  
  uint32_t* lmem_base = reinterpret_cast<uint32_t *>(__local_mem(lmem_capacity_bytes));
  uint32_t* half0_base = lmem_base;
  uint32_t* half1_base = half0_base + (half_lmem_bytes / sizeof(uint32_t));
  // uint32_t* half1_end  = half1_base + (half_lmem_bytes / sizeof(uint32_t));
  static constexpr uint32_t half_lmem_regs = half_lmem_bytes / sizeof(uint32_t);
  // uint32_t* half0_C_base = half0_base + half_lmem_regs - tileC_regs;
  // uint32_t* half1_C_base = half1_base + half_lmem_regs - tileC_regs;
  // uint32_t* C_lmem = half0_C_base;
  // auto pA_gmem = reinterpret_cast<ctx::input_t *>(pA);
  // auto pB_gmem = reinterpret_cast<ctx::input_t *>(pB);
  // const uint32_t gtid = vx_thread_id();
  // const bool lane0 = (gtid == 0);
  // const bool is_dxa_quad = (gtid < 4);
  const bool __UNIFORM__ is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);


  uint32_t current_stage = 0;
  uint32_t next_stage = 0;
  uintptr_t pending_rs1_val = 0;
  uintptr_t pending_rs2_val = 0;

  /* Lambda function is optimized by the compiler */
  auto launch_pending_mma = [&]() __attribute__((always_inline)) {
    tcu_bar[current_stage].arrive_and_wait();
    load_bar[next_stage].arrive_and_wait();

    ctx::mma_op(pending_rs1_val, pending_rs2_val);

    current_stage = next_stage;
    next_stage ^= 1u;
  };


  if constexpr (kDense) 
  {
    uint32_t* A_lmem[2] = {half0_base, half1_base};
    uint32_t* B_lmem[2] = {A_lmem[0] + dense_a_tile_regs, A_lmem[1] + dense_a_tile_regs};

#pragma unroll
    for (uint32_t tile_row_idx = 0; tile_row_idx < tiles_m; ++tile_row_idx) {
      const uint32_t a_tile_base = tile_row_idx * tile_M * K;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        const uint32_t b_tile_base = tile_col_idx * K * tile_N;
        const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
        const uint32_t tile_col_tiles_base = tile_col_idx * tiles_k;

        const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
        const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
        uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);


        // const uint32_t* pC_tile = pC + tile_id * tileC_regs;

        static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);
#pragma unroll
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) 
        {
          const uint32_t k_offset = dense_iter * tile_K;
          const uint32_t k_tile_idx = k_offset / tile_K;
          // const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(use_half0 ? half0_C_base : half1_C_base);

          const uint32_t k_remaining = K - k_offset;
          uint32_t curr_k = std::min(k_remaining, tile_K);

          const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

          tcu_bar[current_stage].arrive_and_wait();

          /* In the very first execution, no data are ready so skip the mma_op */
          if ((tile_row_idx | tile_col_idx | dense_iter) != 0) {
            launch_pending_mma();
          }

          if (is_dxa_warp) {
            // if (first_dense_launch) {
            //   vx_dxa_issue_2d_wg(kDescC, load_bar[next_stage].id(), mma_C, 0, tile_id);
            // }
            // tcu_bar[next_stage].arrive_and_wait();
            vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_lmem[next_stage], 0, tile_row_idx * tiles_k + k_tile_idx);
            vx_dxa_issue_2d_wg(kDescB, load_bar[next_stage].id(), B_lmem[next_stage], 0, tile_col_idx * tiles_k + k_tile_idx);
          }

          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          if (is_dxa_warp) {
            rs1_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_lmem[next_stage],
                      (size_t)(uintptr_t)B_lmem[next_stage],
                      (size_t)(uintptr_t)nullptr /* mma_C */,
                      (size_t)(uintptr_t)mma_D_addr);
                      
            rs2_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)nullptr /*mma_A_bitmap*/,
                      (size_t)(uintptr_t)nullptr /*mma_B_bitmap*/,
                      (size_t)tcu_bar[next_stage].id(),
                      (size_t)((curr_k << 24) /*| (a_blocks << 18) | (b_blocks << 12)*/ | (vt::OTYPE::id << 8) | 
                               (vt::ITYPE::id << 4) | (kConstSparsity << 2) | flags_chunk));
          }

          pending_rs1_val = rs1_val;
          pending_rs2_val = rs2_val;
        }
      }
    }

    launch_pending_mma();
    tcu_bar[current_stage].arrive_and_wait();
  } 
  
  
  
  
else {

  /* Constants for the sparse case only */
  static constexpr uint32_t bitmap_tile_regs = tile_K * tile_M / 32;
  constexpr uint32_t b_bitmap_skew_regs = 16;
  constexpr uint32_t b_tile_align_regs = 16;

  uint32_t a_blocks = 0;
  if constexpr (kSparseA) 
  {
    a_blocks = max_a_blocks;
  }
  const uint32_t b_blocks = max_b_blocks;

  /* LMEM  */
  uint32_t* A_bitmap_lmem[2] = {nullptr, nullptr};
  uint32_t* A_lmem[2]        = {nullptr, nullptr};
  uint32_t* B_bitmap_lmem[2] = {nullptr, nullptr};
  uint32_t* B_lmem[2]        = {nullptr, nullptr};

  if constexpr (kConstSparsity == 2)
  {
    /* LMEM Packing in s = 2 case:
    ||--A_bitmap--|----A_tile----|--(free space)--||--B_skew_regs--|--B_bitmap--|--B_align_regs--|----B_tile----|--(free space)--||
    */
    A_bitmap_lmem[0] = half0_base;
    A_bitmap_lmem[1] = half1_base;
    A_lmem[0] = A_bitmap_lmem[0] + bitmap_tile_regs;
    A_lmem[1] = A_bitmap_lmem[1] + bitmap_tile_regs;
    B_bitmap_lmem[0] = half0_base + dense_a_tile_regs + b_bitmap_skew_regs;
    B_bitmap_lmem[1] = half1_base + dense_a_tile_regs + b_bitmap_skew_regs;
    B_lmem[0] = B_bitmap_lmem[0] + bitmap_tile_regs + b_tile_align_regs;
    B_lmem[1] = B_bitmap_lmem[1] + bitmap_tile_regs + b_tile_align_regs;
  }
  else /* kConstSparsity == 1 */
  {
    /* LMEM Packing in s = 1 case:
    ||----A_tile----|--B_bitmap--|----B_tile----|--(free space)--||
    */
    A_lmem[0] = half0_base;
    A_lmem[1] = half1_base;
    B_bitmap_lmem[0] = half0_base + dense_a_tile_regs;
    B_bitmap_lmem[1] = half1_base + dense_a_tile_regs;
    B_lmem[0] = B_bitmap_lmem[0] + bitmap_tile_regs;
    B_lmem[1] = B_bitmap_lmem[1] + bitmap_tile_regs;
  }
    
#pragma unroll
    for (uint32_t tile_row_idx = 0; tile_row_idx < tiles_m; ++tile_row_idx) {
      const uint32_t a_tile_base = tile_row_idx * tile_M * K;
      const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        const uint32_t b_tile_base = tile_col_idx * K * tile_N;
        const uint32_t tile_col_tiles_base = tile_col_idx * tiles_k;

        const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
        const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
        uint32_t* mma_D = reinterpret_cast<uint32_t*>(mma_D_addr);


        // const uint32_t* pC_tile = pC + tile_id * tileC_regs;

        static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);
#pragma unroll
        for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) 
        {
          const uint32_t k_offset = dense_iter * tile_K;
          const uint32_t k_tile_idx = k_offset / tile_K;
          // const uint32_t* mma_C = reinterpret_cast<const uint32_t *>(use_half0 ? half0_C_base : half1_C_base);

          const uint32_t k_remaining = K - k_offset;
          uint32_t curr_k = std::min(k_remaining, tile_K);

          const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

          tcu_bar[current_stage].arrive_and_wait();

          /* In the very first execution, no data are ready so skip the mma_op */
          if ((tile_row_idx | tile_col_idx | dense_iter) != 0) 
          {
            launch_pending_mma();
          }

          if (is_dxa_warp) {
            if constexpr (kSparseA) {
              const uint32_t a_bitmap_start = tile_row_idx * K + k_offset;
              vx_dxa_issue_1d_wg(kDescABitmap, load_bar[next_stage].id(), A_bitmap_lmem[next_stage], a_bitmap_start);
              vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_lmem[next_stage], 0, tile_row_idx * tiles_k + k_tile_idx);
            } else {
              vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_lmem[next_stage], 0, tile_row_idx * tiles_k + k_tile_idx);
            }

            const uint32_t b_bitmap_start = tile_col_idx * K + k_offset;
            vx_dxa_issue_1d_wg(kDescBBitmap, load_bar[next_stage].id(), B_bitmap_lmem[next_stage], b_bitmap_start);
            vx_dxa_issue_2d_wg(kDescB, load_bar[next_stage].id(), B_lmem[next_stage], 0, tile_col_idx * tiles_k + k_tile_idx);
          }

          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          if (is_dxa_warp) {
            rs1_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_lmem[next_stage],
                      (size_t)(uintptr_t)B_lmem[next_stage],
                      (size_t)(uintptr_t)nullptr /* mma_C */,
                      (size_t)(uintptr_t)mma_D_addr);
            rs2_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_bitmap_lmem[next_stage],
                      (size_t)(uintptr_t)B_bitmap_lmem[next_stage],
                      (size_t)tcu_bar[next_stage].id(),
                      (size_t)((curr_k << 24)       | 
                              (a_blocks << 18)      | 
                              (b_blocks << 12)      | 
                              (vt::OTYPE::id << 8)  | 
                              (vt::ITYPE::id << 4)  | 
                              (kConstSparsity << 2) | 
                              flags_chunk));
          }

          pending_rs1_val = rs1_val;
          pending_rs2_val = rs2_val;
        }
      }
    }

    launch_pending_mma();

    tcu_bar[current_stage].arrive_and_wait();
  }
}
