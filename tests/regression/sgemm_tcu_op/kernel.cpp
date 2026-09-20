// Legacy fork kernel: the dense/sparse/trace-marker variants leave many
// declarations unused in any single build configuration.
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-const-variable"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"

#include "common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
#include <vx_tensor.h>

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
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;
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



__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg)
{
  // Kernel self-timing. The begin values must survive the whole unrolled body,
  // so they are spilled to the per-thread stack -- and a warp-wide stack store
  // is one uncoalesced cache-line miss per thread (stacks are 8KB apart). One
  // warp absorbs that; the multi-engine kernel's warps queue their misses
  // behind each other and delay the first DXA issue by thousands of cycles.
  // -DSGEMM_NO_KERNEL_METRICS drops the self-timing (metrics are reported 0).
#ifndef SGEMM_NO_KERNEL_METRICS
  const uint64_t instret_begin = vx_rdinstret_local();
  const __rdcycle_time cycle_begin = vx_rdcycle_sync_begin();
#endif

  auto pA = reinterpret_cast<uint32_t *>(arg->A_addr);
  auto pB = reinterpret_cast<uint32_t *>(arg->B_addr);
  auto pC = reinterpret_cast<uint32_t *>(arg->C_addr);
  auto pD = reinterpret_cast<uint32_t *>(arg->D_addr);
  // No C (beta == 0, see SGEMM_HAS_C in common.h): skip staging it and pass the
  // engine a null C, which initialises the accumulator from zero instead.
  static constexpr bool has_c = (SGEMM_HAS_C != 0);
  auto pA_bitmap = reinterpret_cast<const uint32_t *>(arg->A_bitmap_addr);
  auto pB_bitmap = reinterpret_cast<const uint32_t *>(arg->B_bitmap_addr);

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

  static constexpr uint32_t i_ratio = sizeof(uint32_t) / sizeof(ctx::input_t);
  static constexpr uint32_t o_ratio = sizeof(uint32_t) / sizeof(ctx::output_t);

  static constexpr uint32_t tile_M = 32;
  static constexpr uint32_t tile_N = 32;
  // K-chunk depth per MMA op. The engine amortizes accumulator init and
  // writeback by keeping the output tile resident across k-chunks, so the
  // intended mode is the default (init on the first chunk, flush on the
  // last). Raising this covers more of K per op, up to the descriptor's
  // 8-bit K limit, at the cost of larger A/B tiles in LMEM; it was used to
  // isolate the cross-op path while the response-path defects that
  // corrupted it were being fixed, and should not be needed going forward.
#ifndef SGEMM_TILE_K_MULT
#define SGEMM_TILE_K_MULT 2
#endif
  static constexpr uint32_t tile_K = 16 * i_ratio * SGEMM_TILE_K_MULT;
  // The MMA_OP descriptor carries K in rs2[3][31:24] -- 8 bits, no hardware
  // range check. curr_k = min(k_remaining, tile_K), so tile_K bounds it; a
  // tile_K of 256 would encode as K=0 and silently compute nothing.
  static_assert(tile_K <= 255,
                "tile_K exceeds the descriptor's 8-bit K field; lower SGEMM_TILE_K_MULT");

  // The descriptor carries fmt_d in rs2[3][11:8] and fmt_s in rs2[3][7:4] --
  // 4 bits each, against TCU_FMT_WIDTH == 5 ids (VX_tcu_pkg.sv). Every integer
  // id (I32=16, I8=17, U8=18, I4=19, U4=20) truncates into a float id: U8
  // aliases FP16 exactly, and the RTL's own TCU_I8_ID/TCU_U8_ID/TCU_I32_ID
  // case labels become unreachable against a 4-bit selector, so the wrong
  // element ratio and the wrong accumulate mode are chosen with no diagnostic.
  // Reject at compile time until the descriptor field is widened.
  static_assert(vt::ITYPE::id < 16,
                "input format id does not fit the descriptor's 4-bit fmt_s field");
  static_assert(vt::OTYPE::id < 16,
                "output format id does not fit the descriptor's 4-bit fmt_d field");
  static constexpr uint32_t tiles_n = (N / tile_N);
  static constexpr uint32_t tiles_m = (M / tile_M);
  static constexpr uint32_t tiles_k = (K / tile_K);
  static constexpr uint32_t total_tiles = tiles_n * tiles_m;
  const uint32_t block_tile_id = blockIdx.y * gridDim.x + blockIdx.x;
  // Multi-engine (see SGEMM_TCU_ENGINES in common.h): this CTA is engine
  // block_tile_id and owns every kEngines-th output tile.
  static constexpr uint32_t kEngines = SGEMM_TCU_ENGINES;
  static_assert(kEngines == 1 || kDense, "multi-engine TCU_OP is dense-only");
  static_assert(total_tiles % kEngines == 0, "output tiles must divide evenly across engines");

#ifdef SGEMM_TCU_COOP
  static constexpr bool kCoop = true;
#else
  static constexpr bool kCoop = false;
#endif
  static constexpr uint32_t num_warps_per_cta = SGEMM_COOP_WARPS;
  vortex::barrier load_bar[2] = { vortex::barrier(0, num_warps_per_cta), vortex::barrier(1, num_warps_per_cta) };
  vortex::barrier tcu_bar[2]  = { vortex::barrier(2, num_warps_per_cta), vortex::barrier(3, num_warps_per_cta) };

  static_assert (VX_CFG_LMEM_ENABLED);

  static constexpr uint32_t tileC_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t tileD_regs = tile_M * tile_N / o_ratio;
  static constexpr uint32_t lmem_capacity_bytes = (1u << VX_CFG_LMEM_LOG_SIZE);
  static constexpr uint32_t half_lmem_bytes = lmem_capacity_bytes >> 1;
  static constexpr uint32_t dense_a_tile_regs = tile_M * tile_K / i_ratio;
  static constexpr uint32_t dense_b_tile_regs = tile_K * tile_N / i_ratio;
  
  uint32_t* lmem_base = reinterpret_cast<uint32_t *>(__local_mem(lmem_capacity_bytes));
  uint32_t* half0_base = lmem_base;
  uint32_t* half1_base = half0_base + (half_lmem_bytes / sizeof(uint32_t));
  static constexpr uint32_t half_lmem_regs = half_lmem_bytes / sizeof(uint32_t);
  // C accumulator-init slot at the top of each half (the TCU op core requires
  // all operands LMEM-resident; C is DXA-staged once per output tile). Needs
  // half_lmem_bytes >= A tile + B tile + C tile (build with LMEM_LOG_SIZE=15).
  uint32_t* half0_C_base = half0_base + half_lmem_regs - tileC_regs;
  uint32_t* half1_C_base = half1_base + half_lmem_regs - tileC_regs;
  uint32_t* C_lmem[2] = {half0_C_base, half1_C_base};
  static_assert(half_lmem_bytes >= (dense_a_tile_regs + dense_b_tile_regs + tileC_regs) * sizeof(uint32_t),
                "LMEM half must hold A + B + C tiles (raise VX_CFG_LMEM_LOG_SIZE)");
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

    vt::mma_op(pending_rs1_val, pending_rs2_val);

    current_stage = next_stage;
    next_stage ^= 1u;
  };


  if constexpr (kDense && kEngines == 1)
  {
    uint32_t* A_lmem[2] = {half0_base, half1_base};
    uint32_t* B_lmem[2] = {A_lmem[0] + dense_a_tile_regs, A_lmem[1] + dense_a_tile_regs};

#pragma unroll
    for (uint32_t tile_row_idx = 0; tile_row_idx < tiles_m; ++tile_row_idx) {
      [[maybe_unused]] const uint32_t a_tile_base = tile_row_idx * tile_M * K;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        [[maybe_unused]] const uint32_t b_tile_base = tile_col_idx * K * tile_N;
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
            // Post-rebase tx-barrier semantics: expectations are armed in
            // software before issue; DXA only delivers done events. The first
            // chunk of each output tile also stages C (accumulator init),
            // unless there is no C.
            const bool stage_c = has_c && (dense_iter == 0);
            load_bar[next_stage].expect_tx(stage_c ? 3 : 2);
            if (stage_c) {
              vx_dxa_issue_2d_wg(kDescC, load_bar[next_stage].id(), C_lmem[next_stage], 0, tile_id);
            }
            vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_lmem[next_stage], 0, tile_row_idx * tiles_k + k_tile_idx);
            vx_dxa_issue_2d_wg(kDescB, load_bar[next_stage].id(), B_lmem[next_stage], 0, tile_col_idx * tiles_k + k_tile_idx);
          }

          uintptr_t rs1_val = 0;
          uintptr_t rs2_val = 0;

          if (is_dxa_warp) {
            // init_flag=1 loads the accumulator's initial value from C; the
            // op core requires it LMEM-resident (DXA-staged above). A null C
            // makes a dense op start the accumulator from zero instead.
            rs1_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_lmem[next_stage],
                      (size_t)(uintptr_t)B_lmem[next_stage],
                      (size_t)(has_c ? (uintptr_t)C_lmem[next_stage] : (uintptr_t)0),
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

#ifdef SGEMM_TCU_COOP
  else if constexpr (kDense && kCoop)
  {
    // Cooperative multi-engine path (see SGEMM_TCU_COOP in common.h): one CTA
    // of kEngines warps covering GM x GN blocks of output tiles. Warp 0 stages
    // the block's GM A tiles and GN B tiles; every warp then issues the MMA_OP
    // for its own tile (A tile wr, B tile wc) on its own engine. Barriers count
    // all warps, so the engines advance stage by stage together.
    static constexpr uint32_t GM = SGEMM_COOP_GM, GN = SGEMM_COOP_GN;
    static_assert(tiles_m % GM == 0 && tiles_n % GN == 0, "output tiles must tile by the cooperative block");
    static_assert(!has_c, "the cooperative prototype stages no C (null C only)");
    static constexpr uint32_t stage_regs = GM * dense_a_tile_regs + GN * dense_b_tile_regs;
    static_assert(2 * stage_regs * sizeof(uint32_t) <= lmem_capacity_bytes,
                  "two stages of GM A + GN B tiles must fit in LMEM");
    const uint32_t wrank = csr_read(VX_CSR_CTA_RANK);
    const uint32_t wr = wrank / GN, wc = wrank % GN;
    auto A_at = [&](uint32_t s, uint32_t i) __attribute__((always_inline)) {
      return lmem_base + s * stage_regs + i * dense_a_tile_regs;
    };
    auto B_at = [&](uint32_t s, uint32_t j) __attribute__((always_inline)) {
      return lmem_base + s * stage_regs + GM * dense_a_tile_regs + j * dense_b_tile_regs;
    };
    static constexpr uint32_t blocks_m = tiles_m / GM, blocks_n = tiles_n / GN;
    static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);

#ifdef SGEMM_TCU_ROLLED
    // Rolled block loop. The stage index must stay a compile-time constant:
    // a runtime-indexed load_bar[]/tcu_bar[]/buffer-pointer array is spilled
    // to the per-thread stack, and every access becomes one uncoalesced
    // cache-line miss per thread (the plain rolled loop ran 2.1x slower). With
    // an even K-chunk count, step s = bi*L + dense_iter has parity
    // dense_iter & 1, so the fully unrolled inner loop sees constant stages.
    // Protocol per step s: wait the op two steps back (tcu_bar[s&1]), launch
    // op s-1 from stage (s-1)&1, then stage step s into stage s&1 -- exactly
    // what launch_pending_mma() does with current/next_stage.
    static_assert(kDenseLaunches % 2 == 0,
                  "rolled cooperative loop needs an even K-chunk count (compile-time stage parity)");
#pragma unroll 1
    for (uint32_t bi = 0; bi < blocks_m * blocks_n; ++bi) {
      const uint32_t block_row = bi / blocks_n, block_col = bi % blocks_n;
      const uint32_t tile_id = (block_row * GM + wr) * tiles_n + (block_col * GN + wc);
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);
#pragma unroll
      for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
        const uint32_t s_cur = dense_iter & 1u;   // stage staged at this step
        const uint32_t s_prv = s_cur ^ 1u;        // stage of the op launched now
        const uint32_t curr_k = std::min(K - dense_iter * tile_K, tile_K);
        const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

        tcu_bar[s_cur].arrive_and_wait();
        if ((bi | dense_iter) != 0) {
          tcu_bar[s_cur].arrive_and_wait();
          load_bar[s_prv].arrive_and_wait();
          vt::mma_op(pending_rs1_val, pending_rs2_val);
        }
        if (is_dxa_warp) {
          load_bar[s_cur].expect_tx(GM + GN);
          for (uint32_t i = 0; i < GM; ++i) {
            vx_dxa_issue_2d_wg(kDescA, load_bar[s_cur].id(), A_at(s_cur, i), 0,
                               (block_row * GM + i) * tiles_k + dense_iter);
          }
          for (uint32_t j = 0; j < GN; ++j) {
            vx_dxa_issue_2d_wg(kDescB, load_bar[s_cur].id(), B_at(s_cur, j), 0,
                               (block_col * GN + j) * tiles_k + dense_iter);
          }
        }
        pending_rs1_val = (uintptr_t)vx_wgather((size_t)(uintptr_t)A_at(s_cur, wr),
                                                (size_t)(uintptr_t)B_at(s_cur, wc),
                                                (size_t)0, (size_t)(uintptr_t)mma_D_addr);
        pending_rs2_val = (uintptr_t)vx_wgather((size_t)(uintptr_t)nullptr, (size_t)(uintptr_t)nullptr,
                                                (size_t)tcu_bar[s_cur].id(),
                                                (size_t)((curr_k << 24) | (vt::OTYPE::id << 8) |
                                                         (vt::ITYPE::id << 4) | (kConstSparsity << 2) | flags_chunk));
      }
    }
    // Last step staged into stage 1 (even chunk count): launch it and drain.
    tcu_bar[0].arrive_and_wait();
    load_bar[1].arrive_and_wait();
    vt::mma_op(pending_rs1_val, pending_rs2_val);
    tcu_bar[1].arrive_and_wait();
    (void)current_stage; (void)next_stage;
#else  // fully unrolled: stage indices fold to constants on their own
#pragma unroll
    for (uint32_t bi = 0; bi < blocks_m * blocks_n; ++bi) {
      const uint32_t block_row = bi / blocks_n, block_col = bi % blocks_n;
      const uint32_t tile_row_idx = block_row * GM + wr;
      const uint32_t tile_col_idx = block_col * GN + wc;
      const uint32_t tile_id = tile_row_idx * tiles_n + tile_col_idx;
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);

#pragma unroll
      for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
        const uint32_t k_tile_idx = dense_iter;
        const uint32_t k_remaining = K - dense_iter * tile_K;
        uint32_t curr_k = std::min(k_remaining, tile_K);
        const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

        tcu_bar[current_stage].arrive_and_wait();
        if ((bi | dense_iter) != 0) {
          launch_pending_mma();
        }

        if (is_dxa_warp) {
          load_bar[next_stage].expect_tx(GM + GN);
          for (uint32_t i = 0; i < GM; ++i) {
            vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_at(next_stage, i), 0,
                               (block_row * GM + i) * tiles_k + k_tile_idx);
          }
          for (uint32_t j = 0; j < GN; ++j) {
            vx_dxa_issue_2d_wg(kDescB, load_bar[next_stage].id(), B_at(next_stage, j), 0,
                               (block_col * GN + j) * tiles_k + k_tile_idx);
          }
        }

        pending_rs1_val = (uintptr_t)vx_wgather(
                          (size_t)(uintptr_t)A_at(next_stage, wr),
                          (size_t)(uintptr_t)B_at(next_stage, wc),
                          (size_t)0,
                          (size_t)(uintptr_t)mma_D_addr);
        pending_rs2_val = (uintptr_t)vx_wgather(
                          (size_t)(uintptr_t)nullptr,
                          (size_t)(uintptr_t)nullptr,
                          (size_t)tcu_bar[next_stage].id(),
                          (size_t)((curr_k << 24) | (vt::OTYPE::id << 8) |
                                   (vt::ITYPE::id << 4) | (kConstSparsity << 2) | flags_chunk));
      }
    }

    launch_pending_mma();
    tcu_bar[current_stage].arrive_and_wait();
#endif // SGEMM_TCU_ROLLED
  }
#endif // SGEMM_TCU_COOP

  else if constexpr (kDense)
  {
    // Multi-engine dense path. Same staging/barrier/launch protocol as the
    // single-engine path above; only the tile assignment and the LMEM base
    // differ. This CTA's LMEM share holds two stages of A + B [+ C], and all
    // kEngines shares must fit at once or the CTAs would not be co-resident
    // and the engines would run one after another.
    static constexpr uint32_t kCtaLmemBytes = SGEMM_CTA_LMEM_BYTES(sizeof(ctx::input_t), tile_K, has_c);
    // kernel_main is not a template, so this branch's asserts are evaluated
    // even when another branch is taken; hence the kCoop guard.
    static_assert(kCoop || kEngines * kCtaLmemBytes <= lmem_capacity_bytes,
                  "all engine CTAs must be LMEM-resident at once (raise VX_CFG_LMEM_LOG_SIZE or lower SGEMM_TILE_K_MULT)");
    static constexpr uint32_t stage_regs = kCtaLmemBytes / 2 / sizeof(uint32_t);
    uint32_t* cta_lmem = reinterpret_cast<uint32_t *>(__local_mem(kCtaLmemBytes));
    uint32_t* A_lmem[2]  = {cta_lmem, cta_lmem + stage_regs};
    uint32_t* B_lmem[2]  = {A_lmem[0] + dense_a_tile_regs, A_lmem[1] + dense_a_tile_regs};
    uint32_t* Ce_lmem[2] = {B_lmem[0] + dense_b_tile_regs, B_lmem[1] + dense_b_tile_regs};
    static constexpr uint32_t tiles_per_cta = total_tiles / kEngines;
    static constexpr uint32_t kDenseLaunches = div_up_constexpr(K, tile_K);

#ifdef SGEMM_TCU_ROLLED
#pragma unroll 1
#else
#pragma unroll
#endif
    for (uint32_t i = 0; i < tiles_per_cta; ++i) {
      const uint32_t tile_id = block_tile_id + i * kEngines;
      const uint32_t tile_row_idx = tile_id / tiles_n;
      const uint32_t tile_col_idx = tile_id % tiles_n;
      const uintptr_t mma_D_addr = static_cast<uintptr_t>(arg->D_addr) + static_cast<uintptr_t>(tile_id) * tileD_regs * sizeof(uint32_t);

#pragma unroll
      for (uint32_t dense_iter = 0; dense_iter < kDenseLaunches; ++dense_iter) {
        const uint32_t k_tile_idx = dense_iter;
        const uint32_t k_remaining = K - dense_iter * tile_K;
        uint32_t curr_k = std::min(k_remaining, tile_K);
        const uint32_t flags_chunk = (uint32_t(dense_iter == 0) << 1) | uint32_t(dense_iter == (kDenseLaunches - 1));

        tcu_bar[current_stage].arrive_and_wait();

        /* In the very first execution, no data are ready so skip the mma_op */
        if ((i | dense_iter) != 0) {
          launch_pending_mma();
        }

        if (is_dxa_warp) {
          const bool stage_c = has_c && (dense_iter == 0);
          load_bar[next_stage].expect_tx(stage_c ? 3 : 2);
          if (stage_c) {
            vx_dxa_issue_2d_wg(kDescC, load_bar[next_stage].id(), Ce_lmem[next_stage], 0, tile_id);
          }
          vx_dxa_issue_2d_wg(kDescA, load_bar[next_stage].id(), A_lmem[next_stage], 0, tile_row_idx * tiles_k + k_tile_idx);
          vx_dxa_issue_2d_wg(kDescB, load_bar[next_stage].id(), B_lmem[next_stage], 0, tile_col_idx * tiles_k + k_tile_idx);
        }

        uintptr_t rs1_val = 0;
        uintptr_t rs2_val = 0;
        if (is_dxa_warp) {
          rs1_val = (uintptr_t)vx_wgather(
                    (size_t)(uintptr_t)A_lmem[next_stage],
                    (size_t)(uintptr_t)B_lmem[next_stage],
                    (size_t)(has_c ? (uintptr_t)Ce_lmem[next_stage] : (uintptr_t)0),
                    (size_t)(uintptr_t)mma_D_addr);
          rs2_val = (uintptr_t)vx_wgather(
                    (size_t)(uintptr_t)nullptr,
                    (size_t)(uintptr_t)nullptr,
                    (size_t)tcu_bar[next_stage].id(),
                    (size_t)((curr_k << 24) | (vt::OTYPE::id << 8) |
                             (vt::ITYPE::id << 4) | (kConstSparsity << 2) | flags_chunk));
        }
        pending_rs1_val = rs1_val;
        pending_rs2_val = rs2_val;
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
      [[maybe_unused]] const uint32_t a_tile_base = tile_row_idx * tile_M * K;
      const uint32_t tile_row_tiles_base = tile_row_idx * tiles_k;
#pragma unroll
      for (uint32_t tile_col_idx = 0; tile_col_idx < tiles_n; ++tile_col_idx) {

        [[maybe_unused]] const uint32_t b_tile_base = tile_col_idx * K * tile_N;
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
            // Post-rebase tx-barrier semantics: arm one expectation per DXA
            // issue in software before issuing; DXA only delivers done events.
            // C is staged once per output tile (on the first k-chunk), exactly
            // as the dense path does: the op core requires every read operand
            // LMEM-resident, so C cannot be handed to it as a global pointer.
            // With no C, nothing is staged and the engine gets a null C.
            const bool stage_c = has_c && (dense_iter == 0);
            load_bar[next_stage].expect_tx((kSparseA ? 4 : 3) + (stage_c ? 1 : 0));
            if (stage_c) {
              vx_dxa_issue_2d_wg(kDescC, load_bar[next_stage].id(), C_lmem[next_stage], 0, tile_id);
            }
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
            // init_flag=1 loads the accumulator's initial value from C. Use the
            // DXA-staged LMEM copy: this used to pass arg->C_addr directly,
            // which is a global address and violates the engine's LMEM-residency
            // contract -- the dcache coalescer then returns partial-mask
            // responses and the engine deadlocks. (That in turn was a workaround
            // for a null C pointer reading DRAM poison at address 0, which the
            // null-C latching fix in VX_tcu_op_core now handles properly.)
            rs1_val = (uintptr_t)vx_wgather(
                      (size_t)(uintptr_t)A_lmem[next_stage],
                      (size_t)(uintptr_t)B_lmem[next_stage],
                      (size_t)(has_c ? (uintptr_t)C_lmem[next_stage] : (uintptr_t)0),
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

#ifndef SGEMM_NO_KERNEL_METRICS
  const __rdcycle_time cycle_end = vx_rdcycle_sync_end();
  const uint64_t instret_end = vx_rdinstret_local();
  const uint64_t total_cycles = vx_rdcycle_sync_diff(cycle_begin, cycle_end);
  const uint64_t total_instructions = instret_end - instret_begin;

  if (vx_thread_id() == 0) {
    uint64_t* metrics = reinterpret_cast<uint64_t*>(arg->metrics_addr);
    metrics[0] = total_cycles;
    metrics[1] = total_instructions;
  }
#endif
}
