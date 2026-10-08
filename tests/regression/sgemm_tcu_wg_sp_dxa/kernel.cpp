#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
// WGMMA accumulator; is_sparse=true (A is 2:4 compressed in smem)
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, true, WGMMA_NRC>;

// Per-warp smem layout: [A_compressed][reserved metadata row], then shared B.
static constexpr uint32_t smem_a_elems     = ctx::xtileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes     = smem_a_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_b_elems     = ctx::tileK * ctx::xtileN;
static constexpr uint32_t smem_b_bytes     = smem_b_elems * sizeof(ctx::input_t);
// SMEM bank-row width = NUM_THREADS × LSU_WORD_SIZE (= XLEN/8).
static constexpr uint32_t smem_bank_bytes  = VX_CFG_NUM_THREADS * (VX_CFG_XLEN / 8);
[[maybe_unused]] static constexpr uint32_t per_warp_section = ((smem_a_bytes + ctx::wg_meta_total_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
[[maybe_unused]] static constexpr uint32_t smem_b_bytes_rounded = ((smem_b_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA    = 0;
constexpr uint32_t kDescB    = 1;
// META_DXA (DEEP_BUF only): stage each pipeline stage's sparse metadata --
// every warp's rows for the stage's KB K-tiles, one 2D tile of
// [num_warps][KB * meta_words] -- through DXA alongside A/B, and read it with
// TCU_LD from LMEM. Hopper's wgmma.sp takes metadata as a register operand
// that software loads ahead of use, so its latency is hidden; here a direct
// per-K-tile GMEM TCU_LD sits in front of every WGMMA, and its per-warp
// latency variance knocks the warpgroup out of lockstep on the shared bbuf
// (measured: removing it closes sparse's gap to dense at 128^3).
// Per-K-tile staging was tried and lost to DXA's fixed per-transfer overhead
// on a 32 B payload; per-stage staging amortizes it over KB K-tiles x warps.
[[maybe_unused]] constexpr uint32_t kDescMeta = 2;
#if defined(META_DXA) && !defined(DEEP_BUF)
#error "META_DXA requires DEEP_BUF"
#endif

// B layout in global/shared memory. Default: B is [K][N] and DXA scatters it
// into the bbuf-native layout (BlockMajor for dense, Flat for sparse).
// B_KMAJOR: the host stores B^T ([N][K] -- NVIDIA's canonical K-major wgmma B,
// i.e. a "TN" GEMM), DXA copies it row-major with no scatter into
// [tileN][tileK], and the B descriptor's ldm selects the bbuf's K-major
// fetch path. Same tile bytes; the DXA writer no longer drains per element.
#if defined(B_KMAJOR) && defined(SW_LOAD_B)
#error "B_KMAJOR is a DXA row-major layout; SW_LOAD_B writes the scattered layout"
#endif
static inline __attribute__((always_inline)) uint32_t b_coord0(uint32_t k, uint32_t col) {
#ifdef B_KMAJOR
  (void)col; return k;
#else
  (void)k; return col;
#endif
}
static inline __attribute__((always_inline)) uint32_t b_coord1(uint32_t k, uint32_t col) {
#ifdef B_KMAJOR
  (void)k; return col;
#else
  (void)col; return k;
#endif
}
// KB: K-tiles per pipeline stage (CUTLASS BLOCK_K analogue). One stage
// fetches BK = KB*tileK of K for A and B in a single DXA transfer each and
// issues KB WGMMAs against it, offsetting the smem descriptors. KB>1 widens
// every A row and every K-major B row (e.g. NT=8 fp16, KB=4: 64 B B-rows =
// one full GMEM line instead of a quarter line), cuts DXA transfers and
// barrier round trips per K by KB. It requires B_KMAJOR -- the scattered
// Flat/BlockMajor layouts are not sub-tile addressable here.
#ifndef KB
#define KB 1
#endif
#if (KB > 1) && (!defined(B_KMAJOR) || !defined(DEEP_BUF))
#error "KB > 1 requires B_KMAJOR and DEEP_BUF"
#endif
#ifdef B_KMAJOR
static constexpr uint32_t kBLdmBytes = KB * ctx::tileK * sizeof(ctx::input_t);
#else
static constexpr uint32_t kBLdmBytes = 0;
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  auto smem_base = reinterpret_cast<uint8_t*>(__local_mem());

  // Initialize accumulator to zero
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t meta_words_per_tile = ctx::wg_meta_total_bytes / 4;
  uint32_t tile_row_idx_w = blockIdx.y * num_warps + warp_rank;
  auto pMetaSp = reinterpret_cast<const uint32_t*>(arg->meta_sp_addr);

#if defined(DEEP_BUF)
  // ── N-stage software-pipelined DXA fetch (sparse) ───────────────────────
  // A is fetched as ONE combined DXA transfer covering the whole CTA's
  // cta_M rows (matching sgemm_tcu_wg_dxa_deepbuf's dense DEEP_BUF pattern),
  // instead of one transfer per warp. Each warp indexes its own row-slice
  // out of the shared buffer, collapsing DXA issue count from (num_warps+1)
  // down to 2 per stage -- the host (main.cpp) programs kDescA's row-tile
  // size as cta_M to match. Each stage covers KB K-tiles (see KB above).
  //
  // Sparse metadata is read straight from global memory via TCU_LD per
  // K-tile, NOT staged through DXA like A/B. Staging it as a third per-stage
  // DXA transfer was measured and reverted: cycles rose 9.1% -- DXA's fixed
  // per-transfer overhead costs more than the stall it hides for a payload
  // this small (meta_words_per_tile words).
  constexpr uint32_t kPipeN = PIPE_N;
  constexpr uint32_t kBK    = KB * ctx::tileK;     // K per pipeline stage
  const uint32_t num_stages = num_k_tiles / KB;
  const uint32_t stage_a_bytes = cta_M * (kBK / 2) * sizeof(ctx::input_t);
  const uint32_t stage_b_off   = ((stage_a_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  const uint32_t stage_b_bytes = ((KB * smem_b_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
#ifdef META_DXA
  const uint32_t stage_meta_off   = stage_b_off + stage_b_bytes;
  const uint32_t stage_meta_bytes = ((num_warps * KB * meta_words_per_tile * 4 + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  const uint32_t stage_bytes   = stage_meta_off + stage_meta_bytes;
  constexpr uint32_t kStageTx  = 3;
#else
  const uint32_t stage_bytes   = stage_b_off + stage_b_bytes;
  constexpr uint32_t kStageTx  = 2;
#endif
  auto stage_base = [&](uint32_t s) { return smem_base + s * stage_bytes; };
  auto stage_A = [&](uint32_t s) {
    return reinterpret_cast<ctx::input_t*>(stage_base(s));
  };
  auto stage_B = [&](uint32_t s) {
    return reinterpret_cast<ctx::input_t*>(stage_base(s) + stage_b_off);
  };
  const uint32_t cta_id = get_local_group_id();
  auto bar_id = [&](uint32_t s) { return cta_id | (s << 8); };
#ifdef META_DXA
  auto stage_Meta = [&](uint32_t s) {
    return reinterpret_cast<const uint32_t*>(stage_base(s) + stage_meta_off);
  };
  // Stage s's metadata tile: rows = this CTA's warps, cols = its KB K-tiles.
  auto issue_meta = [&](uint32_t s, uint32_t stage_idx) {
    vx_dxa_issue_2d_wg(kDescMeta, bar_id(s), stage_Meta(s),
                       stage_idx * KB * meta_words_per_tile, blockIdx.y * num_warps);
  };
#endif

  // Prologue: fill the pipeline kPipeN-1 stages ahead of the first compute.
  if (is_dxa_warp) {
    for (uint32_t s = 0; s < kPipeN - 1 && s < num_stages; ++s) {
      vx_barrier_expect_tx(bar_id(s), kStageTx);
      vx_dxa_issue_2d_wg(kDescA, bar_id(s), stage_A(s), (s * kBK) / 2, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id(s), stage_B(s), b_coord0(s * kBK, tile_col), b_coord1(s * kBK, tile_col));
    #ifdef META_DXA
      issue_meta(s, s);
    #endif
    }
  }

  for (uint32_t i = 0; i < num_stages; ++i) {
    const uint32_t cur         = i % kPipeN;
    const uint32_t issue_i     = i + kPipeN - 1;
    const bool     has_issue   = (issue_i < num_stages);
    const uint32_t issue_stage = issue_i % kPipeN;
    const uint32_t issue_k     = issue_i * kBK;

    // (1) Keep the pipeline full: issue the fetch kPipeN-1 stages ahead.
    if (has_issue && is_dxa_warp) {
      vx_barrier_expect_tx(bar_id(issue_stage), kStageTx);
      vx_dxa_issue_2d_wg(kDescA, bar_id(issue_stage), stage_A(issue_stage), issue_k / 2, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id(issue_stage), stage_B(issue_stage), b_coord0(issue_k, tile_col), b_coord1(issue_k, tile_col));
    #ifdef META_DXA
      issue_meta(issue_stage, issue_i);
    #endif
    }

    // (2) Wait for the current stage's DXA completion.
    vx_barrier(bar_id(cur), num_warps);

    // (3) KB WGMMAs against the stage. A is [cta_M][BK/2] (compressed),
    // K-major B is [tileN][BK]; sub-tile ki offsets the descriptor bases by
    // ki K-tiles and keeps the full-stage row strides.
    for (uint32_t ki = 0; ki < KB; ++ki) {
      const uint32_t kt = i * KB + ki;
    #ifdef META_DXA
      (void)kt; (void)tile_row_idx_w; (void)pMetaSp;
      auto meta_sp = stage_Meta(cur) + (warp_rank * KB + ki) * meta_words_per_tile;
    #else
      auto meta_sp = pMetaSp + (tile_row_idx_w * num_k_tiles + kt) * meta_words_per_tile;
    #endif
      ctx::fragment_a fragA;
      ctx::load_sp_metadata(fragA, meta_sp);

      auto A_warp = stage_A(cur) + warp_rank * ctx::xtileM * (kBK / 2) + ki * (ctx::tileK / 2);
      auto desc_b = vt::vx_make_smem_desc(stage_B(cur) + ki * ctx::tileK, kBLdmBytes);

    #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
      ctx::load_matrix_sync(fragA, A_warp, kBK / 2);
      ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
    #else
      auto desc_a = vt::vx_make_smem_desc(A_warp, (kBK / 2) * sizeof(ctx::input_t));
      ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
    #endif
    }

    // (4) Gate stage reuse: all warps done reading before its next refetch.
    vx_barrier(bar_id(cur), num_warps);
  }
#else
  uint32_t smem_b_off = ((num_warps * per_warp_section + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  auto B_smem = reinterpret_cast<ctx::input_t*>(smem_base + smem_b_off);

  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Loop over K tiles
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;

    // DXA: load per-warp compressed A, plus shared B.
    if (is_dxa_warp) {
      bar.expect_tx(num_warps + 1);
      for (uint32_t w = 0; w < num_warps; ++w) {
        auto A_smem_w = reinterpret_cast<ctx::input_t*>(smem_base + w * per_warp_section);
        vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem_w, k / 2, tile_row + w * ctx::xtileM);
      }
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, b_coord0(k, tile_col), b_coord1(k, tile_col));
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Each warp consumes its A section from smem and metadata from global memory.
    auto A_warp = reinterpret_cast<ctx::input_t*>(smem_base + warp_rank * per_warp_section);
    auto meta_sp = pMetaSp
        + (tile_row_idx_w * num_k_tiles + k_tile) * meta_words_per_tile;
    // B in SMEM: flat candidate-pair (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(B_smem, kBLdmBytes);

    // Sparse metadata is loaded via TCU_LD regardless of A's source —
    // both RS and SS sparse WGMMA use the same metadata path.
    ctx::fragment_a fragA;
    ctx::load_sp_metadata(fragA, meta_sp);

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK / 2);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both A and B from smem descriptors
    auto desc_a = vt::vx_make_smem_desc(A_warp, (ctx::tileK / 2) * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }
#endif

  // Store C tile to global memory
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
