#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_tmem.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
namespace tm = vortex::tmem;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
using wg_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

#ifdef TMEM_SPARSE
constexpr bool     kSparse = true;
constexpr uint32_t kTileK  = wg_cfg::tileK_sp;
#else
constexpr bool     kSparse = false;
constexpr uint32_t kTileK  = ctx::tileK;
#endif
#ifdef TMEM_A
constexpr bool kATmem = true;
#else
constexpr bool kATmem = false;
#endif
// The TCU copies the staged metadata into TMEM (tc.cp) instead of the warps.
#ifdef TMEM_CP
constexpr bool kCopyMeta = kSparse;
#else
constexpr bool kCopyMeta = false;
#endif

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;
[[maybe_unused]] constexpr uint32_t kDescMeta = 2;

#if (KB > 1) && !defined(B_KMAJOR)
#error "KB > 1 requires B_KMAJOR"
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

static_assert(sizeof(ctx::output_t) == sizeof(float), "the TMEM epilogue stores fp32 lanes");

constexpr uint32_t kPipeN = PIPE_N;
constexpr uint32_t kTileN = ctx::xtileN * TMEM_NCHUNK;  // CTA tile N
constexpr uint32_t kBK    = KB * kTileK;  // K per pipeline stage
// tc.mma K-tiles (one FEDP width of A each) per kTileK: a dense tile spans
// k_steps of them, a sparse one exactly one.
constexpr uint32_t kKtPerTile = kSparse ? 1 : wg_cfg::k_steps;
constexpr uint32_t kAK    = kSparse ? (kBK / 2) : kBK;  // A is compressed along K
#ifdef B_KMAJOR
constexpr uint32_t kBLdmBytes = kBK * sizeof(ctx::input_t);
#else
constexpr uint32_t kBLdmBytes = 0;
#endif

// Full, empty, done and TMEM-load barriers of a CTA.
static_assert(2 * kPipeN + 2 <= VX_CFG_NUM_BARRIERS, "not enough barriers for PIPE_N");

constexpr uint32_t kThreadsPerLane = VX_CFG_NUM_THREADS / ctx::xtileM;
constexpr uint32_t kMetaCols = (wg_cfg::fedpK * 2 * (32 / vt::ITYPE::bits) + 31) / 32;
// tmem::store writes whole fragments.
constexpr uint32_t kStoreCols = ctx::fragment_acc::NR * kThreadsPerLane;

// Warp `lane` group: copy `cols` columns from global words [row][col] into TMEM.
[[maybe_unused]] static inline void load_tmem_cols(uint32_t lane, uint32_t col0, uint32_t cols,
                                  const float* src, uint32_t stride, uint32_t row0) {
  const uint32_t t = vx_thread_id();
  const uint32_t row = row0 + t % ctx::xtileM;
  for (uint32_t c = 0; c < cols; c += kStoreCols) {
    ctx::fragment_acc frag;
    #pragma unroll
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) {
      uint32_t col = c + r * kThreadsPerLane + t / ctx::xtileM;
      frag.data[r] = (col < stride) ? src[row * stride + col] : 0.0f;
    }
    tm::store<tm::LANE_PER_THREAD>(tm::make_taddr(lane, col0 + c), frag);
  }
  tm::wait_st();
}

// Warp `lane` group: copy `cols` columns of the CTA's [row][stride] words in
// shared memory into TMEM. Threads read consecutive words of a row; columns
// past `stride` pick up the next row's words, which the MMA never reads.
[[maybe_unused]] static inline void smem_to_tmem_cols(uint32_t lane, uint32_t col0, uint32_t cols,
                                                      const float* src, uint32_t stride) {
  const uint32_t t = vx_thread_id();
  for (uint32_t c = 0; c < cols; c += kStoreCols) {
    ctx::fragment_acc frag;
    #pragma unroll
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) {
      frag.data[r] = src[(lane + r % ctx::xtileM) * stride + c + (r / ctx::xtileM) * VX_CFG_NUM_THREADS + t];
    }
    tm::store<tm::LANE_PER_REG>(tm::make_taddr(lane, col0 + c), frag);
  }
  tm::wait_st();
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // One warp per TCU block: block b computes rows [b*xtileM, (b+1)*xtileM).
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * kTileN;

  auto lmem = reinterpret_cast<uint8_t*>(__local_mem());
  auto tmem_slot = reinterpret_cast<volatile uint32_t*>(lmem);
  auto smem = reinterpret_cast<ctx::input_t *>(lmem + TMEM_SLOT_BYTES);
  const uint32_t stage_a_elems = kATmem ? 0 : (cta_M * kAK);
  const uint32_t stage_elems = stage_a_elems + kBK * kTileN;
  auto stage_A = [&](uint32_t s) { return smem + s * stage_elems; };
  auto stage_B = [&](uint32_t s) { return smem + s * stage_elems + stage_a_elems; };

  const uint32_t cta_id = get_local_group_id();
  auto full_bar  = [&](uint32_t s) { return cta_id | (s << 8); };
  auto empty_bar = [&](uint32_t s) { return cta_id | ((kPipeN + s) << 8); };
  const uint32_t done_bar = cta_id | ((2 * kPipeN) << 8);
  const uint32_t load_bar = cta_id | ((2 * kPipeN + 1) << 8);

  const uint32_t num_steps = K / kBK;
  const uint32_t k_tiles   = K / kTileK;

  // TMEM: D, then A (a_tmem), then the metadata (sparse).
  auto round_store = [](uint32_t cols) { return ((cols + kStoreCols - 1) / kStoreCols) * kStoreCols; };
  const uint32_t a_cols     = k_tiles * kKtPerTile * wg_cfg::fedpK;
  const uint32_t meta_words = k_tiles * kMetaCols;
  const uint32_t tmem_cols  = tm::alloc_cols(kTileN + (kATmem ? round_store(a_cols) : 0)
                                                    + (kSparse ? round_store(meta_words) : 0));
  if (warp_rank == 0) {
    uint32_t col = tm::alloc(tmem_cols);
    if (vx_thread_id() == 0) {
      *tmem_slot = col;
    }
  }
  vx_barrier(load_bar, num_warps);
  const uint32_t taddr    = *tmem_slot;
  const uint32_t a_col    = taddr + kTileN;
  const uint32_t meta_col = a_col + (kATmem ? round_store(a_cols) : 0);
  const uint32_t lane0   = warp_rank * ctx::xtileM;
  const uint32_t row0    = tile_row + lane0;
  // The metadata is staged through the pipeline stages, which no MMA uses yet.
  // Staged rows are padded to NT words (DXA fills past the row with zeros), as
  // tc.cp moves whole NT-word chunks.
  const uint32_t meta_pitch = (meta_words + VX_CFG_NUM_THREADS - 1) / VX_CFG_NUM_THREADS * VX_CFG_NUM_THREADS;
  if constexpr (kSparse) {
    if (warp_rank == 0) {
      vx_barrier_expect_tx(load_bar, 1);
      vx_dxa_issue_2d_wg(kDescMeta, load_bar, smem, 0, tile_row);
    }
  }
  if constexpr (kATmem) {
    const uint32_t a_stride = ((K * sizeof(ctx::input_t)) / (kSparse ? 2 : 1)) / 4;
    load_tmem_cols(lane0, a_col, a_cols, reinterpret_cast<const float*>(arg->A_addr), a_stride, row0);
  }
  if constexpr (kCopyMeta) {
    // With A in TMEM every warp meets on load_bar again, so all wait here too.
    if (kATmem || warp_rank == 0) {
      vx_barrier(load_bar, kATmem ? num_warps : 1);
    }
  } else if constexpr (kSparse) {
    vx_barrier(load_bar, num_warps);
    smem_to_tmem_cols(lane0, meta_col, meta_words, reinterpret_cast<const float*>(smem), meta_pitch);
  }
  if constexpr (kATmem || (kSparse && !kCopyMeta)) {
    vx_barrier(load_bar, num_warps);
  }

  if (warp_rank == 0) {
    if constexpr (kCopyMeta) {
      // The copy runs ahead of every tc.mma; the stages that hold the staged
      // metadata are refilled only once it completes.
      tm::cp(meta_col, smem, meta_pitch * 4, meta_pitch);
      tm::commit(load_bar, 2);
      vx_barrier(load_bar, 2);
    }
    // Producer: this warp and the MMA warp are the two participants of every
    // full and empty barrier.
    for (uint32_t i = 0; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      if (i >= kPipeN) {
        int phase = vx_barrier_arrive(empty_bar(s), 2);
        vx_barrier_wait(empty_bar(s), phase);
      }
      uint32_t k = i * kBK;
      vx_barrier_expect_tx(full_bar(s), kATmem ? 1 : 2);
      if constexpr (!kATmem) {
        vx_dxa_issue_2d_wg(kDescA, full_bar(s), stage_A(s), i * kAK, tile_row);
      }
      vx_dxa_issue_2d_wg(kDescB, full_bar(s), stage_B(s), b_coord0(k, tile_col), b_coord1(k, tile_col));
      vx_barrier_arrive(full_bar(s), 2);
    }
    // Close the last generation of every empty barrier the MMA warp committed.
    uint32_t first = (num_steps > kPipeN) ? (num_steps - kPipeN) : 0;
    for (uint32_t i = first; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      int phase = vx_barrier_arrive(empty_bar(s), 2);
      vx_barrier_wait(empty_bar(s), phase);
    }
  } else if (warp_rank == 1) {
    // Per stage: wait for it, multiply, and commit it back to the producer;
    // the last commit completes the GEMM.
    tm::set_idesc<0>(tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, kTileN / ctx::tcN, KB * kKtPerTile, kSparse, kATmem),
                     tm::make_ld(kAK * sizeof(ctx::input_t), kBLdmBytes));
    for (uint32_t i = 0; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      int phase = vx_barrier_arrive(full_bar(s), 2);
      vx_barrier_wait(full_bar(s), phase);
      tm::mma<0>(taddr, kATmem ? (a_col + i * KB * kKtPerTile * wg_cfg::fedpK) : tm::smem_offset(stage_A(s)),
                 tm::smem_offset(stage_B(s)), i == 0, kSparse ? (meta_col + i * KB * kMetaCols) : 0);
      tm::commit(empty_bar(s), 2);
    }
    tm::commit(done_bar, num_warps + 1);
  }

  // Epilogue: wait for the MMAs; warp r drains TMEM lanes (M rows)
  // [r*xtileM, (r+1)*xtileM), xtileN columns per load.
  vx_barrier(done_bar, num_warps + 1);

  const uint32_t lane = warp_rank * ctx::xtileM;
  const uint32_t t = vx_thread_id();
  for (uint32_t c = 0; c < TMEM_NCHUNK; ++c) {
    ctx::fragment_acc fragC;
    uint32_t col = taddr + c * ctx::xtileN;
    auto pChunkC = pC + (tile_row + lane) * N + tile_col + c * ctx::xtileN;
  #ifdef TMEM_LANE_PER_THREAD
    // Thread t holds one M row; a store writes NT rows.
    constexpr uint32_t kThreadsPerLane = VX_CFG_NUM_THREADS / ctx::xtileM;
    tm::load<tm::LANE_PER_THREAD>(fragC, tm::make_taddr(lane, col));
    auto pRowC = reinterpret_cast<float*>(pChunkC + (t % ctx::xtileM) * N + t / ctx::xtileM);
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) {
      pRowC[r * kThreadsPerLane] = fragC.data[r];
    }
  #else
    // Register r holds NT contiguous columns of one M row; a store writes one row.
    tm::load<tm::LANE_PER_REG>(fragC, tm::make_taddr(lane, col));
    #pragma unroll
    for (uint32_t g = 0; g < ctx::fragment_acc::NR / ctx::xtileM; ++g) {
      auto p = reinterpret_cast<float*>(pChunkC + g * VX_CFG_NUM_THREADS + t);
      #pragma unroll
      for (uint32_t i = 0; i < ctx::xtileM; ++i) {
        *p = fragC.data[g * ctx::xtileM + i];
        p += N;
      }
    }
  #endif
  }

  vx_barrier(load_bar, num_warps);
  if (warp_rank == 0) {
    tm::dealloc(taddr, tmem_cols);
  }
}
