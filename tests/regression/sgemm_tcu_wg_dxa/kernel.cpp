#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
[[maybe_unused]] constexpr uint32_t kDescA = 0;
[[maybe_unused]] constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
#ifdef SW_LOAD_B
  auto pB = reinterpret_cast<const ctx::input_t *>(arg->B_addr);
#endif
#ifdef SW_LOAD_A
  auto pA = reinterpret_cast<const ctx::input_t *>(arg->A_addr);
#endif

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

  // Double-buffered shared memory: two {A, B} tile pairs so the DXA engine
  // stages tile i+1 while the warp-group computes tile i. Opt-in: at the
  // measured configurations the DXA/compute overlap contends for LMEM and
  // DRAM bandwidth and loses to the serialized single-buffer schedule.
  //   buffer p: A [cta_M x tileK] then B [tileK x xtileN]
  uint32_t a_elems   = cta_M * ctx::tileK;
  uint32_t b_elems   = ctx::tileK * ctx::xtileN;
  uint32_t buf_elems = a_elems + b_elems;
  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  ctx::input_t* A_buf[2] = {smem, smem + buf_elems};
  ctx::input_t* B_buf[2] = {smem + a_elems, smem + buf_elems + a_elems};

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // One transaction barrier per buffer: phase a = staging complete for the
  // tile in that buffer, phase b = every warp done reading it (so the next
  // DXA issue may overwrite).
  vortex::barrier bar0(0), bar1(1);
  vortex::barrier* bars[2] = {&bar0, &bar1};

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  // Stage tile k into buffer p.
  // SW_LOAD_B replaces B's DXA with a cooperative SW load (K-major);
  // SW_LOAD_A replaces A's DXA with a cooperative SW load (row-major).
  auto load_tile = [&](uint32_t p, uint32_t k) {
    auto A_smem = A_buf[p];
    auto B_smem = B_buf[p];
    auto& bar   = *bars[p];
    #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
      // both via SW — no DXA needed
    #elif defined(SW_LOAD_A)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k); }
    #elif defined(SW_LOAD_B)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row); }
    #else
      if (is_dxa_warp) {
        bar.expect_tx(2);  // Two pending transactions: A + B
        vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row);
        vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
      }
    #endif
    #ifdef SW_LOAD_A
      // Cooperative load A row-major (matches DXA row-major layout).
      uint32_t a_size = cta_M * ctx::tileK;
      for (uint32_t i = 0; i < a_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::tileK;
        uint32_t c = idx % ctx::tileK;
        A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
      }
    #endif
    #ifdef SW_LOAD_B
      // Cooperative load B block-major (matches DXA BlockMajor LAYOUT / bbuf).
      uint32_t b_size = ctx::tileK * ctx::xtileN;
      for (uint32_t i = 0; i < b_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::xtileN;
        uint32_t c = idx % ctx::xtileN;
        B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
      }
    #endif
    (void)k;
  };

  uint32_t num_k_tiles = K / ctx::tileK;

  // Software pipeline: with DXA_DOUBLE_BUFFER, prefetch the next tile before
  // waiting on the current one so the DXA transfer overlaps the WGMMA
  // compute; otherwise stage and compute serially in buffer 0.
  load_tile(0, 0);
  for (uint32_t i = 0; i < num_k_tiles; ++i) {
#ifdef DXA_DOUBLE_BUFFER
    uint32_t p = i & 1u;
    if (i + 1 < num_k_tiles) {
      load_tile(1u - p, (i + 1) * ctx::tileK);
    }
#else
    uint32_t p = 0;
    if (i != 0) {
      load_tile(0, i * ctx::tileK);
    }
#endif

    // Wait for tile i's staging into buffer p (all warps participate).
    bars[p]->arrive_and_wait();

    auto A_warp = A_buf[p] + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: block-major (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(B_buf[p], 0);

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Release buffer p before it is re-staged: under DXA_DOUBLE_BUFFER
    // tile i+2 is issued into it at the top of the next iteration; single
    // buffered, tile i+1 is.
#ifdef DXA_DOUBLE_BUFFER
    if (i + 2 < num_k_tiles) {
      bars[p]->arrive_and_wait();
    }
#else
    if (i + 1 < num_k_tiles) {
      bars[p]->arrive_and_wait();
    }
#endif
  }

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
