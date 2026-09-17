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

  // Shared memory holds two staging buffers, each A [cta_M x tileK] followed
  // by B [tileK x xtileN], with one transaction barrier per buffer. The
  // current buffer's pointers and packed barrier id are toggled in place by
  // XOR with the other buffer's, so the loop carries no indexed state.
  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  const uint32_t a_elems     = cta_M * ctx::tileK;
  const uint32_t stage_elems = a_elems + ctx::tileK * ctx::xtileN;
  uintptr_t A_cur = reinterpret_cast<uintptr_t>(smem);
  uintptr_t B_cur = reinterpret_cast<uintptr_t>(smem + a_elems);
  const uintptr_t A_xor = A_cur ^ reinterpret_cast<uintptr_t>(smem + stage_elems);
  const uintptr_t B_xor = B_cur ^ reinterpret_cast<uintptr_t>(smem + stage_elems + a_elems);
  const uint32_t cta_id  = get_local_group_id();
  uint32_t bar_cur = (0u << 8) + cta_id;
  const uint32_t bar_xor = bar_cur ^ ((1u << 8) + cta_id);

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  const uint32_t num_cta_warps = get_num_sub_groups();
  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  // Stage the K tile at column k into the buffer (A_dst, B_dst) guarded by
  // bar_id. The DXA warp registers the transfers on the barrier before
  // issuing them; the later arrival of every warp on that barrier completes
  // once the transfers have landed. SW_LOAD_B replaces B's DXA with a
  // cooperative SW load (K-major); SW_LOAD_A replaces A's likewise (row-major).
  auto stage_tile = [&](uintptr_t A_dst, uintptr_t B_dst, uint32_t bar_id, uint32_t k) __attribute__((always_inline)) {
    auto A_smem = reinterpret_cast<ctx::input_t *>(A_dst);
    auto B_smem = reinterpret_cast<ctx::input_t *>(B_dst);
  #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
    // both via SW — no DXA needed
    (void)bar_id;
  #elif defined(SW_LOAD_A)
    if (is_dxa_warp) { vx_barrier_expect_tx(bar_id, 1); vx_dxa_issue_2d_wg(kDescB, bar_id, B_smem, tile_col, k); }
  #elif defined(SW_LOAD_B)
    if (is_dxa_warp) { vx_barrier_expect_tx(bar_id, 1); vx_dxa_issue_2d_wg(kDescA, bar_id, A_smem, k, tile_row); }
  #else
    if (is_dxa_warp) {
      vx_barrier_expect_tx(bar_id, 2);  // Two pending transactions: A + B
      vx_dxa_issue_2d_wg(kDescA, bar_id, A_smem, k, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id, B_smem, tile_col, k);
    }
  #endif
  #ifdef SW_LOAD_A
    // Cooperative load A row-major (matches DXA row-major layout).
    for (uint32_t i = 0; i < a_elems; i += num_threads) {
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
  };

  // Two-stage pipeline over K tiles: tile k+tileK is staged into the idle
  // buffer while tile k computes. Each buffer's barrier alternates between
  // staging complete (transfers plus arrivals) and readers done (arrivals
  // only); the readers-done phase runs only when the buffer is refilled, so
  // the prefetch and release conditions pair off with no tail code.
  stage_tile(A_cur, B_cur, bar_cur, 0);
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    const uint32_t k_next = k + ctx::tileK;
    if (k_next < K) {
      stage_tile(A_cur ^ A_xor, B_cur ^ B_xor, bar_cur ^ bar_xor, k_next);
    }

    // Wait for the current tile (all warps participate).
    vx_barrier(bar_cur, num_cta_warps);

    auto A_warp = reinterpret_cast<ctx::input_t *>(A_cur) + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: block-major (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(reinterpret_cast<ctx::input_t *>(B_cur), 0);

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

    // Release the buffer before the tile after next is staged into it.
    if (k_next + ctx::tileK < K) {
      vx_barrier(bar_cur, num_cta_warps);
    }

    A_cur ^= A_xor;
    B_cur ^= B_xor;
    bar_cur ^= bar_xor;
  }

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
