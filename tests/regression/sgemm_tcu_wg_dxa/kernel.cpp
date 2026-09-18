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

  // Shared-memory tile buffers. Under DXA_DOUBLE_BUFFER two {A, B} pairs are
  // carved out and the loop ping-pongs between them by XOR-TOGGLING the
  // buffer pointers and the barrier id in place. Two earlier attempts failed
  // for register-residency reasons on the SIMT stack (per-lane stacks live in
  // global memory at an 8KB stride, so ONE spilled word costs 16 uncoalesced
  // same-bank DRAM misses):
  //   * `A_buf[2]` / `bars[2]` arrays indexed by a runtime `p` could not be
  //     scalarized — barrier objects reloaded from the stack around every
  //     vx_bar in the hot loop (8.3x slowdown);
  //   * a 2x-unrolled loop with named buffers kept the loop clean but its
  //     larger live set added ~5 callee-saved registers, whose prologue and
  //     epilogue save/restore bursts per CTA wave stalled the dcache bank
  //     ~100k cycles (2.5x slowdown).
  // The XOR toggle needs only two extra loop-invariant constants (the A and
  // B pointer toggles) and mutates state in registers, so the live set stays
  // within caller-saved registers and the stack is never touched.
  //   buffer: A [cta_M x tileK] then B [tileK x xtileN]
  uint32_t a_elems   = cta_M * ctx::tileK;
  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  ctx::input_t* A_cur = smem;
  ctx::input_t* B_cur = smem + a_elems;

  // Barrier id: slot 0/1 selected by bit 8, CTA slot in the low byte (same
  // packing as vortex::barrier). Raw intrinsics are used instead of barrier
  // objects so the id lives in one register.
  uint32_t bar_cur = get_local_group_id();
  const uint32_t num_bar_warps = get_num_sub_groups();
#ifdef DXA_DOUBLE_BUFFER
  uint32_t buf_elems = a_elems + ctx::tileK * ctx::xtileN;
  const uintptr_t a_tog = reinterpret_cast<uintptr_t>(smem) ^
                          reinterpret_cast<uintptr_t>(smem + buf_elems);
  const uintptr_t b_tog = reinterpret_cast<uintptr_t>(smem + a_elems) ^
                          reinterpret_cast<uintptr_t>(smem + buf_elems + a_elems);
  constexpr uint32_t kBarTog = 1u << 8;
#endif

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  // Stage the tile at K-offset k into the {A_smem, B_smem} buffer guarded
  // by barrier id `bar`. All state is passed by value in registers.
  // SW_LOAD_B replaces B's DXA with a cooperative SW load (K-major);
  // SW_LOAD_A replaces A's DXA with a cooperative SW load (row-major).
  auto load_tile = [&](ctx::input_t* A_smem, ctx::input_t* B_smem,
                       uint32_t bar, uint32_t k) {
    #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
      // both via SW — no DXA needed
    #elif defined(SW_LOAD_A)
      if (is_dxa_warp) { vx_barrier_expect_tx(bar, 1); vx_dxa_issue_2d_wg(kDescB, bar, B_smem, tile_col, k); }
    #elif defined(SW_LOAD_B)
      if (is_dxa_warp) { vx_barrier_expect_tx(bar, 1); vx_dxa_issue_2d_wg(kDescA, bar, A_smem, k, tile_row); }
    #else
      if (is_dxa_warp) {
        vx_barrier_expect_tx(bar, 2);  // Two pending transactions: A + B
        vx_dxa_issue_2d_wg(kDescA, bar, A_smem, k, tile_row);
        vx_dxa_issue_2d_wg(kDescB, bar, B_smem, tile_col, k);
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
    (void)A_smem; (void)B_smem; (void)bar; (void)k;
  };

  // One WGMMA accumulation step out of the given buffer pair.
  auto compute_tile = [&](ctx::input_t* A_smem, ctx::input_t* B_smem) {
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: block-major (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0);
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
  };

  uint32_t num_k_tiles = K / ctx::tileK;

#ifdef DXA_DOUBLE_BUFFER
  // Software pipeline over the two buffers: stage tile i+1 into the OTHER
  // buffer, wait for tile i, compute it, release its buffer, toggle.
  // Barrier phase protocol per buffer id (unchanged): wait #1 = staging
  // complete (expect_tx events + all warps), wait #2 = all readers done, so
  // the buffer may be re-staged. Loop-carried state is {A_cur, B_cur,
  // bar_cur} mutated by XOR — no arrays, no unrolling, no object reloads.
  load_tile(A_cur, B_cur, bar_cur, 0);
  for (uint32_t i = 0; i < num_k_tiles; ++i) {
    if (i + 1 < num_k_tiles) {
      // Prefetch tile i+1 into the other buffer while tile i is computed.
      // Its buffer's readers were released at iteration i-1 (or it is
      // untouched when i == 0), so staging may begin immediately.
      load_tile(reinterpret_cast<ctx::input_t*>(
                    reinterpret_cast<uintptr_t>(A_cur) ^ a_tog),
                reinterpret_cast<ctx::input_t*>(
                    reinterpret_cast<uintptr_t>(B_cur) ^ b_tog),
                bar_cur ^ kBarTog, (i + 1) * ctx::tileK);
    }
    vx_barrier(bar_cur, num_bar_warps);   // tile i staged
    compute_tile(A_cur, B_cur);
    if (i + 2 < num_k_tiles) {
      vx_barrier(bar_cur, num_bar_warps); // readers done: buffer may restage
    }
    A_cur = reinterpret_cast<ctx::input_t*>(
                reinterpret_cast<uintptr_t>(A_cur) ^ a_tog);
    B_cur = reinterpret_cast<ctx::input_t*>(
                reinterpret_cast<uintptr_t>(B_cur) ^ b_tog);
    bar_cur ^= kBarTog;
  }
#else
  // Serial single-buffer schedule: stage and compute alternately in buf0.
  load_tile(A_cur, B_cur, bar_cur, 0);
  for (uint32_t i = 0; i < num_k_tiles; ++i) {
    if (i != 0) {
      load_tile(A_cur, B_cur, bar_cur, i * ctx::tileK);
    }
    vx_barrier(bar_cur, num_bar_warps);   // tile i staged
    compute_tile(A_cur, B_cur);
    if (i + 1 < num_k_tiles) {
      vx_barrier(bar_cur, num_bar_warps); // release buf0 for tile i+1
    }
  }
#endif

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
