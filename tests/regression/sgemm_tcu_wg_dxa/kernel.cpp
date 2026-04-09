#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;

// NOTE (2026-04-09 port): this kernel was originally written against an older
// 6-template-arg wmma_context API that no longer exists. Current WGMMA support
// lives in vt::wgmma_context<NT, It, Ot, is_sparse, NRC> with xtileM/xtileN
// geometry names. Ported minimally: same single-warp DXA issue pattern, same
// descriptor slots, just moved to the wgmma_context API + geometry names.
using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Single-warp-per-block geometry (matches original 2026-04 sgemm_tcu_wg_dxa
  // semantics before the port). All warps in the CTA see the same smem data
  // and compute the same WGMMA redundantly — this is wasteful when num_warps>1
  // but preserves functional correctness with the existing main.cpp host setup
  // (which programs DXA descriptors sized for a per-warp tile, not a CTA tile).
  uint32_t tile_row = blockIdx.y * ctx::xtileM;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory: A tile [xtileM x tileK] followed by B tile [tileK x xtileN],
  // both stored row-major.
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::xtileM * ctx::tileK;

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  // Loop over K tiles.
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // DXA: load A tile [tile_row .. tile_row+xtileM, k .. k+tileK] into A_smem.
    // DXA: load B tile [k .. k+tileK, tile_col .. tile_col+xtileN] into B_smem.
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Build smem descriptors: leading_bytes = row stride in bytes.
    auto desc_a = vt::vx_make_smem_desc(A_smem, ctx::tileK  * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

    // Execute WGMMA: C += A * B (all warps do the same work).
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }

  // Store the computed C tile to global memory. All warps store to the same
  // location since they all computed the same result; this is safe because the
  // final value is identical regardless of which warp wrote last.
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
