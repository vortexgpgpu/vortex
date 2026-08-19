// GPU-side program for mode 6 -- the committed single-buffer form of mode 5.
//
// A [cta_M x K] stays resident while the CTA sweeps MOTI_WG_NCOLS output-column
// tiles. B has one [xtileN x kStK] buffer: every K step issues one DXA transfer,
// waits for it, consumes it with WGMMA, then waits before reusing the buffer. This
// intentionally preserves the original architecture target so it can be measured
// beside mode 5's double-buffered B schedule.

#include "k_wg_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>

static constexpr uint32_t kNCols = MOTI_WG_NCOLS;

__kernel __attribute__((aligned(256))) void
moti_tcu_wg_acol_single(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<wgctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<wgctx::output_t*>(arg->D_addr);

  const uint32_t tid       = threadIdx.x;
  const uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;

  const uint32_t cta_M    = num_warps * wgctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t col_base = blockIdx.x * (kNCols * wgctx::xtileN);

  auto smem   = reinterpret_cast<wgctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * K;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  if (is_dxa_warp) {
    bar.expect_tx(1);
    vx_dxa_issue_2d_wg(DESC_A, bar.id(), A_smem, 0, tile_row);
  }
  bar.arrive_and_wait();

  for (uint32_t n = 0; n < kNCols; ++n) {
    const uint32_t tile_col = col_base + n * wgctx::xtileN;

    wgctx::fragment_acc fragC;
    wgctx::fill_fragment(fragC, 0);

    for (uint32_t k = 0; k < K; k += kStK) {
      if (is_dxa_warp) {
        bar.expect_tx(1);
        vx_dxa_issue_2d_wg(DESC_B, bar.id(), B_smem, k, tile_col);
      }
      bar.arrive_and_wait();

      for (uint32_t s = 0; s < kS; ++s) {
        auto A_warp = A_smem + warp_rank * wgctx::xtileM * K
                    + k + s * wgctx::tileK;
        auto desc_b = vt::vx_make_smem_desc(B_smem + s * wgctx::tileK,
                                            kStK * sizeof(wgctx::input_t));
        wgctx::fragment_a fragA;
        wgctx::load_matrix_sync(fragA, A_warp, K);
        wgctx::wgmma_sync(fragC, fragA, desc_b, fragC);
      }

      bar.arrive_and_wait();
    }

    wg_store_epilogue(fragC, pC, pD, tile_row + warp_rank * wgctx::xtileM,
                      tile_col, N, app, arg->M);
  }
}

// Keep standalone reduction kernels after the primary entry so every mode starts at the
// same aligned address. Only the pass selected by MOTI_APP is compiled.
#include "k_epilogue.h"
