#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;

// NOTE (2026-04-09 port): originally used a 6-template-arg wmma_context API
// that no longer exists. Ported to the current vt::wgmma_context<NT, It, Ot,
// is_sparse, NRC> API with xtileM/xtileN geometry. Sparse path uses
// is_sparse=true, so A is 2:4 compressed in smem with metadata side-band.
using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true, WGMMA_NRC>;

// smem layout: [A_compressed][meta][B_dense]
// (A is cta_M rows tall so it must be sized at run-time — use a per-warp
// constant xtileM for the layout math; host sizes local_mem accordingly.)
static constexpr uint32_t smem_a_elems_per_warp = ctx::xtileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes_per_warp = smem_a_elems_per_warp * sizeof(ctx::input_t);
static constexpr uint32_t smem_b_elems  = ctx::tileK * ctx::xtileN;
static constexpr uint32_t smem_b_bytes  = smem_b_elems * sizeof(ctx::input_t);

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA    = 0;
constexpr uint32_t kDescB    = 1;
constexpr uint32_t kDescMeta = 2;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Single-warp-per-block geometry (matches original pre-port semantics). All
  // warps compute the same WGMMA redundantly; wasteful when num_warps>1 but
  // functionally correct with the existing main.cpp host setup that programs
  // DXA descriptors sized for a per-warp tile.
  uint32_t tile_row = blockIdx.y * ctx::xtileM;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // smem layout (per-warp):
  //   [A_compressed: xtileM × (tileK/2)] [meta: wg_meta_total_bytes] [B_dense: tileK × xtileN]
  // meta is bank-row-aligned after A.
  constexpr uint32_t smem_a_bytes_local  = ctx::xtileM * (ctx::tileK / 2) * sizeof(ctx::input_t);
  constexpr uint32_t smem_meta_off_local = smem_a_bytes_local;
  constexpr uint32_t smem_bank_bytes_local = NUM_THREADS * sizeof(float);
  constexpr uint32_t smem_b_off_local =
      ((smem_meta_off_local + ctx::wg_meta_total_bytes + smem_bank_bytes_local - 1) /
       smem_bank_bytes_local) * smem_bank_bytes_local;

  auto smem_base = reinterpret_cast<uint8_t*>(__local_mem());
  auto A_smem    = reinterpret_cast<ctx::input_t*>(smem_base);
  auto meta_smem = reinterpret_cast<uint32_t*>(smem_base + smem_meta_off_local);
  auto B_smem    = reinterpret_cast<ctx::input_t*>(smem_base + smem_b_off_local);

  // Initialize accumulator to zero
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);

  uint32_t meta_words_per_tile = ctx::wg_meta_total_bytes / 4;

  // Loop over K tiles
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;

    // DXA: load compressed A, dense B, and metadata tiles.
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA,    bar.id(), A_smem,    k / 2, tile_row);
      vx_dxa_issue_2d_wg(kDescB,    bar.id(), B_smem,    tile_col, k);
      vx_dxa_issue_2d_wg(kDescMeta, bar.id(), meta_smem, k_tile * meta_words_per_tile, blockIdx.y);
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Build smem descriptors:
    //   desc_a: compressed A with ldm = (tileK/2) * sizeof(input_t)
    //   desc_b: dense B with ldm = xtileN * sizeof(input_t)
    auto desc_a = vt::vx_make_smem_desc(A_smem, (ctx::tileK / 2) * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }

  // Store C tile to global memory (all warps store to the same location with
  // identical data, which is safe — last write wins with same value).
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
