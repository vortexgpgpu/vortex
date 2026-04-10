#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
// WGMMA accumulator; is_sparse=true (A is 2:4 compressed in smem)
using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true, WGMMA_NRC>;

// smem layout: [A_compressed][meta][B_dense]
static constexpr uint32_t smem_a_elems  = ctx::xtileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes  = smem_a_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_b_elems  = ctx::tileK * ctx::xtileN;
static constexpr uint32_t smem_b_bytes  = smem_b_elems * sizeof(ctx::input_t);
// meta immediately follows A; B follows meta, bank-row aligned
static constexpr uint32_t smem_meta_off   = smem_a_bytes;
static constexpr uint32_t smem_bank_bytes = NUM_THREADS * sizeof(float);
static constexpr uint32_t smem_b_off      = ((smem_meta_off + ctx::wg_meta_total_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
static constexpr uint32_t smem_total      = smem_b_off + smem_b_bytes;

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA    = 0;
constexpr uint32_t kDescB    = 1;
constexpr uint32_t kDescMeta = 2;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Tile origin for this block
  uint32_t tile_row = blockIdx.y * ctx::xtileM;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  auto smem_base = reinterpret_cast<uint8_t*>(__local_mem());
  auto A_smem    = reinterpret_cast<ctx::input_t*>(smem_base);
  auto meta_smem = reinterpret_cast<uint32_t*>(smem_base + smem_meta_off);
  auto B_smem    = reinterpret_cast<ctx::input_t*>(smem_base + smem_b_off);

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

    // DXA: load compressed A, dense B, and metadata tiles
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA,    bar.id(), A_smem,    k / 2, tile_row);
      vx_dxa_issue_2d_wg(kDescB,    bar.id(), B_smem,    tile_col, k);
      vx_dxa_issue_2d_wg(kDescMeta, bar.id(), meta_smem, k_tile * meta_words_per_tile, blockIdx.y);
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Build smem descriptors:
    //   desc_a: compressed A with ldm = (tileK/2) * sizeof(input_t)
    //   desc_b: dense B with ldm = tileN * sizeof(input_t)
    auto desc_a = vt::vx_make_smem_desc(A_smem, (ctx::tileK / 2) * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }

  // Store C tile to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
