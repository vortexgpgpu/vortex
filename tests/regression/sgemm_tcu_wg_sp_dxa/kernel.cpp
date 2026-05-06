#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
// WGMMA accumulator; is_sparse=true (A is 2:4 compressed in smem)
using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true, WGMMA_NRC>;

// Per-warp smem layout: [A_compressed][metadata] (bank-aligned section)
// Then shared B after all warp sections
static constexpr uint32_t smem_a_elems     = ctx::xtileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes     = smem_a_elems * sizeof(ctx::input_t);
//static constexpr uint32_t smem_b_elems     = ctx::tileK * ctx::xtileN;
//static constexpr uint32_t smem_b_bytes     = smem_b_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_bank_bytes  = NUM_THREADS * sizeof(float);
static constexpr uint32_t per_warp_section = ((smem_a_bytes + ctx::wg_meta_total_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA    = 0;
constexpr uint32_t kDescB    = 1;
constexpr uint32_t kDescMeta = 2;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / NUM_THREADS;
  uint32_t num_warps = num_threads / NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  auto smem_base = reinterpret_cast<uint8_t*>(__local_mem());
  uint32_t smem_b_off = ((num_warps * per_warp_section + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
  auto B_smem = reinterpret_cast<ctx::input_t*>(smem_base + smem_b_off);

  // Initialize accumulator to zero
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  uint32_t meta_words_per_tile = ctx::wg_meta_total_bytes / 4;

  // Loop over K tiles
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;

    // DXA: load per-warp compressed A and metadata, plus shared B
    if (is_dxa_warp) {
      for (uint32_t w = 0; w < num_warps; ++w) {
        auto A_smem_w = reinterpret_cast<ctx::input_t*>(smem_base + w * per_warp_section);
        auto meta_smem_w = reinterpret_cast<uint32_t*>(smem_base + w * per_warp_section + smem_a_bytes);
        vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem_w, k / 2, tile_row + w * ctx::xtileM);
        uint32_t tile_row_idx_w = blockIdx.y * num_warps + w;
        vx_dxa_issue_2d_wg(kDescMeta, bar.id(), meta_smem_w, k_tile * meta_words_per_tile, tile_row_idx_w);
      }
      vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Each warp's A section in smem
    auto A_warp = reinterpret_cast<ctx::input_t*>(smem_base + warp_rank * per_warp_section);
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A + sparse metadata from registers, B from smem (NRC <= 16 only)
    auto meta_sp = smem_base + warp_rank * per_warp_section + smem_a_bytes;
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK / 2);
    ctx::load_sp_metadata(fragA, meta_sp);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both A and B from smem descriptors
    auto desc_a = vt::vx_make_smem_desc(A_warp, (ctx::tileK / 2) * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }

  // Store C tile to global memory
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
