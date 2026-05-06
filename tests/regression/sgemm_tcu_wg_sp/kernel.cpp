#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;

using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true, WGMMA_NRC>;

// Per-warp smem layout: [A_compressed][metadata] (bank-aligned section)
// Then shared B after all warp sections
static constexpr uint32_t smem_a_elems     = ctx::xtileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes     = smem_a_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_b_elems     = ctx::tileK * ctx::xtileN;
[[maybe_unused]] static constexpr uint32_t smem_b_bytes = smem_b_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_bank_bytes  = NUM_THREADS * sizeof(float);
static constexpr uint32_t per_warp_section = ((smem_a_bytes + ctx::wg_meta_total_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA  = reinterpret_cast<ctx::input_t *>(arg->A_addr);   // compressed A
  auto pB  = reinterpret_cast<ctx::input_t *>(arg->B_addr);   // dense B
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pMetaSp = reinterpret_cast<const uint32_t*>(arg->meta_sp_addr);

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

  // Strides in global memory
  uint32_t a_sp_stride = K / 2;         // compressed A: K/2 elements per row
  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t meta_words_per_tile = ctx::wg_meta_total_bytes / 4;

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;

    // Cooperative load: compressed A (block-major) and metadata (flat) for all warps
    for (uint32_t w = 0; w < num_warps; ++w) {
      auto A_smem_w = reinterpret_cast<ctx::input_t*>(smem_base + w * per_warp_section);
      for (uint32_t i = tid; i < smem_a_elems; i += num_threads) {
        uint32_t r = i / (ctx::tileK / 2);
        uint32_t c = i % (ctx::tileK / 2);
        A_smem_w[ctx::a_sp_blockmajor_idx(r, c)] =
            pA[(tile_row + w * ctx::xtileM + r) * a_sp_stride + (k / 2 + c)];
      }

      uint32_t tile_row_idx_w = blockIdx.y * num_warps + w;
      auto pMeta_tile_w = pMetaSp + (tile_row_idx_w * num_k_tiles + k_tile) * meta_words_per_tile;
      auto meta_smem_w = reinterpret_cast<uint32_t*>(smem_base + w * per_warp_section + smem_a_bytes);
      for (uint32_t i = tid; i < meta_words_per_tile; i += num_threads) {
        meta_smem_w[i] = pMeta_tile_w[i];
      }
    }

    // Cooperative load: dense B [k..k+tileK, tile_col..tile_col+xtileN) — block-major
    for (uint32_t i = tid; i < smem_b_elems; i += num_threads) {
      uint32_t r = i / ctx::xtileN;
      uint32_t c = i % ctx::xtileN;
      B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
    }

    __syncthreads();

    // Each warp's A section in smem
    auto A_warp = reinterpret_cast<ctx::input_t*>(smem_base + warp_rank * per_warp_section);
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0); // stride field unused under block-major

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A + sparse metadata from registers, B from smem (NRC <= 16 only).
    // ldm=0 selects block-major (matches the cooperative-load via
    // a_sp_blockmajor_idx above).
    auto meta_sp = smem_base + warp_rank * per_warp_section + smem_a_bytes;
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, 0);
    ctx::load_sp_metadata(fragA, meta_sp);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both A and B from smem descriptors
    auto desc_a = vt::vx_make_smem_desc(A_warp, 0); // stride field unused under block-major
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    __syncthreads();
  }

  // Store C tile to global memory
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
