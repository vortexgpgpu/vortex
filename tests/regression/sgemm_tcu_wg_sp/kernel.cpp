#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
// NR=32 for WGMMA accumulator; is_sparse=true (A is 2:4 compressed in smem)
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true, 32, 8>;

// smem layout: [A_compressed][meta][B_dense]
static constexpr uint32_t smem_a_elems  = ctx::tileM * (ctx::tileK / 2);
static constexpr uint32_t smem_a_bytes  = smem_a_elems * sizeof(ctx::input_t);
static constexpr uint32_t smem_b_elems  = ctx::tileK * ctx::tileN;
static constexpr uint32_t smem_b_bytes  = smem_b_elems * sizeof(ctx::input_t);
// meta immediately follows A; B follows meta, bank-row aligned
static constexpr uint32_t smem_meta_off   = smem_a_bytes;
static constexpr uint32_t smem_bank_bytes = NUM_THREADS * sizeof(float);
static constexpr uint32_t smem_b_off      = ((smem_meta_off + ctx::wg_meta_total_bytes + smem_bank_bytes - 1) / smem_bank_bytes) * smem_bank_bytes;
static constexpr uint32_t smem_total      = smem_b_off + smem_b_bytes;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA  = reinterpret_cast<ctx::input_t *>(arg->A_addr);   // compressed A
  auto pB  = reinterpret_cast<ctx::input_t *>(arg->B_addr);   // dense B
  auto pC  = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pMetaSp = reinterpret_cast<const uint32_t*>(arg->meta_sp_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Tile origin for this block
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  auto smem_base = reinterpret_cast<uint8_t*>(__local_mem());
  auto A_smem = reinterpret_cast<ctx::input_t*>(smem_base);
  auto B_smem = reinterpret_cast<ctx::input_t*>(smem_base + smem_b_off);

  // Initialize accumulator to zero
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  uint32_t tid = threadIdx.x;

  // Strides in global memory
  uint32_t a_sp_stride = K / 2;         // compressed A: K/2 elements per row
  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t meta_words_per_tile = ctx::wg_meta_total_bytes / 4;

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;

    // Cooperative load: compressed A [tile_row..tile_row+tileM, k/2..k/2+tileK/2)
    for (uint32_t i = tid; i < smem_a_elems; i += CTA_SIZE) {
      uint32_t r = i / (ctx::tileK / 2);
      uint32_t c = i % (ctx::tileK / 2);
      A_smem[r * (ctx::tileK / 2) + c] = pA[(tile_row + r) * a_sp_stride + (k / 2 + c)];
    }

    // Cooperative load: metadata for this (tile_row, k_tile) block
    uint32_t tile_row_idx = blockIdx.y;
    auto pMeta_tile = pMetaSp + (tile_row_idx * num_k_tiles + k_tile) * meta_words_per_tile;
    auto meta_smem = reinterpret_cast<uint32_t*>(smem_base + smem_meta_off);
    for (uint32_t i = tid; i < meta_words_per_tile; i += CTA_SIZE) {
      meta_smem[i] = pMeta_tile[i];
    }

    // Cooperative load: dense B [k..k+tileK, tile_col..tile_col+tileN)
    for (uint32_t i = tid; i < smem_b_elems; i += CTA_SIZE) {
      uint32_t r = i / ctx::tileN;
      uint32_t c = i % ctx::tileN;
      B_smem[r * ctx::tileN + c] = pB[(k + r) * N + (tile_col + c)];
    }

    __syncthreads();

    // Build smem descriptors:
    //   desc_a: compressed A with ldm = (tileK/2) * sizeof(input_t)
    //           metadata is implicitly at A_smem + tileM * ldm_bytes
    //   desc_b: dense B with ldm = tileN * sizeof(input_t)
    auto desc_a = vt::vx_make_smem_desc(A_smem, (ctx::tileK / 2) * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::tileN * sizeof(ctx::input_t));

    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);

    __syncthreads();
  }

  // Store C tile to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
