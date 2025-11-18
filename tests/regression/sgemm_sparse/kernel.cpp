#include "common.h"
#include <vx_spawn.h>
#include <vx_sparse.h>

namespace vt = vortex::sparse;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA_values = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // Tile indices handled by this block
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  if (tile_row >= M || tile_col >= N)
    return;

  // Sparse A layout: values first (M * K / 2 entries), then metadata (M * K / 4 bytes)
  size_t values_per_row = K / 2;
  size_t meta_per_row = K / 4;
  size_t total_values = static_cast<size_t>(M) * values_per_row;
  const uint8_t *meta_base = reinterpret_cast<const uint8_t *>(pA_values + total_values);

  // Initialize accumulator
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k_tile = 0; k_tile < K; k_tile += ctx::tileK) {
    // Keep fragB resident while we iterate over sparse A tiles that consume it.
    /*if constexpr (vt::ITYPE::bits < 8) {
      auto pTileB = pB + tile_col * K + k_tile;
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + k_tile * N + tile_col;
      ctx::load_matrix_sync(fragB, pTileB, N);
    }*/

    // Base pointers for sparse A data/metadata corresponding to this tile
    size_t row_offset_vals = static_cast<size_t>(tile_row) * values_per_row;
    size_t row_offset_meta = static_cast<size_t>(tile_row) * meta_per_row;
    size_t col_offset_vals = k_tile / 2;
    size_t col_offset_meta = k_tile / 4;

    auto pTileA = pA_values + row_offset_vals + col_offset_vals;
    const uint8_t *pTileMeta = meta_base + row_offset_meta + col_offset_meta;

    ctx::load_matrix_sync(fragA, pTileA, K, pTileMeta);

    // Matrix multiply-accumulate while fragB stays in registers
   // ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  //auto pTileC = pC + tile_row * N + tile_col;
  //ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
