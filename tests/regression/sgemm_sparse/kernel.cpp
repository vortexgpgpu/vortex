#include "common.h"
#include <vx_spawn.h>
#include <vx_sparse.h>

namespace vt = vortex::sparse;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static inline size_t align_up_size(size_t value, size_t alignment) {
  if (!alignment)
    return value;
  return (value + alignment - 1) & ~(alignment - 1);
}

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

  // Sparse A layout: data (values) first, then metadata (padded to 4-byte alignment)
  constexpr size_t meta_entry_bytes = sizeof(uint32_t);
  uint32_t sparsity_degree = arg->sparsity_degree;
  uint32_t values_per_row = K * sparsity_degree / 4;
  uint32_t kblocks = K / 4;
  size_t values_size = static_cast<size_t>(M) * values_per_row * sizeof(ctx::input_t);
  size_t meta_offset = align_up_size(values_size, meta_entry_bytes);
  const uint8_t *base_ptr = reinterpret_cast<const uint8_t *>(pA_values);
  const uint32_t *meta_base = reinterpret_cast<const uint32_t *>(base_ptr + meta_offset);

  // Initialize accumulator
  ctx::fill_fragment(fragC, 0);

  auto pTileA_base = pA_values + tile_row * values_per_row;
  const uint32_t *pMeta_base = meta_base + tile_row * kblocks;

  for (uint32_t k_tile = 0; k_tile < K; k_tile += ctx::tileK) {
    // Load dense B tile
    if constexpr (vt::ITYPE::bits < 8) {
      ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + k_tile, K);
    } else {
      ctx::load_matrix_sync(fragB, pB + k_tile * N + tile_col, N);
    }

    // Load sparse A tile
    uint32_t k_block = k_tile / 4;
    ctx::load_matrix_sync(fragA,
                           pTileA_base + k_block * sparsity_degree,
                           values_per_row,
                           reinterpret_cast<const void *>(pMeta_base + k_block),
                           kblocks, 0, sparsity_degree, 0);

    ctx::mma_sync(fragC, fragA, fragB, fragC, sparsity_degree);
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
