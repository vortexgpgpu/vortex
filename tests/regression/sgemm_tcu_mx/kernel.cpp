#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMetaMxBase = reinterpret_cast<const uint32_t*>(arg->meta_mx_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;
  (void)M;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

  uint32_t num_k_steps = K / ctx::tileK;
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  auto pMetaMxTile = pMetaMxBase + block_id * num_k_steps * ctx::mx_meta_words;

  for (int i = 0; i < K; i += ctx::tileK) {
    uint32_t step_idx = static_cast<uint32_t>(i) / ctx::tileK;
    auto pTileA = pA + tile_row * K + i;
    auto pMetaMx = pMetaMxTile + step_idx * ctx::mx_meta_words;

    // Load A tile
    ctx::load_matrix_sync(fragA, pTileA, K);
    ctx::load_mx_metadata(fragA, pMetaMx);

    // Load B tile
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync(fragB, pTileB, N);
    }
    ctx::load_mx_metadata(fragB, pMetaMx);

    // Matrix multiply-accumulate: c += a * b
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
