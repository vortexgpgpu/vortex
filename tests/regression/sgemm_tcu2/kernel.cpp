#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

enum : uint32_t {
  MODE_RS = 0,
  MODE_SS = 1,
};

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t mode = arg->mode;

  vt::smem_matrix_desc a_desc{arg->A_desc};
  vt::smem_matrix_desc b_desc{arg->B_desc};

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

  (void)M;
  uint32_t tile_row_idx = tile_row / ctx::tileM;
  uint32_t tile_col_idx = tile_col / ctx::tileN;

  for (uint32_t i = 0, k_idx = 0; i < K; i += ctx::tileK, ++k_idx) {
    auto pTileA = pA + tile_row * K + i;

    // Load A tile (register source in RS, descriptor source in SS)
    if (mode == MODE_SS) {
      ctx::load_matrix_sync_smem(fragA, a_desc, tile_row_idx, k_idx, K);
    } else
    {
      ctx::load_matrix_sync(fragA, pTileA, K);
    }

    // Load B tile (descriptor source in RS/SS)
    if constexpr (vt::ITYPE::bits < 8) {
      // For sub-byte matrix B must be in col-major format
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync_smem<vt::col_major>(fragB, b_desc, tile_col_idx, k_idx, K);
      (void)pTileB;
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync_smem(fragB, b_desc, tile_col_idx, k_idx, N);
      (void)pTileB;
    }

    // Matrix multiply-accumulate
    if (mode == MODE_SS) {
      ctx::wgmma_sync<vt::wgmma_ss>(fragC, fragA, fragB, fragC);
    } else {
      ctx::wgmma_sync<vt::wgmma_rs>(fragC, fragA, fragB, fragC);
    }
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
