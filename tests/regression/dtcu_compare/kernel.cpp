#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  if (arg->mode == 0) {
    // Copied from kernel.cpp from sgemm_tcu, but B is always col-major

    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

    uint32_t M = arg->M;
    uint32_t N = arg->N;
    uint32_t K = arg->K;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragD;

    // calculate tile row & column based on block index
    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::load_matrix_sync(fragD, pTileC, N);

    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      auto pTileA = pA + tile_row * K + i;
      auto pTileB = pB + tile_col * K + i;

      // A is row-major
      ctx::load_matrix_sync(fragA, pTileA, K);

      // B is always col-major in this test
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);

      ctx::mma_sync(fragD, fragA, fragB, fragD);
    }

    // Store the computed to D tile
    auto pTileD = pD + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileD, fragD, N);
  } else {
    // DTCU only works on core 0
  // Issue the start command from the first thread of the first warp, and wait until completion
    if (vx_warp_id() == 0 && vx_thread_id() == 0) {
      vt::dtensor_start(arg->desc_addr);
      while (0 == vt::dtensor_poll()) {
        // busy wait
      }
    }
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}