#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_dtensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// dtcu_compare: one kernel binary, two modes selected by arg->mode.
//   mode 0 -> in-core TCU: a warp cooperatively computes one output tile D = C + A*B.
//             Launched as a (N/tileN, M/tileM) grid of NUM_THREADS-wide blocks.
//   mode 1 -> DTCU: a single thread fires the whole tiled GEMM descriptor and spins
//             on the done bit. Launched as a 1x1 grid / 1x1 block.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (arg->mode == 0) {
    auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
    auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
    auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
    auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

    uint32_t N = arg->N;
    uint32_t K = arg->K;

    ctx::fragment_a   fragA;
    ctx::fragment_b   fragB;
    ctx::fragment_acc fragD;

    uint32_t tile_row = blockIdx.y * ctx::tileM;
    uint32_t tile_col = blockIdx.x * ctx::tileN;

    // Seed the accumulator from C (D = C + A*B), matching the DTCU flags=0 path.
    auto pTileC = pC + tile_row * N + tile_col;
    ctx::load_matrix_sync(fragD, pTileC, N);

    for (uint32_t i = 0; i < K; i += ctx::tileK) {
      auto pTileA = pA + tile_row * K + i;
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync(fragA, pTileA, K);          // A row-major
      ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K); // B col-major
      ctx::mma_sync(fragD, fragA, fragB, fragD);
    }

    auto pTileD = pD + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileD, fragD, N);
  } else {
    // Single leader thread issues the descriptor; with a 1x1/1x1 launch this runs once.
    if (vx_thread_id() == 0) {
      dtensor_start(arg->desc_addr);
      while (0 == dtensor_poll()) {
        // busy-wait until the DTCU signals completion
      }
    }
  }
}
