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

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

#ifdef PROFILE_ENABLE
  uint32_t cycles = 0;
#endif

  for (int i = 0; i < K; i += ctx::tileK) {
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;

#ifdef PROFILE_ENABLE
    __rdcycle_time t0 = vx_rdcycle_sync_begin();
#endif
    ctx::load_matrix_sync(fragA, pTileA, K);
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
#ifdef PROFILE_ENABLE
    __rdcycle_time t1 = vx_rdcycle_sync_end();
    cycles += vx_rdcycle_sync_diff(t0, t1);
#endif
  }

  // Store the computed C tile
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);

#ifdef PROFILE_ENABLE
  // Write per-block cycle count
  auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  pCycles[block_id] = cycles;
#endif
}
