#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, 8, 0, LDST_MATRIX>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  size_t N = arg->N;
  size_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  // calculate tile row & column based on block index
  size_t tile_row = blockIdx.y * ctx::tileM;
  size_t tile_col = blockIdx.x * ctx::tileN;

  // Initialize accumulator tile to zero
  ctx::fill_fragment(fragC, 0);

#ifdef PROFILE_ENABLE
  uint32_t cycles = 0;
#endif

#if KUNROLL > 1
  // KUNROLL tiles per iteration: each tile's load takes its K offset as a
  // constant, which a vx_ldm folds into its immediate.
  size_t a_lane_off = ctx::lane_offset<ctx::fragment_a>(K);
  size_t b_lane_off = ctx::lane_offset<ctx::fragment_b, vt::col_major>(K);
  auto k_tiles = [&](size_t i, auto count) {
    auto pTileA = pA + tile_row * K + i;
    auto pTileB = pB + tile_col * K + i;
#ifdef PROFILE_ENABLE
    __rdcycle_time t0 = vx_rdcycle_sync_begin();
#endif
    vt::unroll_for<decltype(count)::value>([&](auto u) {
      constexpr int32_t k_off = decltype(u)::value * ctx::tileK;
      ctx::load_matrix_sync<vt::row_major, k_off>(fragA, pTileA, K, a_lane_off);
      ctx::load_matrix_sync<vt::col_major, k_off>(fragB, pTileB, K, b_lane_off);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
    });
#ifdef PROFILE_ENABLE
    __rdcycle_time t1 = vx_rdcycle_sync_end();
    cycles += vx_rdcycle_sync_diff(t0, t1);
#endif
  };
  size_t i = 0;
  for (; i + KUNROLL * ctx::tileK <= K; i += KUNROLL * ctx::tileK) {
    k_tiles(i, std::integral_constant<uint32_t, KUNROLL>{});
  }
  for (; i < K; i += ctx::tileK) {
    k_tiles(i, std::integral_constant<uint32_t, 1>{});
  }
#else
  for (size_t i = 0; i < K; i += ctx::tileK) {
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
#endif

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
