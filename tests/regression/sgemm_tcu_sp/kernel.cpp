#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, true>; // is_sparse=true

__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMetaSpBase = reinterpret_cast<const float *>(arg->meta_sp_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fill_fragment(fragC, 0);
  uint32_t cycles = 0;

  // Per-K-tile metadata reload
  constexpr uint32_t rtl_i_ratio = 32 / vt::ITYPE::bits;
  constexpr uint32_t meta_cols = (NUM_THREADS * 2 * rtl_i_ratio + 31) / 32;
  using kcfg = vt::wmma_config_t<NUM_THREADS>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t meta_cols_per_load = (NUM_THREADS >= PD) ? (NUM_THREADS / PD) : 1;
  constexpr uint32_t num_meta_loads = (PD * meta_cols + NUM_THREADS - 1) / NUM_THREADS;
  constexpr uint32_t per_k_tile_words = num_meta_loads * NUM_THREADS;
  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  uint32_t stride_A = K / 2;

  auto pMetaSp = pMetaSpBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = ctx::tileK / 2;

  auto pTileB = pB + tile_col * K;
  for (int i = 0; i < (int)K; i += (int)ctx::tileK) {
    __rdcycle_time t0 = vx_rdcycle_sync_begin();
    ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, stride_A, nullptr, pMetaSp);
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    __rdcycle_time t1 = vx_rdcycle_sync_end();
    cycles += vx_rdcycle_sync_diff(t0, t1);
    pMetaSp += per_k_tile_words;
    pTileA += a_k_stride;
    pTileB += ctx::tileK;
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);

  // Write per-block cycle count
  auto pCycles = reinterpret_cast<uint32_t*>(arg->cycles_addr);
  uint32_t block_id = blockIdx.y * gridDim.x + blockIdx.x;
  pCycles[block_id] = cycles;
}
