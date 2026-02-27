#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMetaBase = reinterpret_cast<const float *>(arg->meta_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fill_fragment(fragC, 0);

  // Per-K-tile metadata reload
  constexpr uint32_t rtl_i_ratio = 32 / vt::ITYPE::bits;
  constexpr uint32_t meta_cols = (NUM_THREADS * 2 * rtl_i_ratio) / 32;
  using kcfg = vt::wmma_config_t<NUM_THREADS>;
  constexpr uint32_t PD = kcfg::m_steps * (kcfg::k_steps / 2);
  constexpr uint32_t per_k_tile_words = PD * meta_cols;
  uint32_t num_k_tiles = K / ctx::tileK;
  uint32_t tile_row_idx = blockIdx.y;

  uint32_t stride_A = K / 2;

  auto pMeta = pMetaBase + tile_row_idx * num_k_tiles * per_k_tile_words;
  auto pTileA = pA + tile_row * stride_A;
  constexpr uint32_t a_k_stride = ctx::tileK / 2;

  uint32_t cyc_start = csr_read(0xB00);
  if constexpr (vt::ITYPE::bits < 8) {
    auto pTileB = pB + tile_col * K;
    for (int i = 0; i < (int)K; i += (int)ctx::tileK) {
      ctx::load_metadata_sync(pMeta);
      ctx::load_matrix_sync<vt::row_major, true>(fragA, pTileA, stride_A);
      ctx::load_matrix_sync<vt::col_major, true>(fragB, pTileB, K);
      ctx::mma_sync<true>(fragC, fragA, fragB, fragC);
      pMeta += per_k_tile_words;
      pTileA += a_k_stride;
      pTileB += ctx::tileK;
    }
  } else {
    auto pTileB = pB + tile_col;
    uint32_t b_k_stride = ctx::tileK * N;
    for (int i = 0; i < (int)K; i += (int)ctx::tileK) {
      ctx::load_metadata_sync(pMeta);
      ctx::load_matrix_sync<vt::row_major, true>(fragA, pTileA, stride_A);
      ctx::load_matrix_sync<vt::row_major, true>(fragB, pTileB, N);
      ctx::mma_sync<true>(fragC, fragA, fragB, fragC);
      pMeta += per_k_tile_words;
      pTileA += a_k_stride;
      pTileB += b_k_stride;
    }
  }
  uint32_t cyc_end = csr_read(0xB00);
  auto pCycles = reinterpret_cast<uint32_t*>(arg->tcu_cycles_addr);
  uint32_t block_id = blockIdx.y * arg->grid_dim[0] + blockIdx.x;
  pCycles[block_id] = cyc_end - cyc_start;

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
