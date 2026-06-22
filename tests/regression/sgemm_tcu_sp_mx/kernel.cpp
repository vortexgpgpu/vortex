#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, true>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMxA = reinterpret_cast<const uint32_t *>(arg->MX_A_addr);
  auto pMxB = reinterpret_cast<const uint32_t *>(arg->MX_B_addr);
  auto pMetaSp = reinterpret_cast<const uint32_t *>(arg->meta_sp_addr);
#ifdef TCU_MX_TLS
  auto pATensorScale = reinterpret_cast<const float *>(arg->A_tensor_scale_addr);
  auto pBTensorScale = reinterpret_cast<const float *>(arg->B_tensor_scale_addr);
#endif

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;
  uint32_t num_k_tiles = K / ctx::tileK;

  ctx::fill_fragment(fragC, 0);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    uint32_t k_tile = i / ctx::tileK;
    auto pTileA = pA + tile_row * (K / 2) + i / 2;
    auto pTileB = pB + tile_col * K + i;
    auto pTileMxA = pMxA + (blockIdx.y * num_k_tiles + k_tile) * VX_CFG_NUM_THREADS;
    auto pTileMxB = pMxB + (blockIdx.x * num_k_tiles + k_tile) * VX_CFG_NUM_THREADS;
    auto pTileMetaSp = pMetaSp + (blockIdx.y * num_k_tiles + k_tile) * ctx::meta_stride;

    ctx::load_sp_metadata(fragA, pTileMetaSp);
    ctx::load_mx_metadata(fragA, pTileMxA);
    ctx::load_mx_metadata(fragB, pTileMxB);
    ctx::load_matrix_sync<vt::row_major>(fragA, pTileA, K / 2);
    ctx::load_matrix_sync<vt::col_major>(fragB, pTileB, K);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
  }

#ifdef TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    float tensor_scale = (*pATensorScale) * (*pBTensorScale);
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) {
      fragC.data[r] *= tensor_scale;
    }
  }
#endif

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
