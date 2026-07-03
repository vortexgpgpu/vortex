#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE,
                              false, WGMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  static_assert(sizeof(ctx::input_t) == 1, "MX WGMMA inputs use byte storage");
  auto pA = reinterpret_cast<const uint8_t *>(arg->A_addr);
  auto pB = reinterpret_cast<const uint8_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pMxA = reinterpret_cast<const uint32_t *>(arg->MX_A_addr);
  auto pMxB = reinterpret_cast<const uint32_t *>(arg->MX_B_addr);
#ifdef VX_CFG_TCU_MX_TLS
  auto pATensorScale = reinterpret_cast<const float *>(arg->A_tensor_scale_addr);
  auto pBTensorScale = reinterpret_cast<const float *>(arg->B_tensor_scale_addr);
#endif

  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;
  uint32_t cta_m = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_m;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;
  uint32_t num_k_tiles = K / ctx::tileK;

  auto A_smem = reinterpret_cast<uint8_t *>(__local_mem());
  auto B_smem = A_smem + cta_m * ctx::tileK;
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t k_tile = k / ctx::tileK;
    uint32_t a_bytes = cta_m * ctx::tileK;
    for (uint32_t idx = tid; idx < a_bytes; idx += num_threads) {
      uint32_t row = idx / ctx::tileK;
      uint32_t col = idx % ctx::tileK;
      A_smem[idx] = pA[(tile_row + row) * K + k + col];
    }
    uint32_t b_bytes = ctx::xtileN * ctx::tileK;
    for (uint32_t idx = tid; idx < b_bytes; idx += num_threads) {
      uint32_t col = idx / ctx::tileK;
      uint32_t k_byte = idx % ctx::tileK;
      B_smem[idx] = pB[(tile_col + col) * K + k + k_byte];
    }
    __syncthreads();

    uint32_t tile_m = blockIdx.y * num_warps + warp_rank;
    auto pTileMxA = pMxA + (tile_m * num_k_tiles + k_tile) * VX_CFG_NUM_THREADS;
    auto pTileMxB = pMxB + (blockIdx.x * num_k_tiles + k_tile) * VX_CFG_NUM_THREADS;
    ctx::load_mx_metadata(fragA, pTileMxA);
    ctx::load_mx_metadata(fragB, pTileMxB);

    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::tileK);
#if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
#else
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
#endif
    __syncthreads();
  }

#ifdef VX_CFG_TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    float tensor_scale = (*pATensorScale) * (*pBTensorScale);
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
      fragC.data[r] *= tensor_scale;
  }
#endif
  auto out = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(out, fragC, N);
}
