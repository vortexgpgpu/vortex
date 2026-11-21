#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_tmem.h>
#include <vx_intrinsics.h>
#include <vx_barrier.h>

// MX GEMM with the accumulator in TMEM. One warp per TCU block; warp 0's
// thread 0 issues one tc.mma per K step with the step's A and B scales in
// TMEM (vx_tmem.h, MX scales), copied in once with tc.cp.

namespace vt = vortex::tensor;
namespace tm = vortex::tmem;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE,
                              false, WGMMA_NRC, vt::ldst_matrix_enabled>;
using wg_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

static constexpr uint32_t kTileN    = ctx::xtileN;
static constexpr uint32_t kStepB    = wg_cfg::fedpK * 4;   // bytes of K per tc.mma

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  static_assert(sizeof(ctx::input_t) == 1, "MX inputs use byte storage");
  auto pA  = reinterpret_cast<const uint8_t *>(arg->A_addr);
  auto pB  = reinterpret_cast<const uint8_t *>(arg->B_addr);
  auto pC  = reinterpret_cast<ctx::output_t *>(arg->C_addr);
  auto pSF = reinterpret_cast<const uint32_t *>(arg->SF_addr);

  const uint32_t N = arg->N;
  const uint32_t K = arg->K;
  const uint32_t tid = threadIdx.x;
  const uint32_t num_threads = blockDim.x;
  const uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;
  const uint32_t cta_m = num_warps * ctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_m;
  const uint32_t tile_col = blockIdx.x * kTileN;
  const uint32_t ksteps = K / kStepB;

  auto lmem = reinterpret_cast<uint8_t *>(__local_mem());
  auto tmem_slot = reinterpret_cast<volatile uint32_t *>(lmem);
  auto img_smem = reinterpret_cast<uint32_t *>(lmem + TMEM_SLOT_BYTES);
  auto A_smem = reinterpret_cast<uint8_t *>(img_smem + arg->img_words);
  auto B_smem = A_smem + cta_m * ctx::tileK;

  const uint32_t cta_id = get_local_group_id();
  const uint32_t load_bar = cta_id | (0u << 8);
  const uint32_t mma_bar  = cta_id | (1u << 8);

  // TMEM: D, then every K step's scales.
  const uint32_t tmem_cols = tm::alloc_cols(kTileN + ksteps * arg->sf_cols);
  if (warp_rank == 0) {
    uint32_t col = tm::alloc(tmem_cols);
    if (vx_thread_id() == 0) {
      *tmem_slot = col;
    }
  }
  // The CTA's scale image, staged for tc.cp.
  auto pImg = pSF + (blockIdx.y * gridDim.x + blockIdx.x) * arg->img_words;
  for (uint32_t i = tid; i < arg->img_words; i += num_threads) {
    img_smem[i] = pImg[i];
  }
  vx_barrier(load_bar, num_warps);
  const uint32_t taddr  = *tmem_slot;
  const uint32_t sf_col = taddr + kTileN;
  if (warp_rank == 0 && vx_thread_id() == 0) {
    tm::cp(sf_col, img_smem, ksteps * arg->sf_cols);
  }

  const uint32_t n_frag = kTileN / wg_cfg::tcN;
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    for (uint32_t idx = tid; idx < cta_m * ctx::tileK; idx += num_threads) {
      uint32_t row = idx / ctx::tileK;
      uint32_t col = idx % ctx::tileK;
      A_smem[idx] = pA[(tile_row + row) * K + k + col];
    }
    for (uint32_t idx = tid; idx < kTileN * ctx::tileK; idx += num_threads) {
      uint32_t col = idx / ctx::tileK;
      uint32_t kb  = idx % ctx::tileK;
      B_smem[idx] = pB[(tile_col + col) * K + k + kb];
    }
    vx_barrier(load_bar, num_warps);
    if (warp_rank == 0 && vx_thread_id() == 0) {
      const uint32_t a_off = tm::smem_offset(A_smem);
      const uint32_t b_off = tm::smem_offset(B_smem);
      for (uint32_t kt = 0; kt < ctx::tileK / kStepB; ++kt) {
        const uint32_t gk = k / kStepB + kt;
        const auto idesc = tm::make_idesc<vt::ITYPE>(n_frag, taddr, gk == 0, false,
                                                     sf_col + gk * arg->sf_cols);
        tm::mma(idesc, tm::smem_desc(a_off + kt * kStepB, ctx::tileK),
                tm::smem_desc(b_off + kt * kStepB, ctx::tileK));
      }
      // The tiles are refilled only once these MMAs have read them.
      tm::commit(mma_bar, num_warps + 1);
    }
    vx_barrier(mma_bar, num_warps + 1);
  }

  ctx::fragment_acc fragC;
  tm::load<tm::FRAGMENT>(fragC, tm::make_taddr(tm::warp_lane(), taddr));
#ifdef TCU_MX_TLS
  if constexpr (std::is_same<vt::ITYPE, vt::nvfp4>::value) {
    auto pAts = reinterpret_cast<const float *>(arg->A_tensor_scale_addr);
    auto pBts = reinterpret_cast<const float *>(arg->B_tensor_scale_addr);
    float tensor_scale = (*pAts) * (*pBts);
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
      fragC.data[r] *= tensor_scale;
  }
#endif
  ctx::store_matrix_sync(pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col, fragC, N);

  vx_barrier(load_bar, num_warps);
  if (warp_rank == 0) {
    tm::dealloc(taddr, tmem_cols);
  }
}
