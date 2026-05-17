#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;

using ctx = vt::wgmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t lane = tid % NUM_THREADS;
  uint32_t num_threads = blockDim.x;  // warps * NUM_THREADS
  uint32_t warp_rank = tid / NUM_THREADS;
  uint32_t num_warps = num_threads / NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M × tileK] then B [tileK × per_warp_N]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t a_row_base = warp_rank * ctx::xtileM;
    {
      auto A_smem_w = reinterpret_cast<uint32_t *>(A_smem);
      auto pA_w     = reinterpret_cast<const uint32_t *>(pA);
      uint32_t a_xtileK = ctx::tileK / ctx::i_ratio;
      for (uint32_t i = lane; i < ctx::xtileM * a_xtileK; i += NUM_THREADS) {
        uint32_t r = i / a_xtileK;
        uint32_t c = i % a_xtileK;
        A_smem_w[(a_row_base + r) * a_xtileK + c] =
          pA_w[(tile_row + a_row_base + r) * (K / ctx::i_ratio) + (k / ctx::i_ratio + c)];
      }
    }

    // Cooperatively load B [tileK × xtileN] into smem
    {
      auto B_smem_w = reinterpret_cast<uint32_t *>(B_smem);
      auto pB_w     = reinterpret_cast<const uint32_t *>(pB);
      uint32_t b_xtileN = ctx::xtileN / ctx::i_ratio;
      uint32_t b_size_w = ctx::tileK * b_xtileN;
      for (uint32_t i = tid; i < b_size_w; i += num_threads) {
        uint32_t r = i / b_xtileN;
        uint32_t c = i % b_xtileN;
        B_smem_w[r * b_xtileN + c] =
          pB_w[(k + r) * (N / ctx::i_ratio) + (tile_col / ctx::i_ratio + c)];
      }
    }

    __syncthreads();

    // Each warp's A slice starts at warp_rank * per_warp_M * tileK
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    __syncthreads();
  }

  // Store C tile using wgmma_context's n-major store
  auto out = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(out, fragC, N);
}
