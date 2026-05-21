#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

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

  // Shared memory layout: A [cta_M × tileK], B [tileK × xtileN], C [num_warps × xtileM × xtileN]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;
  auto C_smem = reinterpret_cast<ctx::output_t *>(B_smem + ctx::tileK * ctx::xtileN);

  // Per-warp C accumulator region in smem; initialized to zero before k-loop
  auto C_warp = C_smem + warp_rank * ctx::xtileM * ctx::xtileN;
  for (uint32_t i = lane; i < ctx::xtileM * ctx::xtileN; i += NUM_THREADS) {
    C_warp[i] = ctx::output_t(0);
  }

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

    // Each warp's A slice starts at warp_rank * xtileM * tileK
    auto A_warp  = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b  = vt::vx_make_smem_desc(B_smem,  ctx::xtileN * sizeof(ctx::input_t));
    auto desc_cd = vt::vx_make_smem_desc(C_warp, ctx::xtileN * sizeof(ctx::output_t));

    // SS: A and B from smem, C/D accumulator in smem (cd_from_lmem)
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(desc_a, desc_b, desc_cd);

    __syncthreads();
  }

  // Copy C accumulator from smem to global memory
  auto out = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  for (uint32_t i = lane; i < ctx::xtileM * ctx::xtileN; i += NUM_THREADS) {
    uint32_t r = i / ctx::xtileN;
    uint32_t c = i % ctx::xtileN;
    out[r * N + c] = C_warp[i];
  }
}
