// Each CTA does TWO DXA issues:
//   kDescA (row-major): A_smem ← A[cta_row*tileM .. , 0..tileK-1]
//   kDescB (K-major):   B_smem ← B[0..tileK-1, cta_col*tileN ..]
// Then byte-copies both SMEM blocks to dst for host verification.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid     = threadIdx.x;
  const uint32_t cta_id  = blockIdx.x;   // 1D grid — one CTA per (M-tile, N-tile) pair
  const uint32_t grid_n  = arg->N / arg->tileN;
  const uint32_t cta_row = cta_id / grid_n;
  const uint32_t cta_col = cta_id % grid_n;

  auto A_smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto B_smem = A_smem + arg->a_bytes;

  vortex::barrier bar(0);
  if (get_sub_group_id() == 0) {
    bar.expect_tx(2);
    // A: 2D row-major fetch (coord0 = K-axis start, coord1 = M-axis start).
    vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem,
                       /*coord0=*/0, /*coord1=*/cta_row * arg->tileM);
    // B: 2D K-major scatter (coord0 = N-axis start, coord1 = K-axis start).
    vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem,
                       /*coord0=*/cta_col * arg->tileN, /*coord1=*/0);
  }
  bar.arrive_and_wait();

  // Copy both SMEM regions to dst[cta_id * cta_bytes ..]
  auto pDst = reinterpret_cast<uint8_t*>(arg->dst_addr) + cta_id * arg->cta_bytes;
  const uint32_t total = arg->cta_bytes;
  for (uint32_t i = tid; i < total; i += blockDim.x) {
    pDst[i] = A_smem[i];  // A_smem is followed by B_smem in memory
  }
  __syncthreads();
}
