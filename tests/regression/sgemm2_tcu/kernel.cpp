#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE, false, 32>;

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  // Calculate output tile origin for this block
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Shared memory: A tile [tileM x tileK] followed by B tile [tileK x tileN],
  // both stored row-major.
  auto smem  = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + ctx::tileM * ctx::tileK;

  // Initialize accumulator tile to zero
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  uint32_t tid = threadIdx.x;

  // Loop over K tiles
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperative load: A tile [tile_row .. tile_row+tileM, k .. k+tileK] into A_smem
    uint32_t a_size = ctx::tileM * ctx::tileK;
    for (uint32_t i = tid; i < a_size; i += CTA_SIZE) {
      uint32_t r = i / ctx::tileK;
      uint32_t c = i % ctx::tileK;
      A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
    }

    // Cooperative load: B tile [k .. k+tileK, tile_col .. tile_col+tileN] into B_smem
    uint32_t b_size = ctx::tileK * ctx::tileN;
    for (uint32_t i = tid; i < b_size; i += CTA_SIZE) {
      uint32_t r = i / ctx::tileN;
      uint32_t c = i % ctx::tileN;
      B_smem[r * ctx::tileN + c] = pB[(k + r) * N + (tile_col + c)];
    }

    // Ensure all smem writes are visible before WGMMA reads
    __syncthreads();

    // Build smem descriptors: leading_bytes = row stride in bytes
    auto desc_a = vt::vx_make_smem_desc(A_smem, ctx::tileK * sizeof(ctx::input_t));
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::tileN * sizeof(ctx::input_t));

    // Execute WGMMA: C += A * B
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);

    // Sync after WGMMA
    __syncthreads();
  }

  // Store the computed C tile to global memory
  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
