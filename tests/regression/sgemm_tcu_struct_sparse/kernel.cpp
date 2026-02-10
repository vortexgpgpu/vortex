#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fill_fragment(fragC, 0);

  uint32_t stride_A = K / 2;
  for (int i = 0; i < (int)K; i += (int)ctx::tileK) {
    auto pTileA = pA + tile_row * stride_A + (i / 2);
    ctx::load_matrix_sync<vt::row_major, true>(fragA, pTileA, stride_A);

    if constexpr (vt::ITYPE::bits < 8) {
      auto pTileB = pB + tile_col * K + i;
      ctx::load_matrix_sync<vt::col_major, true>(fragB, pTileB, K);
    } else {
      auto pTileB = pB + i * N + tile_col;
      ctx::load_matrix_sync<vt::row_major, true>(fragB, pTileB, N);
    }

    ctx::mma_sync<true>(fragC, fragA, fragB, fragC);
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
