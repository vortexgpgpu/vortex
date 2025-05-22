#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto A = reinterpret_cast<I_TYPE *>(arg->A_addr);
  auto B = reinterpret_cast<I_TYPE *>(arg->B_addr);
  auto C = reinterpret_cast<O_TYPE *>(arg->C_addr);

  tensor::fragment<tensor::matrix_a, I_TYPE, tensor::row_major> fragA;
  tensor::fragment<tensor::matrix_b, I_TYPE, tensor::col_major> fragB;
  tensor::fragment<tensor::matrix_c, O_TYPE, tensor::row_major> fragC;

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * arg->tileM;
  uint32_t tile_col = blockIdx.x * arg->tileN;

  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t tileK = arg->tileK;

  // Initialize accumulator tile to zero
  tensor::fill_fragment(fragC, 0);

  for (int i = 0; i < K; i += tileK) {
    // Load A tile
    auto tileA = A + (tile_row * K + i);
    tensor::load_matrix_sync<tensor::row_major>(fragA, tileA, K);

    // Load B tile
    auto tileB = B + (i * K + tile_col);
    tensor::load_matrix_sync<tensor::row_major>(fragB, tileB, K);

    // Matrix multiply-accumulate: c += a * b
    tensor::mma_sync(fragC, fragA, fragB, fragC);
  }

  // Store the computed C tile
  auto tileC = C + (tile_row * N + tile_col);
  tensor::store_matrix_sync<tensor::row_major>(tileC, fragC, N);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
