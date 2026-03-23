#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  auto size      = arg->size;
  auto tile_size = arg->tile_size;
  auto chunk_k   = arg->chunk_k;

  // Global and local thread coordinates.
  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto g_col = blockIdx.y * blockDim.y + threadIdx.y;
  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;

  // Shared memory layout:
  //   local_A: [tile_size × chunk_k], row-major (row = l_row, col = k_local)
  //   local_B: [chunk_k × tile_size], row-major (row = k_local, col = l_col)
  auto tile_elems_a = tile_size * chunk_k;
  auto tile_elems_b = chunk_k * tile_size;
  auto local_ptr = __local_mem((tile_elems_a + tile_elems_b) * sizeof(TYPE));
  auto local_A = reinterpret_cast<TYPE*>(local_ptr);
  auto local_B = local_A + tile_elems_a;

  TYPE sum(0);

  for (uint32_t k = 0; k < size; k += chunk_k) {
    // Cooperative load: each thread loads chunk_k/tile_size elements per tile.
    // Thread (l_row, l_col) loads columns l_col, l_col+tile_size, ... of its row.
    for (uint32_t kk = l_col; kk < chunk_k; kk += tile_size) {
      local_A[l_row * chunk_k + kk] = A_ptr[g_row * size + (k + kk)];
    }
    for (uint32_t kk = l_row; kk < chunk_k; kk += tile_size) {
      local_B[kk * tile_size + l_col] = B_ptr[(k + kk) * size + g_col];
    }

    __syncthreads();

    for (uint32_t j = 0; j < chunk_k; ++j) {
      sum += local_A[l_row * chunk_k + j] * local_B[j * tile_size + l_col];
    }

    __syncthreads();
  }

  C_ptr[g_row * size + g_col] = sum;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
