#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
	// Setup buffer arguments
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  // Allocate local memory for the tile of matrix A & B
	auto local_ptr = __local_mem(2 * blockDim.x * blockDim.y * sizeof(TYPE));
  auto local_A = (TYPE*)local_ptr;
  auto local_B = (TYPE*)local_ptr + blockDim.x * blockDim.y;

  auto size = arg->size;
  auto tile_size = arg->tile_size;

  // Determine global row and column indices
  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto g_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Determine local row and column indices
  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;

  TYPE sum(0);

  // Loop over tiles
  for (uint32_t k = 0; k < size; k += tile_size) {
    // Load tile of matrix A & B to local memory
    local_A[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
    local_B[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];

    // Synchronize all warps in current group
    __syncthreads();

    // Compute partial sum for the local tile
    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
    }

    // Synchronize all warps in current group
    __syncthreads();
  }

  // Store the computed sum into the result matrix C
  C_ptr[g_row * size + g_col] = sum;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
