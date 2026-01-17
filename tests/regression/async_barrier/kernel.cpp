#include <vx_spawn.h>
#include <math.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
  // Setup buffer arguments
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  // Allocate local memory for double buffering: 2 buffers for A, 2 for B
  auto local_ptr = __local_mem(4 * blockDim.x * blockDim.y * sizeof(TYPE));
  auto local_A0 = (TYPE*)local_ptr;
  auto local_B0 = local_A0 + blockDim.x * blockDim.y;

  auto size = arg->size;
  auto tile_size = arg->tile_size;

  // Determine global row and column indices
  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto g_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Determine local row and column indices
  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;

  TYPE sum(0);
  TYPE local_acc(0);

  // Initialize CTA-level async barrier
  barrier bar;
  bar.init(__warps_per_group);
  __syncthreads();

  uint32_t token_load;
  uint32_t token_compute;

  // Main loop
  for (uint32_t k = 0; k < size; k += tile_size) {
    if (k > 0){
      bar.wait(token_compute);  // Wait for previous compute to finish
    }
    // Load tile to shared memory
    local_A0[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
    local_B0[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];

    // Async arrive: non-blocking, returns token (generation number)
    token_load = bar.arrive();
    
    // Do independent work while waiting for other warps
    TYPE my_val = local_A0[l_row * tile_size + l_col];
    TYPE my_val_b = local_B0[l_row * tile_size + l_col];

    for (int iter = 0; iter < tile_size; ++iter) {
      my_val = sqrt(my_val * my_val + 0.01f);
      my_val_b = sqrt(my_val_b * my_val_b + 0.01f);
    }
    local_acc += my_val + my_val_b;

    // Async wait: blocks until generation > token
    bar.wait(token_load);

    // Now safe to read from shared memory
    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A0[l_row * tile_size + j] * local_B0[j * tile_size + l_col];
    }
    token_compute = bar.arrive();
  }

  // Store the computed sum into the result matrix C
  C_ptr[g_row * size + g_col] = sum + local_acc - local_acc;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}