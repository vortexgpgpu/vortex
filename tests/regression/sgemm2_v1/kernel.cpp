#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t *arg) {
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  size_t size      = arg->size;
  size_t tile_size = arg->tile_size;
  size_t chunk_k   = arg->chunk_k;

  // Global and local thread coordinates.
  size_t g_row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t g_col = blockIdx.y * blockDim.y + threadIdx.y;
  size_t l_row = threadIdx.x;
  size_t l_col = threadIdx.y;

  // Shared memory layout:
  //   local_A: [tile_size × chunk_k], row-major (row = l_row, col = k_local)
  //   local_B: [chunk_k × tile_size], row-major (row = k_local, col = l_col)
  auto tile_elems_a = tile_size * chunk_k;
  auto tile_elems_b = chunk_k * tile_size;
  auto local_ptr = __local_mem((tile_elems_a + tile_elems_b) * sizeof(TYPE));
  auto local_A = reinterpret_cast<TYPE*>(local_ptr);
  auto local_B = local_A + tile_elems_a;

  TYPE sum(0);

  for (size_t k = 0; k < size; k += chunk_k) {
    // Cooperative load: each thread loads chunk_k/tile_size elements per tile.
    // Thread (l_row, l_col) loads columns l_col, l_col+tile_size, ... of its row.
    for (size_t kk = l_col; kk < chunk_k; kk += tile_size) {
      local_A[l_row * chunk_k + kk] = A_ptr[g_row * size + (k + kk)];
    }
    for (size_t kk = l_row; kk < chunk_k; kk += tile_size) {
      local_B[kk * tile_size + l_col] = B_ptr[(k + kk) * size + g_col];
    }

    __syncthreads();

    const TYPE* a_row = &local_A[l_row * chunk_k];
    size_t j = 0;
    for (; j + 4 <= chunk_k; j += 4) {
#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
      static_assert(__is_same(TYPE, float), "the wide loads read float words");
      // B0 and B1 before the quad, B2 and B3 after: the order the compiler gives
      // the scalar loads, which it cannot reach across the quad's asm volatile
      TYPE b0 = local_B[j * tile_size + l_col];
      TYPE b1 = local_B[(j + 1) * tile_size + l_col];
      vx_float4 a = vx_flw_q(&a_row[j]);
      sum += a[0] * b0;
      sum += a[1] * b1;
      sum += a[2] * local_B[(j + 2) * tile_size + l_col];
      sum += a[3] * local_B[(j + 3) * tile_size + l_col];
#else
      TYPE a[4] = {a_row[j], a_row[j + 1], a_row[j + 2], a_row[j + 3]};
      sum += a[0] * local_B[j * tile_size + l_col];
      sum += a[1] * local_B[(j + 1) * tile_size + l_col];
      sum += a[2] * local_B[(j + 2) * tile_size + l_col];
      sum += a[3] * local_B[(j + 3) * tile_size + l_col];
#endif
    }
    for (; j < chunk_k; ++j) {
      sum += a_row[j] * local_B[j * tile_size + l_col];
    }

    __syncthreads();
  }

  C_ptr[g_row * size + g_col] = sum;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
