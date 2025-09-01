#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto* __restrict src0 = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto* __restrict src1 = reinterpret_cast<TYPE*>(arg->src1_addr);
  auto* __restrict dst  = reinterpret_cast<TYPE*>(arg->dst_addr);
  uint32_t n = arg->num_points;

	// Allocate local menory
	auto tbuf = reinterpret_cast<TYPE*>(__local_mem(blockDim.x * sizeof(TYPE)));

	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;

	// grid-stride per-thread accumulate
	TYPE acc(0);
	uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = gid; i < n; i += stride) {
    acc += src0[i] * src1[i];
  }

	// initiqalize shared memory
	tbuf[tid] = acc;

  // Tree reduction in shared memory
  for (int d = blockDim.x / 2; d > 0; d /= 2) {
    __syncthreads();
    if (tid < d) {
			tbuf[tid] += tbuf[tid + d];
		}
  }
	if (tid == 0) {
		dst[blockIdx.x] = tbuf[0];
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
