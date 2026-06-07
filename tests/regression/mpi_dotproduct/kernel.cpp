#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	auto num_points = arg->num_points;

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	auto cache = reinterpret_cast<TYPE*>(__local_mem(blockDim.x * sizeof(TYPE)));

	float temp = 0;
	while (tid < num_points){
		temp += src0_ptr[tid] * src1_ptr[tid];
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex] = temp;

    __syncthreads();

	int i = blockDim.x/2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0)
		dst_ptr[blockIdx.x] = cache[0];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
