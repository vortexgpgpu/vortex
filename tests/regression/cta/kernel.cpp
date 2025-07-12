#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src_ptr  = reinterpret_cast<int*>(arg->src_addr);
	auto dst_ptr  = reinterpret_cast<int*>(arg->dst_addr);
	uint32_t cta_size = blockDim.x * blockDim.y * blockDim.z;
 	uint32_t blockId  = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
 	uint32_t localId  = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  uint32_t globalId = localId + blockId * cta_size;
  dst_ptr[globalId] = globalId + src_ptr[localId];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(3, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}