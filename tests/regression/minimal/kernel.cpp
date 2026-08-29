#include <vx_spawn2.h>
#include "common.h"

// The smallest kernel that still proves the launch path end to end: no inputs,
// no arithmetic worth the name, one store per thread. Anything that fails here
// is the CP / dispatch / memory path, not the workload.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	dst_ptr[idx] = MINIMAL_MAGIC | idx;
}
