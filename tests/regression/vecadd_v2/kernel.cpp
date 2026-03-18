#include <vx_spawn2.h>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	dst_ptr[blockIdx.x] = src0_ptr[blockIdx.x] + src1_ptr[blockIdx.x];
}
