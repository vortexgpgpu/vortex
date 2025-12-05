#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// Simple TGEMM test: C[16x16] = A[16x16] × B[16x16]
// T Tile registers are 16x16 fp32 = 1KB each

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A_ptr = reinterpret_cast<uint8_t*>(arg->A_addr);
	auto B_ptr = reinterpret_cast<uint8_t*>(arg->B_addr);
	auto C_ptr = reinterpret_cast<uint8_t*>(arg->C_addr);

	// Load A tile into T-reg 1
	vx_lt(1, (size_t)A_ptr, 0);

	// Load B tile into T-reg 2
	vx_lt(2, (size_t)B_ptr, 0);

	// TGEMM: T0 = T1 × T2 (accumulate into T0)
	vx_tgemm(0, 1, 2);

	// Store result from T-reg 0 to C
	vx_st((size_t)C_ptr, 0, 0);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, nullptr, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
