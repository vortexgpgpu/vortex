#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

// GEMM kernel supporting three modes:
// - TGEMM: C[16x16] = A[16x16] × B[16x16] (dense × dense)
// - UGEMM: C[16x16] = A[16x16] × B[2:4 sparse] (dense × 2:4 sparse)
// - VGEMM: C[16x16] = A[16x16] × B[1:4 sparse] (dense × 1:4 sparse)

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A_ptr = reinterpret_cast<uint8_t*>(arg->A_addr);
	auto B_ptr = reinterpret_cast<uint8_t*>(arg->B_addr);
	auto M_ptr = reinterpret_cast<uint8_t*>(arg->M_addr);
	auto C_ptr = reinterpret_cast<uint8_t*>(arg->C_addr);
	uint32_t mode = arg->mode;

	// Load A tile into T-reg 1 (always dense 1KB)
	vx_lt(1, (size_t)A_ptr, 0);

	if (mode == GEMM_MODE_TGEMM) {
		// TGEMM: T × T -> T
		// Load B tile into T-reg 2 (1KB dense)
		vx_lt(2, (size_t)B_ptr, 0);
		
		// TGEMM: T0 = T1 × T2 (accumulate into T0)
		vx_tgemm(0, 1, 2);
	}
	else if (mode == GEMM_MODE_UGEMM) {
		// UGEMM: T × U -> T (2:4 sparse)
		// Load metadata into M-reg 1 (128 bytes)
		vx_lm(1, (size_t)M_ptr, 0);
		
		// Load B tile into U-reg 2 (2KB sparse 2:4)
		vx_lu(2, (size_t)B_ptr, 0);
		
		// UGEMM: T0 = T1 × U2 (accumulate into T0)
		vx_ugemm(0, 1, 2);
	}
	else if (mode == GEMM_MODE_VGEMM) {
		// VGEMM: T × V -> T (1:4 sparse)
		// Load metadata into M-reg 1 (128 bytes)
		vx_lm(1, (size_t)M_ptr, 0);
		
		// Load B tile into V-reg 1 (4KB sparse 1:4)
		// Note: V-reg 1 maps to T-regs 4-7, staying within the 8 T-reg limit
		vx_lv(1, (size_t)B_ptr, 0);
		
		// VGEMM: T0 = T1 (sparse with M1 metadata) × V1 (dense)
		vx_vgemm(0, 1, 1);
	}
	else if (mode == GEMM_MODE_RGEMM) {
		// RGEMM: T × U -> U (row-wise N:4 sparse)
		// Load metadata into M-reg 1 (128 bytes)
		vx_lm(1, (size_t)M_ptr, 0);
		
		// Load B tile into U-reg 2 (2KB dense)
		vx_lu(2, (size_t)B_ptr, 0);
		
		// RGEMM: U0 = T1 (row-wise sparse with M1 metadata) × U2 (dense)
		// Output is stored in U-reg 0 = T-reg 0 + T-reg 1 (2KB total)
		// ISA: vx_rgemm computes full U-reg result
		vx_rgemm(0, 1, 2);
	}

	// Store result from T-reg 0 to C (always 1KB)
	// For RGEMM: we only validate first T-reg of U0 (top 16 rows)
	vx_st((size_t)C_ptr, 0, 0);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, nullptr, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
