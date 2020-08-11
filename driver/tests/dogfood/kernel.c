#include <stdint.h>
#include <math.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(void* arg);

void kernel_iadd(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_imul(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv_mul(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a / b;
		int32_t d = a * b;
		int32_t e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fadd(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fsub(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmul(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmadd(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b + 0.5f;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmsub(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b - 0.5f;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = -a * b - 0.5f;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmsub(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = -a * b + 0.5f;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd_madd(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c =-a * b - 0.5f;
		float d = a * b + 0.5f;
		float e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fdiv(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fdiv2(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a / b;
		float d = b / a;
		float e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fsqrt(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = sqrt(a * b);
		dst_ptr[offset+i] = c;
	}
}

void kernel_ftoi(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		int32_t d = (int32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_ftou(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	float* src0_ptr = (float*)_arg->src0_ptr;
	float* src1_ptr = (float*)_arg->src1_ptr;
	uint32_t* dst_ptr  = (uint32_t*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		uint32_t d = (uint32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_itof(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		float d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_utof(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count  = _arg->count;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	float* dst_ptr  = (float*)_arg->dst_ptr;	
	uint32_t offset = vx_thread_gid() * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		uint32_t c = a + b;
		float d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

static const PFN_Kernel sc_tests[] = {
	kernel_iadd,
	kernel_imul,
	kernel_idiv,
	kernel_idiv_mul,
	kernel_fadd,
	kernel_fsub,
	kernel_fmul,
	kernel_fmadd,
	kernel_fmsub,
	kernel_fnmadd,	
	kernel_fnmsub,
	kernel_fnmadd_madd,
	kernel_fdiv,
	kernel_fdiv2,
	kernel_fsqrt,
	kernel_ftoi,
	kernel_ftou,
	kernel_itof,
	kernel_utof,
};

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	int num_warps = vx_num_warps();
	int num_threads = vx_num_threads();
	vx_spawn_warps(num_warps, num_threads, sc_tests[arg->testid], arg);
}