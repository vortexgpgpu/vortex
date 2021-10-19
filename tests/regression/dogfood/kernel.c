#include <stdint.h>
#include <math.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(int task_id, kernel_arg_t* arg);

inline float __ieee754_sqrtf (float x) {
  asm ("fsqrt.s %0, %1" : "=f" (x) : "f" (x));
  return x;
}

void kernel_iadd(int task_id, kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_imul(int task_id, kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv(int task_id, kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv_mul(int task_id, kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a / b;
		int32_t d = a * b;
		int32_t e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fadd(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fsub(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmul(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmadd(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmsub(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c =-a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmsub(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c =-a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd_madd(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c =-a * b - b;
		float d = a * b + b;
		float e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fdiv(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fdiv2(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a / b;
		float d = b / a;
		float e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fsqrt(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = __ieee754_sqrtf(a * b);
		dst_ptr[offset+i] = c;
	}
}

void kernel_ftoi(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		int32_t d = (int32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_ftou(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	float* src0_ptr = (float*)arg->src0_ptr;
	float* src1_ptr = (float*)arg->src1_ptr;
	uint32_t* dst_ptr  = (uint32_t*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		uint32_t d = (uint32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_itof(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		float d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_utof(int task_id, kernel_arg_t* arg) {
	uint32_t count  = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	float* dst_ptr  = (float*)arg->dst_ptr;	
	uint32_t offset = task_id * count;

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
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)sc_tests[arg->testid], arg);
}