#include <stdint.h>
#include <math.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(int task_id, kernel_arg_t* __UNIFORM__ arg);

inline float __ieee754_sqrtf (float x) {
  asm ("fsqrt.s %0, %1" : "=f" (x) : "f" (x));
  return x;
}

void kernel_iadd(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_imul(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv_mul(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		auto d = a * b;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fadd(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fsub(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmul(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmadd(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmsub(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmsub(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd_madd(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b - b;
		auto d = a * b + b;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fdiv(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fdiv2(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		auto d = b / a;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fsqrt(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = __ieee754_sqrtf(a * b);
		dst_ptr[offset+i] = c;
	}
}

void kernel_ftoi(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (int32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_ftou(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (uint32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_itof(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_utof(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset   = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

float fclamp(float a, float b, float c) {
    return fmin(fmax(a, b), c);
}

void kernel_fclamp(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;	
	auto offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		dst_ptr[offset+i] = fclamp(1.0f, a, b);
	}
}

void kernel_bar(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto num_cores = vx_num_cores();
	auto num_warps = vx_num_warps();
	auto num_threads = vx_num_threads();

	auto cid = vx_core_id();
	auto wid = vx_warp_id();
	auto tid = vx_thread_id();

	auto src0_ptr = (uint32_t*)arg->src0_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;

	// update destination using the first threads in core
	if (wid == 0 && tid == 0) {
		int block_size = arg->num_tasks / num_cores;
		int offset = cid * block_size;
		for (int i = 0; i <= block_size; ++i) {
			dst_ptr[i + offset] = src0_ptr[i + offset];
		}
	}	

	// memory fence
	vx_fence();

	// local barrier
	vx_barrier(0, num_warps);
	
	// update destination
	dst_ptr[task_id] += 1;
}

void kernel_gbar(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto num_cores = vx_num_cores();
	auto num_warps = vx_num_warps();
	auto num_threads = vx_num_threads();
	
	auto cid = vx_core_id();
	auto wid = vx_warp_id();
	auto tid = vx_thread_id();

	auto src0_ptr = (uint32_t*)arg->src0_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;

	// update destination using the first threads in processor
	if (cid == 0 && wid == 0 && tid == 0) {
		for (int i = 0, n = arg->num_tasks; i <= n; ++i) {
			dst_ptr[i] = src0_ptr[i];
		}
	}	

	// memory fence
	vx_fence();

	// global barrier
	vx_barrier(0x80000000, num_cores);
	
	// update destination
	dst_ptr[task_id] += 1;
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
	kernel_fclamp,
	kernel_bar,
	kernel_gbar
};

int main() {
	auto arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)sc_tests[arg->testid], arg);
	return 0;
}
