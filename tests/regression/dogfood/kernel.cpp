#include <vx_spawn2.h>
#include <math.h>
#include "common.h"

typedef void (*PFN_Kernel)(kernel_arg_t* __UNIFORM__ arg);

inline float __ieee754_sqrtf (float x) {
  asm ("fsqrt.s %0, %1" : "=f" (x) : "f" (x));
  return x;
}

void kernel_iadd(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_imul(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_idiv_mul(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		auto d = a * b;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fadd(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		float a = src0_ptr[offset+i];
		float b = src1_ptr[offset+i];
		float c = a + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fsub(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmul(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmadd(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fmsub(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b - b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmsub(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b + b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fnmadd_madd(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c =-a * b - b;
		auto d = a * b + b;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fdiv(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		dst_ptr[offset+i] = c;
	}
}

void kernel_fdiv2(kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a / b;
		auto d = b / a;
		auto e = c + d;
		dst_ptr[offset+i] = e;
	}
}

void kernel_fsqrt(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = __ieee754_sqrtf(a * b);
		dst_ptr[offset+i] = c;
	}
}

void kernel_ftoi(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (int32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (int32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_ftou(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (uint32_t)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_itof(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

void kernel_utof(kernel_arg_t* __UNIFORM__ arg) {
	auto count    = arg->task_size;
	auto src0_ptr = (int32_t*)arg->src0_addr;
	auto src1_ptr = (int32_t*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		auto c = a + b;
		auto d = (float)c;
		dst_ptr[offset+i] = d;
	}
}

inline float fclamp(float a, float b, float c) {
  return fmin(fmax(a, b), c);
}

void kernel_fclamp(kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		dst_ptr[offset+i] = fclamp(1.0f, a, b);
	}
}

inline int iclamp(int a, int b, int c) {
  return std::min(std::max(a, b), c);
}

void kernel_iclamp(kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (int*)arg->src0_addr;
	auto src1_ptr = (int*)arg->src1_addr;
	auto dst_ptr  = (int*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		auto a = src0_ptr[offset+i];
		auto b = src1_ptr[offset+i];
		dst_ptr[offset+i] = iclamp(1, a, b);
	}
}

void kernel_trigo(kernel_arg_t* __UNIFORM__ arg) {
	auto count  = arg->task_size;
	auto src0_ptr = (float*)arg->src0_addr;
	auto src1_ptr = (float*)arg->src1_addr;
	auto dst_ptr  = (float*)arg->dst_addr;
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto offset = gid * count;
	for (uint32_t i = 0; i < count; ++i) {
		uint32_t j = offset + i;
		auto a = src0_ptr[j];
		auto b = src1_ptr[j];
		auto c = a * b;
		if ((j % 4) == 0) {
			c = sinf(c);
		}
		dst_ptr[j] = c;
	}
}

void kernel_bar(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto num_cores = vx_num_cores();
	auto num_warps = vx_num_warps();
	auto num_threads = vx_num_threads();

	auto cid = vx_core_id();
	auto wid = vx_warp_id();
	auto tid = vx_thread_id();

	auto src0_ptr = (uint32_t*)arg->src0_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;

	// update destination using the first thread in core
	if (wid == 0 && tid == 0) {
		int block_size = arg->num_tasks / num_cores;
		int offset = cid * block_size;
		for (int i = 0; i < block_size; ++i) {
			dst_ptr[i + offset] = src0_ptr[i + offset];
		}
	}

	// memory fence
	vx_fence();

	// local barrier
	vx_barrier(0, num_warps);

	// update destination
	dst_ptr[gid] += 1;
}

void kernel_gbar(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto num_cores = vx_num_cores();
	auto num_warps = vx_num_warps();
	auto num_threads = vx_num_threads();

	auto cid = vx_core_id();
	auto wid = vx_warp_id();
	auto tid = vx_thread_id();

	auto src0_ptr = (uint32_t*)arg->src0_addr;
	auto dst_ptr  = (uint32_t*)arg->dst_addr;

	// update destination using the first thread in processor
	if (cid == 0 && wid == 0 && tid == 0) {
		for (int i = 0, n = arg->num_tasks; i < n; ++i) {
			dst_ptr[i] = src0_ptr[i];
		}
	}

	// memory fence
	vx_fence();

	// global barrier
	vx_barrier(0x80000000, num_cores);

	// update destination
	dst_ptr[gid] += 1;
}

static const PFN_Kernel sc_tests[] = {
	kernel_iadd, kernel_imul, kernel_idiv, kernel_idiv_mul,
	kernel_fadd, kernel_fsub, kernel_fmul, kernel_fmadd,
	kernel_fmsub, kernel_fnmadd, kernel_fnmsub, kernel_fnmadd_madd,
	kernel_fdiv, kernel_fdiv2, kernel_fsqrt, kernel_ftoi,
	kernel_ftou, kernel_itof, kernel_utof, kernel_fclamp,
	kernel_iclamp, kernel_trigo, kernel_bar, kernel_gbar,
};

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	sc_tests[arg->testid](arg);
}