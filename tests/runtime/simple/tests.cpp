#include "tests.h"
#include <stdio.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

int check_error(const int* buffer, int size) {
	int errors = 0;
	for (int i = 0; i < size; i++)	{
		int value = buffer[i];
		int ref_value = 65 + i;
		if (value == ref_value)	{
			//vx_printf("[%d] %c\n", i, value);
		} else {
			vx_printf("*** error: [%d] %x, expected %x\n", i, value, ref_value);
			++errors;
		}
	}
	return errors;
}

///////////////////////////////////////////////////////////////////////////////

#define GLOBAL_MEM_SZ 8
int global_buffer[GLOBAL_MEM_SZ];

int test_global_memory() {	
	int errors = 0;

	vx_printf("Global Memory test\n");

	for (int i = 0; i < GLOBAL_MEM_SZ; i++) {
		global_buffer[i] = 65 + i;
	}

	return check_error(global_buffer, GLOBAL_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int test_stack_memory() {
	static const int STACK_MEM_SZ = 8;
	int stack_buffer[STACK_MEM_SZ];
	int errors = 0;

	vx_printf("Stack Memory test\n");

	for (int i = 0; i < STACK_MEM_SZ; i++) {
		stack_buffer[i] = 65 + i;
	}

	return check_error(stack_buffer, STACK_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int test_shared_memory() {
	static const int SHARED_MEM_SZ = 8;
	int* shared_buffer = (int*)(SMEM_BASE_ADDR-(SMEM_SIZE-SHARED_MEM_SZ-4));
	int errors = 0;

	vx_printf("Shared Memory test\n");	
	
	for (int i = 0; i < SHARED_MEM_SZ; i++) {
		shared_buffer[i] = 65 + i;
	}

	return check_error(shared_buffer, SHARED_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int tmc_buffer[NUM_THREADS];

int test_tmc() {
	int errors = 0;

	vx_printf("Thread mask test\n");

	vx_tmc(NUM_THREADS);
	unsigned tid = vx_thread_id();
	tmc_buffer[tid] = 65 + tid;
	vx_tmc(1);

	return check_error(tmc_buffer, NUM_THREADS);
}

///////////////////////////////////////////////////////////////////////////////

int wspawn_buffer[NUM_WARPS];

void simple_kernel() {
	unsigned wid = vx_warp_id();
	wspawn_buffer[wid] = 65 + wid;
	vx_tmc(0 == wid);
}

int test_wsapwn() {
	vx_printf("test_wspawn\n");
	vx_wspawn(NUM_WARPS, simple_kernel);
	simple_kernel();

	return check_error(wspawn_buffer, NUM_WARPS);
}

///////////////////////////////////////////////////////////////////////////////

#define DIV_BUF_SZ ((NUM_THREADS > 4) ? 4 : NUM_THREADS)
int div_buffer[DIV_BUF_SZ];

int test_divergence() {
	int errors = 0;

	vx_printf("Control divergence test\n");

	vx_tmc(DIV_BUF_SZ);

	unsigned tid = vx_thread_id();

	bool b = tid < 2;
	__if (b) {
		bool c = tid < 1;
		__if (c) {
			div_buffer[tid] = 65;
		}
		__else {
			div_buffer[tid] = 66;
		}
		__endif
	}
	__else {
		bool c = tid < 3;
		__if (c) {
			div_buffer[tid] = 67;
		}
		__else {
			div_buffer[tid] = 68;
		}
		__endif
	}
	__endif

	vx_tmc(1);

	return check_error(div_buffer, DIV_BUF_SZ);
}

///////////////////////////////////////////////////////////////////////////////

#define ST_BUF_SZ 8
typedef struct {
	int * src;
	int * dst;
} st_args_t;

int st_buffer_src[ST_BUF_SZ];
int st_buffer_dst[ST_BUF_SZ];

void st_kernel(int task_id, void * arg) {
	st_args_t * arguments = (st_args_t *) arg;
  	arguments->dst[task_id] = arguments->src[task_id];
}

int test_spawn_tasks() {
	int error = 0;

	st_args_t arg;
	arg.src = st_buffer_src;
	arg.dst = st_buffer_dst;

	vx_printf("spawning %d tasks\n", ST_BUF_SZ);

	for (int i = 0; i < ST_BUF_SZ; i++) {
		st_buffer_src[i] = 65 + i;
	}

	vx_spawn_tasks(ST_BUF_SZ, st_kernel, &arg);

	return check_error(st_buffer_dst, ST_BUF_SZ);
}