#include "tests.h"
#include <stdio.h>
#include <algorithm>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

int __attribute__ ((noinline)) check_error(const int* buffer, int offset, int size) {
	int errors = 0;
	for (int i = offset; i < size; i++)	{
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

int __attribute__ ((noinline)) make_select_tmask(int tid) {
	return (1 << tid);
}

int __attribute__ ((noinline)) make_full_tmask(int num_threads) {
	return (1 << num_threads) - 1;
}

///////////////////////////////////////////////////////////////////////////////

#define GLOBAL_MEM_SZ 8
int global_buffer[GLOBAL_MEM_SZ];

int test_global_memory() {
	vx_printf("Global Memory Test\n");

	for (int i = 0; i < GLOBAL_MEM_SZ; i++) {
		global_buffer[i] = 65 + i;
	}

	return check_error(global_buffer, 0, GLOBAL_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int test_stack_memory() {
	vx_printf("Stack Memory Test\n");

	static const int STACK_MEM_SZ = 8;
	int stack_buffer[STACK_MEM_SZ];

	for (int i = 0; i < STACK_MEM_SZ; i++) {
		stack_buffer[i] = 65 + i;
	}

	return check_error(stack_buffer, 0, STACK_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int test_shared_memory() {
	static const int SHARED_MEM_SZ = 8;
	int* shared_buffer = (int*)(SMEM_BASE_ADDR-(SMEM_SIZE-SHARED_MEM_SZ-4));

	vx_printf("Shared Memory Test\n");	
	
	for (int i = 0; i < SHARED_MEM_SZ; i++) {
		shared_buffer[i] = 65 + i;
	}

	return check_error(shared_buffer, 0, SHARED_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

int tmc_buffer[8];

void __attribute__ ((noinline)) do_tmc() {
	unsigned tid = vx_thread_id();
	tmc_buffer[tid] = 65 + tid;
}

int test_tmc() {
	vx_printf("TMC Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);	
	do_tmc();
	vx_tmc(1);

	return check_error(tmc_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int pred_buffer[8];

void __attribute__ ((noinline)) do_pred() {
	unsigned tid = vx_thread_id();
	pred_buffer[tid] = 65 + tid;
}

int test_pred() {
	vx_printf("PRED Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);

	for (int i = 0; i < num_threads; i++) {
		pred_buffer[i] = 0;
	}

	vx_pred(~1);
	do_pred();
	vx_tmc(1);

	int status_n0 = (0 == tmc_buffer[0]);
	int status_n1 = check_error(tmc_buffer, 1, num_threads);
	return status_n0 && status_n1;
}

///////////////////////////////////////////////////////////////////////////////

int wspawn_buffer[8];

void wspawn_kernel() {
	unsigned wid = vx_warp_id();
	wspawn_buffer[wid] = 65 + wid;
	vx_tmc(0 == wid);
}

int test_wsapwn() {
	vx_printf("Wspawn Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	vx_wspawn(num_warps, wspawn_kernel);
	wspawn_kernel();

	return check_error(wspawn_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

int dvg_buffer[4];

void __attribute__ ((noinline)) do_divergence() {

	unsigned tid = vx_thread_id();

	__if (tid < 2) {
		__if (tid < 1) {
			dvg_buffer[tid] = 65;			
		}
		__else {
			dvg_buffer[tid] = 66;
		}
		__endif
	}
	__else {
		__if (tid < 3) {
			dvg_buffer[tid] = 67;
		}
		__else {
			dvg_buffer[tid] = 68;
		}
		__endif
	}
	__endif
}

int test_divergence() {
	vx_printf("Control Divergence Test\n");

	int num_threads = std::min(vx_num_threads(), 4);
	int tmask = make_full_tmask(num_threads);	
	vx_tmc(tmask);
	do_divergence();
	vx_tmc(1);

	return check_error(dvg_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

#define ST_BUF_SZ 8
typedef struct {
	int * src;
	int * dst;
} st_args_t;

int st_buffer_src[ST_BUF_SZ];
int st_buffer_dst[ST_BUF_SZ];

void st_kernel(int task_id, const st_args_t * arg) {
  	arg->dst[task_id] = arg->src[task_id];
}

int test_spawn_tasks() {
	vx_printf("SpawnTasks Test\n");

	st_args_t arg;
	arg.src = st_buffer_src;
	arg.dst = st_buffer_dst;

	for (int i = 0; i < ST_BUF_SZ; i++) {
		st_buffer_src[i] = 65 + i;
	}

	vx_spawn_tasks(ST_BUF_SZ, (vx_spawn_tasks_cb)st_kernel, &arg);

	return check_error(st_buffer_dst, 0, ST_BUF_SZ);
}

///////////////////////////////////////////////////////////////////////////////

#define SR_BUF_SZ 8
typedef struct {
	int * buf;
} sr_args_t;

int sr_buffer[SR_BUF_SZ];

void sr_kernel(const sr_args_t * arg) {
	int tid = vx_thread_id();
  	arg->buf[tid] = 65 + tid;
}

void __attribute__ ((noinline)) do_serial() {
	sr_args_t arg;
	arg.buf = sr_buffer;
	vx_serial((vx_serial_cb)sr_kernel, &arg);
}

int test_serial() {
	vx_printf("Serial Test\n");	
	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);	
	vx_tmc(tmask);
	do_serial();
	vx_tmc(1);

	return check_error(sr_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int tmask_buffer[8];

int __attribute__ ((noinline)) do_tmask() {					
	int tid = vx_thread_id();
	int tmask = make_select_tmask(tid);
	int cur_tmask = vx_thread_mask();
	tmask_buffer[tid] = (cur_tmask == tmask) ? (65 + tid) : 0;
	return tid + 1;
}

int test_tmask() {
	vx_printf("Thread Mask Test\n");

	// activate all thread to populate shared variables
	vx_tmc(-1);

	int num_threads = std::min(vx_num_threads(), 8);
	int tid = 0;

l_start:	
	int tmask = make_select_tmask(tid);
	vx_tmc(tmask);	
	tid = do_tmask();	
	if (tid < num_threads)		
		goto l_start;
	vx_tmc(1);

	return check_error(tmask_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int barrier_buffer[8];
volatile int barrier_ctr;
volatile int barrier_stall;

void barrier_kernel() {
	unsigned wid = vx_warp_id();
	for (int i = 0; i <= (wid * 256); ++i) {
		++barrier_stall;
	}	
	barrier_buffer[wid] = 65 + wid;
	vx_barrier(0, barrier_ctr);
	vx_tmc(0 == wid);
}

int test_barrier() {
	vx_printf("Barrier Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	barrier_ctr = num_warps;
	barrier_stall = 0;
	vx_wspawn(num_warps, barrier_kernel);
	barrier_kernel();	
	return check_error(barrier_buffer, 0, num_warps);
}