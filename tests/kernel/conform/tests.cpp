#include "tests.h"
#include <stdio.h>
#include <algorithm>
#include <VX_config.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

int __attribute__((noinline)) check_error(const int* __UNIFORM__  buffer, int __UNIFORM__  offset, int __UNIFORM__  size) {
	int errors = 0;
	for (int i = offset; i < size; i++)	{
		int value = buffer[i];
		int ref_value = 65 + i;
		if (value == ref_value)	{
			//PRINTF("[%d] %c\n", i, value);
		} else {
			PRINTF("*** error: [%d] 0x%x, expected 0x%x\n", i, value, ref_value);
			++errors;
		}
	}
	return errors;
}

int __attribute__((noinline)) make_select_tmask(int tid) {
	return (1 << tid);
}

int __attribute__((noinline)) make_full_tmask(int num_threads) {
	return (1 << num_threads) - 1;
}

///////////////////////////////////////////////////////////////////////////////

#define GLOBAL_MEM_SZ 8
int global_buffer[GLOBAL_MEM_SZ];

int test_global_memory() {
	PRINTF("Global Memory Test\n");

	for (int i = 0; i < GLOBAL_MEM_SZ; i++) {
		global_buffer[i] = 65 + i;
	}

	return check_error(global_buffer, 0, GLOBAL_MEM_SZ);
}

///////////////////////////////////////////////////////////////////////////////

volatile int* lmem_addr = (int*)LMEM_BASE_ADDR;

int lmem_buffer[8];

void __attribute__((noinline)) do_lmem_wr() {
	unsigned tid = vx_thread_id();
	lmem_addr[tid] = 65 + tid;
	int x = lmem_addr[tid];
	lmem_addr[tid] = x;
}

void __attribute__((noinline)) do_lmem_rd() {
	unsigned tid = vx_thread_id();
	lmem_buffer[tid] = lmem_addr[tid];
}

int test_local_memory() {
	PRINTF("Local Memory Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_lmem_wr();
	do_lmem_rd();
	vx_tmc_one();

	return check_error(lmem_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int tmc_buffer[8];

void __attribute__((noinline)) do_tmc() {
	unsigned tid = vx_thread_id();
	tmc_buffer[tid] = 65 + tid;
}

int test_tmc() {
	PRINTF("TMC Test\n");

	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_tmc();
	vx_tmc_one();

	return check_error(tmc_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int pred_buffer[8];

void __attribute__((noinline)) do_pred() {
	unsigned tid = vx_thread_id();
	vx_pred((tid == 0), 1);
	pred_buffer[tid] = 65;
}

int test_pred() {
	PRINTF("PRED Test\n");
	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);

	for (int i = 1; i < num_threads; i++) {
		pred_buffer[i] = 65 + i;
	}

	vx_tmc(tmask);
	do_pred();
	vx_tmc_one();

	return check_error(pred_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

int wspawn_buffer[8];

void wspawn_kernel() {
	unsigned wid = vx_warp_id();
	wspawn_buffer[wid] = 65 + wid;
	vx_tmc(0 == wid);
}

int test_wsapwn() {
	PRINTF("Wspawn Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	vx_wspawn(num_warps, wspawn_kernel);
	wspawn_kernel();

	return check_error(wspawn_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

int dvg_buffer[4];

void __attribute__((noinline)) do_divergence() {
	int tid = vx_thread_id();
	int cond1 = tid < 2;
	int sp1 = vx_split(cond1);
	if (cond1) {
		{
			int cond2 = tid < 1;
			int sp2 = vx_split(cond2);
			if (cond2) {
				dvg_buffer[tid] = 65; // A
			} else {
				dvg_buffer[tid] = 66; // B
			}
			vx_join(sp2);
		}
		{
			int cond3 = tid < 0;
			int sp3 = vx_split(cond3);
			if (cond3) {
				dvg_buffer[tid] = 67; // C
			}
			vx_join(sp3);
		}
	} else {
		{
			int cond2 = tid < 3;
			int sp2 = vx_split(cond2);
			if (cond2) {
				dvg_buffer[tid] = 67; // C
			} else {
				dvg_buffer[tid] = 68; // D
			}
			vx_join(sp2);
		}
	}
	vx_join(sp1);
}

int test_divergence() {
	PRINTF("Control Divergence Test\n");

	int num_threads = std::min(vx_num_threads(), 4);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_divergence();
	vx_tmc_one();

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

void st_kernel(const st_args_t * __UNIFORM__ arg) {
  arg->dst[blockIdx.x] = arg->src[blockIdx.x];
}

int test_spawn_tasks() {
	PRINTF("SpawnTasks Test\n");

	st_args_t arg;
	arg.src = st_buffer_src;
	arg.dst = st_buffer_dst;

	for (int i = 0; i < ST_BUF_SZ; i++) {
		st_buffer_src[i] = 65 + i;
	}

	uint32_t num_tasks(ST_BUF_SZ);
	vx_spawn_threads(1, &num_tasks, nullptr, (vx_kernel_func_cb)st_kernel, &arg);

	return check_error(st_buffer_dst, 0, ST_BUF_SZ);
}

///////////////////////////////////////////////////////////////////////////////

typedef struct {
	int * buf;
} sr_args_t;

int sr_buffer[8];

void sr_kernel(const sr_args_t * arg) {
	int tid = vx_thread_id();
  	arg->buf[tid] = 65 + tid;
}

void __attribute__((noinline)) do_serial() {
	sr_args_t arg;
	arg.buf = sr_buffer;
	vx_serial((vx_serial_cb)sr_kernel, &arg);
}

int test_serial() {
	PRINTF("Serial Test\n");
	int num_threads = std::min(vx_num_threads(), 8);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask);
	do_serial();
	vx_tmc_one();

	return check_error(sr_buffer, 0, num_threads);
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
	PRINTF("Barrier Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	barrier_ctr = num_warps;
	barrier_stall = 0;
	vx_wspawn(num_warps, barrier_kernel);
	barrier_kernel();
	return check_error(barrier_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

int tls_buffer[8];
__thread int tls_var;

__attribute__((noinline)) void print_tls_var() {
	unsigned wid = vx_warp_id();
	tls_buffer[wid] = 65 + tls_var;
}

void tls_kernel() {
	unsigned wid = vx_warp_id();
	tls_var = wid;
	print_tls_var();
	vx_tmc(0 == wid);
}

int test_tls() {
	PRINTF("TLS Test\n");
	int num_warps = std::min(vx_num_warps(), 8);
	vx_wspawn(num_warps, tls_kernel);
	tls_kernel();
	return check_error(tls_buffer, 0, num_warps);
}

///////////////////////////////////////////////////////////////////////////////

#define VOTE_GROUP_SZ 4
int vote_buffer[VOTE_GROUP_SZ];

void __attribute__((noinline)) do_vote() {
	int num_threads = std::min(vx_num_threads(), VOTE_GROUP_SZ);
	int tmask = make_full_tmask(num_threads);
	int tid = vx_thread_id();

	// test predicate
	int predicate1 = (tid != -1);
	int predicate2 = (tid < 2);

	// Execute vote instructions
	int vote1_all = vx_vote_all(predicate1);
	int vote1_any = vx_vote_any(predicate1);
	int vote1_uni = vx_vote_uni(predicate1);
	int vote1_ballot = vx_vote_ballot(predicate1);
	int vote2_all = vx_vote_all(predicate2);
	int vote2_any = vx_vote_any(predicate2);
	int vote2_uni = vx_vote_uni(predicate2);
	int vote2_ballot = vx_vote_ballot(predicate2);

	// evaluate
	bool check_all = (vote1_all == 1) && (vote2_all == 0);
	bool check_any = (vote1_any == 1) && (vote2_any == 1);
	bool check_uni = (vote1_uni == 1) && (vote2_uni == 0);
	bool check_ballot = (vote1_ballot == tmask)
	                 && (vote2_ballot == (tmask & 0b11));
	int passed = (check_all && check_any && check_uni && check_ballot);

	// report per-thread result
	vote_buffer[tid] = passed ? (65 + tid) : 0;
}

int test_vote() {
	PRINTF("Vote Test\n");
	int num_threads = std::min(vx_num_threads(), VOTE_GROUP_SZ);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask); // active all threads
	do_vote();
  vx_tmc_one(); // back to thread0
	return check_error(vote_buffer, 0, num_threads);
}

///////////////////////////////////////////////////////////////////////////////

#define SHFL_GROUP_SZ 4
int shfl_buffer[SHFL_GROUP_SZ];

void __attribute__((noinline)) do_shfl() {
	int num_threads = std::min(vx_num_threads(), VOTE_GROUP_SZ);
	int tmask = make_full_tmask(num_threads);
	int tid = vx_thread_id();
	int value = 65 + tid;

	// compute exactly what each shuffle *should* have returned:
	int exp_up   = (tid >= 1) ? (65 + tid - 1) : value;
	int exp_down = (tid < num_threads - 1) ? (65 + tid + 1) : value;
	int exp_bfly = ((tid ^ 1) < num_threads) ? (65 + (tid ^ 1)) : value;
	int exp_idx  = (num_threads > 1) ? (65+1) : value;

	// Use single subgroup of all threads
  int subgroup_mask = 0;
  int subgroup_clamp = num_threads - 1;

	// lane test for each of the four shuffle ops:
	int v_up   = vx_shfl_up(value,   1, subgroup_clamp, subgroup_mask);
	int v_down = vx_shfl_down(value, 1, subgroup_clamp, subgroup_mask);
	int v_bfly = vx_shfl_bfly(value, 1, subgroup_clamp, subgroup_mask);
	int v_idx  = vx_shfl_idx(value,  1, subgroup_clamp, subgroup_mask);
	//PRINTF("v_up=%d, v_down=%d, v_bfly=%d, v_idx=%d\n", v_up, v_down, v_bfly, v_idx);
  //PRINTF("exp_up=%d, exp_down=%d, exp_bfly=%d, exp_idx=%d\n", exp_up, exp_down, exp_bfly, exp_idx);

  // pass only if *all* four match their expected result:
  int passed = (v_up == exp_up)
            && (v_down == exp_down)
            && (v_bfly == exp_bfly)
            && (v_idx  == exp_idx);

	// report per-thread result
	shfl_buffer[tid] = passed ? (65 + tid) : 0;
}

int test_shfl() {
	PRINTF("Shuffle Test\n");
	int num_threads = std::min(vx_num_threads(), SHFL_GROUP_SZ);
	int tmask = make_full_tmask(num_threads);
	vx_tmc(tmask); // active all threads
	do_shfl();
  vx_tmc_one(); // back to thread0
	return check_error(shfl_buffer, 0, num_threads);
}