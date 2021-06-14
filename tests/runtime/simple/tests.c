#include "tests.h"
#include <stdbool.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

int tmc_array[4] = {5, 5, 5, 5};

void test_tmc_impl() {
	unsigned tid = vx_thread_id(); // Get TID
	tmc_array[tid] = tid;
}

void test_tmc() {
	vx_printf("testing_tmc\n");

	vx_tmc(4);
	test_tmc_impl();
	vx_tmc(1);

	vx_printx(tmc_array[0]);
	vx_printx(tmc_array[1]);
	vx_printx(tmc_array[2]);
	vx_printx(tmc_array[3]);

	return;
}

int div_arr[4];

void test_divergence() {
	vx_tmc(4);

	unsigned tid = vx_thread_id(); // Get TID

	bool b = tid < 2;
	__if (b) {
		bool c = tid < 1;
		__if (c) {
			div_arr[tid] = 10;
		}
		__else {
			div_arr[tid] = 11;
		}
		__endif
	}
	__else {
		bool c = tid < 3;
		__if (c) {
			div_arr[tid] = 12;
		}
		__else {
			div_arr[tid] = 13;
		}
		__endif
	}
	__endif

	vx_tmc(1);

	vx_printx(div_arr[0]);
	vx_printx(div_arr[1]);
	vx_printx(div_arr[2]);
	vx_printx(div_arr[3]);
}

unsigned wsapwn_arr[4];

void simple_kernel() {
	unsigned wid = vx_warp_id();

	wsapwn_arr[wid] = wid;

	vx_tmc(0 == wid);
}

void test_wsapwn() {
	vx_wspawn(4, (unsigned)simple_kernel);
	simple_kernel();
	vx_printx(wsapwn_arr[0]);
	vx_printx(wsapwn_arr[1]);
	vx_printx(wsapwn_arr[2]);
	vx_printx(wsapwn_arr[3]);
}