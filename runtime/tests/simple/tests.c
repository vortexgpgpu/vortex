

#include "tests.h"
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"

int tmc_array[4] = {5,5,5,5};

void test_tmc()
{
	//vx_print_str("testing_tmc\n");

	vx_tmc(4);

	unsigned tid = vx_threadID(); // Get TID

	tmc_array[tid] = tid;

	vx_tmc(1);

	vx_print_hex(tmc_array[0]);
	vx_print_str("\n");
	vx_print_hex(tmc_array[1]);
	vx_print_str("\n");
	vx_print_hex(tmc_array[2]);
	vx_print_str("\n");
	vx_print_hex(tmc_array[3]);
	vx_print_str("\n");

	return;
}

int div_arr[4];

void test_divergence()
{
	unsigned tid = vx_threadID(); // Get TID

	bool b = tid < 2;
	__if (b)
	{
		bool c = tid < 1;
		__if (c)
		{
			div_arr[tid] = 10;
		}
		__else
		{
			div_arr[tid] = 11;
		}
		__endif
	}
	__else
	{
		bool c = tid < 3;
		__if (c)
		{
			div_arr[tid] = 12;
		}
		__else
		{
			div_arr[tid] = 13;
		}
		__endif
	}
	__endif

	vx_print_hex(div_arr[0]);
	vx_print_str("\n");
	vx_print_hex(div_arr[1]);
	vx_print_str("\n");
	vx_print_hex(div_arr[2]);
	vx_print_str("\n");
	vx_print_hex(div_arr[3]);
	vx_print_str("\n");

}


unsigned wsapwn_arr[4];


void simple_kernel()
{
	unsigned wid = vx_warpID();

	wsapwn_arr[wid] = wid;

	wid = vx_warpID();
	if (wid != 0)
	{
		vx_tmc(0);
	}

}

void test_wsapwn()
{
	unsigned func_ptr = (unsigned) simple_kernel;
	vx_wspawn(4, func_ptr);
	simple_kernel();

	for (int i = 0; i < 100; i++) {}

	vx_print_hex(wsapwn_arr[0]);
	vx_print_str("\n");
	vx_print_hex(wsapwn_arr[1]);
	vx_print_str("\n");
	vx_print_hex(wsapwn_arr[2]);
	vx_print_str("\n");
	vx_print_hex(wsapwn_arr[3]);
	vx_print_str("\n");
}

void intrinsics_tests()
{
	// TMC test
	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// Test wspawn
	vx_print_str("test_spawn\n");
	test_wsapwn();
}












