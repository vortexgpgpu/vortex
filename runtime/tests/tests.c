

#include "tests.h"
#include "../intrinsics/vx_intrinsics.h"
#include "../io/vx_io.h"

void test_tmc()
{
	vx_print_str("test_tmc\n");

	vx_tmc(4);

	unsigned tid = vx_threadID(); // Get TID
	arr[tid] = tid;

	vx_tmc(1);

	vx_print_hex(arr[0]);
	vx_print_str("\n");
	vx_print_hex(arr[1]);
	vx_print_str("\n");
	vx_print_hex(arr[2]);
	vx_print_str("\n");
	vx_print_hex(arr[3]);
	vx_print_str("\n");

	return;
}

void test_divergence()
{
	unsigned tid = vx_threadID(); // Get TID

	bool b = tid < 2;
	__if (b)
	{
		bool c = tid < 1;
		__if (c)
		{
			arr[tid] = 10;
		}
		__else
		{
			arr[tid] = 11;
		}
		__endif
	}
	__else
	{
		bool c = tid < 3;
		__if (c)
		{
			arr[tid] = 12;
		}
		__else
		{
			arr[tid] = 13;
		}
		__endif
	}
	__endif

	vx_print_hex(arr[0]);
	vx_print_str("\n");
	vx_print_hex(arr[1]);
	vx_print_str("\n");
	vx_print_hex(arr[2]);
	vx_print_str("\n");
	vx_print_hex(arr[3]);
	vx_print_str("\n");

}


unsigned wsapwn_arr[4];


void simple_kernel()
{
	unsigned wid = vx_warpID();

	wsapwn_arr[wid] = wid;

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
	// // TMC test
	// test_tmc();

	// // Control Divergence Test
	// vx_print_str("test_divergence\n");
	// vx_tmc(4);
	// test_divergence();
	// vx_tmc(1);


	// Test wspawn
	vx_print_str("test_spawn\n");
	test_wsapwn();
}












