
#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"

int arr[4];


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

int main()
{
	vx_tmc(1);

	// TMC test
	test_tmc();

	// Control Divergence Test
	vx_print_str("2new test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	return 0;
}