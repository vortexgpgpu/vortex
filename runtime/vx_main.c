
#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"
#include "tests/tests.h"
#include "vx_api/vx_api.h"

typedef struct
{
	unsigned * x;
	unsigned * y;
	unsigned * z;
	unsigned   numColums;
	unsigned   numRows;
} mat_add_args_t;


unsigned x[] = {1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1};

unsigned y[] = {6, 6, 6, 6,
                6, 6, 6, 6,
                6, 6, 6, 6,
                6, 6, 6, 6};

unsigned z[] = {0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0};

void mat_add_kernel(void * void_arguments)
{
	mat_add_args_t * arguments = (mat_add_args_t *) void_arguments;

	unsigned wid = vx_warpID();
	unsigned tid = vx_threadID();

	bool valid = (wid < arguments->numRows) && (tid < arguments->numColums);

	__if (valid)
	{
		unsigned index = (wid * arguments->numColums) + tid;
		arguments->z[index] = arguments->x[index] + arguments->y[index];
	}
	__endif
}


int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);
	///////////////////////////////////////////////////////////////////////

	// mat_add_args_t arguments;
	// arguments.x         = x;
	// arguments.y         = y;
	// arguments.z         = z;
	// arguments.numColums = 4;
	// arguments.numRows   = 4;


	// int numWarps   = 4;
	// int numThreads = 4;

	// vx_spawnWarps(numWarps, numThreads, mat_add_kernel, &arguments);

	///////////////////////////////////////////////////////////////////////

	/*
		NOTE: * when test_wspawn is called from instrinsic_tests, RA 80000458 is stored at address 6fffefbc,
		      but when read back again it reads zeros even though no other write request is made to that
		      address (when only test_wsapwn is called by itself).

		      * When test_wsapwn is called by itself from main new lines are not printed....

		      * when test_wspawn is called with other tests from main it works fine...
	*/
	// intrinsics_tests(); 

	///////////////////////////////////////////////////////////////////////

	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// Test wspawn
	vx_print_str("test_wspawn\n");
	test_wsapwn();

	return 0;
}