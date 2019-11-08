
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"


typedef struct
{
	unsigned * x;
	unsigned * y;
	unsigned * z;
	unsigned   numColums;
	unsigned   numRows;
} mat_add_args_t;


unsigned x[] = {5, 5, 5, 5,
                6, 6, 6, 6,
                7, 7, 7, 7,
                8, 8, 8, 8};

unsigned y[] = {1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1};

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

	// vx_print_str("Simple Main\n");


	// // // TMC test
	// test_tmc();

	// // Control Divergence Test
	// vx_print_str("test_divergence\n");
	// vx_tmc(4);
	// test_divergence();
	// vx_tmc(1);


	// Test wspawn
	// vx_print_str("test_wspawn\n");
	test_wsapwn();

	// vx_print_str("vx_spawnWarps mat_add_kernel\n");

	// mat_add_args_t arguments;
	// arguments.x         = x;
	// arguments.y         = y;
	// arguments.z         = z;
	// arguments.numColums = 4;
	// arguments.numRows   = 4;


	// int numWarps   = 4;
	// int numThreads = 4;

	// vx_spawnWarps(numWarps, numThreads, mat_add_kernel, &arguments);

	// for (int i = 0; i < arguments.numRows; i++)
	// {
	// 	for (int j = 0; j < arguments.numColums; j++)
	// 	{
	// 		unsigned index = (i * arguments.numColums) + j;
	// 		vx_print_hex(z[index]);
	// 		vx_print_str(" ");
	// 	}
	// 	vx_print_str("\n");
	// }

	return 0;
}