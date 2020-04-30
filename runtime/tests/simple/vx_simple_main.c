
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "tests.h"
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

	// __if (valid)
	// {
		unsigned index = (wid * arguments->numColums) + tid;
		unsigned val = arguments->x[index] + arguments->y[index];
		arguments->z[index] = val;
	// }
	// __endif
}

int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);

	vx_print_str("Let's start... (This might take a while)\n");
	unsigned what[36];
	bool passed = true;
	for (int i = 0; i < 36; i++)
	{
		what[i] = i;
		// vx_print_hex(i);
		// vx_printf(": ", what[i]);
		if (what[i] != i)
		{
			passed = false;
			vx_printf("T1 Fail On ", i);
		}
	}

	for (int i = 0; i < 36; i++)
	{
		// vx_print_hex(i);
		// vx_printf(": ", what[i]);
		if (what[i] != i)
		{
			passed = false;
			vx_printf("T2 Fail on ", i);
		}
	}

	if (passed)
	{
		vx_print_str("Wr->read and repeat(Wr) tests passed!\n");
	}


	vx_print_str("Simple Main\n");


	// // TMC test
	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// Test wspawn
	vx_print_str("test_wspawn\n");
	test_wsapwn();

	vx_print_str("Shared Memory test\n");
	unsigned * ptr = (unsigned *) 0xFFFF0000;
	unsigned value = 0;
	for (int i = 0; i < 5; i++)
	{
		*ptr = value;
		unsigned read_valud = *ptr;
		vx_printf("ptr: ", (unsigned) ptr);
		vx_printf("Original Value: ", value);
		vx_printf("Read Value: ", read_valud);
		vx_print_str("-------------------\n");
		value++;
		ptr++;

	}

	vx_print_str("vx_spawnWarps mat_add_kernel\n");

	mat_add_args_t arguments;
	arguments.x         = x;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;


	int numWarps   = 4;
	int numThreads = 4;

	vx_spawnWarps(numWarps, numThreads, mat_add_kernel, &arguments);

	vx_print_str("Waiting to ensure other warps are done... (Takes a while)\n");
	for (int i = 0; i < 5000; i++) {}

	for (int i = 0; i < numWarps; i++)
	{
		for (int j = 0; j < numThreads; j++)
		{
			unsigned index = (i * arguments.numColums) + j;
			vx_print_hex(z[index]);
			vx_print_str(" ");
		}
		vx_print_str("\n");
	}

	return 0;
}