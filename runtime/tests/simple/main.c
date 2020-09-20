#include "tests.h"
#include <stdbool.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include <VX_config.h>

typedef struct {
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

void mat_add_kernel(void * void_arguments) {
	mat_add_args_t * arguments = (mat_add_args_t *) void_arguments;

	unsigned wid = vx_warp_id();
	unsigned tid = vx_thread_id();

	bool valid = (wid < arguments->numRows) && (tid < arguments->numColums);

	// __if (valid)
	// {
		unsigned index = (wid * arguments->numColums) + tid;
		unsigned val = arguments->x[index] + arguments->y[index];
		arguments->z[index] = val;
	// }
	// __endif
}

int main() {
	vx_printf("Let's start... (This might take a while)\n");

	unsigned what[36];
	bool passed = true;

	for (int i = 0; i < 36; i++) {
		what[i] = i;
		if (what[i] != i) {
			passed = false;
			vx_printf("T1 Fail On %d", i);
		}
	}

	for (int i = 0; i < 36; i++) {
		if (what[i] != i)	{
			passed = false;
			vx_printf("T2 Fail on %d", i);
		}
	}

	if (passed)	{
		vx_printf("Wr->read and repeat(Wr) tests passed!\n");
	}

	vx_printf("Simple Main\n");

	// TMC test
	test_tmc();

	// Control Divergence Test
	vx_printf("test_divergence\n");	
	test_divergence();

	// Test wspawn
	vx_printf("test_wspawn\n");
	test_wsapwn();

	vx_printf("Shared Memory test\n");
	unsigned * ptr = (unsigned *) SHARED_MEM_BASE_ADDR;
	unsigned value = 0;

	for (int i = 0; i < 5; i++)	{
		*ptr = value;
		unsigned read_valud = *ptr;
		vx_printf("ptr: %p\n", ptr);
		vx_printf("Original Value: %x\n", value);
		vx_printf("Read Value: %x\n", read_valud);
		vx_printf("-------------------\n");
		value++;
		ptr++;
	}

	vx_printf("vx_spawn_warps mat_add_kernel\n");

	mat_add_args_t arguments;
	arguments.x         = x;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;

	int numWarps   = 4;
	int numThreads = 4;

	vx_spawn_warps(numWarps, numThreads, mat_add_kernel, &arguments);

	vx_printf("Waiting to ensure other warps are done... (Takes a while)\n");
	for (int i = 0; i < 5000; i++) {}

	for (int i = 0; i < numWarps; i++) {
		for (int j = 0; j < numThreads; j++) {
			unsigned index = (i * arguments.numColums) + j;
			vx_printf("0x%x ", z[index]);
		}
		vx_printf("\n");
	}
	vx_print_str("Passed!\n");
	return 0;
}