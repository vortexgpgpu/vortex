
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"


// #include <utlist.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

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

void vx_print_mat(unsigned * matPtr, int numRows, int numCols)
{
	vx_print_str("---------------------\n");
	for (int i = 0; i < numRows; i++)
	{
		for (int j = 0; j < numCols; j++)
		{
			unsigned index = (i * numCols) + j;
			vx_print_hex(matPtr[index]);
			vx_print_str(" ");
		}
		vx_print_str("\n");
	}
}


int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);

	// void * hellp = malloc(4);
	vx_print_str("Dev Main\n");

	vx_print_str("vx_spawnWarps\n");

	mat_add_args_t arguments;
	arguments.x         = x;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;


	int numWarps   = 4;
	int numThreads = 4;

	// First kernel call
	vx_spawnWarps(numWarps, numThreads, mat_add_kernel, &arguments);
	vx_print_mat(z, arguments.numRows, arguments.numColums);


	arguments.x         = z;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;

	// Second Kernel Call
	vx_spawnWarps(numWarps, numThreads, mat_add_kernel, &arguments);
	vx_print_mat(z, arguments.numRows, arguments.numColums);


	return 0;
}





