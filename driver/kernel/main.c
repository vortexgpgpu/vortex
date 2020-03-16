#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"
#include "vx_api/vx_api.h"

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

	vx_print_str("Demo kernel\n");

	mat_add_args_t arguments;
	arguments.x         = x;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;

	vx_spawnWarps(4, 4, mat_add_kernel, &arguments);

	vx_print_str("done.");

	return 0;
}