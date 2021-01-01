#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>

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

void mat_add_kernel(int task_id, void * void_arguments)
{
	mat_add_args_t * arguments = (mat_add_args_t *) void_arguments;
	arguments->z[task_id] = arguments->x[task_id] + arguments->y[task_id];
}

void vx_print_mat(unsigned * matPtr, int numRows, int numCols)
{
	vx_printf("---------------------\n");
	for (int i = 0; i < numRows; i++)	{
		for (int j = 0; j < numCols; j++) {
			unsigned index = (i * numCols) + j;
			vx_printf("0x%x ", matPtr[index]);
		}
		vx_printf("\n");
	}
}

int main() {
	// void * hellp = malloc(4);
	vx_printf("Confirm Dev Main\n");

	vx_printf("vx_spawn_tasks\n");

	mat_add_args_t arguments;
	arguments.x         = x;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;

	// First kernel call
	vx_spawn_tasks(arguments.numRows * arguments.numColums, mat_add_kernel, &arguments);
	vx_print_mat(z, arguments.numRows, arguments.numColums);


	arguments.x         = z;
	arguments.y         = y;
	arguments.z         = z;
	arguments.numColums = 4;
	arguments.numRows   = 4;

	// Second Kernel Call
	vx_spawn_tasks(arguments.numRows * arguments.numColums, mat_add_kernel, &arguments);
	vx_print_mat(z, arguments.numRows, arguments.numColums);
	
	vx_prints("Passed!\n");

	return 0;
}





