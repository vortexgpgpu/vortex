// #include <stdint.h>
// #include <stdbool.h>
// #include <cstdint>


int main(void);
void matAddition ();

#include "./lib/lib.h"


unsigned x[] = {1, 1,  6, 0, 3, 1, 1, 2, 0, 3, 6, 7, 5, 7, 7, 9};
unsigned y[] = {0, 2,  2, 0, 5, 0, 1, 1, 4, 2, 0, 0, 3, 2, 3, 2};
unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define NUM_WARPS 2
#define NUM_THREADS 8

int main()
{

	initiate_stack();

	createWarps(NUM_WARPS, NUM_THREADS, matAddition, x, y, z);

	return 0;
}


void matAddition(unsigned tid)
{
	unsigned wid     = get_wid();
	unsigned * x_ptr = get_1st_arg();
	unsigned * y_ptr = get_2nd_arg();
	unsigned * z_ptr = get_3rd_arg();

	unsigned i = (wid * 8) + tid;

	z_ptr[i] = x_ptr[i] + y_ptr[i];
}