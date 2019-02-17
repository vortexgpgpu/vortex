// #include <stdint.h>
// #include <stdbool.h>
// #include <cstdint>


int main(void);
void matAddition (unsigned, unsigned);

#include "./lib/lib.h"


unsigned x[] = {1, 1,  6, 0, 3, 1, 1, 2, 0, 3, 6, 7, 5, 7};
unsigned y[] = {0, 2,  2, 0, 5, 0, 1, 1, 4, 2, 0, 0, 3, 2};
unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// unsigned x[] = {1, 1,  6, 0, 3, 1, 1, 2, 0, 3, 6, 7, 5, 7, 7, 9};
// unsigned y[] = {0, 2,  2, 0, 5, 0, 1, 1, 4, 2, 0, 0, 3, 2, 3, 2};
// unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define NUM_WARPS 8
#define NUM_THREADS 2

int main()
{

	initiate_stack();
	queue_initialize();

	createWarps(NUM_WARPS, NUM_THREADS, matAddition, x, y, z);

	while(!queue_isEmpty()) {}

	ECALL;

	return 0;
}


void matAddition(unsigned tid, unsigned wid)
{
	unsigned * x_ptr = get_1st_arg();
	unsigned * y_ptr = get_2nd_arg();
	unsigned * z_ptr = get_3rd_arg();

	unsigned i = (wid * NUM_THREADS) + tid;

	z_ptr[i] = x_ptr[i] + y_ptr[i];
}