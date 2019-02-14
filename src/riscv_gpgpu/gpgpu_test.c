// #include <stdint.h>
// #include <stdbool.h>
// #include <cstdint>


int main(void);

#include "./lib/lib.h"


unsigned x[] = {1, 5, 10, 0, 3, 1, 1, 2, 8, 7, 8, 7, 5, 7, 7, 9};
unsigned y[] = {0, 2,  2, 0, 5, 0, 1, 1, 4, 2, 2, 0, 3, 2, 3, 2};
unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
unsigned N   = 16;

int main()
{

	initiate_stack();

	void mat ();

	createWarps(2, 8, mat);

	return 0;

}


void mat(unsigned warp_id)
{

	unsigned tid   = get_tid();
	unsigned index = (warp_id * 8) + tid;
	asm __volatile("nop");
	asm __volatile("nop");
	asm __volatile("nop");
	asm __volatile("nop");
	z[index] = x[index] + y[index];
}