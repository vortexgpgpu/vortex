// #include <stdint.h>
// #include <cstdint>
extern void  print_consol(char *);
extern void        printc(char);


int main(void);
void matAddition (unsigned, unsigned);

#include "./lib/lib.h"


// unsigned x[] = {2, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
// unsigned y[] = {1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

unsigned x[] = {1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1 , 1 , 1 , 1 , 1 , 1 };
unsigned y[] = {0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

// unsigned x[] = {2, 2,  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
// unsigned y[] = {1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
// unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

#define NUM_WARPS 2
#define NUM_THREADS 8

int main()
{

	for (int i = 0; i < 8; i++)
	{
		queue_initialize(q + i);
	}

	createWarps(NUM_WARPS, NUM_THREADS, matAddition, x, y, z);

	wait_for_done(NUM_WARPS);

	print_consol("-------------------------\n");
	print_consol("FINAL Z\n");
	for (int i = 0; i < 16; i++)
	{
		int_print(i);
		print_consol(": ");
		int_print(z[i]);
		print_consol("\n");
	}
	print_consol("-------------------------------\n");
	return 0;
}


void matAddition(unsigned tid, unsigned wid)
{

	unsigned * x_ptr = get_1st_arg();
	unsigned * y_ptr = get_2nd_arg();
	unsigned * z_ptr = get_3rd_arg();

	unsigned i = (wid * NUM_THREADS) + tid;

	__if((i < 11))
		z_ptr[i] = x_ptr[i] + y_ptr[i];
	__else
	__end_if

	return;
	
}