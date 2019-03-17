// #include <stdint.h>
// #include <cstdint>
extern void  print_consol(char *);
extern void        printc(char);


int main(void);
void matAddition (unsigned, unsigned);

#include "./lib/lib.h"


// unsigned x[] = {1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1 , 1 , 1 , 1 , 1 , 1 };
// unsigned y[] = {0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
// unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};



unsigned x[] = {1,0,0,0,
				0,2,0,0,
				0,0,3,0,
				0,0,0,4};

unsigned y[] = {10,0,0,0,
				0,10,0,0,
				0,0,10,0,
				0,0,0,10};

unsigned z[] = {0,0,0,0,
				0,0,0,0,
				0,0,0,0,
				0,0,0,0};

#define MAT_DIM 4

#define NUM_WARPS MAT_DIM
#define NUM_THREADS MAT_DIM

int main()
{

	for (int i = 0; i < 8; i++)
	{
		queue_initialize(q + i);
	}

	createWarps(NUM_WARPS, NUM_THREADS, matAddition, (void *) x, (void *) y, (void *) z);

	wait_for_done(NUM_WARPS);

	print_consol("-------------------------\n");
	print_consol("FINAL Z\n");

	for (int j = 0; j < (MAT_DIM * MAT_DIM); j++)
	{
		if ((j % MAT_DIM) == 0) print_consol("\n");
		int_print(z[j]);
		print_consol(" ");
	}

	print_consol("\n-------------------------------\n");
	return 0;
}


void matAddition(unsigned tid, unsigned wid)
{

	unsigned * x_ptr = (unsigned *) get_1st_arg();
	unsigned * y_ptr = (unsigned *) get_2nd_arg();
	unsigned * z_ptr = (unsigned *) get_3rd_arg();


	unsigned total = 0;
	for (unsigned place = 0; place < MAT_DIM; place++)
	{
		unsigned x_i = (wid * MAT_DIM)   + place;
		unsigned y_i = (MAT_DIM * place) + tid;

		total += (x_ptr[x_i] * y_ptr[y_i]);
	}

	int final_i = (wid * MAT_DIM) + tid;
	z_ptr[final_i] = total;

	return;
	
}