// #include <stdint.h>
// #include <cstdint>
extern void  print_consol(char *);
extern void        printc(char);


int main(void);
void matMult (unsigned, unsigned);

#include "./lib/lib.h"


// unsigned x[] = {1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1 , 1 , 1 , 1 , 1 , 1 };
// unsigned y[] = {0, 1,  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
// unsigned z[] = {0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};



unsigned x[256] = {0};

unsigned y[256] = {0};

unsigned z[256] = {0};

#define MAT_DIM 16
#define MAX_THREADS 8

#define NUM_WARPS MAT_DIM
#define NUM_THREADS MAX_THREADS

typedef struct
{
	unsigned * x;
	unsigned * y;
	unsigned * z;
	unsigned mat_dim;
	unsigned offset;
	
} matMult_arg_t;

matMult_arg_t args;

int main()
{

	for (int i = 0; i < 8; i++)
	{
		queue_initialize(q + i);
	}

	for (int i = 0; i < (MAT_DIM * MAT_DIM); i++)
	{
		x[i] = 3;
		y[i] = 2;
	}

	
	args.x = x;
	args.y = y;
	args.z = z;
	args.mat_dim = MAT_DIM;
	args.offset = (MAT_DIM/MAX_THREADS);

	createWarps(NUM_WARPS, NUM_THREADS, matMult, (void *) (&args));

	wait_for_done(8);

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


void matMult(unsigned tid, unsigned wid)
{
	matMult_arg_t * args = (matMult_arg_t *) get_1st_arg();

	unsigned * x_ptr = args->x;
	unsigned * y_ptr = args->y;
	unsigned * z_ptr = args->z;

	unsigned off = args->offset;

	unsigned i_index = off * tid;
	unsigned mat_dim = args->mat_dim;

	for (int iter = 0; iter < off; ++iter)
	{
		unsigned total = 0;
		for (unsigned place = 0; place < mat_dim; ++place)
		{
			unsigned x_i = (wid * mat_dim)   + place;
			unsigned y_i = (mat_dim * place) + i_index;

			total += (x_ptr[x_i] * y_ptr[y_i]);
		}

		int final_i = (wid * mat_dim) + i_index;
		z_ptr[final_i] = total;
		i_index++;
	}


	return;
	
}