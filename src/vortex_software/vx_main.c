
#include "./vx_include/vx_front.h"

unsigned x[1024] = {0};
unsigned y[1024] = {0};
unsigned z[1024] = {0};

#define MAT_DIM 16

#define NUM_COLS 16
#define NUM_ROWS 16

void initialize_mats()
{
	for (int i = 0; i < (MAT_DIM * MAT_DIM); i++)
	{
		x[i] = 3;
		y[i] = 2;
	}
}

int main()
{

	initialize_mats();

	// matrix multiplication
	vx_sq_mat_mult(x, y, z, MAT_DIM);
	vx_print_str("-------------------------\n");
	vx_print_str("FINAL MAT\n");

	for (int j = 0; j < (MAT_DIM * MAT_DIM); j++)
	{
		if ((j % MAT_DIM) == 0) vx_print_str("\n");
		vx_print_hex(z[j]);
		vx_print_str(" ");
	}
	vx_print_str("\n-------------------------------\n");


	// matrix addition
	vx_mat_add(x, y, z, NUM_ROWS, NUM_COLS);
	vx_print_str("-------------------------\n");
	vx_print_str("FINAL ADD\n");

	for (int j = 0; j < (NUM_COLS * NUM_ROWS); j++)
	{
		if ((j % NUM_COLS) == 0) vx_print_str("\n");
		vx_print_hex(z[j]);
		vx_print_str(" ");
	}
	vx_print_str("\n-------------------------------\n");


	// matrix sub
	vx_mat_sub(x, y, z, NUM_ROWS, NUM_COLS);
	vx_print_str("-------------------------\n");
	vx_print_str("FINAL Sub\n");

	for (int j = 0; j < (NUM_COLS * NUM_ROWS); j++)
	{
		if ((j % NUM_COLS) == 0) vx_print_str("\n");
		vx_print_hex(z[j]);
		vx_print_str(" ");
	}
	vx_print_str("\n-------------------------------\n");


	return 0;
}