
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

void print_matrix(unsigned * z)
{
	vx_print_str("-------------------------------\n");
	for (int j = 0; j < (MAT_DIM * MAT_DIM); j++)
	{
		if (j!=0) if ((j % MAT_DIM) == 0) vx_print_str("\n");
		vx_print_hex(z[j]);
		vx_print_str(" ");
	}
	vx_print_str("\n-------------------------------\n");
}

int main()
{

	initialize_mats();

	// matrix multiplication
	vx_sq_mat_mult(x, y, z, MAT_DIM);
	vx_print_str("\n\nMatrix multiplication\n");
	print_matrix(z);


	// // matrix addition
	// vx_mat_add(x, y, z, NUM_ROWS, NUM_COLS);
	// vx_print_str("\n\nMatrix Addition\n");
	// print_matrix(z);


	// // matrix sub
	// vx_mat_sub(x, y, z, NUM_ROWS, NUM_COLS);
	// vx_print_str("\n\nMatrix Subtraction\n");
	// print_matrix(z);

	// unsigned scal = 3;

	// // matrix element add
	// vx_e_mat_add(z, &scal, z, NUM_ROWS, NUM_COLS);
	// vx_print_str("\n\nMatrix Element Addition\n");
	// print_matrix(z);

	// // matrix element add
	// vx_e_mat_mult(z, &scal, z, NUM_ROWS, NUM_COLS);
	// vx_print_str("\n\nMatrix Element Addition\n");
	// print_matrix(z);


	return 0;
}