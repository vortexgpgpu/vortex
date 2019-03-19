
#include "vx_front.h"

// -------------------------- Matrix Multiplication --------------------------

static mat_mult_arg_t args;
void _vx_matMult(unsigned, unsigned);

void vx_sq_mat_mult(void * x, void * y, void * z, unsigned mat_dim)
{
	args.x = x;
	args.y = y;
	args.z = z;
	args.mat_dim = mat_dim;

	unsigned off = (mat_dim/MAX_THREADS);

	if ((mat_dim%MAX_THREADS) != 0)
	{
		off += 1;
	}


	args.offset = off;

	vx_print_str("offset: ");
	vx_print_hex(off);
	vx_print_str("\n");

	if (mat_dim >= 8)
	{
		vx_spawnWarps(mat_dim, MAX_THREADS, _vx_matMult, (void *) (&args));
	}
	else
	{
		vx_spawnWarps(mat_dim, mat_dim, _vx_matMult, (void *) (&args));
	}

	if (mat_dim > 7)
	{
		vx_wait_for_warps(MAX_WARPS);
	}
	else
	{
		vx_wait_for_warps(mat_dim);
	}
}

void _vx_matMult(unsigned tid, unsigned wid)
{
	mat_mult_arg_t * args = (mat_mult_arg_t *) vx_get_arg_struct();

	unsigned * x_ptr = args->x;
	unsigned * y_ptr = args->y;
	unsigned * z_ptr = args->z;

	unsigned off = args->offset;

	unsigned i_index = off * tid;

	if (off == 0)
	{
		off = 1;
		i_index = tid;
	}

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
			unsigned cond = i_index < mat_dim;
			__if(cond)
			{
				z_ptr[final_i] = total;
				i_index++;
			}
			__else
			__end_if
	}


	return;
	
}