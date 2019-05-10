
#include "vx_front.h"
#include "../vx_os/vx_back/vx_back.h"

// -------------------------- Matrix Multiplication --------------------------

static mat_mult_arg_t mat_mult_args;

void _vx_mat_mult(unsigned, unsigned);
void vx_sq_mat_mult(void * x, void * y, void * z, unsigned mat_dim)
{
	mat_mult_args.x = x;
	mat_mult_args.y = y;
	mat_mult_args.z = z;
	mat_mult_args.mat_dim = mat_dim;

	unsigned num_avail_threads = vx_available_threads();

	unsigned off = (mat_dim/num_avail_threads);

	if ((mat_dim%num_avail_threads) != 0)
	{
		off += 1;
	}

	vx_printf("Offset: ", off);


	mat_mult_args.offset = off;

	if (mat_dim >= num_avail_threads)
	{
		vx_spawnWarps(mat_dim, num_avail_threads, _vx_mat_mult, (void *) (&mat_mult_args));
	}
	else
	{
		vx_spawnWarps(mat_dim, mat_dim, _vx_mat_mult, (void *) (&mat_mult_args));
	}

	unsigned num_avail_warps = vx_available_warps();

	if (mat_dim > num_avail_warps)
	{
		vx_wait_for_warps(num_avail_warps);
	}
	else
	{
		vx_wait_for_warps(mat_dim);
	}
}

void _vx_mat_mult(unsigned tid, unsigned wid)
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
			unsigned x_i = (wid     * mat_dim)   + place;
			unsigned y_i = (mat_dim * place  )   + i_index;

			total += (x_ptr[x_i] * y_ptr[y_i]);
		}

			int final_i = (wid * mat_dim) + i_index;
			// unsigned cond = i_index < mat_dim;
			// __if(cond)
			// {
				z_ptr[final_i] = total;
				i_index++;
			// }
			// __else
			// __end_if
	}

	// for (int z = 0; z < ((1000 * wid) + 1000); z++);
	return;
}




static mat_r_arg_t mat_r_args;
// -------------------------- Matrix Addition --------------------------
void _vx_mat_add(unsigned, unsigned);
void vx_mat_add(void * x, void * y, void * z, unsigned num_rows, unsigned num_cols)
{
	mat_r_args.x        = x;
	mat_r_args.y        = y;
	mat_r_args.z        = z;
	mat_r_args.num_cols = num_cols;
	mat_r_args.num_rows = num_rows;


	unsigned num_avail_threads = vx_available_threads();

	unsigned off = (num_cols/num_avail_threads);

	if ((num_cols%num_avail_threads) != 0)
	{
		off += 1;
	}


	mat_r_args.offset = off;

	if (num_cols >= num_avail_threads)
	{
		vx_spawnWarps(num_rows, num_avail_threads, _vx_mat_add, (void *) (&mat_r_args));
	}
	else
	{
		vx_spawnWarps(num_rows, num_cols, _vx_mat_add, (void *) (&mat_r_args));
	}

	unsigned num_avail_warps = vx_available_warps();

	if (num_rows > num_avail_warps)
	{
		vx_wait_for_warps(num_avail_warps);
	}
	else
	{
		vx_wait_for_warps(num_rows);
	}
}

void _vx_mat_add(unsigned tid, unsigned wid)
{
	// vx_print_str("*");
	// for (int z = 0; z < ((wid * 1000) + 1000); z++);

	mat_r_arg_t * args = (mat_r_arg_t *) vx_get_arg_struct();

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

	unsigned num_cols = args->num_cols;

	for (int iter = 0; iter < off; ++iter)
	{
		int final_i = (wid * num_cols) + i_index;
		unsigned cond = i_index < num_cols;
		__if(cond)
		{
			z_ptr[final_i] = x_ptr[final_i] + y_ptr[final_i];
			i_index++;
		}
		__else
		__end_if
	}
	return;
	
}



// -------------------------- Matrix Subtraction --------------------------
void _vx_mat_sub(unsigned, unsigned);
void vx_mat_sub(void * x, void * y, void * z, unsigned num_rows, unsigned num_cols)
{
	mat_r_args.x        = x;
	mat_r_args.y        = y;
	mat_r_args.z        = z;
	mat_r_args.num_cols = num_cols;
	mat_r_args.num_rows = num_rows;

	unsigned num_avail_threads = vx_available_threads();

	unsigned off = (num_cols/num_avail_threads);

	if ((num_cols%num_avail_threads) != 0)
	{
		off += 1;
	}


	mat_r_args.offset = off;

	if (num_cols >= num_avail_threads)
	{
		vx_spawnWarps(num_rows, num_avail_threads, _vx_mat_sub, (void *) (&mat_r_args));
	}
	else
	{
		vx_spawnWarps(num_rows, num_cols, _vx_mat_sub, (void *) (&mat_r_args));
	}

	unsigned num_avail_warps = vx_available_warps();

	if (num_rows > num_avail_warps)
	{
		vx_wait_for_warps(num_avail_warps);
	}
	else
	{
		vx_wait_for_warps(num_rows);
	}
}

void _vx_mat_sub(unsigned tid, unsigned wid)
{
	// vx_print_str("*");
	// for (int z = 0; z < ((wid * 1000) + 1000); z++);

	mat_r_arg_t * args = (mat_r_arg_t *) vx_get_arg_struct();

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

	unsigned num_cols = args->num_cols;

	for (int iter = 0; iter < off; ++iter)
	{
		int final_i = (wid * num_cols) + i_index;
		unsigned cond = i_index < num_cols;
		__if(cond)
		{
			z_ptr[final_i] = x_ptr[final_i] - y_ptr[final_i];
			i_index++;
		}
		__else
		__end_if
	}
	return;
	
}



static mat_e_arg_t mat_e_args;
// --------------------------------------------------------------

void _vx_e_mat_add(unsigned, unsigned);
void vx_e_mat_add(void * x, void * scal, void * z, unsigned num_rows, unsigned num_cols)
{
	mat_e_args.x        = x;
	mat_e_args.scal     = scal;
	mat_e_args.z        = z;
	mat_e_args.num_cols = num_cols;
	mat_e_args.num_rows = num_rows;


	unsigned num_avail_threads = vx_available_threads();

	unsigned off = (num_cols/num_avail_threads);

	if ((num_cols%num_avail_threads) != 0)
	{
		off += 1;
	}

	mat_e_args.offset = off;

	if (num_cols >= num_avail_threads)
	{
		vx_spawnWarps(num_rows, num_avail_threads, _vx_e_mat_add, (void *) (&mat_e_args));
	}
	else
	{
		vx_spawnWarps(num_rows, num_cols, _vx_e_mat_add, (void *) (&mat_e_args));
	}

	unsigned num_avail_warps = vx_available_warps();

	if (num_rows > num_avail_warps)
	{
		vx_wait_for_warps(num_avail_warps);
	}
	else
	{
		vx_wait_for_warps(num_rows);
	}
}

void _vx_e_mat_add(unsigned tid, unsigned wid)
{
	// vx_print_str("*");
	// for (int z = 0; z < ((wid * 1000) + 1000); z++);

	mat_e_arg_t * args = (mat_e_arg_t *) vx_get_arg_struct();

	unsigned * x_ptr = args->x;
	unsigned scal    = *((unsigned *) args->scal);

	unsigned * z_ptr = args->z;

	unsigned off = args->offset;

	unsigned i_index = off * tid;

	if (off == 0)
	{
		off = 1;
		i_index = tid;
	}

	unsigned num_cols = args->num_cols;

	for (int iter = 0; iter < off; ++iter)
	{
		int final_i = (wid * num_cols) + i_index;
		unsigned cond = i_index < num_cols;
		__if(cond)
		{
			z_ptr[final_i] = x_ptr[final_i] + scal;
			i_index++;
		}
		__else
		__end_if
	}
	return;
	
}

void _vx_e_mat_mult(unsigned, unsigned);
void vx_e_mat_mult(void * x, void * scal, void * z, unsigned num_rows, unsigned num_cols)
{
	mat_e_args.x        = x;
	mat_e_args.scal     = scal;
	mat_e_args.z        = z;
	mat_e_args.num_cols = num_cols;
	mat_e_args.num_rows = num_rows;


	unsigned num_avail_threads = vx_available_threads();

	unsigned off = (num_cols/num_avail_threads);

	if ((num_cols%num_avail_threads) != 0)
	{
		off += 1;
	}

	mat_e_args.offset = off;

	if (num_cols >= num_avail_threads)
	{
		vx_spawnWarps(num_rows, num_avail_threads, _vx_e_mat_mult, (void *) (&mat_e_args));
	}
	else
	{
		vx_spawnWarps(num_rows, num_cols, _vx_e_mat_mult, (void *) (&mat_e_args));
	}

	unsigned num_avail_warps = vx_available_warps();

	if (num_rows > num_avail_warps)
	{
		vx_wait_for_warps(num_avail_warps);
	}
	else
	{
		vx_wait_for_warps(num_rows);
	}
}

void _vx_e_mat_mult(unsigned tid, unsigned wid)
{
	// vx_print_str("*");
	// for (int z = 0; z < ((wid * 1000) + 1000); z++);

	mat_e_arg_t * args = (mat_e_arg_t *) vx_get_arg_struct();

	unsigned * x_ptr = args->x;
	unsigned scal    = *((unsigned *) args->scal);

	unsigned * z_ptr = args->z;

	unsigned off = args->offset;

	unsigned i_index = off * tid;

	if (off == 0)
	{
		off = 1;
		i_index = tid;
	}

	unsigned num_cols = args->num_cols;

	for (int iter = 0; iter < off; ++iter)
	{
		int final_i = (wid * num_cols) + i_index;
		unsigned cond = i_index < num_cols;
		__if(cond)
		{
			z_ptr[final_i] = x_ptr[final_i] * scal;
			i_index++;
		}
		__else
		__end_if
	}
	return;
	
}



