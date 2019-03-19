#include "../vx_os/vx_back/vx_back.h"
#include "../vx_os/vx_io/vx_io.h"

#define MAX_WARPS 8
#define MAX_THREADS 8


// -------------------------- Matrix Multiplication --------------------------

typedef struct
{
	unsigned * x;
	unsigned * y;
	unsigned * z;
	unsigned mat_dim;
	unsigned offset;
	
} mat_mult_arg_t;
void vx_sq_mat_mult(void *, void *, void *, unsigned);

