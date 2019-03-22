#include "../vx_os/vx_back/vx_back.h"
#include "../vx_os/vx_io/vx_io.h"


// -------------------------- Matrix Multiplication --------------------------

typedef struct
{
	void * x;
	void * y;
	void * z;
	unsigned mat_dim;
	unsigned offset;
	
} mat_mult_arg_t;
void vx_sq_mat_mult(void *, void *, void *, unsigned);


// --------------------------------------------------------------------------

typedef struct
{
	void * x;
	void * y;
	void * z;
	unsigned num_cols;
	unsigned num_rows;
	unsigned offset;
	
} mat_r_arg_t;
// -------------------------- Matrix Addition -----------------------------
void vx_mat_add(void *, void *, void *, unsigned, unsigned);

// -------------------------- Matrix Subtraction --------------------------
void vx_mat_sub(void *, void *, void *, unsigned, unsigned);



// -----------------------------------------------------------------------
typedef struct
{
	void * x;
	void * scal;
	void * z;
	unsigned num_cols;
	unsigned num_rows;
	unsigned offset;
	
} mat_e_arg_t;

// -------------------------- Matrix element Addition ------------------
void vx_e_mat_add(void *, void *, void *, unsigned, unsigned);

// -------------------------- Matrix element Addition ------------------
void vx_e_mat_mult(void *, void *, void *, unsigned, unsigned);