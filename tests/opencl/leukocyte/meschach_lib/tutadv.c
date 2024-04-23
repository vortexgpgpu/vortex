
/* routines from the section 8 of tutorial.txt */

#include "matrix.h"

#define M3D_LIST    3      /* list number */
#define TYPE_MAT3D  0      /* the number of a type */

/* type for 3 dimensional matrices */
typedef struct {
	int l,m,n;    /* actual dimensions */
	int max_l, max_m, max_n;    /* maximal dimensions */
	Real ***me;    /* pointer to matrix elements */
	               /* we do not consider segmented memory */
        Real *base, **me2d;  /* me and me2d are additional pointers 
				to base */
} MAT3D;


/* function for creating a variable of MAT3D type */

MAT3D *m3d_get(l,m,n)
int l,m,n;
{
  MAT3D *mat;
  int i,j,k;

  /* check if arguments are positive */
  if (l <= 0 || m <= 0 || n <= 0)
    error(E_NEG,"m3d_get");

	/* new structure */
  if ((mat = NEW(MAT3D)) == (MAT3D *)NULL)
    error(E_MEM,"m3d_get");
  else if (mem_info_is_on()) {
	/* record how many bytes is allocated */
    mem_bytes_list(TYPE_MAT3D,0,sizeof(MAT3D),M3D_LIST);
	/* record a new allocated variable */
    mem_numvar_list(TYPE_MAT3D,1,M3D_LIST);
  }

  mat->l = mat->max_l = l;
  mat->m = mat->max_m = m;
  mat->n = mat->max_n = n;

	/* allocate memory for 3D array */
  if ((mat->base = NEW_A(l*m*n,Real)) == (Real *)NULL) 
    error(E_MEM,"m3d_get");
  else if (mem_info_is_on())
    mem_bytes_list(TYPE_MAT3D,0,l*m*n*sizeof(Real),M3D_LIST);

	/* allocate memory for 2D pointers */
  if ((mat->me2d = NEW_A(l*m,Real *)) == (Real **)NULL)
    error(E_MEM,"m3d_get");
  else if (mem_info_is_on())
    mem_bytes_list(TYPE_MAT3D,0,l*m*sizeof(Real *),M3D_LIST);  	

	/* allocate  memory for 1D pointers */
  if ((mat->me = NEW_A(l,Real **)) == (Real ***)NULL)
    error(E_MEM,"m3d_get");
  else if (mem_info_is_on())
    mem_bytes_list(TYPE_MAT3D,0,l*sizeof(Real **),M3D_LIST);

  	/* pointers to 2D matrices */
  for (i=0,k=0; i < l; i++)
    for (j=0; j < m; j++)
      mat->me2d[k++] = &mat->base[(i*m+j)*n];

       /* pointers to rows */
  for (i=0; i < l; i++)
    mat->me[i] = &mat->me2d[i*m];

  return mat;
}


/* deallocate a variable of type MAT3D */

int m3d_free(mat)
MAT3D *mat;
{
 	  /* do not try to deallocate the NULL pointer */
  if (mat == (MAT3D *)NULL)
    return -1;
	
	  /* first deallocate base */
  if (mat->base != (Real *)NULL) {
    if (mem_info_is_on())
	/* record how many bytes is deallocated */
      mem_bytes_list(TYPE_MAT3D,mat->max_l*mat->max_m*mat->max_n*sizeof(Real),
		     0,M3D_LIST);
    free((char *)mat->base);
  }

 	/* deallocate array of 2D pointers */
  if (mat->me2d != (Real **)NULL) {
    if (mem_info_is_on())
	/* record how many bytes is deallocated */
      mem_bytes_list(TYPE_MAT3D,mat->max_l*mat->max_m*sizeof(Real *),
		     0,M3D_LIST);
    free((char *)mat->me2d);
  }

 	/* deallocate array of 1D pointers */
  if (mat->me != (Real ***)NULL) {
    if (mem_info_is_on())
	/* record how many bytes is deallocated */
      mem_bytes_list(TYPE_MAT3D,mat->max_l*sizeof(Real **),0,M3D_LIST);
    free((char *)mat->me);
  }

	/* deallocate  MAT3D structure */
  if (mem_info_is_on()) {
    mem_bytes_list(TYPE_MAT3D,sizeof(MAT3D),0,M3D_LIST);
    mem_numvar_list(TYPE_MAT3D,-1,M3D_LIST);
  }
  free((char *)mat);

  return 0;
}

/*=============================================*/

char *m3d_names[] = {
  "MAT3D"
};


#define M3D_NUM  (sizeof(m3d_names)/sizeof(*m3d_names))

int (*m3d_free_funcs[M3D_NUM])() = {
  m3d_free
};

static MEM_ARRAY m3d_sum[M3D_NUM];


/* test routing for allocating/deallocating static variables */
void test_stat(k)
int k;
{
   static MAT3D *work;

   if (!work) {
      work = m3d_get(10,10,10);
      mem_stat_reg_list((void **)&work,TYPE_MAT3D,M3D_LIST);
      work->me[9][9][9] = -3.14;
   }
   
   if (k == 9) 
     printf(" work[9][9][9] = %g\n",work->me[9][9][9]);
}


void main()
{
  MAT3D *M;
  int i,j,k;

  mem_info_on(TRUE);
  /* can be the first command */
  mem_attach_list(M3D_LIST,M3D_NUM,m3d_names,m3d_free_funcs,m3d_sum);

  M = m3d_get(3,4,5);
  mem_info_file(stdout,M3D_LIST);

  /* make use of M->me[i][j][k], where i,j,k are non-negative and 
	i < 3, j < 4, k < 5 */

  mem_stat_mark(1);
  for (i=0; i < 3; i++)
    for (j=0; j < 4; j++)
      for (k=0; k < 5; k++) {
	 test_stat(i+j+k);
	 M->me[i][j][k] = i+j+k;
      }
  mem_stat_free_list(1,M3D_LIST);
  mem_info_file(stdout,M3D_LIST);

  printf(" M[%d][%d][%d] = %g\n",2,3,4,M->me[2][3][4]);

  mem_stat_mark(2);
  test_stat(9);
  mem_stat_free_list(2,M3D_LIST);

  m3d_free(M);  /* if M is not necessary */
  mem_info_file(stdout,M3D_LIST);

}



