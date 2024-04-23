
/**************************************************************************
**
** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
**
**			     Meschach Library
** 
** This Meschach Library is provided "as is" without any express 
** or implied warranty of any kind with respect to this software. 
** In particular the authors shall not be liable for any direct, 
** indirect, special, incidental or consequential damages arising 
** in any way from use of the software.
** 
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.  
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
***************************************************************************/


/* iter.h  14/09/93 */

/* 

  Structures for iterative methods

*/

#ifndef ITERHH

#define ITERHH

/* RCS id: $Id: iter.h,v 1.2 1994/03/08 05:48:27 des Exp $  */


#include	"sparse.h"


/* basic structure for iterative methods */

/* type Fun_Ax for functions to get y = A*x */
#ifdef ANSI_C
typedef VEC  *(*Fun_Ax)(void *,VEC *,VEC *);
#else
typedef VEC *(*Fun_Ax)();
#endif


/* type ITER */
typedef struct Iter_data {
   int shared_x;   /* if TRUE then x is shared and it will not be free'd */ 
   int shared_b;   /* if TRUE then b is shared and it will not be free'd */
   unsigned k;   /* no. of direction (search) vectors; =0 - none */
   int limit;    /* upper bound on the no. of iter. steps */
   int steps;    /* no. of iter. steps done */
   Real eps;     /* accuracy required */
   
   VEC *x;       /* input: initial guess;
		    output: approximate solution */
   VEC *b;       /* right hand side of the equation A*x = b */

   Fun_Ax   Ax;		 /* function computing y = A*x */
   void *A_par;         /* parameters for Ax */

   Fun_Ax  ATx;		 /* function  computing y = A^T*x;
					       T = transpose */
   void *AT_par;         /* parameters for ATx */

   Fun_Ax  Bx; /* function computing y = B*x; B - preconditioner */
   void *B_par;         /* parameters for Bx */

   Fun_Ax  BTx; /* function computing y = B^T*x; B - preconditioner */
   void *BT_par;         /* parameters for BTx */

#ifdef ANSI_C

#ifdef PROTOTYPES_IN_STRUCT
   void (*info)(struct Iter_data *, double, VEC *,VEC *);
            /* function giving some information for a user;
	       nres - a norm of a residual res */
   
   int (*stop_crit)(struct Iter_data *, double, VEC *,VEC *);
           /* stopping criterion:
	      nres - a norm of res;
	      res - residual;
	    if returned value == TRUE then stop;
	    if returned value == FALSE then continue; */
#else
   void (*info)();
   int  (*stop_crit)();
#endif /* PROTOTYPES_IN_STRUCT */

#else

   void (*info)();
            /* function giving some information for a user */
   
   int (*stop_crit)();
           /* stopping criterion:
	    if returned value == TRUE then stop;
	    if returned value == FALSE then continue; */

#endif /* ANSI_C */

   Real init_res;   /* the norm of the initial residual */

}  ITER;


#define INULL   (ITER *)NULL

/* type Fun_info */
#ifdef ANSI_C
typedef void (*Fun_info)(ITER *, double, VEC *,VEC *);
#else
typedef void (*Fun_info)();
#endif

/* type Fun_stp_crt */
#ifdef ANSI_C
typedef int (*Fun_stp_crt)(ITER *, double, VEC *,VEC *);
#else
typedef int (*Fun_stp_crt)();
#endif



/* macros */
/* default values */

#define ITER_LIMIT_DEF  1000
#define ITER_EPS_DEF    1e-6

/* other macros */

/* set ip->Ax=fun and ip->A_par=fun_par */
#define iter_Ax(ip,fun,fun_par) \
  (ip->Ax=(Fun_Ax)(fun),ip->A_par=(void *)(fun_par),0)
#define iter_ATx(ip,fun,fun_par) \
  (ip->ATx=(Fun_Ax)(fun),ip->AT_par=(void *)(fun_par),0)
#define iter_Bx(ip,fun,fun_par) \
  (ip->Bx=(Fun_Ax)(fun),ip->B_par=(void *)(fun_par),0)
#define iter_BTx(ip,fun,fun_par) \
  (ip->BTx=(Fun_Ax)(fun),ip->BT_par=(void *)(fun_par),0)

/* save free macro */
#define ITER_FREE(ip)  (iter_free(ip), (ip)=(ITER *)NULL)


/* prototypes from iter0.c */

#ifdef ANSI_C
/* standard information */
void iter_std_info(const ITER *ip,double nres,VEC *res,VEC *Bres);
/* standard stopping criterion */
int iter_std_stop_crit(const ITER *ip, double nres, VEC *res,VEC *Bres);

/* get, resize and free ITER variable */
ITER *iter_get(int lenb, int lenx);
ITER *iter_resize(ITER *ip,int lenb,int lenx);
int iter_free(ITER *ip);

void iter_dump(FILE *fp,ITER *ip);

/* copy ip1 to ip2 copying also elements of x and b */
ITER *iter_copy(const ITER *ip1, ITER *ip2);
/* copy ip1 to ip2 without copying elements of x and b */
ITER *iter_copy2(ITER *ip1,ITER *ip2);

/* functions for generating sparse matrices with random elements */
SPMAT	*iter_gen_sym(int n, int nrow);
SPMAT	*iter_gen_nonsym(int m,int n,int nrow,double diag);
SPMAT	*iter_gen_nonsym_posdef(int n,int nrow);

#else

void iter_std_info();
int iter_std_stop_crit();
ITER *iter_get();
int iter_free();
ITER *iter_resize();
void iter_dump();
ITER *iter_copy();
ITER *iter_copy2();
SPMAT	*iter_gen_sym();
SPMAT	*iter_gen_nonsym();
SPMAT	*iter_gen_nonsym_posdef();

#endif

/* prototypes from iter.c */

/* different iterative procedures */
#ifdef ANSI_C
VEC  *iter_cg(ITER *ip);
VEC  *iter_cg1(ITER *ip);
VEC  *iter_spcg(SPMAT *A,SPMAT *LLT,VEC *b,double eps,VEC *x,int limit,
		int *steps);
VEC  *iter_cgs(ITER *ip,VEC *r0);
VEC  *iter_spcgs(SPMAT *A,SPMAT *B,VEC *b,VEC *r0,double eps,VEC *x,
		 int limit, int *steps);
VEC  *iter_lsqr(ITER *ip);
VEC  *iter_splsqr(SPMAT *A,VEC *b,double tol,VEC *x,
		  int limit,int *steps);
VEC  *iter_gmres(ITER *ip);
VEC  *iter_spgmres(SPMAT *A,SPMAT *B,VEC *b,double tol,VEC *x,int k,
		   int limit, int *steps);
MAT  *iter_arnoldi_iref(ITER *ip,Real *h,MAT *Q,MAT *H);
MAT  *iter_arnoldi(ITER *ip,Real *h,MAT *Q,MAT *H);
MAT  *iter_sparnoldi(SPMAT *A,VEC *x0,int k,Real *h,MAT *Q,MAT *H);
VEC  *iter_mgcr(ITER *ip);
VEC  *iter_spmgcr(SPMAT *A,SPMAT *B,VEC *b,double tol,VEC *x,int k,
		  int limit, int *steps);
void	iter_lanczos(ITER *ip,VEC *a,VEC *b,Real *beta2,MAT *Q);
void    iter_splanczos(SPMAT *A,int m,VEC *x0,VEC *a,VEC *b,Real *beta2,
		       MAT *Q);
VEC  *iter_lanczos2(ITER *ip,VEC *evals,VEC *err_est);
VEC  *iter_splanczos2(SPMAT *A,int m,VEC *x0,VEC *evals,VEC *err_est);
VEC  *iter_cgne(ITER *ip);
VEC  *iter_spcgne(SPMAT *A,SPMAT *B,VEC *b,double eps,VEC *x,
		  int limit,int *steps);
#else
VEC  *iter_cg();
VEC  *iter_cg1();
VEC  *iter_spcg();
VEC  *iter_cgs();
VEC  *iter_spcgs();
VEC  *iter_lsqr();
VEC  *iter_splsqr();
VEC  *iter_gmres();
VEC  *iter_spgmres();
MAT  *iter_arnoldi_iref();
MAT  *iter_arnoldi();
MAT  *iter_sparnoldi();
VEC  *iter_mgcr();
VEC  *iter_spmgcr();
void  iter_lanczos();
void  iter_splanczos();
VEC  *iter_lanczos2();
VEC  *iter_splanczos2();
VEC  *iter_cgne();
VEC  *iter_spcgne();

#endif


#endif  /* ITERHH */
