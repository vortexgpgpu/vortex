
/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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


/* itersym.c 17/09/93 */


/* 
  ITERATIVE METHODS - implementation of several iterative methods;
  see also iter0.c
  */

#include        <stdio.h>
#include	<math.h>
#include        "matrix.h"
#include        "matrix2.h"
#include	"sparse.h"
#include        "iter.h"

static char rcsid[] = "$Id: itersym.c,v 1.2 1995/01/30 14:55:54 des Exp $";


#ifdef ANSI_C
VEC	*spCHsolve(const SPMAT *,VEC *,VEC *);
VEC	*trieig(VEC *,VEC *,MAT *);
#else
VEC	*spCHsolve();
VEC	*trieig();
#endif



/* iter_spcg -- a simple interface to iter_cg() which uses sparse matrix
   data structures
   -- assumes that LLT contains the Cholesky factorisation of the
   actual preconditioner;
   use always as follows:
   x = iter_spcg(A,LLT,b,eps,x,limit,steps);
   or 
   x = iter_spcg(A,LLT,b,eps,VNULL,limit,steps);
   In the second case the solution vector is created.
   */
#ifndef ANSI_C
VEC  *iter_spcg(A,LLT,b,eps,x,limit,steps)
SPMAT	*A, *LLT;
VEC	*b, *x;
double	eps;
int *steps, limit;
#else
VEC  *iter_spcg(SPMAT *A, SPMAT *LLT, VEC *b, double eps, VEC *x,
		int limit, int *steps)
#endif
{	
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *)A;
   ip->Bx = (Fun_Ax) spCHsolve;
   ip->B_par = (void *)LLT;
   ip->info = (Fun_info) NULL;
   ip->b = b;
   ip->eps = eps;
   ip->limit = limit;
   ip->x = x;
   iter_cg(ip);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return x;		
}

/* 
  Conjugate gradients method;
  */
#ifndef ANSI_C
VEC  *iter_cg(ip)
ITER *ip;
#else
VEC  *iter_cg(ITER *ip)
#endif
{
   STATIC VEC *r = VNULL, *p = VNULL, *q = VNULL, *z = VNULL;
   Real	alpha, beta, inner, old_inner, nres;
   VEC *rr;   /* rr == r or rr == z */
   
   if (ip == INULL)
     error(E_NULL,"iter_cg");
   if (!ip->Ax || !ip->b)
     error(E_NULL,"iter_cg");
   if ( ip->x == ip->b )
     error(E_INSITU,"iter_cg");
   if (!ip->stop_crit)
     error(E_NULL,"iter_cg");
   
   if ( ip->eps <= 0.0 )
     ip->eps = MACHEPS;
   
   r = v_resize(r,ip->b->dim);
   p = v_resize(p,ip->b->dim);
   q = v_resize(q,ip->b->dim);
   
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(p,TYPE_VEC);
   MEM_STAT_REG(q,TYPE_VEC);
   
   if (ip->Bx != (Fun_Ax)NULL) {
      z = v_resize(z,ip->b->dim);
      MEM_STAT_REG(z,TYPE_VEC);
      rr = z;
   }
   else rr = r;
   
   if (ip->x != VNULL) {
      if (ip->x->dim != ip->b->dim)
	error(E_SIZES,"iter_cg");
      ip->Ax(ip->A_par,ip->x,p);    		/* p = A*x */
      v_sub(ip->b,p,r);		 		/* r = b - A*x */
   }
   else {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
      v_copy(ip->b,r);
   }
   
   old_inner = 0.0;
   for ( ip->steps = 0; ip->steps <= ip->limit; ip->steps++ )
   {
      if ( ip->Bx )
	(ip->Bx)(ip->B_par,r,rr);		/* rr = B*r */
      
      inner = in_prod(rr,r);
      nres = sqrt(fabs(inner));
      if (ip->info) ip->info(ip,nres,r,rr);
      if (ip->steps == 0) ip->init_res = nres;
      if ( ip->stop_crit(ip,nres,r,rr) ) break;
      
      if ( ip->steps )	/* if ( ip->steps > 0 ) ... */
      {
	 beta = inner/old_inner;
	 p = v_mltadd(rr,p,beta,p);
      }
      else		/* if ( ip->steps == 0 ) ... */
      {
	 beta = 0.0;
	 p = v_copy(rr,p);
	 old_inner = 0.0;
      }
      (ip->Ax)(ip->A_par,p,q);     /* q = A*p */
      alpha = in_prod(p,q);
      if (sqrt(fabs(alpha)) <= MACHEPS*ip->init_res) 
	error(E_BREAKDOWN,"iter_cg");
      alpha = inner/alpha;
      v_mltadd(ip->x,p,alpha,ip->x);
      v_mltadd(r,q,-alpha,r);
      old_inner = inner;
   }

#ifdef	THREADSAFE
   V_FREE(r);   V_FREE(p);   V_FREE(q);   V_FREE(z);
#endif

   return ip->x;
}



/* iter_lanczos -- raw lanczos algorithm -- no re-orthogonalisation
   -- creates T matrix of size == m,
   but no larger than before beta_k == 0
   -- uses passed routine to do matrix-vector multiplies */
#ifndef ANSI_C
void	iter_lanczos(ip,a,b,beta2,Q)
ITER    *ip;
VEC	*a, *b;
Real	*beta2;
MAT	*Q;
#else
void	iter_lanczos(ITER *ip, VEC *a, VEC *b, Real *beta2, MAT *Q)
#endif
{
   int	j;
   STATIC VEC	*v = VNULL, *w = VNULL, *tmp = VNULL;
   Real	alpha, beta, c;
   
   if ( ! ip )
     error(E_NULL,"iter_lanczos");
   if ( ! ip->Ax || ! ip->x || ! a || ! b )
     error(E_NULL,"iter_lanczos");
   if ( ip->k <= 0 )
     error(E_BOUNDS,"iter_lanczos");
   if ( Q && ( Q->n < ip->x->dim || Q->m < ip->k ) )
     error(E_SIZES,"iter_lanczos");
   
   a = v_resize(a,(unsigned int)ip->k);	
   b = v_resize(b,(unsigned int)(ip->k-1));
   v = v_resize(v,ip->x->dim);
   w = v_resize(w,ip->x->dim);
   tmp = v_resize(tmp,ip->x->dim);
   MEM_STAT_REG(v,TYPE_VEC);
   MEM_STAT_REG(w,TYPE_VEC);
   MEM_STAT_REG(tmp,TYPE_VEC);
   
   beta = 1.0;
   v_zero(a);
   v_zero(b);
   if (Q) m_zero(Q);
   
   /* normalise x as w */
   c = v_norm2(ip->x);
   if (c <= MACHEPS) { /* ip->x == 0 */
      *beta2 = 0.0;
      return;
   }
   else 
     sv_mlt(1.0/c,ip->x,w);
   
   (ip->Ax)(ip->A_par,w,v);
   
   for ( j = 0; j < ip->k; j++ )
   {
      /* store w in Q if Q not NULL */
      if ( Q ) set_row(Q,j,w);
      
      alpha = in_prod(w,v);
      a->ve[j] = alpha;
      v_mltadd(v,w,-alpha,v);
      beta = v_norm2(v);
      if ( beta == 0.0 )
      {
	 *beta2 = 0.0;
	 return;
      }
      
      if ( j < ip->k-1 )
	b->ve[j] = beta;
      v_copy(w,tmp);
      sv_mlt(1/beta,v,w);
      sv_mlt(-beta,tmp,v);
      (ip->Ax)(ip->A_par,w,tmp);
      v_add(v,tmp,v);
   }
   *beta2 = beta;

#ifdef	THREADSAFE
   V_FREE(v);   V_FREE(w);   V_FREE(tmp);
#endif
}

/* iter_splanczos -- version that uses sparse matrix data structure */
#ifndef ANSI_C
void    iter_splanczos(A,m,x0,a,b,beta2,Q)
SPMAT	*A;
int     m;
VEC     *x0, *a, *b;
Real    *beta2;
MAT     *Q;
#else
void    iter_splanczos(SPMAT *A, int m, VEC *x0, 
		       VEC *a, VEC *b, Real *beta2, MAT *Q)
#endif
{	
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->shared_x = ip->shared_b = TRUE;
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   ip->x = x0;
   ip->k = m;
   iter_lanczos(ip,a,b,beta2,Q);	
   iter_free(ip);   /* release only ITER structure */
}


#ifndef ANSI_C
extern	double	frexp(), ldexp();
#else
extern	double	frexp(double num, int *exponent),
  ldexp(double num, int exponent);
#endif

/* product -- returns the product of a long list of numbers
   -- answer stored in mant (mantissa) and expt (exponent) */
#ifndef ANSI_C
static	double	product(a,offset,expt)
VEC	*a;
double	offset;
int	*expt;
#else
static	double	product(VEC *a, double offset, int *expt)
#endif
{
   Real	mant, tmp_fctr;
   int	i, tmp_expt;
   
   if ( ! a )
     error(E_NULL,"product");
   
   mant = 1.0;
   *expt = 0;
   if ( offset == 0.0 )
     for ( i = 0; i < a->dim; i++ )
     {
	mant *= frexp(a->ve[i],&tmp_expt);
	*expt += tmp_expt;
	if ( ! (i % 10) )
	{
	   mant = frexp(mant,&tmp_expt);
	   *expt += tmp_expt;
	}
     }
   else
     for ( i = 0; i < a->dim; i++ )
     {
	tmp_fctr = a->ve[i] - offset;
	tmp_fctr += (tmp_fctr > 0.0 ) ? -MACHEPS*offset :
	  MACHEPS*offset;
	mant *= frexp(tmp_fctr,&tmp_expt);
	*expt += tmp_expt;
	if ( ! (i % 10) )
	{
	   mant = frexp(mant,&tmp_expt);
	   *expt += tmp_expt;
	}
     }
   
   mant = frexp(mant,&tmp_expt);
   *expt += tmp_expt;
   
   return mant;
}

/* product2 -- returns the product of a long list of numbers (except the k'th)
   -- answer stored in mant (mantissa) and expt (exponent) */
#ifndef ANSI_C
static	double	product2(a,k,expt)
VEC	*a;
int	k;	/* entry of a to leave out */
int	*expt;
#else
static	double	product2(VEC *a, int k, int *expt)
#endif
{
   Real	mant, mu, tmp_fctr;
   int	i, tmp_expt;
   
   if ( ! a )
     error(E_NULL,"product2");
   if ( k < 0 || k >= a->dim )
     error(E_BOUNDS,"product2");
   
   mant = 1.0;
   *expt = 0;
   mu = a->ve[k];
   for ( i = 0; i < a->dim; i++ )
   {
      if ( i == k )
	continue;
      tmp_fctr = a->ve[i] - mu;
      tmp_fctr += ( tmp_fctr > 0.0 ) ? -MACHEPS*mu : MACHEPS*mu;
      mant *= frexp(tmp_fctr,&tmp_expt);
      *expt += tmp_expt;
      if ( ! (i % 10) )
      {
	 mant = frexp(mant,&tmp_expt);
	 *expt += tmp_expt;
      }
   }
   mant = frexp(mant,&tmp_expt);
   *expt += tmp_expt;
   
   return mant;
}

/* dbl_cmp -- comparison function to pass to qsort() */
#ifndef ANSI_C
static	int	dbl_cmp(x,y)
Real	*x, *y;
#else
static	int	dbl_cmp(Real *x, Real *y)
#endif
{
   Real	tmp;
   
   tmp = *x - *y;
   return (tmp > 0 ? 1 : tmp < 0 ? -1: 0);
}

/* iter_lanczos2 -- lanczos + error estimate for every e-val
   -- uses Cullum & Willoughby approach, Sparse Matrix Proc. 1978
   -- returns multiple e-vals where multiple e-vals may not exist
   -- returns evals vector */
#ifndef ANSI_C
VEC	*iter_lanczos2(ip,evals,err_est)
ITER 	*ip;            /* ITER structure */
VEC	*evals;		/* eigenvalue vector */
VEC	*err_est;	/* error estimates of eigenvalues */
#else
VEC	*iter_lanczos2(ITER *ip, VEC *evals, VEC *err_est)
#endif
{
   VEC		*a;
   STATIC	VEC	*b=VNULL, *a2=VNULL, *b2=VNULL;
   Real	beta, pb_mant, det_mant, det_mant1, det_mant2;
   int	i, pb_expt, det_expt, det_expt1, det_expt2;
   
   if ( ! ip )
     error(E_NULL,"iter_lanczos2");
   if ( ! ip->Ax || ! ip->x )
     error(E_NULL,"iter_lanczos2");
   if ( ip->k <= 0 )
     error(E_RANGE,"iter_lanczos2");
   
   a = evals;
   a = v_resize(a,(unsigned int)ip->k);
   b = v_resize(b,(unsigned int)(ip->k-1));
   MEM_STAT_REG(b,TYPE_VEC);
   
   iter_lanczos(ip,a,b,&beta,MNULL);
   
   /* printf("# beta =%g\n",beta); */
   pb_mant = 0.0;
   if ( err_est )
   {
      pb_mant = product(b,(double)0.0,&pb_expt);
      /* printf("# pb_mant = %g, pb_expt = %d\n",pb_mant, pb_expt); */
   }
   
   /* printf("# diags =\n");	v_output(a); */
   /* printf("# off diags =\n");	v_output(b); */
   a2 = v_resize(a2,a->dim - 1);
   b2 = v_resize(b2,b->dim - 1);
   MEM_STAT_REG(a2,TYPE_VEC);
   MEM_STAT_REG(b2,TYPE_VEC);
   for ( i = 0; i < a2->dim - 1; i++ )
   {
      a2->ve[i] = a->ve[i+1];
      b2->ve[i] = b->ve[i+1];
   }
   a2->ve[a2->dim-1] = a->ve[a2->dim];
   
   trieig(a,b,MNULL);
   
   /* sort evals as a courtesy */
   qsort((void *)(a->ve),(int)(a->dim),sizeof(Real),(int (*)())dbl_cmp);
   
   /* error estimates */
   if ( err_est )
   {
      err_est = v_resize(err_est,(unsigned int)ip->k);
      
      trieig(a2,b2,MNULL);
      /* printf("# a =\n");	v_output(a); */
      /* printf("# a2 =\n");	v_output(a2); */
      
      for ( i = 0; i < a->dim; i++ )
      {
	 det_mant1 = product2(a,i,&det_expt1);
	 det_mant2 = product(a2,(double)a->ve[i],&det_expt2);
	 /* printf("# det_mant1=%g, det_expt1=%d\n",
	    det_mant1,det_expt1); */
	 /* printf("# det_mant2=%g, det_expt2=%d\n",
	    det_mant2,det_expt2); */
	 if ( det_mant1 == 0.0 )
	 {   /* multiple e-val of T */
	    err_est->ve[i] = 0.0;
	    continue;
	 }
	 else if ( det_mant2 == 0.0 )
	 {
	    err_est->ve[i] = HUGE_VAL;
	    continue;
	 }
	 if ( (det_expt1 + det_expt2) % 2 )
	   /* if odd... */
	   det_mant = sqrt(2.0*fabs(det_mant1*det_mant2));
	 else /* if even... */
	   det_mant = sqrt(fabs(det_mant1*det_mant2));
	 det_expt = (det_expt1+det_expt2)/2;
	 err_est->ve[i] = fabs(beta*
			       ldexp(pb_mant/det_mant,pb_expt-det_expt));
      }
   }

#ifdef	THREADSAFE
   V_FREE(b);   V_FREE(a2);   V_FREE(b2);
#endif

   return a;
}

/* iter_splanczos2 -- version of iter_lanczos2() that uses sparse matrix data
   structure */
#ifndef ANSI_C
VEC    *iter_splanczos2(A,m,x0,evals,err_est)
SPMAT	*A;
int	 m;
VEC	*x0;		/* initial vector */
VEC	*evals;		/* eigenvalue vector */
VEC	*err_est;	/* error estimates of eigenvalues */
#else
VEC    *iter_splanczos2(SPMAT *A, int m, VEC *x0, VEC *evals, VEC *err_est)
#endif
{	
   ITER *ip;
   VEC *a;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   ip->x = x0;
   ip->k = m;
   a = iter_lanczos2(ip,evals,err_est);	
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return a;
}




/*
  Conjugate gradient method
  Another variant - mainly for testing
  */
#ifndef ANSI_C
VEC  *iter_cg1(ip)
ITER *ip;
#else
VEC  *iter_cg1(ITER *ip)
#endif
{
   STATIC VEC *r = VNULL, *p = VNULL, *q = VNULL, *z = VNULL;
   Real	alpha;
   double inner,nres;
   VEC *rr;   /* rr == r or rr == z */
   
   if (ip == INULL)
     error(E_NULL,"iter_cg");
   if (!ip->Ax || !ip->b)
     error(E_NULL,"iter_cg");
   if ( ip->x == ip->b )
     error(E_INSITU,"iter_cg");
   if (!ip->stop_crit)
     error(E_NULL,"iter_cg");
   
   if ( ip->eps <= 0.0 )
     ip->eps = MACHEPS;
   
   r = v_resize(r,ip->b->dim);
   p = v_resize(p,ip->b->dim);
   q = v_resize(q,ip->b->dim);
   
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(p,TYPE_VEC);
   MEM_STAT_REG(q,TYPE_VEC);
   
   if (ip->Bx != (Fun_Ax)NULL) {
      z = v_resize(z,ip->b->dim);
      MEM_STAT_REG(z,TYPE_VEC);
      rr = z;
   }
   else rr = r;
   
   if (ip->x != VNULL) {
      if (ip->x->dim != ip->b->dim)
	error(E_SIZES,"iter_cg");
      ip->Ax(ip->A_par,ip->x,p);    		/* p = A*x */
      v_sub(ip->b,p,r);		 		/* r = b - A*x */
   }
   else {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
      v_copy(ip->b,r);
   }
   
   if (ip->Bx) (ip->Bx)(ip->B_par,r,p);
   else v_copy(r,p);
   
   inner = in_prod(p,r);
   nres = sqrt(fabs(inner));
   if (ip->info) ip->info(ip,nres,r,p);
   if ( nres == 0.0) return ip->x;
   
   for ( ip->steps = 0; ip->steps <= ip->limit; ip->steps++ )
   {
      ip->Ax(ip->A_par,p,q);
      inner = in_prod(q,p);
      if (sqrt(fabs(inner)) <= MACHEPS*ip->init_res)
	error(E_BREAKDOWN,"iter_cg1");

      alpha = in_prod(p,r)/inner;
      v_mltadd(ip->x,p,alpha,ip->x);
      v_mltadd(r,q,-alpha,r);
      
      rr = r;
      if (ip->Bx) {
	 ip->Bx(ip->B_par,r,z);
	 rr = z;
      }
      
      nres = in_prod(r,rr);
      if (nres < 0.0) {
	 warning(WARN_RES_LESS_0,"iter_cg");
	 break;
      }
      nres = sqrt(fabs(nres));
      if (ip->info) ip->info(ip,nres,r,z);
      if (ip->steps == 0) ip->init_res = nres;
      if ( ip->stop_crit(ip,nres,r,z) ) break;
      
      alpha = -in_prod(rr,q)/inner;
      v_mltadd(rr,p,alpha,p);
      
   }

#ifdef	THREADSAFE
   V_FREE(r);   V_FREE(p);   V_FREE(q);   V_FREE(z);
#endif

   return ip->x;
}


