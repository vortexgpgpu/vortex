

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


/* iter.c 17/09/93 */

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

static char rcsid[] = "$Header: iternsym.c,v 1.6 1995/01/30 14:53:01 des Exp $";


#ifdef ANSI_C
VEC	*spCHsolve(SPMAT *,VEC *,VEC *);
#else
VEC	*spCHsolve();
#endif


/* 
  iter_cgs -- uses CGS to compute a solution x to A.x=b
*/
#ifndef ANSI_C
VEC	*iter_cgs(ip,r0)
ITER *ip;
VEC *r0;
#else
VEC	*iter_cgs(ITER *ip, VEC *r0)
#endif
{
   STATIC VEC  *p = VNULL, *q = VNULL, *r = VNULL, *u = VNULL;
   STATIC VEC  *v = VNULL, *z = VNULL;
   VEC  *tmp;
   Real	alpha, beta, nres, rho, old_rho, sigma, inner;

   if (ip == INULL)
     error(E_NULL,"iter_cgs");
   if (!ip->Ax || !ip->b || !r0)
     error(E_NULL,"iter_cgs");
   if ( ip->x == ip->b )
     error(E_INSITU,"iter_cgs");
   if (!ip->stop_crit)
     error(E_NULL,"iter_cgs");
   if ( r0->dim != ip->b->dim )
     error(E_SIZES,"iter_cgs");
   
   if ( ip->eps <= 0.0 ) ip->eps = MACHEPS;
   
   p = v_resize(p,ip->b->dim);
   q = v_resize(q,ip->b->dim);
   r = v_resize(r,ip->b->dim);
   u = v_resize(u,ip->b->dim);
   v = v_resize(v,ip->b->dim);

   MEM_STAT_REG(p,TYPE_VEC);
   MEM_STAT_REG(q,TYPE_VEC);
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(u,TYPE_VEC);
   MEM_STAT_REG(v,TYPE_VEC);

   if (ip->Bx) {
      z = v_resize(z,ip->b->dim);
      MEM_STAT_REG(z,TYPE_VEC); 
   }

   if (ip->x != VNULL) {
      if (ip->x->dim != ip->b->dim)
	error(E_SIZES,"iter_cgs");
      ip->Ax(ip->A_par,ip->x,v);    		/* v = A*x */
      if (ip->Bx) {
	 v_sub(ip->b,v,v);			/* v = b - A*x */
	 (ip->Bx)(ip->B_par,v,r);		/* r = B*(b-A*x) */
      }
      else v_sub(ip->b,v,r);			/* r = b-A*x */
   }
   else {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);		/* x == 0 */
      ip->shared_x = FALSE;
      if (ip->Bx) (ip->Bx)(ip->B_par,ip->b,r);    /* r = B*b */
      else v_copy(ip->b,r);                       /* r = b */
   }

   v_zero(p);	
   v_zero(q);
   old_rho = 1.0;
   
   for (ip->steps = 0; ip->steps <= ip->limit; ip->steps++) {

      inner = in_prod(r,r);
      nres = sqrt(fabs(inner));
      if (ip->steps == 0) ip->init_res = nres;

      if (ip->info) ip->info(ip,nres,r,VNULL);
      if ( ip->stop_crit(ip,nres,r,VNULL) ) break;

      rho = in_prod(r0,r);
      if ( old_rho == 0.0 )
	error(E_BREAKDOWN,"iter_cgs");
      beta = rho/old_rho;
      v_mltadd(r,q,beta,u);
      v_mltadd(q,p,beta,v);
      v_mltadd(u,v,beta,p);
      
      (ip->Ax)(ip->A_par,p,q);
      if (ip->Bx) {
	 (ip->Bx)(ip->B_par,q,z);
	 tmp = z;
      }
      else tmp = q;
      
      sigma = in_prod(r0,tmp);
      if ( sigma == 0.0 )
	error(E_BREAKDOWN,"iter_cgs");
      alpha = rho/sigma;
      v_mltadd(u,tmp,-alpha,q);
      v_add(u,q,v);
      
      (ip->Ax)(ip->A_par,v,u);
      if (ip->Bx) {
	 (ip->Bx)(ip->B_par,u,z);
	 tmp = z;
      }
      else tmp = u;
      
      v_mltadd(r,tmp,-alpha,r);
      v_mltadd(ip->x,v,alpha,ip->x);
      
      old_rho = rho;
   }

#ifdef THREADSAFE
   V_FREE(p);	V_FREE(q);	V_FREE(r);	V_FREE(u);
   V_FREE(v);	V_FREE(z);
#endif

   return ip->x;
}



/* iter_spcgs -- simple interface for SPMAT data structures 
   use always as follows:
      x = iter_spcgs(A,B,b,r0,tol,x,limit,steps);
   or 
      x = iter_spcgs(A,B,b,r0,tol,VNULL,limit,steps);
   In the second case the solution vector is created.  
   If B is not NULL then it is a preconditioner. 
*/
#ifndef ANSI_C
VEC	*iter_spcgs(A,B,b,r0,tol,x,limit,steps)
SPMAT	*A, *B;
VEC	*b, *r0, *x;
double	tol;
int     *steps,limit;
#else
VEC	*iter_spcgs(SPMAT *A, SPMAT *B, VEC *b, VEC *r0, double tol,
		    VEC *x, int limit, int *steps)
#endif
{	
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   if (B) {
      ip->Bx = (Fun_Ax) sp_mv_mlt;
      ip->B_par = (void *) B;
   }
   else {
      ip->Bx = (Fun_Ax) NULL;
      ip->B_par = NULL;
   }
   ip->info = (Fun_info) NULL;
   ip->limit = limit;
   ip->b = b;
   ip->eps = tol;
   ip->x = x;
   iter_cgs(ip,r0);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;   
   iter_free(ip);   /* release only ITER structure */
   return x;		

}

/*
  Routine for performing LSQR -- the least squares QR algorithm
  of Paige and Saunders:
  "LSQR: an algorithm for sparse linear equations and
  sparse least squares", ACM Trans. Math. Soft., v. 8
  pp. 43--71 (1982)
  */
/* iter_lsqr -- sparse CG-like least squares routine:
   -- finds min_x ||A.x-b||_2 using A defined through A & AT
   -- returns x (if x != NULL) */
#ifndef ANSI_C
VEC	*iter_lsqr(ip)
ITER *ip;
#else
VEC	*iter_lsqr(ITER *ip)
#endif
{
   STATIC VEC	*u = VNULL, *v = VNULL, *w = VNULL, *tmp = VNULL;
   Real	alpha, beta, phi, phi_bar;
   Real rho, rho_bar, rho_max, theta, nres;
   Real	s, c;	/* for Givens' rotations */
   int  m, n;
   
   if ( ! ip || ! ip->b || !ip->Ax || !ip->ATx )
     error(E_NULL,"iter_lsqr");
   if ( ip->x == ip->b )
     error(E_INSITU,"iter_lsqr");
   if (!ip->stop_crit || !ip->x)
     error(E_NULL,"iter_lsqr");

   if ( ip->eps <= 0.0 ) ip->eps = MACHEPS;
   
   m = ip->b->dim;	
   n = ip->x->dim;

   u = v_resize(u,(unsigned int)m);
   v = v_resize(v,(unsigned int)n);
   w = v_resize(w,(unsigned int)n);
   tmp = v_resize(tmp,(unsigned int)n);

   MEM_STAT_REG(u,TYPE_VEC);
   MEM_STAT_REG(v,TYPE_VEC);
   MEM_STAT_REG(w,TYPE_VEC);
   MEM_STAT_REG(tmp,TYPE_VEC);  

   if (ip->x != VNULL) {
      ip->Ax(ip->A_par,ip->x,u);    		/* u = A*x */
      v_sub(ip->b,u,u);				/* u = b-A*x */
   }
   else {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
      v_copy(ip->b,u);                       /* u = b */
   }
 
   beta = v_norm2(u); 
   if ( beta == 0.0 ) return ip->x;

   sv_mlt(1.0/beta,u,u);
   (ip->ATx)(ip->AT_par,u,v);
   alpha = v_norm2(v);
   if ( alpha == 0.0 ) return ip->x;

   sv_mlt(1.0/alpha,v,v);
   v_copy(v,w);
   phi_bar = beta;
   rho_bar = alpha;
   
   rho_max = 1.0;
   for (ip->steps = 0; ip->steps <= ip->limit; ip->steps++) {

      tmp = v_resize(tmp,m);
      (ip->Ax)(ip->A_par,v,tmp);
      
      v_mltadd(tmp,u,-alpha,u);
      beta = v_norm2(u);	
      sv_mlt(1.0/beta,u,u);
      
      tmp = v_resize(tmp,n);
      (ip->ATx)(ip->AT_par,u,tmp);
      v_mltadd(tmp,v,-beta,v);
      alpha = v_norm2(v);	
      sv_mlt(1.0/alpha,v,v);
      
      rho = sqrt(rho_bar*rho_bar+beta*beta);
      if ( rho > rho_max )
	rho_max = rho;
      c   = rho_bar/rho;
      s   = beta/rho;
      theta   =  s*alpha;
      rho_bar = -c*alpha;
      phi     =  c*phi_bar;
      phi_bar =  s*phi_bar;
      
      /* update ip->x & w */
      if ( rho == 0.0 )
	error(E_BREAKDOWN,"iter_lsqr");
      v_mltadd(ip->x,w,phi/rho,ip->x);
      v_mltadd(v,w,-theta/rho,w);

      nres = fabs(phi_bar*alpha*c)*rho_max;

      if (ip->info) ip->info(ip,nres,w,VNULL);
      if (ip->steps == 0) ip->init_res = nres;
      if ( ip->stop_crit(ip,nres,w,VNULL) ) break;
   } 

#ifdef THREADSAFE
   V_FREE(u);	V_FREE(v);	V_FREE(w);	V_FREE(tmp);
#endif

   return ip->x;
}

/* iter_splsqr -- simple interface for SPMAT data structures */
#ifndef ANSI_C
VEC	*iter_splsqr(A,b,tol,x,limit,steps)
SPMAT	*A;
VEC	*b, *x;
double	tol;
int *steps,limit;
#else
VEC	*iter_splsqr(SPMAT *A, VEC *b, double tol, 
		     VEC *x, int limit, int *steps)
#endif
{
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   ip->ATx = (Fun_Ax) sp_vm_mlt;
   ip->AT_par = (void *) A;
   ip->Bx = (Fun_Ax) NULL;
   ip->B_par = NULL;

   ip->info = (Fun_info) NULL;
   ip->limit = limit;
   ip->b = b;
   ip->eps = tol;
   ip->x = x;
   iter_lsqr(ip);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return x;		
}



/* iter_arnoldi -- an implementation of the Arnoldi method;
   iterative refinement is applied.
*/
#ifndef ANSI_C
MAT	*iter_arnoldi_iref(ip,h_rem,Q,H)
ITER  *ip;
Real  *h_rem;
MAT   *Q, *H;
#else
MAT	*iter_arnoldi_iref(ITER *ip, Real *h_rem, MAT *Q, MAT *H)
#endif
{
   STATIC VEC *u=VNULL, *r=VNULL, *s=VNULL, *tmp=VNULL;
   VEC v;     /* auxiliary vector */
   int	i,j;
   Real	h_val, c;
   
   if (ip == INULL)
     error(E_NULL,"iter_arnoldi_iref");
   if ( ! ip->Ax || ! Q || ! ip->x )
     error(E_NULL,"iter_arnoldi_iref");
   if ( ip->k <= 0 )
     error(E_BOUNDS,"iter_arnoldi_iref");
   if ( Q->n != ip->x->dim ||	Q->m != ip->k )
     error(E_SIZES,"iter_arnoldi_iref");
   
   m_zero(Q);
   H = m_resize(H,ip->k,ip->k);
   m_zero(H);

   u = v_resize(u,ip->x->dim);
   r = v_resize(r,ip->k);
   s = v_resize(s,ip->k);
   tmp = v_resize(tmp,ip->x->dim);
   MEM_STAT_REG(u,TYPE_VEC);
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(s,TYPE_VEC);
   MEM_STAT_REG(tmp,TYPE_VEC);

   v.dim = v.max_dim = ip->x->dim;

   c = v_norm2(ip->x);
   if ( c <= 0.0)
     return H;
   else {
      v.ve = Q->me[0];
      sv_mlt(1.0/c,ip->x,&v);
   }

   v_zero(r);
   v_zero(s);
   for ( i = 0; i < ip->k; i++ )
   {
      v.ve = Q->me[i];
      u = (ip->Ax)(ip->A_par,&v,u);
      for (j = 0; j <= i; j++) {
	 v.ve = Q->me[j];
	 /* modified Gram-Schmidt */
	 r->ve[j] = in_prod(&v,u);
	 v_mltadd(u,&v,-r->ve[j],u);
      }
      h_val = v_norm2(u);
      /* if u == 0 then we have an exact subspace */
      if ( h_val <= 0.0 )
      {
	 *h_rem = h_val;
	 return H;
      }
      /* iterative refinement -- ensures near orthogonality */
      do {
	 v_zero(tmp);
	 for (j = 0; j <= i; j++) {
	    v.ve = Q->me[j];
	    s->ve[j] = in_prod(&v,u);
	    v_mltadd(tmp,&v,s->ve[j],tmp);
	 }
	 v_sub(u,tmp,u);
         v_add(r,s,r);
      } while ( v_norm2(s) > 0.1*(h_val = v_norm2(u)) );
      /* now that u is nearly orthogonal to Q, update H */
      set_col(H,i,r);
      /* check once again if h_val is zero */
      if ( h_val <= 0.0 )
      {
	 *h_rem = h_val;
	 return H;
      }
      if ( i == ip->k-1 )
      {
	 *h_rem = h_val;
	 continue;
      }
      /* H->me[i+1][i] = h_val; */
      m_set_val(H,i+1,i,h_val);
      v.ve = Q->me[i+1];
      sv_mlt(1.0/h_val,u,&v);
   }

#ifdef THREADSAFE
   V_FREE(u);   V_FREE(r);   V_FREE(s);   V_FREE(tmp);
#endif

   return H;
}

/* iter_arnoldi -- an implementation of the Arnoldi method;
   modified Gram-Schmidt algorithm
*/
#ifndef ANSI_C
MAT	*iter_arnoldi(ip,h_rem,Q,H)
ITER  *ip;
Real  *h_rem;
MAT   *Q, *H;
#else
MAT	*iter_arnoldi(ITER *ip, Real *h_rem, MAT *Q, MAT *H)
#endif
{
   STATIC VEC *u=VNULL, *r=VNULL;
   VEC v;     /* auxiliary vector */
   int	i,j;
   Real	h_val, c;
   
   if (ip == INULL)
     error(E_NULL,"iter_arnoldi");
   if ( ! ip->Ax || ! Q || ! ip->x )
     error(E_NULL,"iter_arnoldi");
   if ( ip->k <= 0 )
     error(E_BOUNDS,"iter_arnoldi");
   if ( Q->n != ip->x->dim ||	Q->m != ip->k )
     error(E_SIZES,"iter_arnoldi");
   
   m_zero(Q);
   H = m_resize(H,ip->k,ip->k);
   m_zero(H);

   u = v_resize(u,ip->x->dim);
   r = v_resize(r,ip->k);
   MEM_STAT_REG(u,TYPE_VEC);
   MEM_STAT_REG(r,TYPE_VEC);

   v.dim = v.max_dim = ip->x->dim;

   c = v_norm2(ip->x);
   if ( c <= 0.0)
     return H;
   else {
      v.ve = Q->me[0];
      sv_mlt(1.0/c,ip->x,&v);
   }

   v_zero(r);
   for ( i = 0; i < ip->k; i++ )
   {
      v.ve = Q->me[i];
      u = (ip->Ax)(ip->A_par,&v,u);
      for (j = 0; j <= i; j++) {
	 v.ve = Q->me[j];
	 /* modified Gram-Schmidt */
	 r->ve[j] = in_prod(&v,u);
	 v_mltadd(u,&v,-r->ve[j],u);
      }
      h_val = v_norm2(u);
      /* if u == 0 then we have an exact subspace */
      if ( h_val <= 0.0 )
      {
	 *h_rem = h_val;
	 return H;
      }
      set_col(H,i,r);
      if ( i == ip->k-1 )
      {
	 *h_rem = h_val;
	 continue;
      }
      /* H->me[i+1][i] = h_val; */
      m_set_val(H,i+1,i,h_val);
      v.ve = Q->me[i+1];
      sv_mlt(1.0/h_val,u,&v);
   }

#ifdef THREADSAFE
   V_FREE(u);	V_FREE(r);
#endif
   
   return H;
}



/* iter_sparnoldi -- uses arnoldi() with an explicit representation of A */
#ifndef ANSI_C
MAT	*iter_sparnoldi(A,x0,m,h_rem,Q,H)
SPMAT	*A;
VEC	*x0;
int	m;
Real	*h_rem;
MAT	*Q, *H;
#else
MAT	*iter_sparnoldi(SPMAT *A, VEC *x0, int m, Real *h_rem, MAT *Q, MAT *H)
#endif
{
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   ip->x = x0;
   ip->k = m;
   iter_arnoldi_iref(ip,h_rem,Q,H);
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return H;	
}


/* for testing gmres */
#ifndef ANSI_C
static void test_gmres(ip,i,Q,R,givc,givs,h_val)
ITER *ip;
int i;
MAT *Q, *R;
VEC *givc, *givs;
double h_val;
#else
static void test_gmres(ITER *ip, int i, MAT *Q, MAT *R,
		       VEC *givc, VEC *givs, double h_val)
#endif
{
   VEC vt, vt1;
   STATIC MAT *Q1=MNULL, *R1=MNULL;
   int j;
   
   /* test Q*A*Q^T = R  */

   Q = m_resize(Q,i+1,ip->b->dim);
   Q1 = m_resize(Q1,i+1,ip->b->dim);
   R1 = m_resize(R1,i+1,i+1);
   MEM_STAT_REG(Q1,TYPE_MAT);
   MEM_STAT_REG(R1,TYPE_MAT);

   vt.dim = vt.max_dim = ip->b->dim;
   vt1.dim = vt1.max_dim = ip->b->dim;
   for (j=0; j <= i; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      ip->Ax(ip->A_par,&vt,&vt1);
   }

   mmtr_mlt(Q,Q1,R1);
   R1 = m_resize(R1,i+2,i+1);
   for (j=0; j < i; j++)
     R1->me[i+1][j] = 0.0;
   R1->me[i+1][i] = h_val;
   
   for (j = 0; j <= i; j++) {
      rot_rows(R1,j,j+1,givc->ve[j],givs->ve[j],R1);
   }

   R1 = m_resize(R1,i+1,i+1);
   m_sub(R,R1,R1);
   /* if (m_norm_inf(R1) > MACHEPS*ip->b->dim)  */
#ifndef MEX
   printf(" %d. ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	  ip->steps,m_norm_inf(R1),MACHEPS);
#endif
   
   /* check Q*Q^T = I */
   
   Q = m_resize(Q,i+1,ip->b->dim);
   mmtr_mlt(Q,Q,R1);
   for (j=0; j <= i; j++)
     R1->me[j][j] -= 1.0;
#ifndef MEX
   if (m_norm_inf(R1) > MACHEPS*ip->b->dim)
     printf(" ! m_norm_inf(Q*Q^T) = %g\n",m_norm_inf(R1));  
#endif
#ifdef THREADSAFE
   M_FREE(Q1);	M_FREE(R1);
#endif
}


/* gmres -- generalised minimum residual algorithm of Saad & Schultz
   SIAM J. Sci. Stat. Comp. v.7, pp.856--869 (1986)
*/
#ifndef ANSI_C
VEC	*iter_gmres(ip)
ITER *ip;
#else
VEC	*iter_gmres(ITER *ip)
#endif
{
   STATIC VEC *u=VNULL, *r=VNULL, *rhs = VNULL;
   STATIC VEC *givs=VNULL, *givc=VNULL, *z = VNULL;
   STATIC MAT *Q = MNULL, *R = MNULL;
   VEC *rr, v, v1;   /* additional pointers (not real vectors) */
   int	i,j, done;
   Real	nres;
/*   Real last_h;  */
   
   if (ip == INULL)
     error(E_NULL,"iter_gmres");
   if ( ! ip->Ax || ! ip->b )
     error(E_NULL,"iter_gmres");
   if ( ! ip->stop_crit )
     error(E_NULL,"iter_gmres");
   if ( ip->k <= 0 )
     error(E_BOUNDS,"iter_gmres");
   if (ip->x != VNULL && ip->x->dim != ip->b->dim)
     error(E_SIZES,"iter_gmres");
   if (ip->eps <= 0.0) ip->eps = MACHEPS;

   r = v_resize(r,ip->k+1);
   u = v_resize(u,ip->b->dim);
   rhs = v_resize(rhs,ip->k+1);
   givs = v_resize(givs,ip->k);  /* Givens rotations */
   givc = v_resize(givc,ip->k); 
   
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(u,TYPE_VEC);
   MEM_STAT_REG(rhs,TYPE_VEC);
   MEM_STAT_REG(givs,TYPE_VEC);
   MEM_STAT_REG(givc,TYPE_VEC);
   
   R = m_resize(R,ip->k+1,ip->k);
   Q = m_resize(Q,ip->k,ip->b->dim);
   MEM_STAT_REG(R,TYPE_MAT);
   MEM_STAT_REG(Q,TYPE_MAT);		

   if (ip->x == VNULL) {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
   }   

   v.dim = v.max_dim = ip->b->dim;      /* v and v1 are pointers to rows */
   v1.dim = v1.max_dim = ip->b->dim;  	/* of matrix Q */
   
   if (ip->Bx != (Fun_Ax)NULL) {    /* if precondition is defined */
      z = v_resize(z,ip->b->dim);
      MEM_STAT_REG(z,TYPE_VEC);
   }
   
   done = FALSE;
   for (ip->steps = 0; ip->steps < ip->limit; ) {

      /* restart */

      ip->Ax(ip->A_par,ip->x,u);    		/* u = A*x */
      v_sub(ip->b,u,u);		 		/* u = b - A*x */
      rr = u;				/* rr is a pointer only */
      
      if (ip->Bx) {
	 (ip->Bx)(ip->B_par,u,z);            /* tmp = B*(b-A*x)  */
	 rr = z;
      }
      
      nres = v_norm2(rr);
      if (ip->steps == 0) {
	 if (ip->info) ip->info(ip,nres,VNULL,VNULL);
	 ip->init_res = nres;
      }

      if ( nres == 0.0 ) {
	 done = TRUE;
	 break;
      }

      v.ve = Q->me[0];
      sv_mlt(1.0/nres,rr,&v);
      
      v_zero(r);
      v_zero(rhs);
      rhs->ve[0] = nres;

      for ( i = 0; i < ip->k && ip->steps < ip->limit; i++ ) {
	 ip->steps++;
	 v.ve = Q->me[i];	
	 (ip->Ax)(ip->A_par,&v,u);
	 rr = u;
	 if (ip->Bx) {
	    (ip->Bx)(ip->B_par,u,z);
	    rr = z;
	 }
	 
	 if (i < ip->k - 1) {
	    v1.ve = Q->me[i+1];
	    v_copy(rr,&v1);
	    for (j = 0; j <= i; j++) {
	       v.ve = Q->me[j];
	       /* r->ve[j] = in_prod(&v,rr); */
	       /* modified Gram-Schmidt algorithm */
	       r->ve[j] = in_prod(&v,&v1);
	       v_mltadd(&v1,&v,-r->ve[j],&v1);
	    }
	    
	    r->ve[i+1] = nres = v_norm2(&v1);
	    if (nres <= MACHEPS*ip->init_res) {
	       for (j = 0; j < i; j++) 
		 rot_vec(r,j,j+1,givc->ve[j],givs->ve[j],r);
	       set_col(R,i,r);
	       done = TRUE;
	       break;
	    }
	    sv_mlt(1.0/nres,&v1,&v1);
	 }
	 else {  /* i == ip->k - 1 */
	    /* Q->me[ip->k] need not be computed */

	    for (j = 0; j <= i; j++) {
	       v.ve = Q->me[j];
	       r->ve[j] = in_prod(&v,rr);
	    }
	    
	    nres = in_prod(rr,rr) - in_prod(r,r);
	    if (sqrt(fabs(nres)) <= MACHEPS*ip->init_res) { 
	       for (j = 0; j < i; j++) 
		 rot_vec(r,j,j+1,givc->ve[j],givs->ve[j],r);
	       set_col(R,i,r);
	       done = TRUE;
	       break;
	    }
	    if (nres < 0.0) { /* do restart */
	       i--; 
	       ip->steps--;
	       break;
	    } 
	    r->ve[i+1] = sqrt(nres);
	 }

	 /* QR update */

	 /* last_h = r->ve[i+1]; */ /* for test only */
	 for (j = 0; j < i; j++) 
	   rot_vec(r,j,j+1,givc->ve[j],givs->ve[j],r);
	 givens(r->ve[i],r->ve[i+1],&givc->ve[i],&givs->ve[i]);
	 rot_vec(r,i,i+1,givc->ve[i],givs->ve[i],r);
	 rot_vec(rhs,i,i+1,givc->ve[i],givs->ve[i],rhs);
	 
	 set_col(R,i,r);

	 nres = fabs((double) rhs->ve[i+1]);
	 if (ip->info) ip->info(ip,nres,VNULL,VNULL);
	 if ( ip->stop_crit(ip,nres,VNULL,VNULL) ) {
	    done = TRUE;
	    break;
	 }
      }
      
      /* use ixi submatrix of R */

      if (i >= ip->k) i = ip->k - 1;

      R = m_resize(R,i+1,i+1);
      rhs = v_resize(rhs,i+1);
      
      /* test only */
      /* test_gmres(ip,i,Q,R,givc,givs,last_h);  */
      
      Usolve(R,rhs,rhs,0.0); 	 /* solve a system: R*x = rhs */

      /* new approximation */

      for (j = 0; j <= i; j++) {
	 v.ve = Q->me[j]; 
	 v_mltadd(ip->x,&v,rhs->ve[j],ip->x);
      }

      if (done) break;

      /* back to old dimensions */

      rhs = v_resize(rhs,ip->k+1);
      R = m_resize(R,ip->k+1,ip->k);

   }

#ifdef THREADSAFE
   V_FREE(u);		V_FREE(r);	V_FREE(rhs);
   V_FREE(givs);	V_FREE(givc);	V_FREE(z);
   M_FREE(Q);		M_FREE(R);
#endif

   return ip->x;
}

/* iter_spgmres - a simple interface to iter_gmres */
#ifndef ANSI_C
VEC	*iter_spgmres(A,B,b,tol,x,k,limit,steps)
SPMAT	*A, *B;
VEC	*b, *x;
double	tol;
int *steps,k,limit;
#else
VEC	*iter_spgmres(SPMAT *A, SPMAT *B, VEC *b, double tol,
		      VEC *x, int k, int limit, int *steps)
#endif
{
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   if (B) {
      ip->Bx = (Fun_Ax) sp_mv_mlt;
      ip->B_par = (void *) B;
   }
   else {
      ip->Bx = (Fun_Ax) NULL;
      ip->B_par = NULL;
   }
   ip->k = k;
   ip->limit = limit;
   ip->info = (Fun_info) NULL;
   ip->b = b;
   ip->eps = tol;
   ip->x = x;
   iter_gmres(ip);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return x;		
}


/* for testing mgcr */
#ifndef ANSI_C
static void test_mgcr(ip,i,Q,R)
ITER *ip;
int i;
MAT *Q, *R;
#else
static void test_mgcr(ITER *ip, int i, MAT *Q, MAT *R)
#endif
{
   VEC vt, vt1;
   static MAT *R1=MNULL;
   static VEC *r=VNULL, *r1=VNULL;
   VEC *rr;
   int k,j;
   Real sm;
   
   
   /* check Q*Q^T = I */
   vt.dim = vt.max_dim = ip->b->dim;
   vt1.dim = vt1.max_dim = ip->b->dim;
   
   Q = m_resize(Q,i+1,ip->b->dim);
   R1 = m_resize(R1,i+1,i+1);
   r = v_resize(r,ip->b->dim);
   r1 = v_resize(r1,ip->b->dim);
   MEM_STAT_REG(R1,TYPE_MAT);
   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(r1,TYPE_VEC);

   m_zero(R1);
   for (k=1; k <= i; k++)
     for (j=1; j <= i; j++) {
	vt.ve = Q->me[k];
	vt1.ve = Q->me[j];
	R1->me[k][j] = in_prod(&vt,&vt1);
     }
   for (j=1; j <= i; j++)
     R1->me[j][j] -= 1.0;
#ifndef MEX
   if (m_norm_inf(R1) > MACHEPS*ip->b->dim)
     printf(" ! (mgcr:) m_norm_inf(Q*Q^T) = %g\n",m_norm_inf(R1));  
#endif

   /* check (r_i,Ap_j) = 0 for j <= i */
   
   ip->Ax(ip->A_par,ip->x,r);
   v_sub(ip->b,r,r);
   rr = r;
   if (ip->Bx) {
      ip->Bx(ip->B_par,r,r1);
      rr = r1;
   }
   
#ifndef MEX
   printf(" ||r|| = %g\n",v_norm2(rr));
#endif
   sm = 0.0;
   for (j = 1; j <= i; j++) {
      vt.ve = Q->me[j];
      sm = max(sm,in_prod(&vt,rr));
   }
#ifndef MEX
   if (sm >= MACHEPS*ip->b->dim)
     printf(" ! (mgcr:) max_j (r,Ap_j) = %g\n",sm);
#endif

}




/* 
  iter_mgcr -- modified generalized conjugate residual algorithm;
  fast version of GCR;
*/
#ifndef ANSI_C
VEC *iter_mgcr(ip)
ITER *ip;
#else
VEC *iter_mgcr(ITER *ip)
#endif
{
   STATIC VEC *As=VNULL, *beta=VNULL, *alpha=VNULL, *z=VNULL;
   STATIC MAT *N=MNULL, *H=MNULL;
   
   VEC *rr, v, s;  /* additional pointer and structures */
   Real nres;      /* norm of a residual */
   Real dd;        /* coefficient d_i */
   int i,j;
   int done;      /* if TRUE then stop the iterative process */
   int dim;       /* dimension of the problem */
   
   /* ip cannot be NULL */
   if (ip == INULL) error(E_NULL,"mgcr");
   /* Ax, b and stopping criterion must be given */
   if (! ip->Ax || ! ip->b || ! ip->stop_crit) 
     error(E_NULL,"mgcr");
   /* at least one direction vector must exist */
   if ( ip->k <= 0) error(E_BOUNDS,"mgcr");
   /* if the vector x is given then b and x must have the same dimension */
   if ( ip->x && ip->x->dim != ip->b->dim)
     error(E_SIZES,"mgcr");
   if (ip->eps <= 0.0) ip->eps = MACHEPS;
   
   dim = ip->b->dim;
   As = v_resize(As,dim);
   alpha = v_resize(alpha,ip->k);
   beta = v_resize(beta,ip->k);
   
   MEM_STAT_REG(As,TYPE_VEC);
   MEM_STAT_REG(alpha,TYPE_VEC);
   MEM_STAT_REG(beta,TYPE_VEC);
   
   H = m_resize(H,ip->k,ip->k);
   N = m_resize(N,ip->k,dim);
   
   MEM_STAT_REG(H,TYPE_MAT);
   MEM_STAT_REG(N,TYPE_MAT);
   
   /* if a preconditioner is defined */
   if (ip->Bx) {
      z = v_resize(z,dim);
      MEM_STAT_REG(z,TYPE_VEC);
   }
   
   /* if x is NULL then it is assumed that x has 
      entries with value zero */
   if ( ! ip->x ) {
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
   }
   
   /* v and s are additional pointers to rows of N */
   /* they must have the same dimension as rows of N */
   v.dim = v.max_dim = s.dim = s.max_dim = dim;
   
   
   done = FALSE;
   for (ip->steps = 0; ip->steps < ip->limit; ) {
      (*ip->Ax)(ip->A_par,ip->x,As);         /* As = A*x */
      v_sub(ip->b,As,As);                    /* As = b - A*x */
      rr = As;                               /* rr is an additional pointer */
      
      /* if a preconditioner is defined */
      if (ip->Bx) {
	 (*ip->Bx)(ip->B_par,As,z);               /* z = B*(b-A*x)  */
	 rr = z;                                  
      }
      
      /* norm of the residual */
      nres = v_norm2(rr);
      dd = nres;                            /* dd = ||r_i||  */
      
      /* check if the norm of the residual is zero */
      if (ip->steps == 0) {                
	 /* information for a user */
	 if (ip->info) (*ip->info)(ip,nres,As,rr); 
	 ip->init_res = fabs(nres);
      }

      if (nres == 0.0) { 
	 /* iterative process is finished */
	 done = TRUE; 
	 break;
      }
      
      /* save this residual in the first row of N */
      v.ve = N->me[0];
      v_copy(rr,&v);
      
      for (i = 0; i < ip->k && ip->steps < ip->limit; i++) {
	 ip->steps++;
	 v.ve = N->me[i];                /* pointer to a row of N (=s_i) */
	 /* note that we must use here &v, not v */
	 (*ip->Ax)(ip->A_par,&v,As); 
	 rr = As;                        /* As = A*s_i */
	 if (ip->Bx) {
	    (*ip->Bx)(ip->B_par,As,z);    /* z = B*A*s_i  */
	    rr = z;
	 }
	 
	 if (i < ip->k - 1) {
	    s.ve = N->me[i+1];         /* pointer to a row of N (=s_{i+1}) */
	    v_copy(rr,&s);                   /* s_{i+1} = B*A*s_i */
	    for (j = 0; j <= i-1; j++) {
	       v.ve = N->me[j+1];      /* pointer to a row of N (=s_{j+1}) */
	       /* beta->ve[j] = in_prod(&v,rr); */      /* beta_{j,i} */
	       /* modified Gram-Schmidt algorithm */
	       beta->ve[j] = in_prod(&v,&s);  	         /* beta_{j,i} */
	                                 /* s_{i+1} -= beta_{j,i}*s_{j+1} */
	       v_mltadd(&s,&v,- beta->ve[j],&s);    
	    }
	    
	     /* beta_{i,i} = ||s_{i+1}||_2 */
	    beta->ve[i] = nres = v_norm2(&s);     
	    if ( nres <= MACHEPS*ip->init_res) { 
	       /* s_{i+1} == 0 */
	       i--;
	       done = TRUE;
	       break;
	    }
	    sv_mlt(1.0/nres,&s,&s);           /* normalize s_{i+1} */
	    
	    v.ve = N->me[0];
	    alpha->ve[i] = in_prod(&v,&s);     /* alpha_i = (s_0 , s_{i+1}) */
	    
	 }
	 else {
	    for (j = 0; j <= i-1; j++) {
	       v.ve = N->me[j+1];      /* pointer to a row of N (=s_{j+1}) */
	       beta->ve[j] = in_prod(&v,rr);       /* beta_{j,i} */
	    }
	    
	    nres = in_prod(rr,rr);                 /* rr = B*A*s_{k-1} */
	    for (j = 0; j <= i-1; j++)
              nres -= beta->ve[j]*beta->ve[j];

	    if (sqrt(fabs(nres)) <= MACHEPS*ip->init_res)  {
	       /* s_k is zero */
	       i--;
	       done = TRUE;
	       break;
	    }
	    if (nres < 0.0) { /* do restart */
	       i--; 
	       ip->steps--;
	       break; 
	    }   
	    beta->ve[i] = sqrt(nres);         /* beta_{k-1,k-1} */
	    
	    v.ve = N->me[0];
	    alpha->ve[i] = in_prod(&v,rr); 
	    for (j = 0; j <= i-1; j++)
              alpha->ve[i] -= beta->ve[j]*alpha->ve[j];
	    alpha->ve[i] /= beta->ve[i];                /* alpha_{k-1} */
	    
	 }
	 
	 set_col(H,i,beta);

	 /* other method of computing dd */
	/* if (fabs((double)alpha->ve[i]) > dd)  {     
	    nres = - dd*dd + alpha->ve[i]*alpha->ve[i];
	    nres = sqrt((double) nres); 
	    if (ip->info) (*ip->info)(ip,-nres,VNULL,VNULL);  	
	    break;     
	 }  */
	 /* to avoid overflow/underflow in computing dd */
	 /* dd *= cos(asin((double)(alpha->ve[i]/dd))); */
	 
	 nres = alpha->ve[i]/dd;
	 if (fabs(nres-1.0) <= MACHEPS*ip->init_res) 
	   dd = 0.0;
	 else {
	    nres = 1.0 - nres*nres;
	    if (nres < 0.0) {
	       nres = sqrt((double) -nres); 
	       if (ip->info) (*ip->info)(ip,-dd*nres,VNULL,VNULL);  	
	       break;
	    }
	    dd *= sqrt((double) nres);  
	 }

	 if (ip->info) (*ip->info)(ip,dd,VNULL,VNULL);     
	 if ( ip->stop_crit(ip,dd,VNULL,VNULL) ) {
	    /* stopping criterion is satisfied */
	    done = TRUE;
	    break;
	 }
	 
      } /* end of for */
      
      if (i >= ip->k) i = ip->k - 1;
      
      /* use (i+1) by (i+1) submatrix of H */
      H = m_resize(H,i+1,i+1);
      alpha = v_resize(alpha,i+1);
      Usolve(H,alpha,alpha,0.0);       /* c_i is saved in alpha */
      
      for (j = 0; j <= i; j++) {
	 v.ve = N->me[j];
	 v_mltadd(ip->x,&v,alpha->ve[j],ip->x);
      }
      
      
      if (done) break;              /* stop the iterative process */
      alpha = v_resize(alpha,ip->k);
      H = m_resize(H,ip->k,ip->k);
      
   }  /* end of while */

#ifdef THREADSAFE
   V_FREE(As);		V_FREE(beta);	V_FREE(alpha);	V_FREE(z);
   M_FREE(N);		M_FREE(H);
#endif

   return ip->x;                    /* return the solution */
}



/* iter_spmgcr - a simple interface to iter_mgcr */
/* no preconditioner */
#ifndef ANSI_C
VEC	*iter_spmgcr(A,B,b,tol,x,k,limit,steps)
SPMAT	*A, *B;
VEC	*b, *x;
double	tol;
int *steps,k,limit;
#else
VEC	*iter_spmgcr(SPMAT *A, SPMAT *B, VEC *b, double tol,
		     VEC *x, int k, int limit, int *steps)
#endif
{
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *) A;
   if (B) {
      ip->Bx = (Fun_Ax) sp_mv_mlt;
      ip->B_par = (void *) B;
   }
   else {
      ip->Bx = (Fun_Ax) NULL;
      ip->B_par = NULL;
   }

   ip->k = k;
   ip->limit = limit;
   ip->info = (Fun_info) NULL;
   ip->b = b;
   ip->eps = tol;
   ip->x = x;
   iter_mgcr(ip);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return x;		
}



/* 
  Conjugate gradients method for a normal equation
  a preconditioner B must be symmetric !!
*/
#ifndef ANSI_C
VEC  *iter_cgne(ip)
ITER *ip;
#else
VEC  *iter_cgne(ITER *ip)
#endif
{
   STATIC VEC *r = VNULL, *p = VNULL, *q = VNULL, *z = VNULL;
   Real	alpha, beta, inner, old_inner, nres;
   VEC *rr1;   /* pointer only */
   
   if (ip == INULL)
     error(E_NULL,"iter_cgne");
   if (!ip->Ax || ! ip->ATx || !ip->b)
     error(E_NULL,"iter_cgne");
   if ( ip->x == ip->b )
     error(E_INSITU,"iter_cgne");
   if (!ip->stop_crit)
     error(E_NULL,"iter_cgne");
   
   if ( ip->eps <= 0.0 ) ip->eps = MACHEPS;
   
   r = v_resize(r,ip->b->dim);
   p = v_resize(p,ip->b->dim);
   q = v_resize(q,ip->b->dim);

   MEM_STAT_REG(r,TYPE_VEC);
   MEM_STAT_REG(p,TYPE_VEC);
   MEM_STAT_REG(q,TYPE_VEC);

   z = v_resize(z,ip->b->dim);
   MEM_STAT_REG(z,TYPE_VEC);

   if (ip->x) {
      if (ip->x->dim != ip->b->dim)
	error(E_SIZES,"iter_cgne");
      ip->Ax(ip->A_par,ip->x,p);    		/* p = A*x */
      v_sub(ip->b,p,z);		 		/* z = b - A*x */
   }
   else {  /* ip->x == 0 */
      ip->x = v_get(ip->b->dim);
      ip->shared_x = FALSE;
      v_copy(ip->b,z);
   }
   rr1 = z;
   if (ip->Bx) {
      (ip->Bx)(ip->B_par,rr1,p);
      rr1 = p;
   }
   (ip->ATx)(ip->AT_par,rr1,r);		/* r = A^T*B*(b-A*x)  */


   old_inner = 0.0;
   for ( ip->steps = 0; ip->steps <= ip->limit; ip->steps++ )
   {
      rr1 = r;
      if ( ip->Bx ) {
	 (ip->Bx)(ip->B_par,r,z);		/* rr = B*r */
	 rr1 = z;
      }

      inner = in_prod(r,rr1);
      nres = sqrt(fabs(inner));
      if (ip->info) ip->info(ip,nres,r,rr1);
      if (ip->steps == 0) ip->init_res = nres;
      if ( ip->stop_crit(ip,nres,r,rr1) ) break;

      if ( ip->steps )	/* if ( ip->steps > 0 ) ... */
      {
	 beta = inner/old_inner;
	 p = v_mltadd(rr1,p,beta,p);
      }
      else		/* if ( ip->steps == 0 ) ... */
      {
	 beta = 0.0;
	 p = v_copy(rr1,p);
	 old_inner = 0.0;
      }
      (ip->Ax)(ip->A_par,p,q);     /* q = A*p */
      if (ip->Bx) {
	 (ip->Bx)(ip->B_par,q,z);
	 (ip->ATx)(ip->AT_par,z,q);
	 rr1 = q;			/* q = A^T*B*A*p */
      }
      else {
	 (ip->ATx)(ip->AT_par,q,z);	/* z = A^T*A*p */
	 rr1 = z;
      }

      alpha = inner/in_prod(rr1,p);
      v_mltadd(ip->x,p,alpha,ip->x);
      v_mltadd(r,rr1,-alpha,r);
      old_inner = inner;
   }

#ifdef THREADSAFE
   V_FREE(r);   V_FREE(p);   V_FREE(q);   V_FREE(z);
#endif

   return ip->x;
}

/* iter_spcgne -- a simple interface to iter_cgne() which 
   uses sparse matrix data structures
   -- assumes that B contains an actual preconditioner (or NULL)
   use always as follows:
      x = iter_spcgne(A,B,b,eps,x,limit,steps);
   or 
      x = iter_spcgne(A,B,b,eps,VNULL,limit,steps);
   In the second case the solution vector is created.
*/
#ifndef ANSI_C
VEC  *iter_spcgne(A,B,b,eps,x,limit,steps)
SPMAT	*A, *B;
VEC	*b, *x;
double	eps;
int *steps, limit;
#else
VEC  *iter_spcgne(SPMAT *A,SPMAT *B, VEC *b, double eps,
		  VEC *x, int limit, int *steps)
#endif
{	
   ITER *ip;
   
   ip = iter_get(0,0);
   ip->Ax = (Fun_Ax) sp_mv_mlt;
   ip->A_par = (void *)A;
   ip->ATx = (Fun_Ax) sp_vm_mlt;
   ip->AT_par = (void *)A;
   if (B) {
      ip->Bx = (Fun_Ax) sp_mv_mlt;
      ip->B_par = (void *)B;
   }
   else {
      ip->Bx = (Fun_Ax) NULL;
      ip->B_par = NULL;
   }
   ip->info = (Fun_info) NULL;
   ip->b = b;
   ip->eps = eps;
   ip->limit = limit;
   ip->x = x;
   iter_cgne(ip);
   x = ip->x;
   if (steps) *steps = ip->steps;
   ip->shared_x = ip->shared_b = TRUE;
   iter_free(ip);   /* release only ITER structure */
   return x;		
}



