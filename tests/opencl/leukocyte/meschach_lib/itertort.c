
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


/*  iter_tort.c  16/09/93 */

/*
  This file contains tests for the iterative part of Meschach
*/

#include	<stdio.h>
#include	"matrix2.h"
#include	"sparse2.h"
#include	"iter.h"
#include	<math.h>

#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
  
  /* for iterative methods */
  
#if REAL == DOUBLE
#define	EPS	1e-7
#define KK	20
#elif REAL == FLOAT
#define EPS   1e-5
#define KK	8
#endif

#define ANON  513
#define ASYM  ANON   

  
static VEC *ex_sol = VNULL;

/* new iter information */
void iter_mod_info(ip,nres,res,Bres)
ITER *ip;
double nres;
VEC *res, *Bres;
{
   static VEC *tmp;

   if (ip->b == VNULL) return;
   tmp = v_resize(tmp,ip->b->dim);
   MEM_STAT_REG(tmp,TYPE_VEC);

   if (nres >= 0.0) {
      printf(" %d. residual = %g\n",ip->steps,nres);
   }
   else 
     printf(" %d. residual = %g (WARNING !!! should be >= 0) \n",
	    ip->steps,nres);
   if (ex_sol != VNULL)
     printf("    ||u_ex - u_approx||_2 = %g\n",
	    v_norm2(v_sub(ip->x,ex_sol,tmp)));
}


/* out = A^T*A*x */
VEC *norm_equ(A,x,out)
SPMAT *A;
VEC *x, *out;
{
   static VEC * tmp;

   tmp = v_resize(tmp,x->dim);
   MEM_STAT_REG(tmp,TYPE_VEC);
   sp_mv_mlt(A,x,tmp);
   sp_vm_mlt(A,tmp,out);
   return out;

}


/* 
  make symmetric preconditioner for nonsymmetric matrix A;
   B = 0.5*(A+A^T) and then B is factorized using 
   incomplete Choleski factorization
*/

SPMAT *gen_sym_precond(A)
SPMAT *A;
{
   SPMAT *B;
   SPROW *row;
   int i,j,k;
   Real val;
   
   B = sp_get(A->m,A->n,A->row[0].maxlen);
   for (i=0; i < A->m; i++) {
      row = &(A->row[i]);
      for (j = 0; j < row->len; j++) {
	k = row->elt[j].col;
	if (i != k) {
	   val = 0.5*(sp_get_val(A,i,k) + sp_get_val(A,k,i));
	   sp_set_val(B,i,k,val);
	   sp_set_val(B,k,i,val);
	}
	else { /* i == k */
	  val = sp_get_val(A,i,i);
	  sp_set_val(B,i,i,val);
       }
     }
   }

   spICHfactor(B);
   return B;
}

/* Dv_mlt -- diagonal by vector multiply; the diagonal matrix is represented
		by a vector d */
VEC	*Dv_mlt(d, x, out)
VEC	*d, *x, *out;
{
    int		i;

    if ( ! d || ! x )
	error(E_NULL,"Dv_mlt");
    if ( d->dim != x->dim )
	error(E_SIZES,"Dv_mlt");
    out = v_resize(out,x->dim);

    for ( i = 0; i < x->dim; i++ )
	out->ve[i] = d->ve[i]*x->ve[i];

    return out;
}



/************************************************/
void	main(argc, argv)
int	argc;
char	*argv[];
{
   VEC		*x, *y, *z, *u, *v, *xn, *yn;
   SPMAT	*A = NULL, *B = NULL;
   SPMAT	*An = NULL, *Bn = NULL;
   int		i, k, kk, j;
   ITER        *ips, *ips1, *ipns, *ipns1;
   MAT         *Q, *H, *Q1, *H1;
   VEC         vt, vt1;
   Real        hh;


   mem_info_on(TRUE);
   notice("allocating sparse matrices");
   
   printf(" dim of A = %dx%d\n",ASYM,ASYM);
   
   A = iter_gen_sym(ASYM,8);   
   B = sp_copy(A);
   spICHfactor(B);
   
   u = v_get(A->n);
   x = v_get(A->n);
   y = v_get(A->n);
   v = v_get(A->n);

   v_rand(x);
   sp_mv_mlt(A,x,y);
   ex_sol = x;
   
   notice(" initialize ITER variables");
   /* ips for symmetric matrices with precondition */
   ips = iter_get(A->m,A->n);

   /*  printf(" ips:\n");
   iter_dump(stdout,ips);   */

   ips->limit = 500;
   ips->eps = EPS;
   
   iter_Ax(ips,sp_mv_mlt,A);
   iter_Bx(ips,spCHsolve,B);

   ips->b = v_copy(y,ips->b);
   v_rand(ips->x);
   /* test of iter_resize */
   ips = iter_resize(ips,2*A->m,2*A->n);
   ips = iter_resize(ips,A->m,A->n);

   /*  printf(" ips:\n");
   iter_dump(stdout,ips); */
   
   /* ips1 for symmetric matrices without precondition */
   ips1 = iter_get(0,0);
   /*   printf(" ips1:\n");
   iter_dump(stdout,ips1);   */
   ITER_FREE(ips1);

   ips1 = iter_copy2(ips,ips1);
   iter_Bx(ips1,NULL,NULL);
   ips1->b = ips->b;
   ips1->shared_b = TRUE;
   /*    printf(" ips1:\n");
   iter_dump(stdout,ips1);   */

   /* ipns for nonsymetric matrices with precondition */
   ipns = iter_copy(ips,INULL);
   ipns->k = KK;
   ipns->limit = 500;
   ipns->info = NULL;

   An = iter_gen_nonsym_posdef(ANON,8);   
   Bn = gen_sym_precond(An);
   xn = v_get(An->n);
   yn = v_get(An->n);
   v_rand(xn);
   sp_mv_mlt(An,xn,yn);
   ipns->b = v_copy(yn,ipns->b);

   iter_Ax(ipns, sp_mv_mlt,An);
   iter_ATx(ipns, sp_vm_mlt,An);
   iter_Bx(ipns, spCHsolve,Bn);

   /*  printf(" ipns:\n");
   iter_dump(stdout,ipns); */
   
   /* ipns1 for nonsymmetric matrices without precondition */
   ipns1 = iter_copy2(ipns,INULL);
   ipns1->b = ipns->b;
   ipns1->shared_b = TRUE;
   iter_Bx(ipns1,NULL,NULL);

   /*   printf(" ipns1:\n");
   iter_dump(stdout,ipns1);  */


   /*******  CG  ********/

   notice(" CG method without preconditioning");
   ips1->info = NULL;
   mem_stat_mark(1);
   iter_cg(ips1);

   k = ips1->steps;
   z = ips1->x;
   printf(" cg: no. of iter.steps = %d\n",k);
   v_sub(z,x,u);
   printf(" (cg:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",
	  v_norm2(u),EPS);
   
   notice(" CG method with ICH preconditioning");

   ips->info = NULL;
   v_zero(ips->x);  
   iter_cg(ips);  

   k = ips->steps;
   printf(" cg: no. of iter.steps = %d\n",k);
   v_sub(ips->x,x,u);
   printf(" (cg:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",
	  v_norm2(u),EPS);
   
   V_FREE(v);
   if ((v = iter_spcg(A,B,y,EPS,VNULL,1000,&k)) == VNULL)
     errmesg("CG method with precond.: NULL solution"); 
   
   v_sub(ips->x,v,u);
   if (v_norm2(u) >= EPS) {
      errmesg("CG method with precond.: different solutions");
      printf(" diff. = %g\n",v_norm2(u));
   }   
   

   mem_stat_free(1);
   printf(" spcg: # of iter. steps = %d\n",k);
   v_sub(v,x,u);
   printf(" (spcg:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(u),EPS);  


   /***  CG FOR NORMAL EQUATION *****/

   notice("CGNE method with ICH preconditioning (nonsymmetric case)");

   /* ipns->info = iter_std_info;  */
   ipns->info = NULL;
   v_zero(ipns->x);
 
   mem_stat_mark(1);
   iter_cgne(ipns);

   k = ipns->steps;
   z = ipns->x;
   printf(" cgne: no. of iter.steps = %d\n",k);
   v_sub(z,xn,u);
   printf(" (cgne:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(u),EPS);

   notice("CGNE method without preconditioning (nonsymmetric case)");

   v_rand(u);
   u = iter_spcgne(An,NULL,yn,EPS,u,1000,&k);

   mem_stat_free(1);
   printf(" spcgne: no. of iter.steps = %d\n",k);
   v_sub(u,xn,u);
   printf(" (spcgne:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(u),EPS);

   /***  CGS  *****/

   notice("CGS method with ICH preconditioning (nonsymmetric case)");

   v_zero(ipns->x);   /* new init guess == 0 */
 
   mem_stat_mark(1);
   ipns->info = NULL;
   v_rand(u);
   iter_cgs(ipns,u);

   k = ipns->steps;
   z = ipns->x;
   printf(" cgs: no. of iter.steps = %d\n",k);
   v_sub(z,xn,u);
   printf(" (cgs:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(u),EPS);

   notice("CGS method without preconditioning (nonsymmetric case)");

   v_rand(u);
   v_rand(v);
   v = iter_spcgs(An,NULL,yn,u,EPS,v,1000,&k);

   mem_stat_free(1);
   printf(" cgs: no. of iter.steps = %d\n",k);
   v_sub(v,xn,u);
   printf(" (cgs:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(u),EPS);
   


   /*** LSQR ***/

   notice("LSQR method (without preconditioning)");

   v_rand(u);
   v_free(ipns1->x);
   ipns1->x = u;
   ipns1->shared_x = TRUE;
   ipns1->info = NULL;
   mem_stat_mark(2);
   z = iter_lsqr(ipns1);
   
   v_sub(xn,z,v);
   k = ipns1->steps;
   printf(" lsqr: # of iter. steps = %d\n",k);
   printf(" (lsqr:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(v),EPS);

   v_rand(u);
   u = iter_splsqr(An,yn,EPS,u,1000,&k);
   mem_stat_free(2);
   
   v_sub(xn,u,v);
   printf(" splsqr: # of iter. steps = %d\n",k);
   printf(" (splsqr:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",	
	  v_norm2(v),EPS);



   /***** GMRES ********/

   notice("GMRES method with ICH preconditioning (nonsymmetric case)");

   v_zero(ipns->x);
/*   ipns->info = iter_std_info;  */
   ipns->info = NULL;  

   mem_stat_mark(2);
   z = iter_gmres(ipns);
   v_sub(xn,z,v);
   k = ipns->steps;
   printf(" gmres: # of iter. steps = %d\n",k);
   printf(" (gmres:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(v),EPS);

   notice("GMRES method without preconditioning (nonsymmetric case)");
   V_FREE(v);
   v = iter_spgmres(An,NULL,yn,EPS,VNULL,10,1004,&k);
   mem_stat_free(2);
   
   v_sub(xn,v,v);
   printf(" spgmres: # of iter. steps = %d\n",k);
   printf(" (spgmres:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(v),EPS);



   /**** MGCR *****/

   notice("MGCR method with ICH preconditioning (nonsymmetric case)");

   v_zero(ipns->x);
   mem_stat_mark(2);
   z = iter_mgcr(ipns);
   v_sub(xn,z,v);
   k = ipns->steps;
   printf(" mgcr: # of iter. steps = %d\n",k);
   printf(" (mgcr:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",
	  v_norm2(v),EPS);

   notice("MGCR method without  preconditioning (nonsymmetric case)");
   V_FREE(v);
   v = iter_spmgcr(An,NULL,yn,EPS,VNULL,10,1004,&k);
   mem_stat_free(2);
   
   v_sub(xn,v,v);
   printf(" spmgcr: # of iter. steps = %d\n",k);
   printf(" (spmgcr:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",
	  v_norm2(v),EPS);


   /***** ARNOLDI METHOD ********/


   notice("arnoldi method");

   kk = ipns1->k = KK;
   Q = m_get(kk,x->dim);
   Q1 = m_get(kk,x->dim);
   H = m_get(kk,kk);
   v_rand(u);
   ipns1->x = u;
   ipns1->shared_x = TRUE;
   mem_stat_mark(3);
   iter_arnoldi_iref(ipns1,&hh,Q,H);
   mem_stat_free(3);

   /* check the equality:
      Q*A*Q^T = H; */

   vt.dim = vt.max_dim = x->dim;
   vt1.dim = vt1.max_dim = x->dim;
   for (j=0; j < kk; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      sp_mv_mlt(An,&vt,&vt1);
   }
   H1 = m_get(kk,kk);
   mmtr_mlt(Q,Q1,H1);
   m_sub(H,H1,H1);
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (arnoldi_iref) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);

   /* check Q*Q^T = I  */

   mmtr_mlt(Q,Q,H1);
   for (j=0; j < kk; j++)
     H1->me[j][j] -= 1.0;
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (arnoldi_iref) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);

   ipns1->x = u;
   ipns1->shared_x = TRUE;
   mem_stat_mark(3);
   iter_arnoldi(ipns1,&hh,Q,H);
   mem_stat_free(3);

   /* check the equality:
      Q*A*Q^T = H; */

   vt.dim = vt.max_dim = x->dim;
   vt1.dim = vt1.max_dim = x->dim;
   for (j=0; j < kk; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      sp_mv_mlt(An,&vt,&vt1);
   }

   mmtr_mlt(Q,Q1,H1);
   m_sub(H,H1,H1);
  if (m_norm_inf(H1) > MACHEPS*x->dim)  
     printf(" (arnoldi) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);
   /* check Q*Q^T = I  */
   mmtr_mlt(Q,Q,H1);
   for (j=0; j < kk; j++)
     H1->me[j][j] -= 1.0;
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (arnoldi) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);

   v_rand(u);
   mem_stat_mark(3);
   iter_sparnoldi(An,u,kk,&hh,Q,H);
   mem_stat_free(3);

   /* check the equality:
      Q*A*Q^T = H; */

   vt.dim = vt.max_dim = x->dim;
   vt1.dim = vt1.max_dim = x->dim;
   for (j=0; j < kk; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      sp_mv_mlt(An,&vt,&vt1);
   }

   mmtr_mlt(Q,Q1,H1);
   m_sub(H,H1,H1);
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (sparnoldi) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);
   /* check Q*Q^T = I  */
   mmtr_mlt(Q,Q,H1);
   for (j=0; j < kk; j++)
     H1->me[j][j] -= 1.0;
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (sparnoldi) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);



   /****** LANCZOS METHOD ******/

   notice("lanczos method");
   kk = ipns1->k; 
   Q = m_resize(Q,kk,x->dim);
   Q1 = m_resize(Q1,kk,x->dim);
   H = m_resize(H,kk,kk);
   ips1->k = kk;
   v_rand(u);
   v_free(ips1->x);
   ips1->x = u;
   ips1->shared_x = TRUE;

   mem_stat_mark(3);
   iter_lanczos(ips1,x,y,&hh,Q);
   mem_stat_free(3);

   /* check the equality:
      Q*A*Q^T = H; */

   vt.dim = vt1.dim = Q->n;
   vt.max_dim = vt1.max_dim = Q->max_n;
   Q1 = m_resize(Q1,Q->m,Q->n);
   for (j=0; j < Q->m; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      sp_mv_mlt(A,&vt,&vt1);
   }
   H1 = m_resize(H1,Q->m,Q->m);
   H = m_resize(H,Q->m,Q->m);
   mmtr_mlt(Q,Q1,H1);

   m_zero(H);
   for (j=0; j < Q->m-1; j++) {
      H->me[j][j] = x->ve[j];
      H->me[j][j+1] = H->me[j+1][j] = y->ve[j];
   }
   H->me[Q->m-1][Q->m-1] = x->ve[Q->m-1];

   m_sub(H,H1,H1);
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (lanczos) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);

   /* check Q*Q^T = I  */

   mmtr_mlt(Q,Q,H1);
   for (j=0; j < Q->m; j++)
     H1->me[j][j] -= 1.0;
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (lanczos) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);

   mem_stat_mark(3);
   v_rand(u);
   iter_splanczos(A,kk,u,x,y,&hh,Q);
   mem_stat_free(3);

   /* check the equality:
      Q*A*Q^T = H; */

   vt.dim = vt1.dim = Q->n;
   vt.max_dim = vt1.max_dim = Q->max_n;
   Q1 = m_resize(Q1,Q->m,Q->n);
   for (j=0; j < Q->m; j++) {
      vt.ve = Q->me[j];
      vt1.ve = Q1->me[j];
      sp_mv_mlt(A,&vt,&vt1);
   }
   H1 = m_resize(H1,Q->m,Q->m);
   H = m_resize(H,Q->m,Q->m);
   mmtr_mlt(Q,Q1,H1);
   for (j=0; j < Q->m-1; j++) {
      H->me[j][j] = x->ve[j];
      H->me[j][j+1] = H->me[j+1][j] = y->ve[j];
   }
   H->me[Q->m-1][Q->m-1] = x->ve[Q->m-1];

   m_sub(H,H1,H1);
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (splanczos) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);
   /* check Q*Q^T = I  */
   mmtr_mlt(Q,Q,H1);
   for (j=0; j < Q->m; j++)
     H1->me[j][j] -= 1.0;
   if (m_norm_inf(H1) > MACHEPS*x->dim)
     printf(" (splanczos) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",
	    m_norm_inf(H1),MACHEPS);



   /***** LANCZOS2 ****/

   notice("lanczos2 method");
   kk = 50;  		/* # of dir. vectors */
   ips1->k = kk;
   v_rand(u);
   ips1->x = u;
   ips1->shared_x = TRUE;

   for ( i = 0; i < xn->dim; i++ )
	xn->ve[i] = i;
   iter_Ax(ips1,Dv_mlt,xn);
   mem_stat_mark(3);
   iter_lanczos2(ips1,y,v);
   mem_stat_free(3);

   printf("# Number of steps of Lanczos algorithm = %d\n", kk);
   printf("# Exact eigenvalues are 0, 1, 2, ..., %d\n",ANON-1);
   printf("# Extreme eigenvalues should be accurate; \n");
   printf("# interior values usually are not.\n");
   printf("# approx e-vals =\n");	v_output(y);
   printf("# Error in estimate of bottom e-vec (Lanczos) = %g\n",
	  fabs(v->ve[0]));

   mem_stat_mark(3);
   v_rand(u);
   iter_splanczos2(A,kk,u,y,v);
   mem_stat_free(3);


   /***** FINISHING *******/

   notice("release ITER variables");
   
   M_FREE(Q);
   M_FREE(Q1);
   M_FREE(H);
   M_FREE(H1);

   ITER_FREE(ipns);
   ITER_FREE(ips);
   ITER_FREE(ipns1);
   ITER_FREE(ips1);
   SP_FREE(A);
   SP_FREE(B);
   SP_FREE(An);
   SP_FREE(Bn);
   
   V_FREE(x);
   V_FREE(y);
   V_FREE(u);
   V_FREE(v); 
   V_FREE(xn);
   V_FREE(yn);

   printf("# Done testing (%s)\n",argv[0]);
   mem_info();
}
