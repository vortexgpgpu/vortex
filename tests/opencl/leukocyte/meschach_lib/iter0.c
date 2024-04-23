
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


/* iter0.c  14/09/93 */

/* ITERATIVE METHODS - service functions */

/* functions for creating and releasing ITER structures;
   for memory information;
   for getting some values from an ITER variable;
   for changing values in an ITER variable;
   see also iter.c
*/

#include        <stdio.h>
#include	<math.h>
#include        "iter.h"


static char rcsid[] = "$Id: iter0.c,v 1.3 1995/01/30 14:50:56 des Exp $";


/* standard functions */

/* standard information */
#ifndef ANSI_C
void iter_std_info(ip,nres,res,Bres)
ITER *ip;
double nres;
VEC *res, *Bres;
#else
void iter_std_info(const ITER *ip, double nres, VEC *res, VEC *Bres)
#endif
{
   if (nres >= 0.0)
#ifndef MEX
     printf(" %d. residual = %g\n",ip->steps,nres);
#else
     mexPrintf(" %d. residual = %g\n",ip->steps,nres);
#endif
   else 
#ifndef MEX
     printf(" %d. residual = %g (WARNING !!! should be >= 0) \n",
	    ip->steps,nres);
#else
     mexPrintf(" %d. residual = %g (WARNING !!! should be >= 0) \n",
	       ip->steps,nres);
#endif
}

/* standard stopping criterion */
#ifndef ANSI_C
int iter_std_stop_crit(ip, nres, res, Bres)
ITER *ip;
double nres;
VEC *res, *Bres;
#else
int iter_std_stop_crit(const ITER *ip, double nres, VEC *res, VEC *Bres)
#endif
{
   /* standard stopping criterium */
   if (nres <= ip->init_res*ip->eps) return TRUE; 
   return FALSE;
}


/* iter_get - create a new structure pointing to ITER */
#ifndef ANSI_C
ITER *iter_get(lenb, lenx)
int lenb, lenx;
#else
ITER *iter_get(int lenb, int lenx)
#endif
{
   ITER *ip;

   if ((ip = NEW(ITER)) == (ITER *) NULL)
     error(E_MEM,"iter_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ITER,0,sizeof(ITER));
      mem_numvar(TYPE_ITER,1);
   }

   /* default values */
   
   ip->shared_x = FALSE;
   ip->shared_b = FALSE;
   ip->k = 0;
   ip->limit = ITER_LIMIT_DEF;
   ip->eps = ITER_EPS_DEF;
   ip->steps = 0;

   if (lenb > 0) ip->b = v_get(lenb);
   else ip->b = (VEC *)NULL;

   if (lenx > 0) ip->x = v_get(lenx);
   else ip->x = (VEC *)NULL;

   ip->Ax = (Fun_Ax) NULL;
   ip->A_par = NULL;	
   ip->ATx = (Fun_Ax) NULL;
   ip->AT_par = NULL;
   ip->Bx = (Fun_Ax) NULL;
   ip->B_par = NULL;
   ip->info = iter_std_info;
   ip->stop_crit = iter_std_stop_crit;
   ip->init_res = 0.0;
   
   return ip;
}


/* iter_free - release memory */
#ifndef ANSI_C
int iter_free(ip)
ITER *ip;
#else
int iter_free(ITER *ip)
#endif
{
   if (ip == (ITER *)NULL) return -1;
   
   if (mem_info_is_on()) {
      mem_bytes(TYPE_ITER,sizeof(ITER),0);
      mem_numvar(TYPE_ITER,-1);
   }

   if ( !ip->shared_x && ip->x != NULL ) v_free(ip->x);
   if ( !ip->shared_b && ip->b != NULL ) v_free(ip->b);

   free((char *)ip);

   return 0;
}

#ifndef ANSI_C
ITER *iter_resize(ip,new_lenb,new_lenx)
ITER *ip;
int new_lenb, new_lenx;
#else
ITER *iter_resize(ITER *ip, int new_lenb, int new_lenx)
#endif
{
   VEC *old;

   if ( ip == (ITER *) NULL)
     error(E_NULL,"iter_resize");

   old = ip->x;
   ip->x = v_resize(ip->x,new_lenx);
   if ( ip->shared_x && old != ip->x )
     warning(WARN_SHARED_VEC,"iter_resize");
   old = ip->b;
   ip->b = v_resize(ip->b,new_lenb);
   if ( ip->shared_b && old != ip->b )
     warning(WARN_SHARED_VEC,"iter_resize");

   return ip;
}

#ifndef MEX
/* print out ip structure - for diagnostic purposes mainly */
#ifndef ANSI_C
void iter_dump(fp,ip)
ITER *ip;
FILE *fp;
#else
void iter_dump(FILE *fp, ITER *ip)
#endif
{
   if (ip == NULL) {
      fprintf(fp," ITER structure: NULL\n");
      return;
   }

   fprintf(fp,"\n ITER structure:\n");
   fprintf(fp," ip->shared_x = %s, ip->shared_b = %s\n",
	   (ip->shared_x ? "TRUE" : "FALSE"),
	   (ip->shared_b ? "TRUE" : "FALSE") );
   fprintf(fp," ip->k = %d, ip->limit = %d, ip->steps = %d, ip->eps = %g\n",
	   ip->k,ip->limit,ip->steps,ip->eps);
   fprintf(fp," ip->x = 0x%p, ip->b = 0x%p\n",ip->x,ip->b);
   fprintf(fp," ip->Ax = 0x%p, ip->A_par = 0x%p\n",ip->Ax,ip->A_par);
   fprintf(fp," ip->ATx = 0x%p, ip->AT_par = 0x%p\n",ip->ATx,ip->AT_par);
   fprintf(fp," ip->Bx = 0x%p, ip->B_par = 0x%p\n",ip->Bx,ip->B_par);
   fprintf(fp," ip->info = 0x%p, ip->stop_crit = 0x%p, ip->init_res = %g\n",
	   ip->info,ip->stop_crit,ip->init_res);
   fprintf(fp,"\n");
   
}
#endif

/* copy the structure ip1 to ip2 preserving vectors x and b of ip2
   (vectors x and b in ip2 are the same before and after iter_copy2)
   if ip2 == NULL then a new structure is created with x and b being NULL
   and other members are taken from ip1
*/
#ifndef ANSI_C
ITER *iter_copy2(ip1,ip2)
ITER *ip1, *ip2;
#else
ITER *iter_copy2(ITER *ip1, ITER *ip2)
#endif
{
   VEC *x, *b;
   int shx, shb;

   if (ip1 == (ITER *)NULL) 
     error(E_NULL,"iter_copy2");

   if (ip2 == (ITER *)NULL) {
      if ((ip2 = NEW(ITER)) == (ITER *) NULL)
	error(E_MEM,"iter_copy2");
      else if (mem_info_is_on()) {
	 mem_bytes(TYPE_ITER,0,sizeof(ITER));
	 mem_numvar(TYPE_ITER,1);
      }
      ip2->x = ip2->b = NULL;
      ip2->shared_x = ip2->shared_x = FALSE;
   }

   x = ip2->x;
   b = ip2->b;
   shb = ip2->shared_b;
   shx = ip2->shared_x;
   MEM_COPY(ip1,ip2,sizeof(ITER));
   ip2->x = x;
   ip2->b = b;
   ip2->shared_x = shx;
   ip2->shared_b = shb;

   return ip2;
}


/* copy the structure ip1 to ip2 copying also the vectors x and b */
#ifndef ANSI_C
ITER *iter_copy(ip1,ip2)
ITER *ip1, *ip2;
#else
ITER *iter_copy(const ITER *ip1, ITER *ip2)
#endif
{
   VEC *x, *b;

   if (ip1 == (ITER *)NULL) 
     error(E_NULL,"iter_copy");

   if (ip2 == (ITER *)NULL) {
      if ((ip2 = NEW(ITER)) == (ITER *) NULL)
	error(E_MEM,"iter_copy2");
      else if (mem_info_is_on()) {
	 mem_bytes(TYPE_ITER,0,sizeof(ITER));
	 mem_numvar(TYPE_ITER,1);
      }
   }

   x = ip2->x;
   b = ip2->b;

   MEM_COPY(ip1,ip2,sizeof(ITER));
   if (ip1->x)
     ip2->x = v_copy(ip1->x,x);
   if (ip1->b)
     ip2->b = v_copy(ip1->b,b);

   ip2->shared_x = ip2->shared_b = FALSE;

   return ip2;
}


/*** functions to generate sparse matrices with random entries ***/


/* iter_gen_sym -- generate symmetric positive definite
   n x n matrix, 
   nrow - number of nonzero entries in a row
   */
#ifndef ANSI_C
SPMAT	*iter_gen_sym(n,nrow)
int	n, nrow;
#else
SPMAT	*iter_gen_sym(int n, int nrow)
#endif
{
   SPMAT	*A;
   VEC	        *u;
   Real       s1;
   int		i, j, k, k_max;
   
   if (nrow <= 1) nrow = 2;
   /* nrow should be even */
   if ((nrow & 1)) nrow -= 1;
   A = sp_get(n,n,nrow);
   u = v_get(A->m);
   v_zero(u);
   for ( i = 0; i < A->m; i++ )
   {
      k_max = ((rand() >> 8) % (nrow/2));
      for ( k = 0; k <= k_max; k++ )
      {
	 j = (rand() >> 8) % A->n;
	 s1 = mrand();
	 sp_set_val(A,i,j,s1);
	 sp_set_val(A,j,i,s1);
	 u->ve[i] += fabs(s1);
	 u->ve[j] += fabs(s1);
      }
   }
   /* ensure that A is positive definite */
   for ( i = 0; i < A->m; i++ )
     sp_set_val(A,i,i,u->ve[i] + 1.0);
   
   V_FREE(u);
   return A;
}


/* iter_gen_nonsym -- generate non-symmetric m x n sparse matrix, m >= n 
   nrow - number of entries in a row;
   diag - number which is put in diagonal entries and then permuted
   (if diag is zero then 1.0 is there)
*/
#ifndef ANSI_C
SPMAT	*iter_gen_nonsym(m,n,nrow,diag)
int	m, n, nrow;
double diag;
#else
SPMAT	*iter_gen_nonsym(int m, int n, int nrow, double diag)
#endif
{
   SPMAT	*A;
   PERM		*px;
   int		i, j, k, k_max;
   Real		s1;
   
   if (nrow <= 1) nrow = 2;
   if (diag == 0.0) diag = 1.0;
   A = sp_get(m,n,nrow);
   px = px_get(n);
   for ( i = 0; i < A->m; i++ )
   {
      k_max = (rand() >> 8) % (nrow-1);
      for ( k = 0; k <= k_max; k++ )
      {
	 j = (rand() >> 8) % A->n;
	 s1 = mrand();
	 sp_set_val(A,i,j,-s1);
      }
   }
   /* to make it likely that A is nonsingular, use pivot... */
   for ( i = 0; i < 2*A->n; i++ )
   {
      j = (rand() >> 8) % A->n;
      k = (rand() >> 8) % A->n;
      px_transp(px,j,k);
   }
   for ( i = 0; i < A->n; i++ )
     sp_set_val(A,i,px->pe[i],diag);  
   
   PX_FREE(px);
   return A;
}

#if ( 0 )
/* iter_gen_nonsym -- generate non-symmetric positive definite 
   n x n sparse matrix;
   nrow - number of entries in a row
*/
#ifndef ANSI_C
SPMAT	*iter_gen_nonsym_posdef(n,nrow)
int	n, nrow;
#else
SPMAT	*iter_gen_nonsym(int m, int n, int nrow, double diag)
#endif
{
   SPMAT	*A;
   PERM		*px;
   VEC          *u;
   int		i, j, k, k_max;
   Real		s1;
   
   if (nrow <= 1) nrow = 2;
   A = sp_get(n,n,nrow);
   px = px_get(n);
   u = v_get(A->m);
   v_zero(u);
   for ( i = 0; i < A->m; i++ )
   {
      k_max = (rand() >> 8) % (nrow-1);
      for ( k = 0; k <= k_max; k++ )
      {
	 j = (rand() >> 8) % A->n;
	 s1 = mrand();
	 sp_set_val(A,i,j,-s1);
	 u->ve[i] += fabs(s1);
      }
   }
   /* ensure that A is positive definite */
   for ( i = 0; i < A->m; i++ )
     sp_set_val(A,i,i,u->ve[i] + 1.0);
   
   PX_FREE(px);
   V_FREE(u);
   return A;
}
#endif




