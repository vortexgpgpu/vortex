
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


/*
  This file contains routines for computing functions of matrices
  especially polynomials and exponential functions
  Copyright (C) Teresa Leyk and David Stewart, 1993
  */

#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "matrix2.h"

static char	rcsid[] = "$Id: mfunc.c,v 1.2 1994/11/01 05:57:56 des Exp $";



/* _m_pow -- computes integer powers of a square matrix A, A^p
   -- uses tmp as temporary workspace */
#ifndef ANSI_C
MAT	*_m_pow(A, p, tmp, out)
MAT	*A, *tmp, *out;
int	p;
#else
MAT	*_m_pow(const MAT *A, int p, MAT *tmp, MAT *out)
#endif
{
   int		it_cnt, k, max_bit;
   
   /*
     File containing routines for evaluating matrix functions
     esp. the exponential function
     */

#define	Z(k)	(((k) & 1) ? tmp : out)
   
   if ( ! A )
     error(E_NULL,"_m_pow");
   if ( A->m != A->n )
     error(E_SQUARE,"_m_pow");
   if ( p < 0 )
     error(E_NEG,"_m_pow");
   out = m_resize(out,A->m,A->n);
   tmp = m_resize(tmp,A->m,A->n);
   
   if ( p == 0 )
     m_ident(out);
   else if ( p > 0 )
   {
      it_cnt = 1;
      for ( max_bit = 0; ; max_bit++ )
	if ( (p >> (max_bit+1)) == 0 )
	  break;
      tmp = m_copy(A,tmp);
      
      for ( k = 0; k < max_bit; k++ )
      {
	 m_mlt(Z(it_cnt),Z(it_cnt),Z(it_cnt+1));
	 it_cnt++;
	 if ( p & (1 << (max_bit-1)) )
	 {
	    m_mlt(A,Z(it_cnt),Z(it_cnt+1));
	    /* m_copy(Z(it_cnt),out); */
	    it_cnt++;
	 }
	 p <<= 1;
      }
      if (it_cnt & 1)
	out = m_copy(Z(it_cnt),out);
   }

   return out;

#undef Z   
}

/* m_pow -- computes integer powers of a square matrix A, A^p */
#ifndef ANSI_C
MAT	*m_pow(A, p, out)
MAT	*A, *out;
int	p;
#else
MAT	*m_pow(const MAT *A, int p, MAT *out)
#endif
{
   STATIC MAT	*wkspace=MNULL, *tmp=MNULL;
   
   if ( ! A )
     error(E_NULL,"m_pow");
   if ( A->m != A->n )
     error(E_SQUARE,"m_pow");
   
   wkspace = m_resize(wkspace,A->m,A->n);
   MEM_STAT_REG(wkspace,TYPE_MAT);
   if ( p < 0 )
   {
       tmp = m_resize(tmp,A->m,A->n);
       MEM_STAT_REG(tmp,TYPE_MAT);
       tracecatch(m_inverse(A,tmp),"m_pow");
       out = _m_pow(tmp, -p, wkspace, out);
   }
   else
       out = _m_pow(A, p, wkspace, out);

#ifdef	THREADSAFE
   M_FREE(wkspace);	M_FREE(tmp);
#endif

   return out;
}

/**************************************************/

/* _m_exp -- compute matrix exponential of A and save it in out
   -- uses Pade approximation followed by repeated squaring
   -- eps is the tolerance used for the Pade approximation 
   -- A is not changed
   -- q_out - degree of the Pade approximation (q_out,q_out)
   -- j_out - the power of 2 for scaling the matrix A
              such that ||A/2^j_out|| <= 0.5
*/
#ifndef ANSI_C
MAT *_m_exp(A,eps,out,q_out,j_out)
MAT *A,*out;
double eps;
int *q_out, *j_out;
#else
MAT *_m_exp(MAT *A, double eps, MAT *out, int *q_out, int *j_out)
#endif
{
   STATIC MAT *D = MNULL, *Apow = MNULL, *N = MNULL, *Y = MNULL;
   STATIC VEC *c1 = VNULL, *tmp = VNULL;
   VEC y0, y1;  /* additional structures */
   STATIC PERM *pivot = PNULL;
   int j, k, l, q, r, s, j2max, t;
   double inf_norm, eqq, power2, c, sign;
   
   if ( ! A )
     error(E_SIZES,"_m_exp");
   if ( A->m != A->n )
     error(E_SIZES,"_m_exp");
   if ( A == out )
     error(E_INSITU,"_m_exp");
   if ( eps < 0.0 )
     error(E_RANGE,"_m_exp");
   else if (eps == 0.0)
     eps = MACHEPS;
      
   N = m_resize(N,A->m,A->n);
   D = m_resize(D,A->m,A->n);
   Apow = m_resize(Apow,A->m,A->n);
   out = m_resize(out,A->m,A->n);

   MEM_STAT_REG(N,TYPE_MAT);
   MEM_STAT_REG(D,TYPE_MAT);
   MEM_STAT_REG(Apow,TYPE_MAT);
   
   /* normalise A to have ||A||_inf <= 1 */
   inf_norm = m_norm_inf(A);
   if (inf_norm <= 0.0) {
      m_ident(out);
      *q_out = -1;
      *j_out = 0;
      return out;
   }
   else {
      j2max = floor(1+log(inf_norm)/log(2.0));
      j2max = max(0, j2max);
   }
   
   power2 = 1.0;
   for ( k = 1; k <= j2max; k++ )
     power2 *= 2;
   power2 = 1.0/power2;
   if ( j2max > 0 )
     sm_mlt(power2,A,A);
   
   /* compute order for polynomial approximation */
   eqq = 1.0/6.0;
   for ( q = 1; eqq > eps; q++ )
     eqq /= 16.0*(2.0*q+1.0)*(2.0*q+3.0);
   
   /* construct vector of coefficients */
   c1 = v_resize(c1,q+1);
   MEM_STAT_REG(c1,TYPE_VEC);
   c1->ve[0] = 1.0;
   for ( k = 1; k <= q; k++ ) 
     c1->ve[k] = c1->ve[k-1]*(q-k+1)/((2*q-k+1)*(double)k);
   
   tmp = v_resize(tmp,A->n);
   MEM_STAT_REG(tmp,TYPE_VEC);
   
   s = (int)floor(sqrt((double)q/2.0));
   if ( s <= 0 )  s = 1;
   _m_pow(A,s,out,Apow);
   r = q/s;
   
   Y = m_resize(Y,s,A->n);
   MEM_STAT_REG(Y,TYPE_MAT);
   /* y0 and y1 are pointers to rows of Y, N and D */
   y0.dim = y0.max_dim = A->n;   
   y1.dim = y1.max_dim = A->n;
   
   m_zero(Y);
   m_zero(N);
   m_zero(D);
   
   for( j = 0; j < A->n; j++ )
   {
      if (j > 0)
	Y->me[0][j-1] = 0.0;
      y0.ve = Y->me[0];
      y0.ve[j] = 1.0;
      for ( k = 0; k < s-1; k++ )
      {
	 y1.ve = Y->me[k+1];
	 mv_mlt(A,&y0,&y1);
	 y0.ve = y1.ve;
      }

      y0.ve = N->me[j];
      y1.ve = D->me[j];
      t = s*r;
      for ( l = 0; l <= q-t; l++ )
      {
	 c = c1->ve[t+l];
	 sign = ((t+l) & 1) ? -1.0 : 1.0;
	 __mltadd__(y0.ve,Y->me[l],c,     Y->n);
	 __mltadd__(y1.ve,Y->me[l],c*sign,Y->n);
      }
      
      for (k=1; k <= r; k++)
      {
	 v_copy(mv_mlt(Apow,&y0,tmp),&y0);
	 v_copy(mv_mlt(Apow,&y1,tmp),&y1);
	 t = s*(r-k);
	 for (l=0; l < s; l++)
	 {
	    c = c1->ve[t+l];
	    sign = ((t+l) & 1) ? -1.0 : 1.0;
	    __mltadd__(y0.ve,Y->me[l],c,     Y->n);
	    __mltadd__(y1.ve,Y->me[l],c*sign,Y->n);
	 }
      }
   }

   pivot = px_resize(pivot,A->m);
   MEM_STAT_REG(pivot,TYPE_PERM);
   
   /* note that N and D are transposed,
      therefore we use LUTsolve;
      out is saved row-wise, and must be transposed 
      after this */

   LUfactor(D,pivot);
   for (k=0; k < A->n; k++)
   {
      y0.ve = N->me[k];
      y1.ve = out->me[k];
      LUTsolve(D,pivot,&y0,&y1);
   }
   m_transp(out,out); 


   /* Use recursive squaring to turn the normalised exponential to the
      true exponential */

#define Z(k)    ((k) & 1 ? Apow : out)

   for( k = 1; k <= j2max; k++)
      m_mlt(Z(k-1),Z(k-1),Z(k));

   if (Z(k) == out)
     m_copy(Apow,out);
   
   /* output parameters */
   *j_out = j2max;
   *q_out = q;

   /* restore the matrix A */
   sm_mlt(1.0/power2,A,A);

#ifdef	THREADSAFE
   M_FREE(D);	M_FREE(Apow);	M_FREE(N);	M_FREE(Y);
   V_FREE(c1); 	V_FREE(tmp);
   PX_FREE(pivot);
#endif

   return out;

#undef Z
}


/* simple interface for _m_exp */
#ifndef ANSI_C
MAT *m_exp(A,eps,out)
MAT *A,*out;
double eps;
#else
MAT *m_exp(MAT *A, double eps, MAT *out)
#endif
{
   int q_out, j_out;

   return _m_exp(A,eps,out,&q_out,&j_out);
}


/*--------------------------------*/

/* m_poly -- computes sum_i a[i].A^i, where i=0,1,...dim(a);
   -- uses C. Van Loan's fast and memory efficient method  */
#ifndef ANSI_C
MAT *m_poly(A,a,out)
MAT *A,*out;
VEC *a;
#else
MAT *m_poly(const MAT *A, const VEC *a, MAT *out)
#endif
{
   STATIC MAT	*Apow = MNULL, *Y = MNULL;
   STATIC VEC   *tmp = VNULL;
   VEC y0, y1;  /* additional vectors */
   int j, k, l, q, r, s, t;
   
   if ( ! A || ! a )
     error(E_NULL,"m_poly");
   if ( A->m != A->n )
     error(E_SIZES,"m_poly");
   if ( A == out )
     error(E_INSITU,"m_poly");
   
   out = m_resize(out,A->m,A->n);
   Apow = m_resize(Apow,A->m,A->n);
   MEM_STAT_REG(Apow,TYPE_MAT);
   tmp = v_resize(tmp,A->n);
   MEM_STAT_REG(tmp,TYPE_VEC);

   q = a->dim - 1;
   if ( q == 0 ) {
      m_zero(out);
      for (j=0; j < out->n; j++)
	out->me[j][j] = a->ve[0];
      return out;
   }
   else if ( q == 1) {
      sm_mlt(a->ve[1],A,out);
      for (j=0; j < out->n; j++)
	out->me[j][j] += a->ve[0];
      return out;
   }
   
   s = (int)floor(sqrt((double)q/2.0));
   if ( s <= 0 ) s = 1;
   _m_pow(A,s,out,Apow);
   r = q/s;
   
   Y = m_resize(Y,s,A->n);
   MEM_STAT_REG(Y,TYPE_MAT);
   /* pointers to rows of Y */
   y0.dim = y0.max_dim = A->n;
   y1.dim = y1.max_dim = A->n;

   m_zero(Y);
   m_zero(out);
   
#define Z(k)     ((k) & 1 ? tmp : &y0)
#define ZZ(k)    ((k) & 1 ? tmp->ve : y0.ve)

   for( j = 0; j < A->n; j++)
   {
      if( j > 0 )
	Y->me[0][j-1] = 0.0;
      Y->me[0][j] = 1.0;

      y0.ve = Y->me[0];
      for (k = 0; k < s-1; k++)
      {
	 y1.ve = Y->me[k+1];
	 mv_mlt(A,&y0,&y1);
	 y0.ve = y1.ve;
      }
      
      y0.ve = out->me[j];

      t = s*r;
      for ( l = 0; l <= q-t; l++ )
	__mltadd__(y0.ve,Y->me[l],a->ve[t+l],Y->n);
      
      for (k=1; k <= r; k++)
      {
	 mv_mlt(Apow,Z(k-1),Z(k)); 
	 t = s*(r-k);
	 for (l=0; l < s; l++)
	   __mltadd__(ZZ(k),Y->me[l],a->ve[t+l],Y->n);
      }
      if (Z(k) == &y0) v_copy(tmp,&y0);
   }

   m_transp(out,out);

#ifdef	THREADSAFE
   M_FREE(Apow);	M_FREE(Y);	V_FREE(tmp);	
#endif
   
   return out;
}


