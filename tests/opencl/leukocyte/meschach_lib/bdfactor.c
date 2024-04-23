

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
  Band matrix factorisation routines
  */

/* bdfactor.c  18/11/93 */
static	char	rcsid[] = "$Id: ";

#include	<stdio.h>
#include	<math.h>
#include        "matrix2.h"


/* generate band matrix 
   for a matrix  with n columns,
   lb subdiagonals and ub superdiagonals;

   Way of saving a band of a matrix:
   first we save subdiagonals (from 0 to lb-1);
   then main diagonal (in the lb row)
   and then superdiagonals (from lb+1 to lb+ub)
   in such a way that the elements which were previously
   in one column are now also in one column
*/
#ifndef ANSI_C
BAND *bd_get(lb,ub,n)
int lb, ub, n;
#else
BAND *bd_get(int lb, int ub, int n)
#endif
{
   BAND *A;

   if (lb < 0 || ub < 0 || n <= 0)
     error(E_NEG,"bd_get");

   if ((A = NEW(BAND)) == (BAND *)NULL)
     error(E_MEM,"bd_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_BAND,0,sizeof(BAND));
      mem_numvar(TYPE_BAND,1);
   }

   lb = A->lb = min(n-1,lb);
   ub = A->ub = min(n-1,ub);
   A->mat = m_get(lb+ub+1,n);
   return A;
}

/* bd_free -- frees BAND matrix -- returns (-1) on error and 0 otherwise */
#ifndef ANSI_C
int bd_free(A)
BAND *A;
#else
int bd_free(BAND *A)
#endif
{
   if ( A == (BAND *)NULL || A->lb < 0 || A->ub < 0 )
     /* don't trust it */
     return (-1);

   if (A->mat) m_free(A->mat);

   if (mem_info_is_on()) {
      mem_bytes(TYPE_BAND,sizeof(BAND),0);
      mem_numvar(TYPE_BAND,-1);
   }

   free((char *)A);
   return 0;
}


/* resize band matrix */
#ifndef ANSI_C
BAND *bd_resize(A,new_lb,new_ub,new_n)
BAND *A;
int new_lb,new_ub,new_n;
#else
BAND *bd_resize(BAND *A, int new_lb, int new_ub, int new_n)
#endif
{
   int lb,ub,i,j,l,shift,umin;
   Real **Av;

   if (new_lb < 0 || new_ub < 0 || new_n <= 0)
     error(E_NEG,"bd_resize");
   if ( ! A )
     return bd_get(new_lb,new_ub,new_n);
    if ( A->lb+A->ub+1 > A->mat->m )
	error(E_INTERN,"bd_resize");

   if ( A->lb == new_lb && A->ub == new_ub && A->mat->n == new_n )
	return A;

   lb = A->lb;
   ub = A->ub;
   Av = A->mat->me;
   umin = min(ub,new_ub);

    /* ensure that unused triangles at edges are zero'd */

   for ( i = 0; i < lb; i++ )
      for ( j = A->mat->n - lb + i; j < A->mat->n; j++ )
	Av[i][j] = 0.0;  
    for ( i = lb+1,l=1; l <= umin; i++,l++ )
      for ( j = 0; j < l; j++ )
	Av[i][j] = 0.0; 

   new_lb = A->lb = min(new_lb,new_n-1);
   new_ub = A->ub = min(new_ub,new_n-1);
   A->mat = m_resize(A->mat,new_lb+new_ub+1,new_n);
   Av = A->mat->me;

   /* if new_lb != lb then move the rows to get the main diag 
      in the new_lb row */

   if (new_lb > lb) {
      shift = new_lb-lb;

      for (i=lb+umin, l=i+shift; i >= 0; i--,l--)
	MEM_COPY(Av[i],Av[l],new_n*sizeof(Real));
      for (l=shift-1; l >= 0; l--)
	__zero__(Av[l],new_n);
   }
   else if (new_lb < lb) { 
      shift = lb - new_lb;

      for (i=shift, l=0; i <= lb+umin; i++,l++)
	MEM_COPY(Av[i],Av[l],new_n*sizeof(Real));
      for (i=lb+umin+1; i <= new_lb+new_ub; i++)
	__zero__(Av[i],new_n);
   }

   return A;
}


/* bd_copy -- copies band matrix A to B, returning B
	-- if B is NULL, create
	-- B is set to the correct size */
#ifndef ANSI_C
BAND *bd_copy(A,B)
BAND *A,*B;
#else
BAND *bd_copy(const BAND *A, BAND *B)
#endif
{
   int lb,ub,i,j,n;
   
   if ( !A )
     error(E_NULL,"bd_copy");

   if (A == B) return B;
   
   n = A->mat->n;
   if ( !B )
     B = bd_get(A->lb,A->ub,n);
   else if (B->lb != A->lb || B->ub != A->ub || B->mat->n != n )
     B = bd_resize(B,A->lb,A->ub,n);
   
   if (A->mat == B->mat) return B;
   ub = B->ub = A->ub;
   lb = B->lb = A->lb;

   for ( i=0, j=n-lb; i <= lb; i++, j++ )
     MEM_COPY(A->mat->me[i],B->mat->me[i],j*sizeof(Real));   

   for ( i=lb+1, j=1; i <= lb+ub; i++, j++ )
     MEM_COPY(A->mat->me[i]+j,B->mat->me[i]+j,(n - j)*sizeof(Real));     

   return B;
}


/* copy band matrix bA to a square matrix A returning A */
#ifndef ANSI_C
MAT *band2mat(bA,A)
BAND *bA;
MAT *A;
#else
MAT *band2mat(const BAND *bA, MAT *A)
#endif
{
   int i,j,l,n,n1;
   int lb, ub;
   Real **bmat;

   if ( !bA )
     error(E_NULL,"band2mat");
   if ( bA->mat == A )
     error(E_INSITU,"band2mat");

   ub = bA->ub;
   lb = bA->lb;
   n = bA->mat->n;
   n1 = n-1;
   bmat = bA->mat->me;

   A = m_resize(A,n,n);
   m_zero(A);

   for (j=0; j < n; j++)
     for (i=min(n1,j+lb),l=lb+j-i; i >= max(0,j-ub); i--,l++)
       A->me[i][j] = bmat[l][j];

   return A;
}

/* copy a square matrix to a band matrix with 
   lb subdiagonals and ub superdiagonals */
#ifndef ANSI_C
BAND *mat2band(A,lb,ub,bA)
BAND *bA;
MAT *A;
int lb, ub;
#else
BAND *mat2band(const MAT *A, int lb, int ub,BAND *bA)
#endif
{
   int i, j, l, n1;
   Real **bmat;
   
   if (! A )
     error(E_NULL,"mat2band");
   if (ub < 0 || lb < 0)
     error(E_SIZES,"mat2band");
   if ( bA != (BAND *)NULL && bA->mat == A )
     error(E_INSITU,"mat2band");

   n1 = A->n-1;
   lb = min(n1,lb);
   ub = min(n1,ub);
   bA = bd_resize(bA,lb,ub,n1+1);
   bmat = bA->mat->me;

   for (j=0; j <= n1; j++)
     for (i=min(n1,j+lb),l=lb+j-i; i >= max(0,j-ub); i--,l++)
       bmat[l][j] = A->me[i][j];

   return bA;
}



/* transposition of matrix in;
   out - matrix after transposition;
   can be done in situ
*/
#ifndef ANSI_C
BAND *bd_transp(in,out)
BAND *in, *out;
#else
BAND *bd_transp(const BAND *in, BAND *out)
#endif
{
   int i, j, jj, l, k, lb, ub, lub, n, n1;
   int in_situ;
   Real  **in_v, **out_v;
   
   if ( in == (BAND *)NULL || in->mat == (MAT *)NULL )
     error(E_NULL,"bd_transp");

   lb = in->lb;
   ub = in->ub;
   lub = lb+ub;
   n = in->mat->n;
   n1 = n-1;

   in_situ = ( in == out );
   if ( ! in_situ )
       out = bd_resize(out,ub,lb,n);
   else
   {   /* only need to swap lb and ub fields */
       out->lb = ub;
       out->ub = lb;
   }

   in_v = in->mat->me;
   
   if (! in_situ) {
      int sh_in,sh_out; 

      out_v = out->mat->me;
      for (i=0, l=lub, k=lb-i; i <= lub; i++,l--,k--) {
	 sh_in = max(-k,0);
	 sh_out = max(k,0);
	 MEM_COPY(&(in_v[i][sh_in]),&(out_v[l][sh_out]),
		  (n-sh_in-sh_out)*sizeof(Real));
	 /**********************************
	 for (j=n1-sh_out, jj=n1-sh_in; j >= sh_in; j--,jj--) {
	    out_v[l][jj] = in_v[i][j];
	 }
	 **********************************/
      }
   }
   else if (ub == lb) {
      Real tmp;

      for (i=0, l=lub, k=lb-i; i < lb; i++,l--,k--) {
	 for (j=n1-k, jj=n1; j >= 0; j--,jj--) {
	    tmp = in_v[l][jj];
	    in_v[l][jj] = in_v[i][j];
	    in_v[i][j] = tmp;
	 }
      }
   }
   else if (ub > lb) {  /* hence i-ub <= 0 & l-lb >= 0 */
      int p,pp,lbi;
      
      for (i=0, l=lub; i < (lub+1)/2; i++,l--) {
	 lbi = lb-i;
	 for (j=l-lb, jj=0, p=max(-lbi,0), pp = max(l-ub,0); j <= n1; 
	      j++,jj++,p++,pp++) {
	    in_v[l][pp] = in_v[i][p];
	    in_v[i][jj] = in_v[l][j];
	 }
	 for (  ; p <= n1-max(lbi,0); p++,pp++)
	   in_v[l][pp] = in_v[i][p];
      }
      
      if (lub%2 == 0) { /* shift only */
	 i = lub/2;
	 for (j=max(i-lb,0), jj=0; jj <= n1-ub+i; j++,jj++) 
	   in_v[i][jj] = in_v[i][j];
      }
   }
   else {      /* ub < lb, hence ub-l <= 0 & lb-i >= 0 */
      int p,pp,ubi;

      for (i=0, l=lub; i < (lub+1)/2; i++,l--) {
	 ubi = i-ub;
	 for (j=n1-max(lb-l,0), jj=n1-max(-ubi,0), p=n1-lb+i, pp=n1;
	      p >= 0; j--, jj--, pp--, p--) {
	    in_v[i][jj] = in_v[l][j];
	    in_v[l][pp] = in_v[i][p];
	 }
	 for (  ; jj >= max(ubi,0); j--, jj--)
	   in_v[i][jj] = in_v[l][j];
      }

      if (lub%2 == 0) {  /* shift only */
	 i = lub/2;
	 for (j=n1-lb+i, jj=n1-max(ub-i,0); j >= 0; j--, jj--) 
	    in_v[i][jj] = in_v[i][j];
      }
   }

   return out;
}

/* bdv_mltadd -- band matrix-vector multiply and add
   -- returns out <- x + s.bA.y
   -- if y is NULL then create y (as zero vector)
   -- error if either A or x is NULL */
#ifndef ANSI_C
VEC	*bdv_mltadd(x,y,bA,s,out)
     BAND	*bA;
     VEC	*x, *y;
     double	s;
     VEC *out;
#else
VEC	*bdv_mltadd(const VEC *x, const VEC *y, const BAND *bA,
		    double s, VEC *out)
#endif
{
  int	i, j;

  if ( ! bA || ! x || ! y )
    error(E_NULL,"bdv_mltadd");
  if ( bA->mat->n != x->dim || y->dim != x->dim )
    error(E_SIZES,"bdv_mltadd");
  if ( ! out || out->dim != x->dim )
    out = v_resize(out,x->dim);
  out = v_copy(x,out);

  for ( j = 0; j < x->dim; j++ )
    for ( i = max(j-bA->ub,0); i <= j+bA->lb && i < x->dim; i++ )
      out->ve[i] += s*bd_get_val(bA,i,j)*y->ve[j];

  return out;
}

/* vbd_mltadd -- band matrix-vector multiply and add
   -- returns out^T <- x^T + s.y^T.bA
   -- if out is NULL then create out (as zero vector)
   -- error if either bA or x is NULL */
#ifndef ANSI_C
VEC	*vbd_mltadd(x,y,bA,s,out)
     BAND	*bA;
     VEC	*x, *y;
     double	s;
     VEC *out;
#else
VEC	*vbd_mltadd(const VEC *x, const VEC *y, const BAND *bA,
		    double s, VEC *out)
#endif
{
  int	i, j;

  if ( ! bA || ! x || ! y )
    error(E_NULL,"vbd_mltadd");
  if ( bA->mat->n != x->dim || y->dim != x->dim )
    error(E_SIZES,"vbd_mltadd");
  if ( ! out || out->dim != x->dim )
    out = v_resize(out,x->dim);
  out = v_copy(x,out);

  for ( j = 0; j < x->dim; j++ )
    for ( i = max(j-bA->ub,0); i <= j+bA->lb && i < x->dim; i++ )
      out->ve[j] += s*bd_get_val(bA,i,j)*y->ve[i];

  return out;
}

/* bd_zero -- zeros band matrix A which is returned */
#ifndef ANSI_C
BAND	*bd_zero(A)
BAND	*A;
#else
BAND	*bd_zero(BAND *A)
#endif
{
  if ( ! A )
    error(E_NULL,"bd_zero");

  m_zero(A->mat);
  return A;
}

/* bds_mltadd -- returns OUT <- A+alpha*B
	-- OUT is created (as zero) if NULL
	-- if OUT is not the correct size, it is re-sized before the operation
	-- if A or B are null, and error is generated */
#ifndef ANSI_C
BAND	*bds_mltadd(A,B,alpha,OUT)
BAND	*A, *B, *OUT;
Real	alpha;
#else
BAND	*bds_mltadd(const BAND *A, const BAND *B, double alpha, BAND *OUT)
#endif
{
  int	i;

  if ( ! A || ! B )
    error(E_NULL,"bds_mltadd");
  if ( A->mat->n != B->mat->n )
    error(E_SIZES,"bds_mltadd");
  if ( A == OUT || B == OUT )
    error(E_INSITU,"bds_mltadd");

  OUT = bd_copy(A,OUT);
  OUT = bd_resize(OUT,max(A->lb,B->lb),max(A->ub,B->ub),A->mat->n);
  for ( i = 0; i <= B->lb + B->ub; i++ )
    __mltadd__(OUT->mat->me[i+OUT->lb-B->lb],B->mat->me[i],alpha,B->mat->n);
  
  return OUT;
}

/* sbd_mlt -- returns OUT <- s.A */
#ifndef ANSI_C
BAND	*sbd_mlt(Real s, BAND *A, BAND *OUT)
#else
BAND	*sbd_mlt(Real s, const BAND *A, BAND *OUT)
#endif
{
  if ( ! A )
    error(E_NULL,"sbd_mlt");

  OUT = bd_resize(OUT,A->lb,A->ub,A->mat->n);
  sm_mlt(s,A->mat,OUT->mat);

  return OUT;
}

/* bdLUfactor -- gaussian elimination with partial pivoting
   -- on entry, the matrix A in band storage with elements 
      in rows 0 to lb+ub; 
      The jth column of A is stored in the jth column of 
      band A (bA) as follows:
      bA->mat->me[lb+j-i][j] = A->me[i][j] for 
      max(0,j-lb) <= i <= min(A->n-1,j+ub);
   -- on exit: U is stored as an upper triangular matrix
      with lb+ub superdiagonals in rows lb to 2*lb+ub, 
      and the matrix L is stored in rows 0 to lb-1.
      Matrix U is permuted, whereas L is not permuted !!!
      Therefore we save some memory.
   */
#ifndef ANSI_C
BAND	*bdLUfactor(bA,pivot)
BAND	*bA;
PERM	*pivot;
#else
BAND	*bdLUfactor(BAND *bA, PERM *pivot)
#endif
{
   int	i, j, k, l, n, n1, lb, ub, lub, k_end, k_lub;
   int	i_max, shift;
   Real	**bA_v;
   Real max1, temp;
   
   if ( bA==(BAND *)NULL || pivot==(PERM *)NULL )
     error(E_NULL,"bdLUfactor");

   lb = bA->lb;
   ub = bA->ub;
   lub = lb+ub;
   n = bA->mat->n;
   n1 = n-1;
   lub = lb+ub;

   if ( pivot->size != n )
     error(E_SIZES,"bdLUfactor");

   
   /* initialise pivot with identity permutation */
   for ( i=0; i < n; i++ )
     pivot->pe[i] = i;

   /* extend band matrix */
   /* extended part is filled with zeros */
   bA = bd_resize(bA,lb,min(n1,lub),n);
   bA_v = bA->mat->me;


   /* main loop */

   for ( k=0; k < n1; k++ )
   {
      k_end = max(0,lb+k-n1);
      k_lub = min(k+lub,n1);

      /* find the best pivot row */
      
      max1 = 0.0;	
      i_max = -1;
      for ( i=lb; i >= k_end; i-- ) {
	 temp = fabs(bA_v[i][k]);
	 if ( temp > max1 )
	 { max1 = temp;	i_max = i; }
      }
      
      /* if no pivot then ignore column k... */
      if ( i_max == -1 )
	continue;
      
      /* do we pivot ? */
      if ( i_max != lb )	/* yes we do... */
      {
	 /* save transposition using non-shifted indices */
	 shift = lb-i_max;
	 px_transp(pivot,k+shift,k);
	 for ( i=lb, j=k; j <= k_lub; i++,j++ )
	 {
	    temp = bA_v[i][j];
	    bA_v[i][j] = bA_v[i-shift][j];
	    bA_v[i-shift][j] = temp;
	 }
      }
      
      /* row operations */
      for ( i=lb-1; i >= k_end; i-- ) {
	 temp = bA_v[i][k] /= bA_v[lb][k];
	 shift = lb-i;
	 for ( j=k+1,l=i+1; j <= k_lub; l++,j++ )
	   bA_v[l][j] -= temp*bA_v[l+shift][j];
      }
   }
   
   return bA;
}


/* bdLUsolve -- given an LU factorisation in bA, solve bA*x=b */
/* pivot is changed upon return  */
#ifndef ANSI_C
VEC	*bdLUsolve(bA,pivot,b,x)
BAND	*bA;
PERM	*pivot;
VEC	*b,*x;
#else
VEC	*bdLUsolve(const BAND *bA, PERM *pivot, const VEC *b, VEC *x)
#endif
{
   int i,j,l,n,n1,pi,lb,ub,jmin, maxj;
   Real c;
   Real **bA_v;

   if ( bA==(BAND *)NULL || b==(VEC *)NULL || pivot==(PERM *)NULL )
     error(E_NULL,"bdLUsolve");
   if ( bA->mat->n != b->dim || bA->mat->n != pivot->size)
     error(E_SIZES,"bdLUsolve");
 
   lb = bA->lb;
   ub = bA->ub;
   n = b->dim;
   n1 = n-1;
   bA_v = bA->mat->me;

   x = v_resize(x,b->dim);
   px_vec(pivot,b,x);

   /* solve Lx = b; implicit diagonal = 1 
      L is not permuted, therefore it must be permuted now
    */
   
   px_inv(pivot,pivot);
   for (j=0; j < n; j++) {
      jmin = j+1;
      c = x->ve[j];
      maxj = max(0,j+lb-n1);
      for (i=jmin,l=lb-1; l >= maxj; i++,l--) {
	 if ( (pi = pivot->pe[i]) < jmin) 
	   pi = pivot->pe[i] = pivot->pe[pi];
	 x->ve[pi] -= bA_v[l][j]*c;
      }
   }

   /* solve Ux = b; explicit diagonal */

   x->ve[n1] /= bA_v[lb][n1];
   for (i=n-2; i >= 0; i--) {
      c = x->ve[i];
      for (j=min(n1,i+ub), l=lb+j-i; j > i; j--,l--)
	c -= bA_v[l][j]*x->ve[j];
      x->ve[i] = c/bA_v[lb][i];
   }
   
   return (x);
}

/* LDLfactor -- L.D.L' factorisation of A in-situ;
   A is a band matrix
   it works using only lower bandwidth & main diagonal
   so it is possible to set A->ub = 0
 */
#ifndef ANSI_C
BAND *bdLDLfactor(A)
BAND *A;
#else
BAND *bdLDLfactor(BAND *A)
#endif
{
   int i,j,k,n,n1,lb,ki,jk,ji,lbkm,lbkp;
   Real **Av;
   Real c, cc;

   if ( ! A )
     error(E_NULL,"bdLDLfactor");

   if (A->lb == 0) return A;

   lb = A->lb;
   n = A->mat->n;
   n1 = n-1;
   Av = A->mat->me;
   
   for (k=0; k < n; k++) {    
      lbkm = lb-k;
      lbkp = lb+k;

      /* matrix D */
      c = Av[lb][k];
      for (j=max(0,-lbkm), jk=lbkm+j; j < k; j++, jk++) {
	 cc = Av[jk][j];
	 c -= Av[lb][j]*cc*cc;
      }
      if (c == 0.0)
	error(E_SING,"bdLDLfactor");
      Av[lb][k] = c;

      /* matrix L */
      
      for (i=min(n1,lbkp), ki=lbkp-i; i > k; i--,ki++) {
	 c = Av[ki][k];
	 for (j=max(0,i-lb), ji=lb+j-i, jk=lbkm+j; j < k;
	      j++, ji++, jk++)
	   c -= Av[lb][j]*Av[ji][j]*Av[jk][j];
	 Av[ki][k] = c/Av[lb][k];
      }
   }
   
   return A;
}

/* solve A*x = b, where A is factorized by 
   Choleski LDL^T factorization */
#ifndef ANSI_C
VEC    *bdLDLsolve(A,b,x)
BAND   *A;
VEC    *b, *x;
#else
VEC    *bdLDLsolve(const BAND *A, const VEC *b, VEC *x)
#endif
{
   int i,j,l,n,n1,lb,ilb;
   Real **Av, *Avlb;
   Real c;

   if ( ! A || ! b )
     error(E_NULL,"bdLDLsolve");
   if ( A->mat->n != b->dim )
     error(E_SIZES,"bdLDLsolve");

   n = A->mat->n;
   n1 = n-1;
   x = v_resize(x,n);
   lb = A->lb;
   Av = A->mat->me;  
   Avlb = Av[lb];
   
   /* solve L*y = b */
   x->ve[0] = b->ve[0];
   for (i=1; i < n; i++) {
      ilb = i-lb;
      c = b->ve[i];
      for (j=max(0,ilb), l=j-ilb; j < i; j++,l++)
	c -= Av[l][j]*x->ve[j];
      x->ve[i] = c;
   }

   /* solve D*z = y */
   for (i=0; i < n; i++) 
     x->ve[i] /= Avlb[i];

   /* solve L^T*x = z */
   for (i=n-2; i >= 0; i--) {
      ilb = i+lb;
      c = x->ve[i];
      for (j=min(n1,ilb), l=ilb-j; j > i; j--,l++)
	c -= Av[l][i]*x->ve[j];
      x->ve[i] = c;
   }

   return x;
}


/* ******************************************************
  This function is a contribution from Ruediger Franke.
   His e-mail addres is: Ruediger.Franke@rz.tu-ilmenau.de
   
   ******************************************************
*/

/* bd_mv_mlt --
 *   computes out = A * x
 *   may not work in situ (x != out)
 */

VEC *bd_mv_mlt(A, x, out)
BAND *A;
VEC *x, *out;
{
  int i, j, j_end, k;
  int start_idx, end_idx;
  int n, m, lb, ub;
  Real **A_me;
  Real *x_ve;
  Real sum;

  if (!A || !x)
    error(E_NULL,"bd_mv_mlt");
  if (x->dim != A->mat->n)
    error(E_SIZES,"bd_mv_mlt");
  if (!out || out->dim != A->mat->n)
    out = v_resize(out, A->mat->n);
  if (out == x)
    error(E_INSITU,"bd_mv_mlt");

  n = A->mat->n;
  m = A->mat->m;
  lb = A->lb;
  ub = A->ub;
  A_me = A->mat->me;
  start_idx = lb;
  end_idx = m + n-1 - ub;
  for (i=0; i<n; i++, start_idx--, end_idx--) {
    j = max(0, start_idx);
    k = max(0, -start_idx);
    j_end = min(m, end_idx);
    x_ve = x->ve + k;
    sum = 0.0;	     
    for (; j < j_end; j++, k++)
      sum += A_me[j][k] * *x_ve++;
    out->ve[i] = sum;
  }

  return out;
}



