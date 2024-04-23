
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



#include	<stdio.h>
#include	"zmatrix.h"

static	char	rcsid[] = "$Id: zmatop.c,v 1.2 1995/03/27 15:49:03 des Exp $";


#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)

/* zm_add -- matrix addition -- may be in-situ */
ZMAT	*zm_add(mat1,mat2,out)
ZMAT	*mat1,*mat2,*out;
{
    unsigned int	m,n,i;
    
    if ( mat1==ZMNULL || mat2==ZMNULL )
	error(E_NULL,"zm_add");
    if ( mat1->m != mat2->m || mat1->n != mat2->n )
	error(E_SIZES,"zm_add");
    if ( out==ZMNULL || out->m != mat1->m || out->n != mat1->n )
	out = zm_resize(out,mat1->m,mat1->n);
    m = mat1->m;	n = mat1->n;
    for ( i=0; i<m; i++ )
    {
	__zadd__(mat1->me[i],mat2->me[i],out->me[i],(int)n);
	/**************************************************
	  for ( j=0; j<n; j++ )
	  out->me[i][j] = mat1->me[i][j]+mat2->me[i][j];
	  **************************************************/
    }
    
    return (out);
}

/* zm_sub -- matrix subtraction -- may be in-situ */
ZMAT	*zm_sub(mat1,mat2,out)
ZMAT	*mat1,*mat2,*out;
{
    unsigned int	m,n,i;
    
    if ( mat1==ZMNULL || mat2==ZMNULL )
	error(E_NULL,"zm_sub");
    if ( mat1->m != mat2->m || mat1->n != mat2->n )
	error(E_SIZES,"zm_sub");
    if ( out==ZMNULL || out->m != mat1->m || out->n != mat1->n )
	out = zm_resize(out,mat1->m,mat1->n);
    m = mat1->m;	n = mat1->n;
    for ( i=0; i<m; i++ )
    {
	__zsub__(mat1->me[i],mat2->me[i],out->me[i],(int)n);
	/**************************************************
	  for ( j=0; j<n; j++ )
	  out->me[i][j] = mat1->me[i][j]-mat2->me[i][j];
	**************************************************/
    }
    
    return (out);
}

/*
  Note: In the following routines, "adjoint" means complex conjugate
  transpose:
  A* = conjugate(A^T)
  */

/* zm_mlt -- matrix-matrix multiplication */
ZMAT	*zm_mlt(A,B,OUT)
ZMAT	*A,*B,*OUT;
{
    unsigned int	i, /* j, */ k, m, n, p;
    complex	**A_v, **B_v /*, *B_row, *OUT_row, sum, tmp */;
    
    if ( A==ZMNULL || B==ZMNULL )
	error(E_NULL,"zm_mlt");
    if ( A->n != B->m )
	error(E_SIZES,"zm_mlt");
    if ( A == OUT || B == OUT )
	error(E_INSITU,"zm_mlt");
    m = A->m;	n = A->n;	p = B->n;
    A_v = A->me;		B_v = B->me;
    
    if ( OUT==ZMNULL || OUT->m != A->m || OUT->n != B->n )
	OUT = zm_resize(OUT,A->m,B->n);
    
    /****************************************************************
      for ( i=0; i<m; i++ )
      for  ( j=0; j<p; j++ )
      {
      sum = 0.0;
      for ( k=0; k<n; k++ )
      sum += A_v[i][k]*B_v[k][j];
      OUT->me[i][j] = sum;
      }
    ****************************************************************/
    zm_zero(OUT);
    for ( i=0; i<m; i++ )
	for ( k=0; k<n; k++ )
	{
	    if ( ! is_zero(A_v[i][k]) )
		__zmltadd__(OUT->me[i],B_v[k],A_v[i][k],(int)p,Z_NOCONJ);
	    /**************************************************
	      B_row = B_v[k];	OUT_row = OUT->me[i];
	      for ( j=0; j<p; j++ )
	      (*OUT_row++) += tmp*(*B_row++);
	    **************************************************/
	}
    
    return OUT;
}

/* zmma_mlt -- matrix-matrix adjoint multiplication
   -- A.B* is returned, and stored in OUT */
ZMAT	*zmma_mlt(A,B,OUT)
ZMAT	*A, *B, *OUT;
{
    int	i, j, limit;
    /* complex	*A_row, *B_row, sum; */
    
    if ( ! A || ! B )
	error(E_NULL,"zmma_mlt");
    if ( A == OUT || B == OUT )
	error(E_INSITU,"zmma_mlt");
    if ( A->n != B->n )
	error(E_SIZES,"zmma_mlt");
    if ( ! OUT || OUT->m != A->m || OUT->n != B->m )
	OUT = zm_resize(OUT,A->m,B->m);
    
    limit = A->n;
    for ( i = 0; i < A->m; i++ )
	for ( j = 0; j < B->m; j++ )
	{
	    OUT->me[i][j] = __zip__(B->me[j],A->me[i],(int)limit,Z_CONJ);
	    /**************************************************
	      sum = 0.0;
	      A_row = A->me[i];
	      B_row = B->me[j];
	      for ( k = 0; k < limit; k++ )
	      sum += (*A_row++)*(*B_row++);
	      OUT->me[i][j] = sum;
	      **************************************************/
	}
    
    return OUT;
}

/* zmam_mlt -- matrix adjoint-matrix multiplication
   -- A*.B is returned, result stored in OUT */
ZMAT	*zmam_mlt(A,B,OUT)
ZMAT	*A, *B, *OUT;
{
    int	i, k, limit;
    /* complex	*B_row, *OUT_row, multiplier; */
    complex	tmp;
    
    if ( ! A || ! B )
	error(E_NULL,"zmam_mlt");
    if ( A == OUT || B == OUT )
	error(E_INSITU,"zmam_mlt");
    if ( A->m != B->m )
	error(E_SIZES,"zmam_mlt");
    if ( ! OUT || OUT->m != A->n || OUT->n != B->n )
	OUT = zm_resize(OUT,A->n,B->n);
    
    limit = B->n;
    zm_zero(OUT);
    for ( k = 0; k < A->m; k++ )
	for ( i = 0; i < A->n; i++ )
	{
	    tmp.re =   A->me[k][i].re;
	    tmp.im = - A->me[k][i].im;
	    if ( ! is_zero(tmp) )
		__zmltadd__(OUT->me[i],B->me[k],tmp,(int)limit,Z_NOCONJ);
	}
    
    return OUT;
}

/* zmv_mlt -- matrix-vector multiplication 
   -- Note: b is treated as a column vector */
ZVEC	*zmv_mlt(A,b,out)
ZMAT	*A;
ZVEC	*b,*out;
{
    unsigned int	i, m, n;
    complex	**A_v, *b_v /*, *A_row */;
    /* register complex	sum; */
    
    if ( A==ZMNULL || b==ZVNULL )
	error(E_NULL,"zmv_mlt");
    if ( A->n != b->dim )
	error(E_SIZES,"zmv_mlt");
    if ( b == out )
	error(E_INSITU,"zmv_mlt");
    if ( out == ZVNULL || out->dim != A->m )
	out = zv_resize(out,A->m);
    
    m = A->m;		n = A->n;
    A_v = A->me;		b_v = b->ve;
    for ( i=0; i<m; i++ )
    {
	/* for ( j=0; j<n; j++ )
	   sum += A_v[i][j]*b_v[j]; */
	out->ve[i] = __zip__(A_v[i],b_v,(int)n,Z_NOCONJ);
	/**************************************************
	  A_row = A_v[i];		b_v = b->ve;
	  for ( j=0; j<n; j++ )
	  sum += (*A_row++)*(*b_v++);
	  out->ve[i] = sum;
	**************************************************/
    }
    
    return out;
}

/* zsm_mlt -- scalar-matrix multiply -- may be in-situ */
ZMAT	*zsm_mlt(scalar,matrix,out)
complex	scalar;
ZMAT	*matrix,*out;
{
    unsigned int	m,n,i;
    
    if ( matrix==ZMNULL )
	error(E_NULL,"zsm_mlt");
    if ( out==ZMNULL || out->m != matrix->m || out->n != matrix->n )
	out = zm_resize(out,matrix->m,matrix->n);
    m = matrix->m;	n = matrix->n;
    for ( i=0; i<m; i++ )
	__zmlt__(matrix->me[i],scalar,out->me[i],(int)n);
    /**************************************************
      for ( j=0; j<n; j++ )
      out->me[i][j] = scalar*matrix->me[i][j];
      **************************************************/
    return (out);
}

/* zvm_mlt -- vector adjoint-matrix multiplication */
ZVEC	*zvm_mlt(A,b,out)
ZMAT	*A;
ZVEC	*b,*out;
{
    unsigned int	j,m,n;
    /* complex	sum,**A_v,*b_v; */
    
    if ( A==ZMNULL || b==ZVNULL )
	error(E_NULL,"zvm_mlt");
    if ( A->m != b->dim )
	error(E_SIZES,"zvm_mlt");
    if ( b == out )
	error(E_INSITU,"zvm_mlt");
    if ( out == ZVNULL || out->dim != A->n )
	out = zv_resize(out,A->n);
    
    m = A->m;		n = A->n;
    
    zv_zero(out);
    for ( j = 0; j < m; j++ )
	if ( b->ve[j].re != 0.0 || b->ve[j].im != 0.0  )
	    __zmltadd__(out->ve,A->me[j],b->ve[j],(int)n,Z_CONJ);
    /**************************************************
      A_v = A->me;		b_v = b->ve;
      for ( j=0; j<n; j++ )
      {
      sum = 0.0;
      for ( i=0; i<m; i++ )
      sum += b_v[i]*A_v[i][j];
      out->ve[j] = sum;
      }
      **************************************************/
    
    return out;
}

/* zm_adjoint -- adjoint matrix */
ZMAT	*zm_adjoint(in,out)
ZMAT	*in, *out;
{
    int	i, j;
    int	in_situ;
    complex	tmp;
    
    if ( in == ZMNULL )
	error(E_NULL,"zm_adjoint");
    if ( in == out && in->n != in->m )
	error(E_INSITU2,"zm_adjoint");
    in_situ = ( in == out );
    if ( out == ZMNULL || out->m != in->n || out->n != in->m )
	out = zm_resize(out,in->n,in->m);
    
    if ( ! in_situ )
    {
	for ( i = 0; i < in->m; i++ )
	    for ( j = 0; j < in->n; j++ )
	    {
		out->me[j][i].re =   in->me[i][j].re;
		out->me[j][i].im = - in->me[i][j].im;
	    }
    }
    else
    {
	for ( i = 0 ; i < in->m; i++ )
	{
	    for ( j = 0; j < i; j++ )
	    {
		tmp.re = in->me[i][j].re;
		tmp.im = in->me[i][j].im;
		in->me[i][j].re =   in->me[j][i].re;
		in->me[i][j].im = - in->me[j][i].im;
		in->me[j][i].re =   tmp.re;
		in->me[j][i].im = - tmp.im;
	    }
	    in->me[i][i].im = - in->me[i][i].im;
	}
    }
    
    return out;
}

/* zswap_rows -- swaps rows i and j of matrix A upto column lim */
ZMAT	*zswap_rows(A,i,j,lo,hi)
ZMAT	*A;
int	i, j, lo, hi;
{
    int	k;
    complex	**A_me, tmp;
    
    if ( ! A )
	error(E_NULL,"swap_rows");
    if ( i < 0 || j < 0 || i >= A->m || j >= A->m )
	error(E_SIZES,"swap_rows");
    lo = max(0,lo);
    hi = min(hi,A->n-1);
    A_me = A->me;
    
    for ( k = lo; k <= hi; k++ )
    {
	tmp = A_me[k][i];
	A_me[k][i] = A_me[k][j];
	A_me[k][j] = tmp;
    }
    return A;
}

/* zswap_cols -- swap columns i and j of matrix A upto row lim */
ZMAT	*zswap_cols(A,i,j,lo,hi)
ZMAT	*A;
int	i, j, lo, hi;
{
    int	k;
    complex	**A_me, tmp;
    
    if ( ! A )
	error(E_NULL,"swap_cols");
    if ( i < 0 || j < 0 || i >= A->n || j >= A->n )
	error(E_SIZES,"swap_cols");
    lo = max(0,lo);
    hi = min(hi,A->m-1);
    A_me = A->me;
    
    for ( k = lo; k <= hi; k++ )
    {
	tmp = A_me[i][k];
	A_me[i][k] = A_me[j][k];
	A_me[j][k] = tmp;
    }
    return A;
}

/* mz_mltadd -- matrix-scalar multiply and add
   -- may be in situ
   -- returns out == A1 + s*A2 */
ZMAT	*mz_mltadd(A1,A2,s,out)
ZMAT	*A1, *A2, *out;
complex	s;
{
    /* register complex	*A1_e, *A2_e, *out_e; */
    /* register int	j; */
    int	i, m, n;
    
    if ( ! A1 || ! A2 )
	error(E_NULL,"mz_mltadd");
    if ( A1->m != A2->m || A1->n != A2->n )
	error(E_SIZES,"mz_mltadd");

    if ( out != A1 && out != A2 )
        out = zm_resize(out,A1->m,A1->n);
    
    if ( s.re == 0.0 && s.im == 0.0 )
	return zm_copy(A1,out);
    if ( s.re == 1.0 && s.im == 0.0 )
	return zm_add(A1,A2,out);
    
    out = zm_copy(A1,out);
    
    m = A1->m;	n = A1->n;
    for ( i = 0; i < m; i++ )
    {
	__zmltadd__(out->me[i],A2->me[i],s,(int)n,Z_NOCONJ);
	/**************************************************
	  A1_e = A1->me[i];
	  A2_e = A2->me[i];
	  out_e = out->me[i];
	  for ( j = 0; j < n; j++ )
	  out_e[j] = A1_e[j] + s*A2_e[j];
	  **************************************************/
    }
    
    return out;
}

/* zmv_mltadd -- matrix-vector multiply and add
   -- may not be in situ
   -- returns out == v1 + alpha*A*v2 */
ZVEC	*zmv_mltadd(v1,v2,A,alpha,out)
ZVEC	*v1, *v2, *out;
ZMAT	*A;
complex	alpha;
{
    /* register	int	j; */
    int	i, m, n;
    complex	tmp, *v2_ve, *out_ve;
    
    if ( ! v1 || ! v2 || ! A )
	error(E_NULL,"zmv_mltadd");
    if ( out == v2 )
	error(E_INSITU,"zmv_mltadd");
    if ( v1->dim != A->m || v2->dim != A-> n )
	error(E_SIZES,"zmv_mltadd");
    
    tracecatch(out = zv_copy(v1,out),"zmv_mltadd");
    
    v2_ve = v2->ve;	out_ve = out->ve;
    m = A->m;	n = A->n;
    
    if ( alpha.re == 0.0 && alpha.im == 0.0 )
	return out;
    
    for ( i = 0; i < m; i++ )
    {
	tmp = __zip__(A->me[i],v2_ve,(int)n,Z_NOCONJ);
	out_ve[i].re += alpha.re*tmp.re - alpha.im*tmp.im;
	out_ve[i].im += alpha.re*tmp.im + alpha.im*tmp.re;
	/**************************************************
	  A_e = A->me[i];
	  sum = 0.0;
	  for ( j = 0; j < n; j++ )
	  sum += A_e[j]*v2_ve[j];
	  out_ve[i] = v1->ve[i] + alpha*sum;
	  **************************************************/
    }
    
    return out;
}

/* zvm_mltadd -- vector-matrix multiply and add a la zvm_mlt()
   -- may not be in situ
   -- returns out == v1 + v2*.A */
ZVEC	*zvm_mltadd(v1,v2,A,alpha,out)
ZVEC	*v1, *v2, *out;
ZMAT	*A;
complex	alpha;
{
    int	/* i, */ j, m, n;
    complex	tmp, /* *A_e, */ *out_ve;
    
    if ( ! v1 || ! v2 || ! A )
	error(E_NULL,"zvm_mltadd");
    if ( v2 == out )
	error(E_INSITU,"zvm_mltadd");
    if ( v1->dim != A->n || A->m != v2->dim )
	error(E_SIZES,"zvm_mltadd");
    
    tracecatch(out = zv_copy(v1,out),"zvm_mltadd");
    
    out_ve = out->ve;	m = A->m;	n = A->n;
    for ( j = 0; j < m; j++ )
    {
	/* tmp = zmlt(v2->ve[j],alpha); */
	tmp.re =   v2->ve[j].re*alpha.re - v2->ve[j].im*alpha.im;
	tmp.im =   v2->ve[j].re*alpha.im + v2->ve[j].im*alpha.re;
	if ( tmp.re != 0.0 || tmp.im != 0.0 )
	    __zmltadd__(out_ve,A->me[j],tmp,(int)n,Z_CONJ);
	/**************************************************
	  A_e = A->me[j];
	  for ( i = 0; i < n; i++ )
	  out_ve[i] += A_e[i]*tmp;
	**************************************************/
    }
    
    return out;
}

/* zget_col -- gets a specified column of a matrix; returned as a vector */
ZVEC	*zget_col(mat,col,vec)
int	col;
ZMAT	*mat;
ZVEC	*vec;
{
	unsigned int	i;

	if ( mat==ZMNULL )
		error(E_NULL,"zget_col");
	if ( col < 0 || col >= mat->n )
		error(E_RANGE,"zget_col");
	if ( vec==ZVNULL || vec->dim<mat->m )
		vec = zv_resize(vec,mat->m);

	for ( i=0; i<mat->m; i++ )
	    vec->ve[i] = mat->me[i][col];

	return (vec);
}

/* zget_row -- gets a specified row of a matrix and retruns it as a vector */
ZVEC	*zget_row(mat,row,vec)
int	row;
ZMAT	*mat;
ZVEC	*vec;
{
	int	/* i, */ lim;

	if ( mat==ZMNULL )
		error(E_NULL,"zget_row");
	if ( row < 0 || row >= mat->m )
		error(E_RANGE,"zget_row");
	if ( vec==ZVNULL || vec->dim<mat->n )
		vec = zv_resize(vec,mat->n);

	lim = min(mat->n,vec->dim);

	/* for ( i=0; i<mat->n; i++ ) */
	/*     vec->ve[i] = mat->me[row][i]; */
	MEMCOPY(mat->me[row],vec->ve,lim,complex);

	return (vec);
}

/* zset_col -- sets column of matrix to values given in vec (in situ) */
ZMAT	*zset_col(mat,col,vec)
ZMAT	*mat;
ZVEC	*vec;
int	col;
{
	unsigned int	i,lim;

	if ( mat==ZMNULL || vec==ZVNULL )
		error(E_NULL,"zset_col");
	if ( col < 0 || col >= mat->n )
		error(E_RANGE,"zset_col");
	lim = min(mat->m,vec->dim);
	for ( i=0; i<lim; i++ )
	    mat->me[i][col] = vec->ve[i];

	return (mat);
}

/* zset_row -- sets row of matrix to values given in vec (in situ) */
ZMAT	*zset_row(mat,row,vec)
ZMAT	*mat;
ZVEC	*vec;
int	row;
{
	unsigned int	/* j, */ lim;

	if ( mat==ZMNULL || vec==ZVNULL )
		error(E_NULL,"zset_row");
	if ( row < 0 || row >= mat->m )
		error(E_RANGE,"zset_row");
	lim = min(mat->n,vec->dim);
	/* for ( j=j0; j<lim; j++ ) */
	/*     mat->me[row][j] = vec->ve[j]; */
	MEMCOPY(vec->ve,mat->me[row],lim,complex);

	return (mat);
}

/* zm_rand -- randomise a complex matrix; uniform in [0,1)+[0,1)*i */
ZMAT	*zm_rand(A)
ZMAT	*A;
{
    int		i;

    if ( ! A )
	error(E_NULL,"zm_rand");

    for ( i = 0; i < A->m; i++ )
	mrandlist((Real *)(A->me[i]),2*A->n);

    return A;
}
