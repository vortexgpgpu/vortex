
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
	Matrix factorisation routines to work with the other matrix files.
	Complex case
*/

static	char	rcsid[] = "$Id: zsolve.c,v 1.1 1994/01/13 04:20:33 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "zmatrix2.h"


#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0 )

/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* zUsolve -- back substitution with optional over-riding diagonal
		-- can be in-situ but doesn't need to be */
ZVEC	*zUsolve(matrix,b,out,diag)
ZMAT	*matrix;
ZVEC	*b, *out;
double	diag;
{
    unsigned int	dim /* , j */;
    int	i, i_lim;
    complex	**mat_ent, *mat_row, *b_ent, *out_ent, *out_col, sum;
    
    if ( matrix==ZMNULL || b==ZVNULL )
	error(E_NULL,"zUsolve");
    dim = min(matrix->m,matrix->n);
    if ( b->dim < dim )
	error(E_SIZES,"zUsolve");
    if ( out==ZVNULL || out->dim < dim )
	out = zv_resize(out,matrix->n);
    mat_ent = matrix->me;	b_ent = b->ve;	out_ent = out->ve;
    
    for ( i=dim-1; i>=0; i-- )
	if ( ! is_zero(b_ent[i]) )
	    break;
	else
	    out_ent[i].re = out_ent[i].im = 0.0;
    i_lim = i;
    
    for ( i = i_lim; i>=0; i-- )
    {
	sum = b_ent[i];
	mat_row = &(mat_ent[i][i+1]);
	out_col = &(out_ent[i+1]);
	sum = zsub(sum,__zip__(mat_row,out_col,i_lim-i,Z_NOCONJ));
	/******************************************************
	  for ( j=i+1; j<=i_lim; j++ )
	  sum -= mat_ent[i][j]*out_ent[j];
	  sum -= (*mat_row++)*(*out_col++);
	******************************************************/
	if ( diag == 0.0 )
	{
	    if ( is_zero(mat_ent[i][i]) )
		error(E_SING,"zUsolve");
	    else
		/* out_ent[i] = sum/mat_ent[i][i]; */
		out_ent[i] = zdiv(sum,mat_ent[i][i]);
	}
	else
	{
	    /* out_ent[i] = sum/diag; */
	    out_ent[i].re = sum.re / diag;
	    out_ent[i].im = sum.im / diag;
	}
    }
    
    return (out);
}

/* zLsolve -- forward elimination with (optional) default diagonal value */
ZVEC	*zLsolve(matrix,b,out,diag)
ZMAT	*matrix;
ZVEC	*b,*out;
double	diag;
{
    unsigned int	dim, i, i_lim /* , j */;
    complex	**mat_ent, *mat_row, *b_ent, *out_ent, *out_col, sum;
    
    if ( matrix==ZMNULL || b==ZVNULL )
	error(E_NULL,"zLsolve");
    dim = min(matrix->m,matrix->n);
    if ( b->dim < dim )
	error(E_SIZES,"zLsolve");
    if ( out==ZVNULL || out->dim < dim )
	out = zv_resize(out,matrix->n);
    mat_ent = matrix->me;	b_ent = b->ve;	out_ent = out->ve;
    
    for ( i=0; i<dim; i++ )
	if ( ! is_zero(b_ent[i]) )
	    break;
	else
	    out_ent[i].re = out_ent[i].im = 0.0;
    i_lim = i;
    
    for ( i = i_lim; i<dim; i++ )
    {
	sum = b_ent[i];
	mat_row = &(mat_ent[i][i_lim]);
	out_col = &(out_ent[i_lim]);
	sum = zsub(sum,__zip__(mat_row,out_col,(int)(i-i_lim),Z_NOCONJ));
	/*****************************************************
	  for ( j=i_lim; j<i; j++ )
	  sum -= mat_ent[i][j]*out_ent[j];
	  sum -= (*mat_row++)*(*out_col++);
	******************************************************/
	if ( diag == 0.0 )
	{
	    if ( is_zero(mat_ent[i][i]) )
		error(E_SING,"zLsolve");
	    else
		out_ent[i] = zdiv(sum,mat_ent[i][i]);
	}
	else
	{
	    out_ent[i].re = sum.re / diag;
	    out_ent[i].im = sum.im / diag;
	}
    }
    
    return (out);
}


/* zUAsolve -- forward elimination with (optional) default diagonal value
		using UPPER triangular part of matrix */
ZVEC	*zUAsolve(U,b,out,diag)
ZMAT	*U;
ZVEC	*b,*out;
double	diag;
{
    unsigned int	dim, i, i_lim /* , j */;
    complex	**U_me, *b_ve, *out_ve, tmp;
    Real	invdiag;
    
    if ( ! U || ! b )
	error(E_NULL,"zUAsolve");
    dim = min(U->m,U->n);
    if ( b->dim < dim )
	error(E_SIZES,"zUAsolve");
    out = zv_resize(out,U->n);
    U_me = U->me;	b_ve = b->ve;	out_ve = out->ve;
    
    for ( i=0; i<dim; i++ )
	if ( ! is_zero(b_ve[i]) )
	    break;
	else
	    out_ve[i].re = out_ve[i].im = 0.0;
    i_lim = i;
    if ( b != out )
    {
	__zzero__(out_ve,out->dim);
	/* MEM_COPY(&(b_ve[i_lim]),&(out_ve[i_lim]),
	   (dim-i_lim)*sizeof(complex)); */
	MEMCOPY(&(b_ve[i_lim]),&(out_ve[i_lim]),dim-i_lim,complex);
    }

    if ( diag == 0.0 )
    {
	for (    ; i<dim; i++ )
	{
	    tmp = zconj(U_me[i][i]);
	    if ( is_zero(tmp) )
		error(E_SING,"zUAsolve");
	    /* out_ve[i] /= tmp; */
	    out_ve[i] = zdiv(out_ve[i],tmp);
	    tmp.re = - out_ve[i].re;
	    tmp.im = - out_ve[i].im;
	    __zmltadd__(&(out_ve[i+1]),&(U_me[i][i+1]),tmp,dim-i-1,Z_CONJ);
	}
    }
    else
    {
	invdiag = 1.0/diag;
	for (    ; i<dim; i++ )
	{
	    out_ve[i].re *= invdiag;
	    out_ve[i].im *= invdiag;
	    tmp.re = - out_ve[i].re;
	    tmp.im = - out_ve[i].im;
	    __zmltadd__(&(out_ve[i+1]),&(U_me[i][i+1]),tmp,dim-i-1,Z_CONJ);
	}
    }
    return (out);
}

/* zDsolve -- solves Dx=b where D is the diagonal of A -- may be in-situ */
ZVEC	*zDsolve(A,b,x)
ZMAT	*A;
ZVEC	*b,*x;
{
    unsigned int	dim, i;
    
    if ( ! A || ! b )
	error(E_NULL,"zDsolve");
    dim = min(A->m,A->n);
    if ( b->dim < dim )
	error(E_SIZES,"zDsolve");
    x = zv_resize(x,A->n);
    
    dim = b->dim;
    for ( i=0; i<dim; i++ )
	if ( is_zero(A->me[i][i]) )
	    error(E_SING,"zDsolve");
	else
	    x->ve[i] = zdiv(b->ve[i],A->me[i][i]);
    
    return (x);
}

/* zLAsolve -- back substitution with optional over-riding diagonal
		using the LOWER triangular part of matrix
		-- can be in-situ but doesn't need to be */
ZVEC	*zLAsolve(L,b,out,diag)
ZMAT	*L;
ZVEC	*b, *out;
double	diag;
{
    unsigned int	dim;
    int		i, i_lim;
    complex	**L_me, *b_ve, *out_ve, tmp;
    Real	invdiag;
    
    if ( ! L || ! b )
	error(E_NULL,"zLAsolve");
    dim = min(L->m,L->n);
    if ( b->dim < dim )
	error(E_SIZES,"zLAsolve");
    out = zv_resize(out,L->n);
    L_me = L->me;	b_ve = b->ve;	out_ve = out->ve;
    
    for ( i=dim-1; i>=0; i-- )
	if ( ! is_zero(b_ve[i]) )
	    break;
    i_lim = i;

    if ( b != out )
    {
	__zzero__(out_ve,out->dim);
	/* MEM_COPY(b_ve,out_ve,(i_lim+1)*sizeof(complex)); */
	MEMCOPY(b_ve,out_ve,i_lim+1,complex);
    }

    if ( diag == 0.0 )
    {
	for (        ; i>=0; i-- )
	{
	    tmp = zconj(L_me[i][i]);
	    if ( is_zero(tmp) )
		error(E_SING,"zLAsolve");
	    out_ve[i] = zdiv(out_ve[i],tmp);
	    tmp.re = - out_ve[i].re;
	    tmp.im = - out_ve[i].im;
	    __zmltadd__(out_ve,L_me[i],tmp,i,Z_CONJ);
	}
    }
    else
    {
	invdiag = 1.0/diag;
	for (        ; i>=0; i-- )
	{
	    out_ve[i].re *= invdiag;
	    out_ve[i].im *= invdiag;
	    tmp.re = - out_ve[i].re;
	    tmp.im = - out_ve[i].im;
	    __zmltadd__(out_ve,L_me[i],tmp,i,Z_CONJ);
	}
    }
    
    return (out);
}
