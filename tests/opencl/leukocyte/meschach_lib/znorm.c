
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
	A collection of functions for computing norms: scaled and unscaled
	Complex version
*/
static	char	rcsid[] = "$Id: znorm.c,v 1.1 1994/01/13 04:21:31 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"



/* _zv_norm1 -- computes (scaled) 1-norms of vectors */
double	_zv_norm1(x,scale)
ZVEC	*x;
VEC	*scale;
{
    int	i, dim;
    Real	s, sum;
    
    if ( x == ZVNULL )
	error(E_NULL,"_zv_norm1");
    dim = x->dim;
    
    sum = 0.0;
    if ( scale == VNULL )
	for ( i = 0; i < dim; i++ )
	    sum += zabs(x->ve[i]);
    else if ( scale->dim < dim )
	error(E_SIZES,"_zv_norm1");
    else
	for ( i = 0; i < dim; i++ )
	{
	    s = scale->ve[i];
	    sum += ( s== 0.0 ) ? zabs(x->ve[i]) : zabs(x->ve[i])/fabs(s);
	}
    
    return sum;
}

/* square -- returns x^2 */
/******************************
double	square(x)
double	x;
{	return x*x;	}
******************************/

#define	square(x)	((x)*(x))

/* _zv_norm2 -- computes (scaled) 2-norm (Euclidean norm) of vectors */
double	_zv_norm2(x,scale)
ZVEC	*x;
VEC	*scale;
{
    int	i, dim;
    Real	s, sum;
    
    if ( x == ZVNULL )
	error(E_NULL,"_zv_norm2");
    dim = x->dim;
    
    sum = 0.0;
    if ( scale == VNULL )
	for ( i = 0; i < dim; i++ )
	    sum += square(x->ve[i].re) + square(x->ve[i].im);
    else if ( scale->dim < dim )
	error(E_SIZES,"_v_norm2");
    else
	for ( i = 0; i < dim; i++ )
	{
	    s = scale->ve[i];
	    sum += ( s== 0.0 ) ? square(x->ve[i].re) + square(x->ve[i].im) :
		(square(x->ve[i].re) + square(x->ve[i].im))/square(s);
	}
    
    return sqrt(sum);
}

#define	max(a,b)	((a) > (b) ? (a) : (b))

/* _zv_norm_inf -- computes (scaled) infinity-norm (supremum norm) of vectors */
double	_zv_norm_inf(x,scale)
ZVEC	*x;
VEC	*scale;
{
    int	i, dim;
    Real	s, maxval, tmp;
    
    if ( x == ZVNULL )
	error(E_NULL,"_zv_norm_inf");
    dim = x->dim;
    
    maxval = 0.0;
    if ( scale == VNULL )
	for ( i = 0; i < dim; i++ )
	{
	    tmp = zabs(x->ve[i]);
	    maxval = max(maxval,tmp);
	}
    else if ( scale->dim < dim )
	error(E_SIZES,"_zv_norm_inf");
    else
	for ( i = 0; i < dim; i++ )
	{
	    s = scale->ve[i];
	    tmp = ( s == 0.0 ) ? zabs(x->ve[i]) : zabs(x->ve[i])/fabs(s);
	    maxval = max(maxval,tmp);
	}
    
    return maxval;
}

/* zm_norm1 -- compute matrix 1-norm -- unscaled
	-- complex version */
double	zm_norm1(A)
ZMAT	*A;
{
    int	i, j, m, n;
    Real	maxval, sum;
    
    if ( A == ZMNULL )
	error(E_NULL,"zm_norm1");

    m = A->m;	n = A->n;
    maxval = 0.0;
    
    for ( j = 0; j < n; j++ )
    {
	sum = 0.0;
	for ( i = 0; i < m; i ++ )
	    sum += zabs(A->me[i][j]);
	maxval = max(maxval,sum);
    }
    
    return maxval;
}

/* zm_norm_inf -- compute matrix infinity-norm -- unscaled
	-- complex version */
double	zm_norm_inf(A)
ZMAT	*A;
{
    int	i, j, m, n;
    Real	maxval, sum;
    
    if ( A == ZMNULL )
	error(E_NULL,"zm_norm_inf");
    
    m = A->m;	n = A->n;
    maxval = 0.0;
    
    for ( i = 0; i < m; i++ )
    {
	sum = 0.0;
	for ( j = 0; j < n; j ++ )
	    sum += zabs(A->me[i][j]);
	maxval = max(maxval,sum);
    }
    
    return maxval;
}

/* zm_norm_frob -- compute matrix frobenius-norm -- unscaled */
double	zm_norm_frob(A)
ZMAT	*A;
{
    int	i, j, m, n;
    Real	sum;
    
    if ( A == ZMNULL )
	error(E_NULL,"zm_norm_frob");
    
    m = A->m;	n = A->n;
    sum = 0.0;
    
    for ( i = 0; i < m; i++ )
	for ( j = 0; j < n; j ++ )
	    sum += square(A->me[i][j].re) + square(A->me[i][j].im);
    
    return sqrt(sum);
}

