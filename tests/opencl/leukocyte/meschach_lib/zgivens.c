
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
	Givens operations file. Contains routines for calculating and
	applying givens rotations for/to vectors and also to matrices by
	row and by column.

	Complex version.
*/

static	char	rcsid[] = "$Id: ";

#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"

/*
	(Complex) Givens rotation matrix:
		[ c   -s ]
		[ s*   c ]
	Note that c is real and s is complex
*/

/* zgivens -- returns c,s parameters for Givens rotation to
		eliminate y in the **column** vector [ x y ] */
void	zgivens(x,y,c,s)
complex	x,y,*s;
Real	*c;
{
	Real	inv_norm, norm;
	complex	tmp;

	/* this is a safe way of computing sqrt(|x|^2+|y|^2) */
	tmp.re = zabs(x);	tmp.im = zabs(y);
	norm = zabs(tmp);

	if ( norm == 0.0 )
	{	*c = 1.0;	s->re = s->im = 0.0;	} /* identity */
	else
	{
	    inv_norm = 1.0 / tmp.re;	/* inv_norm = 1/|x| */
	    x.re *= inv_norm;
	    x.im *= inv_norm;		/* normalise x */
	    inv_norm = 1.0/norm;		/* inv_norm = 1/||[x,y]||2 */
	    *c = tmp.re * inv_norm;
	    /* now compute - conj(normalised x).y/||[x,y]||2 */
	    s->re = - inv_norm*(x.re*y.re + x.im*y.im);
	    s->im =   inv_norm*(x.re*y.im - x.im*y.re);
	}
}

/* rot_zvec -- apply Givens rotation to x's i & k components */
ZVEC	*rot_zvec(x,i,k,c,s,out)
ZVEC	*x,*out;
int	i,k;
double	c;
complex	s;
{

	complex	temp1, temp2;

	if ( x==ZVNULL )
		error(E_NULL,"rot_zvec");
	if ( i < 0 || i >= x->dim || k < 0 || k >= x->dim )
		error(E_RANGE,"rot_zvec");
	if ( x != out )
	    out = zv_copy(x,out);

	/* temp1 = c*out->ve[i] - s*out->ve[k]; */
	temp1.re = c*out->ve[i].re
	    - s.re*out->ve[k].re + s.im*out->ve[k].im;
	temp1.im = c*out->ve[i].im
	    - s.re*out->ve[k].im - s.im*out->ve[k].re;

	/* temp2 = c*out->ve[k] + zconj(s)*out->ve[i]; */
	temp2.re = c*out->ve[k].re
		+ s.re*out->ve[i].re + s.im*out->ve[i].im;
	temp2.im = c*out->ve[k].im
		+ s.re*out->ve[i].im - s.im*out->ve[i].re;

	out->ve[i] = temp1;
	out->ve[k] = temp2;

	return (out);
}

/* zrot_rows -- premultiply mat by givens rotation described by c,s */
ZMAT	*zrot_rows(mat,i,k,c,s,out)
ZMAT	*mat,*out;
int	i,k;
double	c;
complex	s;
{
	unsigned int	j;
	complex	temp1, temp2;

	if ( mat==ZMNULL )
		error(E_NULL,"zrot_rows");
	if ( i < 0 || i >= mat->m || k < 0 || k >= mat->m )
		error(E_RANGE,"zrot_rows");

	if ( mat != out )
		out = zm_copy(mat,zm_resize(out,mat->m,mat->n));

	/* temp1 = c*out->me[i][j] - s*out->me[k][j]; */
	for ( j=0; j<mat->n; j++ )
	{
	    /* temp1 = c*out->me[i][j] - s*out->me[k][j]; */
	    temp1.re = c*out->me[i][j].re
		- s.re*out->me[k][j].re + s.im*out->me[k][j].im;
	    temp1.im = c*out->me[i][j].im
		- s.re*out->me[k][j].im - s.im*out->me[k][j].re;
	    
	    /* temp2 = c*out->me[k][j] + conj(s)*out->me[i][j]; */
	    temp2.re = c*out->me[k][j].re
		+ s.re*out->me[i][j].re + s.im*out->me[i][j].im;
	    temp2.im = c*out->me[k][j].im
		+ s.re*out->me[i][j].im - s.im*out->me[i][j].re;
	    
	    out->me[i][j] = temp1;
	    out->me[k][j] = temp2;
	}

	return (out);
}

/* zrot_cols -- postmultiply mat by adjoint Givens rotation described by c,s */
ZMAT	*zrot_cols(mat,i,k,c,s,out)
ZMAT	*mat,*out;
int	i,k;
double	c;
complex	s;
{
	unsigned int	j;
	complex	x, y;

	if ( mat==ZMNULL )
		error(E_NULL,"zrot_cols");
	if ( i < 0 || i >= mat->n || k < 0 || k >= mat->n )
		error(E_RANGE,"zrot_cols");

	if ( mat != out )
		out = zm_copy(mat,zm_resize(out,mat->m,mat->n));

	for ( j=0; j<mat->m; j++ )
	{
	    x = out->me[j][i];	y = out->me[j][k];
	    /* out->me[j][i] = c*x - conj(s)*y; */
	    out->me[j][i].re = c*x.re - s.re*y.re - s.im*y.im;
	    out->me[j][i].im = c*x.im - s.re*y.im + s.im*y.re;
	    
	    /* out->me[j][k] = c*y + s*x; */
	    out->me[j][k].re = c*y.re + s.re*x.re - s.im*x.im;
	    out->me[j][k].im = c*y.im + s.re*x.im + s.im*x.re;
	}

	return (out);
}

