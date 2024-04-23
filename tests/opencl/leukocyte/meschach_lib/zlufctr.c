
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
	Complex version
*/

static	char	rcsid[] = "$Id: zlufctr.c,v 1.3 1996/08/20 20:07:09 stewart Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"

#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)


/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* zLUfactor -- Gaussian elimination with scaled partial pivoting
		-- Note: returns LU matrix which is A */
ZMAT	*zLUfactor(A,pivot)
ZMAT	*A;
PERM	*pivot;
{
	unsigned int	i, j, m, n;
	int	i_max, k, k_max;
	Real	dtemp, max1;
	complex	**A_v, *A_piv, *A_row, temp;
	STATIC	VEC	*scale = VNULL;

	if ( A==ZMNULL || pivot==PNULL )
		error(E_NULL,"zLUfactor");
	if ( pivot->size != A->m )
		error(E_SIZES,"zLUfactor");
	m = A->m;	n = A->n;
	scale = v_resize(scale,A->m);
	MEM_STAT_REG(scale,TYPE_VEC);
	A_v = A->me;

	/* initialise pivot with identity permutation */
	for ( i=0; i<m; i++ )
	    pivot->pe[i] = i;

	/* set scale parameters */
	for ( i=0; i<m; i++ )
	{
		max1 = 0.0;
		for ( j=0; j<n; j++ )
		{
			dtemp = zabs(A_v[i][j]);
			max1 = max(max1,dtemp);
		}
		scale->ve[i] = max1;
	}

	/* main loop */
	k_max = min(m,n)-1;
	for ( k=0; k<k_max; k++ )
	{
	    /* find best pivot row */
	    max1 = 0.0;	i_max = -1;
	    for ( i=k; i<m; i++ )
		if ( scale->ve[i] > 0.0 )
		{
		    dtemp = zabs(A_v[i][k])/scale->ve[i];
		    if ( dtemp > max1 )
		    { max1 = dtemp;	i_max = i;	}
		}
	    
	    /* if no pivot then ignore column k... */
	    if ( i_max == -1 )
		continue;

	    /* do we pivot ? */
	    if ( i_max != k )	/* yes we do... */
	    {
		px_transp(pivot,i_max,k);
		for ( j=0; j<n; j++ )
		{
		    temp = A_v[i_max][j];
		    A_v[i_max][j] = A_v[k][j];
		    A_v[k][j] = temp;
		}
	    }
	    
	    /* row operations */
	    for ( i=k+1; i<m; i++ )	/* for each row do... */
	    {	/* Note: divide by zero should never happen */
		temp = A_v[i][k] = zdiv(A_v[i][k],A_v[k][k]);
		A_piv = &(A_v[k][k+1]);
		A_row = &(A_v[i][k+1]);
		temp.re = - temp.re;
		temp.im = - temp.im;
		if ( k+1 < n )
		    __zmltadd__(A_row,A_piv,temp,(int)(n-(k+1)),Z_NOCONJ);
		/*********************************************
		  for ( j=k+1; j<n; j++ )
		  A_v[i][j] -= temp*A_v[k][j];
		  (*A_row++) -= temp*(*A_piv++);
		*********************************************/
	    }
	}

#ifdef	THREADSAFE
	V_FREE(scale);
#endif

	return A;
}


/* zLUsolve -- given an LU factorisation in A, solve Ax=b */
ZVEC	*zLUsolve(A,pivot,b,x)
ZMAT	*A;
PERM	*pivot;
ZVEC	*b,*x;
{
	if ( A==ZMNULL || b==ZVNULL || pivot==PNULL )
		error(E_NULL,"zLUsolve");
	if ( A->m != A->n || A->n != b->dim )
		error(E_SIZES,"zLUsolve");

	x = px_zvec(pivot,b,x);	/* x := P.b */
	zLsolve(A,x,x,1.0);	/* implicit diagonal = 1 */
	zUsolve(A,x,x,0.0);	/* explicit diagonal */

	return (x);
}

/* zLUAsolve -- given an LU factorisation in A, solve A^*.x=b */
ZVEC	*zLUAsolve(LU,pivot,b,x)
ZMAT	*LU;
PERM	*pivot;
ZVEC	*b,*x;
{
	if ( ! LU || ! b || ! pivot )
		error(E_NULL,"zLUAsolve");
	if ( LU->m != LU->n || LU->n != b->dim )
		error(E_SIZES,"zLUAsolve");

	x = zv_copy(b,x);
	zUAsolve(LU,x,x,0.0);	/* explicit diagonal */
	zLAsolve(LU,x,x,1.0);	/* implicit diagonal = 1 */
	pxinv_zvec(pivot,x,x);	/* x := P^*.x */

	return (x);
}

/* zm_inverse -- returns inverse of A, provided A is not too rank deficient
	-- uses LU factorisation */
ZMAT	*zm_inverse(A,out)
ZMAT	*A, *out;
{
	int	i;
	STATIC ZVEC	*tmp=ZVNULL, *tmp2=ZVNULL;
	STATIC ZMAT	*A_cp=ZMNULL;
	STATIC PERM	*pivot=PNULL;

	if ( ! A )
	    error(E_NULL,"zm_inverse");
	if ( A->m != A->n )
	    error(E_SQUARE,"zm_inverse");
	if ( ! out || out->m < A->m || out->n < A->n )
	    out = zm_resize(out,A->m,A->n);

	A_cp = zm_resize(A_cp,A->m,A->n);
	A_cp = zm_copy(A,A_cp);
	tmp = zv_resize(tmp,A->m);
	tmp2 = zv_resize(tmp2,A->m);
	pivot = px_resize(pivot,A->m);
	MEM_STAT_REG(A_cp,TYPE_ZMAT);
	MEM_STAT_REG(tmp, TYPE_ZVEC);
	MEM_STAT_REG(tmp2,TYPE_ZVEC);
	MEM_STAT_REG(pivot,TYPE_PERM);
	tracecatch(zLUfactor(A_cp,pivot),"zm_inverse");
	for ( i = 0; i < A->n; i++ )
	{
	    zv_zero(tmp);
	    tmp->ve[i].re = 1.0;
	    tmp->ve[i].im = 0.0;
	    tracecatch(zLUsolve(A_cp,pivot,tmp,tmp2),"zm_inverse");
	    zset_col(out,i,tmp2);
	}

#ifdef	THREADSAFE
	ZV_FREE(tmp);	ZV_FREE(tmp2);
	ZM_FREE(A_cp);	PX_FREE(pivot);
#endif

	return out;
}

/* zLUcondest -- returns an estimate of the condition number of LU given the
	LU factorisation in compact form */
double	zLUcondest(LU,pivot)
ZMAT	*LU;
PERM	*pivot;
{
    STATIC	ZVEC	*y = ZVNULL, *z = ZVNULL;
    Real	cond_est, L_norm, U_norm, norm, sn_inv;
    complex	sum;
    int		i, j, n;

    if ( ! LU || ! pivot )
	error(E_NULL,"zLUcondest");
    if ( LU->m != LU->n )
	error(E_SQUARE,"zLUcondest");
    if ( LU->n != pivot->size )
	error(E_SIZES,"zLUcondest");

    n = LU->n;
    y = zv_resize(y,n);
    z = zv_resize(z,n);
    MEM_STAT_REG(y,TYPE_ZVEC);
    MEM_STAT_REG(z,TYPE_ZVEC);

    cond_est = 0.0;		/* should never be returned */

    for ( i = 0; i < n; i++ )
    {
	sum.re = 1.0;
	sum.im = 0.0;
	for ( j = 0; j < i; j++ )
	    /* sum -= LU->me[j][i]*y->ve[j]; */
	    sum = zsub(sum,zmlt(LU->me[j][i],y->ve[j]));
	/* sum -= (sum < 0.0) ? 1.0 : -1.0; */
	sn_inv = 1.0 / zabs(sum);
	sum.re += sum.re * sn_inv;
	sum.im += sum.im * sn_inv;
	if ( is_zero(LU->me[i][i]) )
	    return HUGE_VAL;
	/* y->ve[i] = sum / LU->me[i][i]; */
	y->ve[i] = zdiv(sum,LU->me[i][i]);
    }

    zLAsolve(LU,y,y,1.0);
    zLUsolve(LU,pivot,y,z);

    /* now estimate norm of A (even though it is not directly available) */
    /* actually computes ||L||_inf.||U||_inf */
    U_norm = 0.0;
    for ( i = 0; i < n; i++ )
    {
	norm = 0.0;
	for ( j = i; j < n; j++ )
	    norm += zabs(LU->me[i][j]);
	if ( norm > U_norm )
	    U_norm = norm;
    }
    L_norm = 0.0;
    for ( i = 0; i < n; i++ )
    {
	norm = 1.0;
	for ( j = 0; j < i; j++ )
	    norm += zabs(LU->me[i][j]);
	if ( norm > L_norm )
	    L_norm = norm;
    }

    tracecatch(cond_est = U_norm*L_norm*zv_norm_inf(z)/zv_norm_inf(y),
	       "zLUcondest");
#ifdef	THREADSAFE
    ZV_FREE(y);		ZV_FREE(z);
#endif

    return cond_est;
}
