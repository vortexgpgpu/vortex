
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
	File containing routines for computing the SVD of matrices
*/

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"


static char rcsid[] = "$Id: svd.c,v 1.7 1995/09/08 14:45:43 des Exp $";



#define	sgn(x)	((x) >= 0 ? 1 : -1)
#define	MAX_STACK	100

/* fixsvd -- fix minor details about SVD
	-- make singular values non-negative
	-- sort singular values in decreasing order
	-- variables as for bisvd()
	-- no argument checking */
#ifndef ANSI_C
static void	fixsvd(d,U,V)
VEC	*d;
MAT	*U, *V;
#else
static void	fixsvd(VEC *d, MAT *U, MAT *V)
#endif
{
    int		i, j, k, l, r, stack[MAX_STACK], sp;
    Real	tmp, v;

    /* make singular values non-negative */
    for ( i = 0; i < d->dim; i++ )
	if ( d->ve[i] < 0.0 )
	{
	    d->ve[i] = - d->ve[i];
	    if ( U != MNULL )
		for ( j = 0; j < U->m; j++ )
		    U->me[i][j] = - U->me[i][j];
	}

    /* sort singular values */
    /* nonrecursive implementation of quicksort due to R.Sedgewick,
       "Algorithms in C", p. 122 (1990) */
    sp = -1;
    l = 0;	r = d->dim - 1;
    for ( ; ; )
    {
	while ( r > l )
	{
	    /* i = partition(d->ve,l,r) */
	    v = d->ve[r];

	    i = l - 1;	    j = r;
	    for ( ; ; )
	    {	/* inequalities are "backwards" for **decreasing** order */
		while ( d->ve[++i] > v )
		    ;
		while ( d->ve[--j] < v )
		    ;
		if ( i >= j )
		    break;
		/* swap entries in d->ve */
		tmp = d->ve[i];	  d->ve[i] = d->ve[j];	d->ve[j] = tmp;
		/* swap rows of U & V as well */
		if ( U != MNULL )
		    for ( k = 0; k < U->n; k++ )
		    {
			tmp = U->me[i][k];
			U->me[i][k] = U->me[j][k];
			U->me[j][k] = tmp;
		    }
		if ( V != MNULL )
		    for ( k = 0; k < V->n; k++ )
		    {
			tmp = V->me[i][k];
			V->me[i][k] = V->me[j][k];
			V->me[j][k] = tmp;
		    }
	    }
	    tmp = d->ve[i];    d->ve[i] = d->ve[r];    d->ve[r] = tmp;
	    if ( U != MNULL )
		for ( k = 0; k < U->n; k++ )
		{
		    tmp = U->me[i][k];
		    U->me[i][k] = U->me[r][k];
		    U->me[r][k] = tmp;
		}
	    if ( V != MNULL )
		for ( k = 0; k < V->n; k++ )
		{
		    tmp = V->me[i][k];
		    V->me[i][k] = V->me[r][k];
		    V->me[r][k] = tmp;
		}
	    /* end i = partition(...) */
	    if ( i - l > r - i )
	    {	stack[++sp] = l;    stack[++sp] = i-1;	l = i+1;    }
	    else
	    {	stack[++sp] = i+1;  stack[++sp] = r;	r = i-1;    }
	}
	if ( sp < 0 )
	    break;
	r = stack[sp--];	l = stack[sp--];
    }
}


/* bisvd -- svd of a bidiagonal m x n matrix represented by d (diagonal) and
			f (super-diagonals)
	-- returns with d set to the singular values, f zeroed
	-- if U, V non-NULL, the orthogonal operations are accumulated
		in U, V; if U, V == I on entry, then SVD == U^T.A.V
		where A is initial matrix
	-- returns d on exit */
#ifndef ANSI_C
VEC	*bisvd(d,f,U,V)
VEC	*d, *f;
MAT	*U, *V;
#else
VEC	*bisvd(VEC *d, VEC *f, MAT *U, MAT *V)
#endif
{
	int	i, j, n;
	int	i_min, i_max, split;
	Real	c, s, shift, size, z;
	Real	d_tmp, diff, t11, t12, t22, *d_ve, *f_ve;

	if ( ! d || ! f )
		error(E_NULL,"bisvd");
	if ( d->dim != f->dim + 1 )
		error(E_SIZES,"bisvd");
	n = d->dim;
	if ( ( U && U->n < n ) || ( V && V->m < n ) )
		error(E_SIZES,"bisvd");
	if ( ( U && U->m != U->n ) || ( V && V->m != V->n ) )
		error(E_SQUARE,"bisvd");


	if ( n == 1 )
	  {
	    if ( d->ve[0] < 0.0 )
	      {
		d->ve[0] = - d->ve[0];
		if ( U != MNULL )
		  sm_mlt(-1.0,U,U);
	      }
	    return d;
	  }
	d_ve = d->ve;	f_ve = f->ve;

	size = v_norm_inf(d) + v_norm_inf(f);

	i_min = 0;
	while ( i_min < n )	/* outer while loop */
	{
	    /* find i_max to suit;
		submatrix i_min..i_max should be irreducible */
	    i_max = n - 1;
	    for ( i = i_min; i < n - 1; i++ )
		if ( d_ve[i] == 0.0 || f_ve[i] == 0.0 )
		{   i_max = i;
		    if ( f_ve[i] != 0.0 )
		    {
			/* have to ``chase'' f[i] element out of matrix */
			z = f_ve[i];	f_ve[i] = 0.0;
			for ( j = i; j < n-1 && z != 0.0; j++ )
			{
			    givens(d_ve[j+1],z, &c, &s);
			    s = -s;
			    d_ve[j+1] =  c*d_ve[j+1] - s*z;
			    if ( j+1 < n-1 )
			    {
				z         = s*f_ve[j+1];
				f_ve[j+1] = c*f_ve[j+1];
			    }
			    if ( U )
				rot_rows(U,i,j+1,c,s,U);
			}
		    }
		    break;
		}
	    if ( i_max <= i_min )
	    {
		i_min = i_max + 1;
		continue;
	    }
	    /* printf("bisvd: i_min = %d, i_max = %d\n",i_min,i_max); */

	    split = FALSE;
	    while ( ! split )
	    {
		/* compute shift */
		t11 = d_ve[i_max-1]*d_ve[i_max-1] +
			(i_max > i_min+1 ? f_ve[i_max-2]*f_ve[i_max-2] : 0.0);
		t12 = d_ve[i_max-1]*f_ve[i_max-1];
		t22 = d_ve[i_max]*d_ve[i_max] + f_ve[i_max-1]*f_ve[i_max-1];
		/* use e-val of [[t11,t12],[t12,t22]] matrix
				closest to t22 */
		diff = (t11-t22)/2;
		shift = t22 - t12*t12/(diff +
			sgn(diff)*sqrt(diff*diff+t12*t12));

		/* initial Givens' rotation */
		givens(d_ve[i_min]*d_ve[i_min]-shift,
			d_ve[i_min]*f_ve[i_min], &c, &s);

		/* do initial Givens' rotations */
		d_tmp         = c*d_ve[i_min] + s*f_ve[i_min];
		f_ve[i_min]   = c*f_ve[i_min] - s*d_ve[i_min];
		d_ve[i_min]   = d_tmp;
		z             = s*d_ve[i_min+1];
		d_ve[i_min+1] = c*d_ve[i_min+1];
		if ( V )
		    rot_rows(V,i_min,i_min+1,c,s,V);
		/* 2nd Givens' rotation */
		givens(d_ve[i_min],z, &c, &s);
		d_ve[i_min]   = c*d_ve[i_min] + s*z;
		d_tmp         = c*d_ve[i_min+1] - s*f_ve[i_min];
		f_ve[i_min]   = s*d_ve[i_min+1] + c*f_ve[i_min];
		d_ve[i_min+1] = d_tmp;
		if ( i_min+1 < i_max )
		{
		    z             = s*f_ve[i_min+1];
		    f_ve[i_min+1] = c*f_ve[i_min+1];
		}
		if ( U )
		    rot_rows(U,i_min,i_min+1,c,s,U);

		for ( i = i_min+1; i < i_max; i++ )
		{
		    /* get Givens' rotation for zeroing z */
		    givens(f_ve[i-1],z, &c, &s);
		    f_ve[i-1] = c*f_ve[i-1] + s*z;
		    d_tmp     = c*d_ve[i] + s*f_ve[i];
		    f_ve[i]   = c*f_ve[i] - s*d_ve[i];
		    d_ve[i]   = d_tmp;
		    z         = s*d_ve[i+1];
		    d_ve[i+1] = c*d_ve[i+1];
		    if ( V )
			rot_rows(V,i,i+1,c,s,V);
		    /* get 2nd Givens' rotation */
		    givens(d_ve[i],z, &c, &s);
		    d_ve[i]   = c*d_ve[i] + s*z;
		    d_tmp     = c*d_ve[i+1] - s*f_ve[i];
		    f_ve[i]   = c*f_ve[i] + s*d_ve[i+1];
		    d_ve[i+1] = d_tmp;
		    if ( i+1 < i_max )
		    {
			z         = s*f_ve[i+1];
			f_ve[i+1] = c*f_ve[i+1];
		    }
		    if ( U )
			rot_rows(U,i,i+1,c,s,U);
		}
		/* should matrix be split? */
		for ( i = i_min; i < i_max; i++ )
		    if ( fabs(f_ve[i]) <
				MACHEPS*(fabs(d_ve[i])+fabs(d_ve[i+1])) )
		    {
			split = TRUE;
			f_ve[i] = 0.0;
		    }
		    else if ( fabs(d_ve[i]) < MACHEPS*size )
		    {
			split = TRUE;
			d_ve[i] = 0.0;
		    }
		    /* printf("bisvd: d =\n");	v_output(d); */
		    /* printf("bisvd: f = \n");	v_output(f); */
		}
	}
	fixsvd(d,U,V);

	return d;
}

/* bifactor -- perform preliminary factorisation for bisvd
	-- updates U and/or V, which ever is not NULL */
#ifndef ANSI_C
MAT	*bifactor(A,U,V)
MAT	*A, *U, *V;
#else
MAT	*bifactor(MAT *A, MAT *U, MAT *V)
#endif
{
	int	k;
	STATIC VEC	*tmp1=VNULL, *tmp2=VNULL, *w=VNULL;
	Real	beta;

	if ( ! A )
		error(E_NULL,"bifactor");
	if ( ( U && ( U->m != U->n ) ) || ( V && ( V->m != V->n ) ) )
		error(E_SQUARE,"bifactor");
	if ( ( U && U->m != A->m ) || ( V && V->m != A->n ) )
		error(E_SIZES,"bifactor");
	tmp1 = v_resize(tmp1,A->m);
	tmp2 = v_resize(tmp2,A->n);
	w    = v_resize(w,   max(A->m,A->n));
	MEM_STAT_REG(tmp1,TYPE_VEC);
	MEM_STAT_REG(tmp2,TYPE_VEC);
	MEM_STAT_REG(w,   TYPE_VEC);

	if ( A->m >= A->n )
	    for ( k = 0; k < A->n; k++ )
	    {
		get_col(A,k,tmp1);
		hhvec(tmp1,k,&beta,tmp1,&(A->me[k][k]));
		_hhtrcols(A,k,k+1,tmp1,beta,w);
		if ( U )
		    _hhtrcols(U,k,0,tmp1,beta,w);
		if ( k+1 >= A->n )
		    continue;
		get_row(A,k,tmp2);
		hhvec(tmp2,k+1,&beta,tmp2,&(A->me[k][k+1]));
		hhtrrows(A,k+1,k+1,tmp2,beta);
		if ( V )
		    _hhtrcols(V,k+1,0,tmp2,beta,w);
	    }
	else
	    for ( k = 0; k < A->m; k++ )
	    {
		get_row(A,k,tmp2);
		hhvec(tmp2,k,&beta,tmp2,&(A->me[k][k]));
		hhtrrows(A,k+1,k,tmp2,beta);
		if ( V )
		    _hhtrcols(V,k,0,tmp2,beta,w);
		if ( k+1 >= A->m )
		    continue;
		get_col(A,k,tmp1);
		hhvec(tmp1,k+1,&beta,tmp1,&(A->me[k+1][k]));
		_hhtrcols(A,k+1,k+1,tmp1,beta,w);
		if ( U )
		    _hhtrcols(U,k+1,0,tmp1,beta,w);
	    }

#ifdef	THREADSAFE
	V_FREE(tmp1);	V_FREE(tmp2);
#endif

	return A;
}

/* svd -- returns vector of singular values in d
	-- also updates U and/or V, if one or the other is non-NULL
	-- destroys A */
#ifndef ANSI_C
VEC	*svd(A,U,V,d)
MAT	*A, *U, *V;
VEC	*d;
#else
VEC	*svd(MAT *A, MAT *U, MAT *V, VEC *d)
#endif
{
	STATIC VEC	*f=VNULL;
	int	i, limit;
	MAT	*A_tmp;

	if ( ! A )
		error(E_NULL,"svd");
	if ( ( U && ( U->m != U->n ) ) || ( V && ( V->m != V->n ) ) )
		error(E_SQUARE,"svd");
	if ( ( U && U->m != A->m ) || ( V && V->m != A->n ) )
		error(E_SIZES,"svd");

	A_tmp = m_copy(A,MNULL);
	if ( U != MNULL )
	    m_ident(U);
	if ( V != MNULL )
	    m_ident(V);
	limit = min(A_tmp->m,A_tmp->n);
	d = v_resize(d,limit);
	f = v_resize(f,limit-1);
	MEM_STAT_REG(f,TYPE_VEC);

	bifactor(A_tmp,U,V);
	if ( A_tmp->m >= A_tmp->n )
	    for ( i = 0; i < limit; i++ )
	    {
		d->ve[i] = A_tmp->me[i][i];
		if ( i+1 < limit )
		    f->ve[i] = A_tmp->me[i][i+1];
	    }
	else
	    for ( i = 0; i < limit; i++ )
	    {
		d->ve[i] = A_tmp->me[i][i];
		if ( i+1 < limit )
		    f->ve[i] = A_tmp->me[i+1][i];
	    }


	if ( A_tmp->m >= A_tmp->n )
	    bisvd(d,f,U,V);
	else
	    bisvd(d,f,V,U);

	M_FREE(A_tmp);
#ifdef	THREADSAFE
	V_FREE(f);
#endif

	return d;
}

