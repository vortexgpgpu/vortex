
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


/* pxop.c 1.5 12/03/87 */


#include	<stdio.h>
#include	"matrix.h"

static	char	rcsid[] = "$Id: pxop.c,v 1.6 1995/06/08 14:57:11 des Exp $";

/**********************************************************************
Note: A permutation is often interpreted as a matrix
		(i.e. a permutation matrix).
	A permutation px represents a permutation matrix P where
		P[i][j] == 1 if and only if px->pe[i] == j
**********************************************************************/


/* px_inv -- invert permutation -- in situ
	-- taken from ACM Collected Algorithms #250 */
#ifndef ANSI_C
PERM	*px_inv(px,out)
PERM	*px, *out;
#else
PERM	*px_inv(const PERM *px, PERM *out)
#endif
{
    int	i, j, k, n, *p;
    
    out = px_copy(px, out);
    n = out->size;
    p = (int *)(out->pe);
    for ( n--; n>=0; n-- )
    {
	i = p[n];
	if ( i < 0 )	p[n] = -1 - i;
	else if ( i != n )
	{
	    k = n;
	    while (TRUE)
	    {
		if ( i < 0 || i >= out->size )
		    error(E_BOUNDS,"px_inv");
		j = p[i];	p[i] = -1 - k;
		if ( j == n )
		{	p[n] = i;	break;		}
		k = i;		i = j;
	    }
	}
    }
    return out;
}

/* px_mlt -- permutation multiplication (composition) */
#ifndef ANSI_C
PERM	*px_mlt(px1,px2,out)
PERM	*px1,*px2,*out;
#else
PERM	*px_mlt(const PERM *px1, const PERM *px2, PERM *out)
#endif
{
    unsigned int	i,size;
    
    if ( px1==(PERM *)NULL || px2==(PERM *)NULL )
	error(E_NULL,"px_mlt");
    if ( px1->size != px2->size )
	error(E_SIZES,"px_mlt");
    if ( px1 == out || px2 == out )
	error(E_INSITU,"px_mlt");
    if ( out==(PERM *)NULL || out->size < px1->size )
	out = px_resize(out,px1->size);
    
    size = px1->size;
    for ( i=0; i<size; i++ )
	if ( px2->pe[i] >= size )
	    error(E_BOUNDS,"px_mlt");
	else
	    out->pe[i] = px1->pe[px2->pe[i]];
    
    return out;
}

/* px_vec -- permute vector */
#ifndef ANSI_C
VEC	*px_vec(px,vector,out)
PERM	*px;
VEC	*vector,*out;
#else
VEC	*px_vec(PERM *px, const VEC *vector, VEC *out)
#endif
{
    unsigned int	old_i, i, size, start;
    Real	tmp;
    
    if ( px==PNULL || vector==VNULL )
	error(E_NULL,"px_vec");
    if ( px->size > vector->dim )
	error(E_SIZES,"px_vec");
    if ( out==VNULL || out->dim < vector->dim )
	out = v_resize(out,vector->dim);
    
    size = px->size;
    if ( size == 0 )
	return v_copy(vector,out);
    if ( out != vector )
    {
	for ( i=0; i<size; i++ )
	    if ( px->pe[i] >= size )
		error(E_BOUNDS,"px_vec");
	    else
		out->ve[i] = vector->ve[px->pe[i]];
    }
    else
    {	/* in situ algorithm */
	start = 0;
	while ( start < size )
	{
	    old_i = start;
	    i = px->pe[old_i];
	    if ( i >= size )
	    {
		start++;
		continue;
	    }
	    tmp = vector->ve[start];
	    while ( TRUE )
	    {
		vector->ve[old_i] = vector->ve[i];
		px->pe[old_i] = i+size;
		old_i = i;
		i = px->pe[old_i];
		if ( i >= size )
		    break;
		if ( i == start )
		{
		    vector->ve[old_i] = tmp;
		    px->pe[old_i] = i+size;
		    break;
		}
	    }
	    start++;
	}

	for ( i = 0; i < size; i++ )
	    if ( px->pe[i] < size )
		error(E_BOUNDS,"px_vec");
	    else
		px->pe[i] = px->pe[i]-size;
    }
    
    return out;
}

/* pxinv_vec -- apply the inverse of px to x, returning the result in out */
#ifndef ANSI_C
VEC	*pxinv_vec(px,x,out)
PERM	*px;
VEC	*x, *out;
#else
VEC	*pxinv_vec(PERM *px, const VEC *x, VEC *out)
#endif
{
    unsigned int	i, size;
    
    if ( ! px || ! x )
	error(E_NULL,"pxinv_vec");
    if ( px->size > x->dim )
	error(E_SIZES,"pxinv_vec");
    /* if ( x == out )
	error(E_INSITU,"pxinv_vec"); */
    if ( ! out || out->dim < x->dim )
	out = v_resize(out,x->dim);
    
    size = px->size;
    if ( size == 0 )
	return v_copy(x,out);
    if ( out != x )
    {
	for ( i=0; i<size; i++ )
	    if ( px->pe[i] >= size )
		error(E_BOUNDS,"pxinv_vec");
	    else
		out->ve[px->pe[i]] = x->ve[i];
    }
    else
    {	/* in situ algorithm --- cheat's way out */
	px_inv(px,px);
	px_vec(px,x,out);
	px_inv(px,px);
    }

    return out;
}



/* px_transp -- transpose elements of permutation
		-- Really multiplying a permutation by a transposition */
#ifndef ANSI_C
PERM	*px_transp(px,i1,i2)
PERM	*px;		/* permutation to transpose */
unsigned int	i1,i2;		/* elements to transpose */
#else
PERM	*px_transp(PERM *px, unsigned int i1, unsigned int i2)
#endif
{
	unsigned int	temp;

	if ( px==(PERM *)NULL )
		error(E_NULL,"px_transp");

	if ( i1 < px->size && i2 < px->size )
	{
		temp = px->pe[i1];
		px->pe[i1] = px->pe[i2];
		px->pe[i2] = temp;
	}

	return px;
}

/* myqsort -- a cheap implementation of Quicksort on integers
		-- returns number of swaps */
#ifndef ANSI_C
static int myqsort(a,num)
int	*a, num;
#else
static int myqsort(int *a, int num)
#endif
{
	int	i, j, tmp, v;
	int	numswaps;

	numswaps = 0;
	if ( num <= 1 )
		return 0;

	i = 0;	j = num;	v = a[0];
	for ( ; ; )
	{
		while ( a[++i] < v )
			;
		while ( a[--j] > v )
			;
		if ( i >= j )	break;

		tmp = a[i];
		a[i] = a[j];
		a[j] = tmp;
		numswaps++;
	}

	tmp = a[0];
	a[0] = a[j];
	a[j] = tmp;
	if ( j != 0 )
		numswaps++;

	numswaps += myqsort(&a[0],j);
	numswaps += myqsort(&a[j+1],num-(j+1));

	return numswaps;
}


/* px_sign -- compute the ``sign'' of a permutation = +/-1 where
		px is the product of an even/odd # transpositions */
#ifndef ANSI_C
int	px_sign(px)
PERM	*px;
#else
int	px_sign(const PERM *px)
#endif
{
	int	numtransp;
	PERM	*px2;

	if ( px==(PERM *)NULL )
		error(E_NULL,"px_sign");
	px2 = px_copy(px,PNULL);
	numtransp = myqsort((int *)px2->pe,px2->size);
	px_free(px2);

	return ( numtransp % 2 ) ? -1 : 1;
}


/* px_cols -- permute columns of matrix A; out = A.px'
	-- May NOT be in situ */
#ifndef ANSI_C
MAT	*px_cols(px,A,out)
PERM	*px;
MAT	*A, *out;
#else
MAT	*px_cols(const PERM *px, const MAT *A, MAT *out)
#endif
{
	int	i, j, m, n, px_j;
	Real	**A_me, **out_me;
#ifdef ANSI_C
	MAT	*m_get(int, int);
#else
	extern MAT	*m_get();
#endif

	if ( ! A || ! px )
		error(E_NULL,"px_cols");
	if ( px->size != A->n )
		error(E_SIZES,"px_cols");
	if ( A == out )
		error(E_INSITU,"px_cols");
	m = A->m;	n = A->n;
	if ( ! out || out->m != m || out->n != n )
		out = m_get(m,n);
	A_me = A->me;	out_me = out->me;

	for ( j = 0; j < n; j++ )
	{
		px_j = px->pe[j];
		if ( px_j >= n )
		    error(E_BOUNDS,"px_cols");
		for ( i = 0; i < m; i++ )
		    out_me[i][px_j] = A_me[i][j];
	}

	return out;
}

/* px_rows -- permute columns of matrix A; out = px.A
	-- May NOT be in situ */
#ifndef ANSI_C
MAT	*px_rows(px,A,out)
PERM	*px;
MAT	*A, *out;
#else
MAT	*px_rows(const PERM *px, const MAT *A, MAT *out)
#endif
{
	int	i, j, m, n, px_i;
	Real	**A_me, **out_me;
#ifdef ANSI_C
	MAT	*m_get(int, int);
#else
	extern MAT	*m_get();
#endif

	if ( ! A || ! px )
		error(E_NULL,"px_rows");
	if ( px->size != A->m )
		error(E_SIZES,"px_rows");
	if ( A == out )
		error(E_INSITU,"px_rows");
	m = A->m;	n = A->n;
	if ( ! out || out->m != m || out->n != n )
		out = m_get(m,n);
	A_me = A->me;	out_me = out->me;

	for ( i = 0; i < m; i++ )
	{
		px_i = px->pe[i];
		if ( px_i >= m )
		    error(E_BOUNDS,"px_rows");
		for ( j = 0; j < n; j++ )
		    out_me[i][j] = A_me[px_i][j];
	}

	return out;
}

