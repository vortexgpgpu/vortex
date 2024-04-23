
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
*/

static	char	rcsid[] = "$Id: bkpfacto.c,v 1.7 1994/01/13 05:45:50 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"

#define	btos(x)	((x) ? "TRUE" : "FALSE")

/* Most matrix factorisation routines are in-situ unless otherwise specified */

#define alpha	0.6403882032022076 /* = (1+sqrt(17))/8 */

/* sqr -- returns square of x -- utility function */
double	sqr(x)
double	x;
{	return x*x;	}

/* interchange -- a row/column swap routine */
static void interchange(A,i,j)
MAT	*A;	/* assumed != NULL & also SQUARE */
int	i, j;	/* assumed in range */
{
	Real	**A_me, tmp;
	int	k, n;

	A_me = A->me;	n = A->n;
	if ( i == j )
		return;
	if ( i > j )
	{	k = i;	i = j;	j = k;	}
	for ( k = 0; k < i; k++ )
	{
		/* tmp = A_me[k][i]; */
		tmp = m_entry(A,k,i);
		/* A_me[k][i] = A_me[k][j]; */
		m_set_val(A,k,i,m_entry(A,k,j));
		/* A_me[k][j] = tmp; */
		m_set_val(A,k,j,tmp);
	}
	for ( k = j+1; k < n; k++ )
	{
		/* tmp = A_me[j][k]; */
		tmp = m_entry(A,j,k);
		/* A_me[j][k] = A_me[i][k]; */
		m_set_val(A,j,k,m_entry(A,i,k));
		/* A_me[i][k] = tmp; */
		m_set_val(A,i,k,tmp);
	}
	for ( k = i+1; k < j; k++ )
	{
		/* tmp = A_me[k][j]; */
		tmp = m_entry(A,k,j);
		/* A_me[k][j] = A_me[i][k]; */
		m_set_val(A,k,j,m_entry(A,i,k));
		/* A_me[i][k] = tmp; */
		m_set_val(A,i,k,tmp);
	}
	/* tmp = A_me[i][i]; */
	tmp = m_entry(A,i,i);
	/* A_me[i][i] = A_me[j][j]; */
	m_set_val(A,i,i,m_entry(A,j,j));
	/* A_me[j][j] = tmp; */
	m_set_val(A,j,j,tmp);
}

/* BKPfactor -- Bunch-Kaufman-Parlett factorisation of A in-situ
	-- A is factored into the form P'AP = MDM' where 
	P is a permutation matrix, M lower triangular and D is block
	diagonal with blocks of size 1 or 2
	-- P is stored in pivot; blocks[i]==i iff D[i][i] is a block */
#ifndef ANSI_C
MAT	*BKPfactor(A,pivot,blocks)
MAT	*A;
PERM	*pivot, *blocks;
#else
MAT	*BKPfactor(MAT *A, PERM *pivot, PERM *blocks)
#endif
{
	int	i, j, k, n, onebyone, r;
	Real	**A_me, aii, aip1, aip1i, lambda, sigma, tmp;
	Real	det, s, t;

	if ( ! A || ! pivot || ! blocks )
		error(E_NULL,"BKPfactor");
	if ( A->m != A->n )
		error(E_SQUARE,"BKPfactor");
	if ( A->m != pivot->size || pivot->size != blocks->size )
		error(E_SIZES,"BKPfactor");

	n = A->n;
	A_me = A->me;
	px_ident(pivot);	px_ident(blocks);

	for ( i = 0; i < n; i = onebyone ? i+1 : i+2 )
	{
		/* printf("# Stage: %d\n",i); */
		aii = fabs(m_entry(A,i,i));
		lambda = 0.0;	r = (i+1 < n) ? i+1 : i;
		for ( k = i+1; k < n; k++ )
		{
		    tmp = fabs(m_entry(A,i,k));
		    if ( tmp >= lambda )
		    {
			lambda = tmp;
			r = k;
		    }
		}
		/* printf("# lambda = %g, r = %d\n", lambda, r); */
		/* printf("# |A[%d][%d]| = %g\n",r,r,fabs(m_entry(A,r,r))); */

		/* determine if 1x1 or 2x2 block, and do pivoting if needed */
		if ( aii >= alpha*lambda )
		{
		    onebyone = TRUE;
		    goto dopivot;
		}
		/* compute sigma */
		sigma = 0.0;
		for ( k = i; k < n; k++ )
		{
		    if ( k == r )
			continue;
		    tmp = ( k > r ) ? fabs(m_entry(A,r,k)) :
			fabs(m_entry(A,k,r));
		    if ( tmp > sigma )
			sigma = tmp;
		}
		if ( aii*sigma >= alpha*sqr(lambda) )
		    onebyone = TRUE;
		else if ( fabs(m_entry(A,r,r)) >= alpha*sigma )
		{
		    /* printf("# Swapping rows/cols %d and %d\n",i,r); */
		    interchange(A,i,r);
		    px_transp(pivot,i,r);
		    onebyone = TRUE;
		}
		else
		{
		    /* printf("# Swapping rows/cols %d and %d\n",i+1,r); */
		    interchange(A,i+1,r);
		    px_transp(pivot,i+1,r);
		    px_transp(blocks,i,i+1);
		    onebyone = FALSE;
		}
		/* printf("onebyone = %s\n",btos(onebyone)); */
		/* printf("# Matrix so far (@checkpoint A) =\n"); */
		/* m_output(A); */
		/* printf("# pivot =\n");	px_output(pivot); */
		/* printf("# blocks =\n");	px_output(blocks); */

dopivot:
		if ( onebyone )
		{   /* do one by one block */
		    if ( m_entry(A,i,i) != 0.0 )
		    {
			aii = m_entry(A,i,i);
			for ( j = i+1; j < n; j++ )
			{
			    tmp = m_entry(A,i,j)/aii;
			    for ( k = j; k < n; k++ )
				m_sub_val(A,j,k,tmp*m_entry(A,i,k));
			    m_set_val(A,i,j,tmp);
			}
		    }
		}
		else /* onebyone == FALSE */
		{   /* do two by two block */
		    det = m_entry(A,i,i)*m_entry(A,i+1,i+1)-sqr(m_entry(A,i,i+1));
		    /* Must have det < 0 */
		    /* printf("# det = %g\n",det); */
		    aip1i = m_entry(A,i,i+1)/det;
		    aii = m_entry(A,i,i)/det;
		    aip1 = m_entry(A,i+1,i+1)/det;
		    for ( j = i+2; j < n; j++ )
		    {
			s = - aip1i*m_entry(A,i+1,j) + aip1*m_entry(A,i,j);
			t = - aip1i*m_entry(A,i,j) + aii*m_entry(A,i+1,j);
			for ( k = j; k < n; k++ )
			    m_sub_val(A,j,k,m_entry(A,i,k)*s + m_entry(A,i+1,k)*t);
			m_set_val(A,i,j,s);
			m_set_val(A,i+1,j,t);
		    }
		}
		/* printf("# Matrix so far (@checkpoint B) =\n"); */
		/* m_output(A); */
		/* printf("# pivot =\n");	px_output(pivot); */
		/* printf("# blocks =\n");	px_output(blocks); */
	}

	/* set lower triangular half */
	for ( i = 0; i < A->m; i++ )
	    for ( j = 0; j < i; j++ )
		m_set_val(A,i,j,m_entry(A,j,i));

	return A;
}

/* BKPsolve -- solves A.x = b where A has been factored a la BKPfactor()
	-- returns x, which is created if NULL */
#ifndef ANSI_C
VEC	*BKPsolve(A,pivot,block,b,x)
MAT	*A;
PERM	*pivot, *block;
VEC	*b, *x;
#else
VEC	*BKPsolve(const MAT *A, PERM *pivot, const PERM *block,
		  const VEC *b, VEC *x)
#endif
{
	STATIC VEC	*tmp=VNULL;	/* dummy storage needed */
	int	i, j, n, onebyone;
	Real	**A_me, a11, a12, a22, b1, b2, det, sum, *tmp_ve, tmp_diag;

	if ( ! A || ! pivot || ! block || ! b )
		error(E_NULL,"BKPsolve");
	if ( A->m != A->n )
		error(E_SQUARE,"BKPsolve");
	n = A->n;
	if ( b->dim != n || pivot->size != n || block->size != n )
		error(E_SIZES,"BKPsolve");
	x = v_resize(x,n);
	tmp = v_resize(tmp,n);
	MEM_STAT_REG(tmp,TYPE_VEC);

	A_me = A->me;	tmp_ve = tmp->ve;

	px_vec(pivot,b,tmp);
	/* solve for lower triangular part */
	for ( i = 0; i < n; i++ )
	{
		sum = v_entry(tmp,i);
		if ( block->pe[i] < i )
		    for ( j = 0; j < i-1; j++ )
			sum -= m_entry(A,i,j)*v_entry(tmp,j);
		else
		    for ( j = 0; j < i; j++ )
			sum -= m_entry(A,i,j)*v_entry(tmp,j);
		v_set_val(tmp,i,sum);
	}
	/* printf("# BKPsolve: solving L part: tmp =\n");	v_output(tmp); */
	/* solve for diagonal part */
	for ( i = 0; i < n; i = onebyone ? i+1 : i+2 )
	{
		onebyone = ( block->pe[i] == i );
		if ( onebyone )
		{
		    tmp_diag = m_entry(A,i,i);
		    if ( tmp_diag == 0.0 )
			error(E_SING,"BKPsolve");
		    /* tmp_ve[i] /= tmp_diag; */
		    v_set_val(tmp,i,v_entry(tmp,i) / tmp_diag);
		}
		else
		{
		    a11 = m_entry(A,i,i);
		    a22 = m_entry(A,i+1,i+1);
		    a12 = m_entry(A,i+1,i);
		    b1 = v_entry(tmp,i);	b2 = v_entry(tmp,i+1);
		    det = a11*a22-a12*a12;	/* < 0 : see BKPfactor() */
		    if ( det == 0.0 )
			error(E_SING,"BKPsolve");
		    det = 1/det;
		    v_set_val(tmp,i,det*(a22*b1-a12*b2));
		    v_set_val(tmp,i+1,det*(a11*b2-a12*b1));
		}
	}
	/* printf("# BKPsolve: solving D part: tmp =\n");	v_output(tmp); */
	/* solve for transpose of lower traingular part */
	for ( i = n-1; i >= 0; i-- )
	{	/* use symmetry of factored form to get stride 1 */
		sum = v_entry(tmp,i);
		if ( block->pe[i] > i )
		    for ( j = i+2; j < n; j++ )
			sum -= m_entry(A,i,j)*v_entry(tmp,j);
		else
		    for ( j = i+1; j < n; j++ )
			sum -= m_entry(A,i,j)*v_entry(tmp,j);
		v_set_val(tmp,i,sum);
	}

	/* printf("# BKPsolve: solving L^T part: tmp =\n");v_output(tmp); */
	/* and do final permutation */
	x = pxinv_vec(pivot,tmp,x);

#ifdef THREADSAFE
	V_FREE(tmp);
#endif

	return x;
}

		

