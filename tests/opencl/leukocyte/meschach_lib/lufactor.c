
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

/* LUfactor.c 1.5 11/25/87 */
static	char	rcsid[] = "$Id: lufactor.c,v 1.10 1995/05/16 17:26:44 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"



/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* LUfactor -- gaussian elimination with scaled partial pivoting
		-- Note: returns LU matrix which is A */
#ifndef ANSI_C
MAT	*LUfactor(A,pivot)
MAT	*A;
PERM	*pivot;
#else
MAT	*LUfactor(MAT *A, PERM *pivot)
#endif
{
	unsigned int	i, j, m, n;
	int	i_max, k, k_max;
	Real	**A_v, *A_piv, *A_row;
	Real	max1, temp, tiny;
	STATIC	VEC	*scale = VNULL;

	if ( A==(MAT *)NULL || pivot==(PERM *)NULL )
		error(E_NULL,"LUfactor");
	if ( pivot->size != A->m )
		error(E_SIZES,"LUfactor");
	m = A->m;	n = A->n;
	scale = v_resize(scale,A->m);
	MEM_STAT_REG(scale,TYPE_VEC);
	A_v = A->me;

	tiny = 10.0/HUGE_VAL;

	/* initialise pivot with identity permutation */
	for ( i=0; i<m; i++ )
		pivot->pe[i] = i;

	/* set scale parameters */
	for ( i=0; i<m; i++ )
	{
		max1 = 0.0;
		for ( j=0; j<n; j++ )
		{
			temp = fabs(A_v[i][j]);
			max1 = max(max1,temp);
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
		if ( fabs(scale->ve[i]) >= tiny*fabs(A_v[i][k]) )
		{
		    temp = fabs(A_v[i][k])/scale->ve[i];
		    if ( temp > max1 )
		    { max1 = temp;	i_max = i;	}
		}
	    
	    /* if no pivot then ignore column k... */
	    if ( i_max == -1 )
	    {
		/* set pivot entry A[k][k] exactly to zero,
		   rather than just "small" */
		A_v[k][k] = 0.0;
		continue;
	    }
	    
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
		temp = A_v[i][k] = A_v[i][k]/A_v[k][k];
		A_piv = &(A_v[k][k+1]);
		A_row = &(A_v[i][k+1]);
		if ( k+1 < n )
		    __mltadd__(A_row,A_piv,-temp,(int)(n-(k+1)));
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


/* LUsolve -- given an LU factorisation in A, solve Ax=b */
#ifndef ANSI_C
VEC	*LUsolve(LU,pivot,b,x)
MAT	*LU;
PERM	*pivot;
VEC	*b,*x;
#else
VEC	*LUsolve(const MAT *LU, PERM *pivot, const VEC *b, VEC *x)
#endif
{
	if ( ! LU || ! b || ! pivot )
		error(E_NULL,"LUsolve");
	if ( LU->m != LU->n || LU->n != b->dim )
		error(E_SIZES,"LUsolve");

	x = v_resize(x,b->dim);
	px_vec(pivot,b,x);	/* x := P.b */
	Lsolve(LU,x,x,1.0);	/* implicit diagonal = 1 */
	Usolve(LU,x,x,0.0);	/* explicit diagonal */

	return (x);
}

/* LUTsolve -- given an LU factorisation in A, solve A^T.x=b */
#ifndef ANSI_C
VEC	*LUTsolve(LU,pivot,b,x)
MAT	*LU;
PERM	*pivot;
VEC	*b,*x;
#else
VEC	*LUTsolve(const MAT *LU, PERM *pivot, const VEC *b, VEC *x)
#endif
{
	if ( ! LU || ! b || ! pivot )
		error(E_NULL,"LUTsolve");
	if ( LU->m != LU->n || LU->n != b->dim )
		error(E_SIZES,"LUTsolve");

	x = v_copy(b,x);
	UTsolve(LU,x,x,0.0);	/* explicit diagonal */
	LTsolve(LU,x,x,1.0);	/* implicit diagonal = 1 */
	pxinv_vec(pivot,x,x);	/* x := P^T.tmp */

	return (x);
}

/* m_inverse -- returns inverse of A, provided A is not too rank deficient
	-- uses LU factorisation */
#ifndef ANSI_C
MAT	*m_inverse(A,out)
MAT	*A, *out;
#else
MAT	*m_inverse(const MAT *A, MAT *out)
#endif
{
	int	i;
	STATIC VEC	*tmp = VNULL, *tmp2 = VNULL;
	STATIC MAT	*A_cp = MNULL;
	STATIC PERM	*pivot = PNULL;

	if ( ! A )
	    error(E_NULL,"m_inverse");
	if ( A->m != A->n )
	    error(E_SQUARE,"m_inverse");
	if ( ! out || out->m < A->m || out->n < A->n )
	    out = m_resize(out,A->m,A->n);

	A_cp = m_resize(A_cp,A->m,A->n);
	A_cp = m_copy(A,A_cp);
	tmp = v_resize(tmp,A->m);
	tmp2 = v_resize(tmp2,A->m);
	pivot = px_resize(pivot,A->m);
	MEM_STAT_REG(A_cp,TYPE_MAT);
	MEM_STAT_REG(tmp, TYPE_VEC);
	MEM_STAT_REG(tmp2,TYPE_VEC);
	MEM_STAT_REG(pivot,TYPE_PERM);
	tracecatch(LUfactor(A_cp,pivot),"m_inverse");
	for ( i = 0; i < A->n; i++ )
	{
	    v_zero(tmp);
	    tmp->ve[i] = 1.0;
	    tracecatch(LUsolve(A_cp,pivot,tmp,tmp2),"m_inverse");
	    set_col(out,i,tmp2);
	}

#ifdef	THREADSAFE
	V_FREE(tmp);	V_FREE(tmp2);
	M_FREE(A_cp);	PX_FREE(pivot);
#endif

	return out;
}

/* LUcondest -- returns an estimate of the condition number of LU given the
	LU factorisation in compact form */
#ifndef ANSI_C
double	LUcondest(LU,pivot)
MAT	*LU;
PERM	*pivot;
#else
double	LUcondest(const MAT *LU, PERM *pivot)
#endif
{
    STATIC	VEC	*y = VNULL, *z = VNULL;
    Real	cond_est, L_norm, U_norm, sum, tiny;
    int		i, j, n;

    if ( ! LU || ! pivot )
	error(E_NULL,"LUcondest");
    if ( LU->m != LU->n )
	error(E_SQUARE,"LUcondest");
    if ( LU->n != pivot->size )
	error(E_SIZES,"LUcondest");

    tiny = 10.0/HUGE_VAL;

    n = LU->n;
    y = v_resize(y,n);
    z = v_resize(z,n);
    MEM_STAT_REG(y,TYPE_VEC);
    MEM_STAT_REG(z,TYPE_VEC);

    for ( i = 0; i < n; i++ )
    {
	sum = 0.0;
	for ( j = 0; j < i; j++ )
	    sum -= LU->me[j][i]*y->ve[j];
	sum -= (sum < 0.0) ? 1.0 : -1.0;
	if ( fabs(LU->me[i][i]) <= tiny*fabs(sum) )
	    return HUGE_VAL;
	y->ve[i] = sum / LU->me[i][i];
    }

    catch(E_SING,
	  LTsolve(LU,y,y,1.0);
	  LUsolve(LU,pivot,y,z);
	  ,
	  return HUGE_VAL);

    /* now estimate norm of A (even though it is not directly available) */
    /* actually computes ||L||_inf.||U||_inf */
    U_norm = 0.0;
    for ( i = 0; i < n; i++ )
    {
	sum = 0.0;
	for ( j = i; j < n; j++ )
	    sum += fabs(LU->me[i][j]);
	if ( sum > U_norm )
	    U_norm = sum;
    }
    L_norm = 0.0;
    for ( i = 0; i < n; i++ )
    {
	sum = 1.0;
	for ( j = 0; j < i; j++ )
	    sum += fabs(LU->me[i][j]);
	if ( sum > L_norm )
	    L_norm = sum;
    }

    tracecatch(cond_est = U_norm*L_norm*v_norm_inf(z)/v_norm_inf(y),
	       "LUcondest");

#ifdef	THREADSAFE
    V_FREE(y);    V_FREE(z);
#endif

    return cond_est;
}
