
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

/* solve.c 1.2 11/25/87 */
static	char	rcsid[] = "$Id: solve.c,v 1.3 1994/01/13 05:29:57 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "matrix2.h"





/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* Usolve -- back substitution with optional over-riding diagonal
		-- can be in-situ but doesn't need to be */
#ifndef ANSI_C
VEC	*Usolve(matrix,b,out,diag)
MAT	*matrix;
VEC	*b, *out;
double	diag;
#else
VEC	*Usolve(const MAT *matrix, const VEC *b, VEC *out, double diag)
#endif
{
	unsigned int	dim /* , j */;
	int	i, i_lim;
	Real	**mat_ent, *mat_row, *b_ent, *out_ent, *out_col, sum, tiny;

	if ( matrix==MNULL || b==VNULL )
		error(E_NULL,"Usolve");
	dim = min(matrix->m,matrix->n);
	if ( b->dim < dim )
		error(E_SIZES,"Usolve");
	if ( out==VNULL || out->dim < dim )
		out = v_resize(out,matrix->n);
	mat_ent = matrix->me;	b_ent = b->ve;	out_ent = out->ve;

	tiny = 10.0/HUGE_VAL;

	for ( i=dim-1; i>=0; i-- )
		if ( b_ent[i] != 0.0 )
		    break;
		else
		    out_ent[i] = 0.0;
	i_lim = i;

	for (    ; i>=0; i-- )
	{
		sum = b_ent[i];
		mat_row = &(mat_ent[i][i+1]);
		out_col = &(out_ent[i+1]);
		sum -= __ip__(mat_row,out_col,i_lim-i);
		/******************************************************
		for ( j=i+1; j<=i_lim; j++ )
			sum -= mat_ent[i][j]*out_ent[j];
			sum -= (*mat_row++)*(*out_col++);
		******************************************************/
		if ( diag==0.0 )
		{
			if ( fabs(mat_ent[i][i]) <= tiny*fabs(sum) )
				error(E_SING,"Usolve");
			else
				out_ent[i] = sum/mat_ent[i][i];
		}
		else
			out_ent[i] = sum/diag;
	}

	return (out);
}

/* Lsolve -- forward elimination with (optional) default diagonal value */
#ifndef ANSI_C
VEC	*Lsolve(matrix,b,out,diag)
MAT	*matrix;
VEC	*b,*out;
double	diag;
#else
VEC	*Lsolve(const MAT *matrix, const VEC *b, VEC *out, double diag)
#endif
{
	unsigned int	dim, i, i_lim /* , j */;
	Real	**mat_ent, *mat_row, *b_ent, *out_ent, *out_col, sum, tiny;

	if ( matrix==(MAT *)NULL || b==(VEC *)NULL )
		error(E_NULL,"Lsolve");
	dim = min(matrix->m,matrix->n);
	if ( b->dim < dim )
		error(E_SIZES,"Lsolve");
	if ( out==(VEC *)NULL || out->dim < dim )
		out = v_resize(out,matrix->n);
	mat_ent = matrix->me;	b_ent = b->ve;	out_ent = out->ve;

	for ( i=0; i<dim; i++ )
		if ( b_ent[i] != 0.0 )
		    break;
		else
		    out_ent[i] = 0.0;
	i_lim = i;

	tiny = 10.0/HUGE_VAL;

	for (    ; i<dim; i++ )
	{
		sum = b_ent[i];
		mat_row = &(mat_ent[i][i_lim]);
		out_col = &(out_ent[i_lim]);
		sum -= __ip__(mat_row,out_col,(int)(i-i_lim));
		/*****************************************************
		for ( j=i_lim; j<i; j++ )
			sum -= mat_ent[i][j]*out_ent[j];
			sum -= (*mat_row++)*(*out_col++);
		******************************************************/
		if ( diag==0.0 )
		{
			if ( fabs(mat_ent[i][i]) <= tiny*fabs(sum) )
				error(E_SING,"Lsolve");
			else
				out_ent[i] = sum/mat_ent[i][i];
		}
		else
			out_ent[i] = sum/diag;
	}

	return (out);
}


/* UTsolve -- forward elimination with (optional) default diagonal value
		using UPPER triangular part of matrix */
#ifndef ANSI_C
VEC	*UTsolve(U,b,out,diag)
MAT	*U;
VEC	*b,*out;
double	diag;
#else
VEC	*UTsolve(const MAT *U, const VEC *b, VEC *out, double diag)
#endif
{
    unsigned int	dim, i, i_lim;
    Real	**U_me, *b_ve, *out_ve, tmp, invdiag, tiny;
    
    if ( ! U || ! b )
	error(E_NULL,"UTsolve");
    dim = min(U->m,U->n);
    if ( b->dim < dim )
	error(E_SIZES,"UTsolve");
    out = v_resize(out,U->n);
    U_me = U->me;	b_ve = b->ve;	out_ve = out->ve;

    tiny = 10.0/HUGE_VAL;

    for ( i=0; i<dim; i++ )
	if ( b_ve[i] != 0.0 )
	    break;
	else
	    out_ve[i] = 0.0;
    i_lim = i;
    if ( b != out )
    {
	__zero__(out_ve,out->dim);
	MEM_COPY(&(b_ve[i_lim]),&(out_ve[i_lim]),(dim-i_lim)*sizeof(Real));
    }

    if ( diag == 0.0 )
    {
	for (    ; i<dim; i++ )
	{
	    tmp = U_me[i][i];
	    if ( fabs(tmp) <= tiny*fabs(out_ve[i]) )
		error(E_SING,"UTsolve");
	    out_ve[i] /= tmp;
	    __mltadd__(&(out_ve[i+1]),&(U_me[i][i+1]),-out_ve[i],dim-i-1);
	}
    }
    else
    {
	invdiag = 1.0/diag;
	for (    ; i<dim; i++ )
	{
	    out_ve[i] *= invdiag;
	    __mltadd__(&(out_ve[i+1]),&(U_me[i][i+1]),-out_ve[i],dim-i-1);
	}
    }
    return (out);
}

/* Dsolve -- solves Dx=b where D is the diagonal of A -- may be in-situ */
#ifndef ANSI_C
VEC	*Dsolve(A,b,x)
MAT	*A;
VEC	*b,*x;
#else
VEC	*Dsolve(const MAT *A, const VEC *b, VEC *x)
#endif
{
    unsigned int	dim, i;
    Real	tiny;
    
    if ( ! A || ! b )
	error(E_NULL,"Dsolve");
    dim = min(A->m,A->n);
    if ( b->dim < dim )
	error(E_SIZES,"Dsolve");
    x = v_resize(x,A->n);

    tiny = 10.0/HUGE_VAL;

    dim = b->dim;
    for ( i=0; i<dim; i++ )
	if ( fabs(A->me[i][i]) <= tiny*fabs(b->ve[i]) )
	    error(E_SING,"Dsolve");
	else
	    x->ve[i] = b->ve[i]/A->me[i][i];
    
    return (x);
}

/* LTsolve -- back substitution with optional over-riding diagonal
		using the LOWER triangular part of matrix
		-- can be in-situ but doesn't need to be */
#ifndef ANSI_C
VEC	*LTsolve(L,b,out,diag)
MAT	*L;
VEC	*b, *out;
double	diag;
#else
VEC	*LTsolve(const MAT *L, const VEC *b, VEC *out, double diag)
#endif
{
    unsigned int	dim;
    int		i, i_lim;
    Real	**L_me, *b_ve, *out_ve, tmp, invdiag, tiny;
    
    if ( ! L || ! b )
	error(E_NULL,"LTsolve");
    dim = min(L->m,L->n);
    if ( b->dim < dim )
	error(E_SIZES,"LTsolve");
    out = v_resize(out,L->n);
    L_me = L->me;	b_ve = b->ve;	out_ve = out->ve;

    tiny = 10.0/HUGE_VAL;
    
    for ( i=dim-1; i>=0; i-- )
	if ( b_ve[i] != 0.0 )
	    break;
    i_lim = i;

    if ( b != out )
    {
	__zero__(out_ve,out->dim);
	MEM_COPY(b_ve,out_ve,(i_lim+1)*sizeof(Real));
    }

    if ( diag == 0.0 )
    {
	for (        ; i>=0; i-- )
	{
	    tmp = L_me[i][i];
	    if ( fabs(tmp) <= tiny*fabs(out_ve[i]) )
		error(E_SING,"LTsolve");
	    out_ve[i] /= tmp;
	    __mltadd__(out_ve,L_me[i],-out_ve[i],i);
	}
    }
    else
    {
	invdiag = 1.0/diag;
	for (        ; i>=0; i-- )
	{
	    out_ve[i] *= invdiag;
	    __mltadd__(out_ve,L_me[i],-out_ve[i],i);
	}
    }
    
    return (out);
}
