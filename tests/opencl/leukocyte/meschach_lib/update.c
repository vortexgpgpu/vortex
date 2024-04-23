
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

/* update.c 1.3 11/25/87 */
static	char	rcsid[] = "$Id: update.c,v 1.2 1994/01/13 05:26:06 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"




/* Most matrix factorisation routines are in-situ unless otherwise specified */

/* LDLupdate -- updates a CHolesky factorisation, replacing LDL' by
	MD~M' = LDL' + alpha.w.w' Note: w is overwritten
	Ref: Gill et al Math Comp 28, p516 Algorithm C1 */
#ifndef ANSI_C
MAT	*LDLupdate(CHmat,w,alpha)
MAT	*CHmat;
VEC	*w;
double	alpha;
#else
MAT	*LDLupdate(MAT *CHmat, VEC *w, double alpha)
#endif
{
	unsigned int	i,j;
	Real	diag,new_diag,beta,p;

	if ( CHmat==(MAT *)NULL || w==(VEC *)NULL )
		error(E_NULL,"LDLupdate");
	if ( CHmat->m != CHmat->n || w->dim != CHmat->m )
		error(E_SIZES,"LDLupdate");

	for ( j=0; j < w->dim; j++ )
	{
		p = w->ve[j];
		diag = CHmat->me[j][j];
		new_diag = CHmat->me[j][j] = diag + alpha*p*p;
		if ( new_diag <= 0.0 )
			error(E_POSDEF,"LDLupdate");
		beta = p*alpha/new_diag;
		alpha *= diag/new_diag;

		for ( i=j+1; i < w->dim; i++ )
		{
			w->ve[i] -= p*CHmat->me[i][j];
			CHmat->me[i][j] += beta*w->ve[i];
			CHmat->me[j][i] = CHmat->me[i][j];
		}
	}

	return (CHmat);
}


/* QRupdate -- updates QR factorisation in expanded form (seperate matrices)
	Finds Q+, R+ s.t. Q+.R+ = Q.(R+u.v') and Q+ orthogonal, R+ upper triang
	Ref: Golub & van Loan Matrix Computations pp437-443
	-- does not update Q if it is NULL */
#ifndef ANSI_C
MAT	*QRupdate(Q,R,u,v)
MAT	*Q,*R;
VEC	*u,*v;
#else
MAT	*QRupdate(MAT *Q, MAT *R, VEC *u, VEC *v)
#endif
{
	int	i,j,k;
	Real	c,s,temp;

	if ( ! R || ! u || ! v )
		error(E_NULL,"QRupdate");
	if ( ( Q && ( Q->m != Q->n || R->m != Q->n ) ) ||
					u->dim != R->m || v->dim != R->n )
		error(E_SIZES,"QRupdate");

	/* find largest k s.t. u[k] != 0 */
	for ( k=R->m-1; k>=0; k-- )
		if ( u->ve[k] != 0.0 )
			break;

	/* transform R+u.v' to Hessenberg form */
	for ( i=k-1; i>=0; i-- )
	{
		/* get Givens rotation */
		givens(u->ve[i],u->ve[i+1],&c,&s);
		rot_rows(R,i,i+1,c,s,R);
		if ( Q )
			rot_cols(Q,i,i+1,c,s,Q);
		rot_vec(u,i,i+1,c,s,u);
	}

	/* add into R */
	temp = u->ve[0];
	for ( j=0; j<R->n; j++ )
		R->me[0][j] += temp*v->ve[j];

	/* transform Hessenberg to upper triangular */
	for ( i=0; i<k; i++ )
	{
		givens(R->me[i][i],R->me[i+1][i],&c,&s);
		rot_rows(R,i,i+1,c,s,R);
		if ( Q )
			rot_cols(Q,i,i+1,c,s,Q);
	}

	return R;
}

