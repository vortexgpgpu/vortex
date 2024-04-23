
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
	File containing Lanczos type routines for finding eigenvalues
	of large, sparse, symmetic matrices
*/

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include	"sparse.h"

static char rcsid[] = "$Id: lanczos.c,v 1.4 1994/01/13 05:28:24 des Exp $";

#ifdef ANSI_C
extern	VEC	*trieig(VEC *,VEC *,MAT *);
#else
extern	VEC	*trieig();
#endif

/* lanczos -- raw lanczos algorithm -- no re-orthogonalisation
	-- creates T matrix of size == m,
		but no larger than before beta_k == 0
	-- uses passed routine to do matrix-vector multiplies */
void	lanczos(A_fn,A_params,m,x0,a,b,beta2,Q)
VEC	*(*A_fn)();	/* VEC *(*A_fn)(void *A_params,VEC *in, VEC *out) */
void	*A_params;
int	m;
VEC	*x0, *a, *b;
Real	*beta2;
MAT	*Q;
{
	int	j;
	VEC	*v, *w, *tmp;
	Real	alpha, beta;

	if ( ! A_fn || ! x0 || ! a || ! b )
		error(E_NULL,"lanczos");
	if ( m <= 0 )
		error(E_BOUNDS,"lanczos");
	if ( Q && ( Q->m < x0->dim || Q->n < m ) )
		error(E_SIZES,"lanczos");

	a = v_resize(a,(unsigned int)m);
	b = v_resize(b,(unsigned int)(m-1));
	v = v_get(x0->dim);
	w = v_get(x0->dim);
	tmp = v_get(x0->dim);

	beta = 1.0;
	/* normalise x0 as w */
	sv_mlt(1.0/v_norm2(x0),x0,w);

	(*A_fn)(A_params,w,v);

	for ( j = 0; j < m; j++ )
	{
		/* store w in Q if Q not NULL */
		if ( Q )
		    set_col(Q,j,w);

		alpha = in_prod(w,v);
		a->ve[j] = alpha;
		v_mltadd(v,w,-alpha,v);
		beta = v_norm2(v);
		if ( beta == 0.0 )
		{
		    v_resize(a,(unsigned int)j+1);
		    v_resize(b,(unsigned int)j);
		    *beta2 = 0.0;
		    if ( Q )
			Q = m_resize(Q,Q->m,j+1);
		    return;
		}
		if ( j < m-1 )
		    b->ve[j] = beta;
		v_copy(w,tmp);
		sv_mlt(1/beta,v,w);
		sv_mlt(-beta,tmp,v);
		(*A_fn)(A_params,w,tmp);
		v_add(v,tmp,v);
	}
	*beta2 = beta;


	V_FREE(v);	V_FREE(w);	V_FREE(tmp);
}

extern	double	frexp(), ldexp();

/* product -- returns the product of a long list of numbers
	-- answer stored in mant (mantissa) and expt (exponent) */
static	double	product(a,offset,expt)
VEC	*a;
double	offset;
int	*expt;
{
	Real	mant, tmp_fctr;
	int	i, tmp_expt;

	if ( ! a )
		error(E_NULL,"product");

	mant = 1.0;
	*expt = 0;
	if ( offset == 0.0 )
		for ( i = 0; i < a->dim; i++ )
		{
			mant *= frexp(a->ve[i],&tmp_expt);
			*expt += tmp_expt;
			if ( ! (i % 10) )
			{
			    mant = frexp(mant,&tmp_expt);
			    *expt += tmp_expt;
			}
		}
	else
		for ( i = 0; i < a->dim; i++ )
		{
			tmp_fctr = a->ve[i] - offset;
			tmp_fctr += (tmp_fctr > 0.0 ) ? -MACHEPS*offset :
							 MACHEPS*offset;
			mant *= frexp(tmp_fctr,&tmp_expt);
			*expt += tmp_expt;
			if ( ! (i % 10) )
			{
			    mant = frexp(mant,&tmp_expt);
			    *expt += tmp_expt;
			}
		}

	mant = frexp(mant,&tmp_expt);
	*expt += tmp_expt;

	return mant;
}

/* product2 -- returns the product of a long list of numbers
	-- answer stored in mant (mantissa) and expt (exponent) */
static	double	product2(a,k,expt)
VEC	*a;
int	k;	/* entry of a to leave out */
int	*expt;
{
	Real	mant, mu, tmp_fctr;
	int	i, tmp_expt;

	if ( ! a )
		error(E_NULL,"product2");
	if ( k < 0 || k >= a->dim )
		error(E_BOUNDS,"product2");

	mant = 1.0;
	*expt = 0;
	mu = a->ve[k];
	for ( i = 0; i < a->dim; i++ )
	{
		if ( i == k )
			continue;
		tmp_fctr = a->ve[i] - mu;
		tmp_fctr += ( tmp_fctr > 0.0 ) ? -MACHEPS*mu : MACHEPS*mu;
		mant *= frexp(tmp_fctr,&tmp_expt);
		*expt += tmp_expt;
		if ( ! (i % 10) )
		{
		    mant = frexp(mant,&tmp_expt);
		    *expt += tmp_expt;
		}
	}
	mant = frexp(mant,&tmp_expt);
	*expt += tmp_expt;

	return mant;
}

/* dbl_cmp -- comparison function to pass to qsort() */
static	int	dbl_cmp(x,y)
Real	*x, *y;
{
	Real	tmp;

	tmp = *x - *y;
	return (tmp > 0 ? 1 : tmp < 0 ? -1: 0);
}

/* lanczos2 -- lanczos + error estimate for every e-val
	-- uses Cullum & Willoughby approach, Sparse Matrix Proc. 1978
	-- returns multiple e-vals where multiple e-vals may not exist
	-- returns evals vector */
VEC	*lanczos2(A_fn,A_params,m,x0,evals,err_est)
VEC	*(*A_fn)();
void	*A_params;
int	m;
VEC	*x0;		/* initial vector */
VEC	*evals;		/* eigenvalue vector */
VEC	*err_est;	/* error estimates of eigenvalues */
{
	VEC		*a;
	STATIC	VEC	*b=VNULL, *a2=VNULL, *b2=VNULL;
	Real	beta, pb_mant, det_mant, det_mant1, det_mant2;
	int	i, pb_expt, det_expt, det_expt1, det_expt2;

	if ( ! A_fn || ! x0 )
		error(E_NULL,"lanczos2");
	if ( m <= 0 )
		error(E_RANGE,"lanczos2");

	a = evals;
	a = v_resize(a,(unsigned int)m);
	b = v_resize(b,(unsigned int)(m-1));
	MEM_STAT_REG(b,TYPE_VEC);

	lanczos(A_fn,A_params,m,x0,a,b,&beta,MNULL);

	/* printf("# beta =%g\n",beta); */
	pb_mant = 0.0;
	if ( err_est )
	{
		pb_mant = product(b,(double)0.0,&pb_expt);
		/* printf("# pb_mant = %g, pb_expt = %d\n",pb_mant, pb_expt); */
	}

	/* printf("# diags =\n");	out_vec(a); */
	/* printf("# off diags =\n");	out_vec(b); */
	a2 = v_resize(a2,a->dim - 1);
	b2 = v_resize(b2,b->dim - 1);
	MEM_STAT_REG(a2,TYPE_VEC);
	MEM_STAT_REG(b2,TYPE_VEC);
	for ( i = 0; i < a2->dim - 1; i++ )
	{
		a2->ve[i] = a->ve[i+1];
		b2->ve[i] = b->ve[i+1];
	}
	a2->ve[a2->dim-1] = a->ve[a2->dim];

	trieig(a,b,MNULL);

	/* sort evals as a courtesy */
	qsort((void *)(a->ve),(int)(a->dim),sizeof(Real),(int (*)())dbl_cmp);

	/* error estimates */
	if ( err_est )
	{
		err_est = v_resize(err_est,(unsigned int)m);

		trieig(a2,b2,MNULL);
		/* printf("# a =\n");	out_vec(a); */
		/* printf("# a2 =\n");	out_vec(a2); */

		for ( i = 0; i < a->dim; i++ )
		{
			det_mant1 = product2(a,i,&det_expt1);
			det_mant2 = product(a2,(double)a->ve[i],&det_expt2);
			/* printf("# det_mant1=%g, det_expt1=%d\n",
					det_mant1,det_expt1); */
			/* printf("# det_mant2=%g, det_expt2=%d\n",
					det_mant2,det_expt2); */
			if ( det_mant1 == 0.0 )
			{   /* multiple e-val of T */
			    err_est->ve[i] = 0.0;
			    continue;
			}
			else if ( det_mant2 == 0.0 )
			{
			    err_est->ve[i] = HUGE_VAL;
			    continue;
			}
			if ( (det_expt1 + det_expt2) % 2 )
			    /* if odd... */
			    det_mant = sqrt(2.0*fabs(det_mant1*det_mant2));
			else /* if even... */
			    det_mant = sqrt(fabs(det_mant1*det_mant2));
			det_expt = (det_expt1+det_expt2)/2;
			err_est->ve[i] = fabs(beta*
				ldexp(pb_mant/det_mant,pb_expt-det_expt));
		}
	}

#ifdef	THREADSAFE
	V_FREE(b);	V_FREE(a2);	V_FREE(b2);
#endif

	return a;
}

/* sp_lanczos -- version that uses sparse matrix data structure */
void    sp_lanczos(A,m,x0,a,b,beta2,Q)
SPMAT	*A;
int     m;
VEC     *x0, *a, *b;
Real  *beta2;
MAT     *Q;
{	lanczos(sp_mv_mlt,A,m,x0,a,b,beta2,Q);	}

/* sp_lanczos2 -- version of lanczos2() that uses sparse matrix data
					structure */
VEC	*sp_lanczos2(A,m,x0,evals,err_est)
SPMAT	*A;
int	m;
VEC	*x0;		/* initial vector */
VEC	*evals;		/* eigenvalue vector */
VEC	*err_est;	/* error estimates of eigenvalues */
{	return lanczos2(sp_mv_mlt,A,m,x0,evals,err_est);	}

