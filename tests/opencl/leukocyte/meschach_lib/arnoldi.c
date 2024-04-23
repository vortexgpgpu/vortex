
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
	Arnoldi method for finding eigenvalues of large non-symmetric
		matrices
*/
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include	"matrix2.h"
#include	"sparse.h"

static char rcsid[] = "$Id: arnoldi.c,v 1.3 1994/01/13 05:45:40 des Exp $";


/* arnoldi -- an implementation of the Arnoldi method */
MAT	*arnoldi(A,A_param,x0,m,h_rem,Q,H)
VEC	*(*A)();
void	*A_param;
VEC	*x0;
int	m;
Real	*h_rem;
MAT	*Q, *H;
{
	STATIC VEC	*v=VNULL, *u=VNULL, *r=VNULL, *s=VNULL, *tmp=VNULL;
	int	i;
	Real	h_val;

	if ( ! A || ! Q || ! x0 )
	    error(E_NULL,"arnoldi");
	if ( m <= 0 )
	    error(E_BOUNDS,"arnoldi");
	if ( Q->n != x0->dim ||	Q->m != m )
	    error(E_SIZES,"arnoldi");

	m_zero(Q);
	H = m_resize(H,m,m);
	m_zero(H);
	u = v_resize(u,x0->dim);
	v = v_resize(v,x0->dim);
	r = v_resize(r,m);
	s = v_resize(s,m);
	tmp = v_resize(tmp,x0->dim);
	MEM_STAT_REG(u,TYPE_VEC);
	MEM_STAT_REG(v,TYPE_VEC);
	MEM_STAT_REG(r,TYPE_VEC);
	MEM_STAT_REG(s,TYPE_VEC);
	MEM_STAT_REG(tmp,TYPE_VEC);
	sv_mlt(1.0/v_norm2(x0),x0,v);
	for ( i = 0; i < m; i++ )
	{
	    set_row(Q,i,v);
	    u = (*A)(A_param,v,u);
	    r = mv_mlt(Q,u,r);
	    tmp = vm_mlt(Q,r,tmp);
	    v_sub(u,tmp,u);
	    h_val = v_norm2(u);
	    /* if u == 0 then we have an exact subspace */
	    if ( h_val == 0.0 )
	    {
		*h_rem = h_val;
		return H;
	    }
	    /* iterative refinement -- ensures near orthogonality */
	    do {
		s = mv_mlt(Q,u,s);
		tmp = vm_mlt(Q,s,tmp);
		v_sub(u,tmp,u);
		v_add(r,s,r);
	    } while ( v_norm2(s) > 0.1*(h_val = v_norm2(u)) );
	    /* now that u is nearly orthogonal to Q, update H */
	    set_col(H,i,r);
	    if ( i == m-1 )
	    {
		*h_rem = h_val;
		continue;
	    }
	    /* H->me[i+1][i] = h_val; */
	    m_set_val(H,i+1,i,h_val);
	    sv_mlt(1.0/h_val,u,v);
	}

#ifdef THREADSAFE
	V_FREE(v);	V_FREE(u);	V_FREE(r);
	V_FREE(r);	V_FREE(s);	V_FREE(tmp);
#endif
	return H;
}

/* sp_arnoldi -- uses arnoldi() with an explicit representation of A */
MAT	*sp_arnoldi(A,x0,m,h_rem,Q,H)
SPMAT	*A;
VEC	*x0;
int	m;
Real	*h_rem;
MAT	*Q, *H;
{	return arnoldi(sp_mv_mlt,A,x0,m,h_rem,Q,H);	}

/* gmres -- generalised minimum residual algorithm of Saad & Schultz
		SIAM J. Sci. Stat. Comp. v.7, pp.856--869 (1986)
	-- y is overwritten with the solution */
VEC	*gmres(A,A_param,m,Q,R,b,tol,x)
VEC	*(*A)();
void	*A_param;
VEC	*b, *x;
int	m;
MAT	*Q, *R;
double	tol;
{
    STATIC VEC	*v=VNULL, *u=VNULL, *r=VNULL, *tmp=VNULL, *rhs=VNULL;
    STATIC VEC	*diag=VNULL, *beta=VNULL;
    int	i;
    Real	h_val, norm_b;
    
    if ( ! A || ! Q || ! b || ! R )
	error(E_NULL,"gmres");
    if ( m <= 0 )
	error(E_BOUNDS,"gmres");
    if ( Q->n != b->dim || Q->m != m )
	error(E_SIZES,"gmres");
    
    x = v_copy(b,x);
    m_zero(Q);
    R = m_resize(R,m+1,m);
    m_zero(R);
    u = v_resize(u,x->dim);
    v = v_resize(v,x->dim);
    tmp = v_resize(tmp,x->dim);
    rhs = v_resize(rhs,m+1);
    MEM_STAT_REG(u,TYPE_VEC);
    MEM_STAT_REG(v,TYPE_VEC);
    MEM_STAT_REG(r,TYPE_VEC);
    MEM_STAT_REG(tmp,TYPE_VEC);
    MEM_STAT_REG(rhs,TYPE_VEC);
    norm_b = v_norm2(x);
    if ( norm_b == 0.0 )
	error(E_RANGE,"gmres");
    sv_mlt(1.0/norm_b,x,v);
    
    for ( i = 0; i < m; i++ )
    {
	set_row(Q,i,v);
	tracecatch(u = (*A)(A_param,v,u),"gmres");
	r = mv_mlt(Q,u,r);
	tmp = vm_mlt(Q,r,tmp);
	v_sub(u,tmp,u);
	h_val = v_norm2(u);
	set_col(R,i,r);
	R->me[i+1][i] = h_val;
	sv_mlt(1.0/h_val,u,v);
    }
    
    /* use i x i submatrix of R */
    R = m_resize(R,i+1,i);
    rhs = v_resize(rhs,i+1);
    v_zero(rhs);
    rhs->ve[0] = norm_b;
    tmp = v_resize(tmp,i);
    diag = v_resize(diag,i+1);
    beta = v_resize(beta,i+1);
    MEM_STAT_REG(beta,TYPE_VEC);
    MEM_STAT_REG(diag,TYPE_VEC);
    QRfactor(R,diag /* ,beta */);
    tmp = QRsolve(R,diag, /* beta, */ rhs,tmp);
    v_resize(tmp,m);
    vm_mlt(Q,tmp,x);

#ifdef THREADSAFE
    V_FREE(v);		V_FREE(u);	V_FREE(r);
    V_FREE(tmp);	V_FREE(rhs);
    V_FREE(diag);	V_FREE(beta);
#endif

    return x;
}
