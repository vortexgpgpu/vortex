
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
  This file contains the routines needed to perform QR factorisation
  of matrices, as well as Householder transformations.
  The internal "factored form" of a matrix A is not quite standard.
  The diagonal of A is replaced by the diagonal of R -- not by the 1st non-zero
  entries of the Householder vectors. The 1st non-zero entries are held in
  the diag parameter of QRfactor(). The reason for this non-standard
  representation is that it enables direct use of the Usolve() function
  rather than requiring that  a seperate function be written just for this case.
  See, e.g., QRsolve() below for more details.
  
*/


static	char	rcsid[] = "$Id: qrfactor.c,v 1.5 1994/01/13 05:35:07 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "matrix2.h"





#define		sign(x)	((x) > 0.0 ? 1 : ((x) < 0.0 ? -1 : 0 ))

extern	VEC	*Usolve();	/* See matrix2.h */

/* Note: The usual representation of a Householder transformation is taken
   to be:
   P = I - beta.u.uT
   where beta = 2/(uT.u) and u is called the Householder vector
   */

/* QRfactor -- forms the QR factorisation of A -- factorisation stored in
   compact form as described above ( not quite standard format ) */
#ifndef ANSI_C
MAT	*QRfactor(A,diag)
MAT	*A;
VEC	*diag;
#else
MAT	*QRfactor(MAT *A, VEC *diag)
#endif
{
    unsigned int	k,limit;
    Real	beta;
    STATIC	VEC	*hh=VNULL, *w=VNULL;
    
    if ( ! A || ! diag )
	error(E_NULL,"QRfactor");
    limit = min(A->m,A->n);
    if ( diag->dim < limit )
	error(E_SIZES,"QRfactor");
    
    hh = v_resize(hh,A->m);
    w  = v_resize(w, A->n);
    MEM_STAT_REG(hh,TYPE_VEC);
    MEM_STAT_REG(w, TYPE_VEC);
    
    for ( k=0; k<limit; k++ )
    {
	/* get H/holder vector for the k-th column */
	get_col(A,k,hh);
	/* hhvec(hh,k,&beta->ve[k],hh,&A->me[k][k]); */
	hhvec(hh,k,&beta,hh,&A->me[k][k]);
	diag->ve[k] = hh->ve[k];
	
	/* apply H/holder vector to remaining columns */
	/* hhtrcols(A,k,k+1,hh,beta->ve[k]); */
	_hhtrcols(A,k,k+1,hh,beta,w);
    }

#ifdef	THREADSAFE
    V_FREE(hh);	V_FREE(w);
#endif

    return (A);
}

/* QRCPfactor -- forms the QR factorisation of A with column pivoting
   -- factorisation stored in compact form as described above
   ( not quite standard format )				*/
#ifndef ANSI_C
MAT	*QRCPfactor(A,diag,px)
MAT	*A;
VEC	*diag;
PERM	*px;
#else
MAT	*QRCPfactor(MAT *A, VEC *diag, PERM *px)
#endif
{
    unsigned int	i, i_max, j, k, limit;
    STATIC	VEC	*gamma=VNULL, *tmp1=VNULL, *tmp2=VNULL, *w=VNULL;
    Real	beta, maxgamma, sum, tmp;
    
    if ( ! A || ! diag || ! px )
	error(E_NULL,"QRCPfactor");
    limit = min(A->m,A->n);
    if ( diag->dim < limit || px->size != A->n )
	error(E_SIZES,"QRCPfactor");
    
    tmp1 = v_resize(tmp1,A->m);
    tmp2 = v_resize(tmp2,A->m);
    gamma = v_resize(gamma,A->n);
    w    = v_resize(w,   A->n);
    MEM_STAT_REG(tmp1,TYPE_VEC);
    MEM_STAT_REG(tmp2,TYPE_VEC);
    MEM_STAT_REG(gamma,TYPE_VEC);
    MEM_STAT_REG(w,   TYPE_VEC);
    
    /* initialise gamma and px */
    for ( j=0; j<A->n; j++ )
    {
	px->pe[j] = j;
	sum = 0.0;
	for ( i=0; i<A->m; i++ )
	    sum += square(A->me[i][j]);
	gamma->ve[j] = sum;
    }
    
    for ( k=0; k<limit; k++ )
    {
	/* find "best" column to use */
	i_max = k;	maxgamma = gamma->ve[k];
	for ( i=k+1; i<A->n; i++ )
	    /* Loop invariant:maxgamma=gamma[i_max]
	       >=gamma[l];l=k,...,i-1 */
	    if ( gamma->ve[i] > maxgamma )
	    {	maxgamma = gamma->ve[i]; i_max = i;	}
	
	/* swap columns if necessary */
	if ( i_max != k )
	{
	    /* swap gamma values */
	    tmp = gamma->ve[k];
	    gamma->ve[k] = gamma->ve[i_max];
	    gamma->ve[i_max] = tmp;
	    
	    /* update column permutation */
	    px_transp(px,k,i_max);
	    
	    /* swap columns of A */
	    for ( i=0; i<A->m; i++ )
	    {
		tmp = A->me[i][k];
		A->me[i][k] = A->me[i][i_max];
		A->me[i][i_max] = tmp;
	    }
	}
	
	/* get H/holder vector for the k-th column */
	get_col(A,k,tmp1);
	/* hhvec(tmp1,k,&beta->ve[k],tmp1,&A->me[k][k]); */
	hhvec(tmp1,k,&beta,tmp1,&A->me[k][k]);
	diag->ve[k] = tmp1->ve[k];
	
	/* apply H/holder vector to remaining columns */
	/* hhtrcols(A,k,k+1,tmp1,beta->ve[k]); */
	_hhtrcols(A,k,k+1,tmp1,beta,w);
	
	/* update gamma values */
	for ( j=k+1; j<A->n; j++ )
	    gamma->ve[j] -= square(A->me[k][j]);
    }

#ifdef	THREADSAFE
    V_FREE(gamma);	V_FREE(tmp1);	V_FREE(tmp2);	V_FREE(w);
#endif

    return (A);
}

/* Qsolve -- solves Qx = b, Q is an orthogonal matrix stored in compact
   form a la QRfactor() -- may be in-situ */
#ifndef ANSI_C
VEC	*_Qsolve(QR,diag,b,x,tmp)
MAT	*QR;
VEC	*diag, *b, *x, *tmp;
#else
VEC	*_Qsolve(const MAT *QR, const VEC *diag, const VEC *b, 
		 VEC *x, VEC *tmp)
#endif
{
    unsigned int	dynamic;
    int		k, limit;
    Real	beta, r_ii, tmp_val;
    
    limit = min(QR->m,QR->n);
    dynamic = FALSE;
    if ( ! QR || ! diag || ! b )
	error(E_NULL,"_Qsolve");
    if ( diag->dim < limit || b->dim != QR->m )
	error(E_SIZES,"_Qsolve");
    x = v_resize(x,QR->m);
    if ( tmp == VNULL )
	dynamic = TRUE;
    tmp = v_resize(tmp,QR->m);
    
    /* apply H/holder transforms in normal order */
    x = v_copy(b,x);
    for ( k = 0 ; k < limit ; k++ )
    {
	get_col(QR,k,tmp);
	r_ii = fabs(tmp->ve[k]);
	tmp->ve[k] = diag->ve[k];
	tmp_val = (r_ii*fabs(diag->ve[k]));
	beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	/* hhtrvec(tmp,beta->ve[k],k,x,x); */
	hhtrvec(tmp,beta,k,x,x);
    }
    
    if ( dynamic )
	V_FREE(tmp);
    
    return (x);
}

/* makeQ -- constructs orthogonal matrix from Householder vectors stored in
   compact QR form */
#ifndef ANSI_C
MAT	*makeQ(QR,diag,Qout)
MAT	*QR,*Qout;
VEC	*diag;
#else
MAT	*makeQ(const MAT *QR,const VEC *diag, MAT *Qout)
#endif
{
    STATIC	VEC	*tmp1=VNULL,*tmp2=VNULL;
    unsigned int	i, limit;
    Real	beta, r_ii, tmp_val;
    int	j;
    
    limit = min(QR->m,QR->n);
    if ( ! QR || ! diag )
	error(E_NULL,"makeQ");
    if ( diag->dim < limit )
	error(E_SIZES,"makeQ");
    if ( Qout==(MAT *)NULL || Qout->m < QR->m || Qout->n < QR->m )
	Qout = m_get(QR->m,QR->m);
    
    tmp1 = v_resize(tmp1,QR->m);	/* contains basis vec & columns of Q */
    tmp2 = v_resize(tmp2,QR->m);	/* contains H/holder vectors */
    MEM_STAT_REG(tmp1,TYPE_VEC);
    MEM_STAT_REG(tmp2,TYPE_VEC);
    
    for ( i=0; i<QR->m ; i++ )
    {	/* get i-th column of Q */
	/* set up tmp1 as i-th basis vector */
	for ( j=0; j<QR->m ; j++ )
	    tmp1->ve[j] = 0.0;
	tmp1->ve[i] = 1.0;
	
	/* apply H/h transforms in reverse order */
	for ( j=limit-1; j>=0; j-- )
	{
	    get_col(QR,j,tmp2);
	    r_ii = fabs(tmp2->ve[j]);
	    tmp2->ve[j] = diag->ve[j];
	    tmp_val = (r_ii*fabs(diag->ve[j]));
	    beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	    /* hhtrvec(tmp2,beta->ve[j],j,tmp1,tmp1); */
	    hhtrvec(tmp2,beta,j,tmp1,tmp1);
	}
	
	/* insert into Q */
	set_col(Qout,i,tmp1);
    }

#ifdef	THREADSAFE
    V_FREE(tmp1);	V_FREE(tmp2);
#endif

    return (Qout);
}

/* makeR -- constructs upper triangular matrix from QR (compact form)
   -- may be in-situ (all it does is zero the lower 1/2) */
#ifndef ANSI_C
MAT	*makeR(QR,Rout)
MAT	*QR,*Rout;
#else
MAT	*makeR(const MAT *QR, MAT *Rout)
#endif
{
    unsigned int	i,j;
    
    if ( QR==MNULL )
	error(E_NULL,"makeR");
    Rout = m_copy(QR,Rout);
    
    for ( i=1; i<QR->m; i++ )
	for ( j=0; j<QR->n && j<i; j++ )
	    Rout->me[i][j] = 0.0;
    
    return (Rout);
}

/* QRsolve -- solves the system Q.R.x=b where Q & R are stored in compact form
   -- returns x, which is created if necessary */
#ifndef ANSI_C
VEC	*QRsolve(QR,diag,b,x)
MAT	*QR;
VEC	*diag /* , *beta */ , *b, *x;
#else
VEC	*QRsolve(const MAT *QR, const VEC *diag, const VEC *b, VEC *x)
#endif
{
    int	limit;
    STATIC	VEC	*tmp = VNULL;
    
    if ( ! QR || ! diag || ! b )
	error(E_NULL,"QRsolve");
    limit = min(QR->m,QR->n);
    if ( diag->dim < limit || b->dim != QR->m )
	error(E_SIZES,"QRsolve");
    tmp = v_resize(tmp,limit);
    MEM_STAT_REG(tmp,TYPE_VEC);

    x = v_resize(x,QR->n);
    _Qsolve(QR,diag,b,x,tmp);
    x = Usolve(QR,x,x,0.0);
    v_resize(x,QR->n);

#ifdef	THREADSAFE
    V_FREE(tmp);
#endif

    return x;
}

/* QRCPsolve -- solves A.x = b where A is factored by QRCPfactor()
   -- assumes that A is in the compact factored form */
#ifndef ANSI_C
VEC	*QRCPsolve(QR,diag,pivot,b,x)
MAT	*QR;
VEC	*diag;
PERM	*pivot;
VEC	*b, *x;
#else
VEC	*QRCPsolve(const MAT *QR, const VEC *diag, PERM *pivot,
		   const VEC *b, VEC *x)
#endif
{
    STATIC	VEC	*tmp=VNULL;
    
    if ( ! QR || ! diag || ! pivot || ! b )
	error(E_NULL,"QRCPsolve");
    if ( (QR->m > diag->dim &&QR->n > diag->dim) || QR->n != pivot->size )
	error(E_SIZES,"QRCPsolve");
    
    tmp = QRsolve(QR,diag,b,tmp);
    MEM_STAT_REG(tmp,TYPE_VEC);
    x = pxinv_vec(pivot,tmp,x);

#ifdef	THREADSAFE
    V_FREE(tmp);
#endif

    return x;
}

/* Umlt -- compute out = upper_triang(U).x
	-- may be in situ */
#ifndef ANSI_C
static	VEC	*Umlt(U,x,out)
MAT	*U;
VEC	*x, *out;
#else
static	VEC	*Umlt(const MAT *U, const VEC *x, VEC *out)
#endif
{
    int		i, limit;

    if ( U == MNULL || x == VNULL )
	error(E_NULL,"Umlt");
    limit = min(U->m,U->n);
    if ( limit != x->dim )
	error(E_SIZES,"Umlt");
    if ( out == VNULL || out->dim < limit )
	out = v_resize(out,limit);

    for ( i = 0; i < limit; i++ )
	out->ve[i] = __ip__(&(x->ve[i]),&(U->me[i][i]),limit - i);
    return out;
}

/* UTmlt -- returns out = upper_triang(U)^T.x */
#ifndef ANSI_C
static	VEC	*UTmlt(U,x,out)
MAT	*U;
VEC	*x, *out;
#else
static	VEC	*UTmlt(const MAT *U, const VEC *x, VEC *out)
#endif
{
    Real	sum;
    int		i, j, limit;

    if ( U == MNULL || x == VNULL )
	error(E_NULL,"UTmlt");
    limit = min(U->m,U->n);
    if ( out == VNULL || out->dim < limit )
	out = v_resize(out,limit);

    for ( i = limit-1; i >= 0; i-- )
    {
	sum = 0.0;
	for ( j = 0; j <= i; j++ )
	    sum += U->me[j][i]*x->ve[j];
	out->ve[i] = sum;
    }
    return out;
}

/* QRTsolve -- solve A^T.sc = c where the QR factors of A are stored in
	compact form
	-- returns sc
	-- original due to Mike Osborne modified Wed 09th Dec 1992 */
#ifndef ANSI_C
VEC *QRTsolve(A,diag,c,sc)
MAT *A;
VEC *diag, *c, *sc;
#else
VEC *QRTsolve(const MAT *A, const VEC *diag, const VEC *c, VEC *sc)
#endif
{
    int		i, j, k, n, p;
    Real	beta, r_ii, s, tmp_val;

    if ( ! A || ! diag || ! c )
	error(E_NULL,"QRTsolve");
    if ( diag->dim < min(A->m,A->n) )
	error(E_SIZES,"QRTsolve");
    sc = v_resize(sc,A->m);
    n = sc->dim;
    p = c->dim;
    if ( n == p )
	k = p-2;
    else
	k = p-1;
    v_zero(sc);
    sc->ve[0] = c->ve[0]/A->me[0][0];
    if ( n ==  1)
	return sc;
    if ( p > 1)
    {
	for ( i = 1; i < p; i++ )
	{
	    s = 0.0;
	    for ( j = 0; j < i; j++ )
		s += A->me[j][i]*sc->ve[j];
	    if ( A->me[i][i] == 0.0 )
		error(E_SING,"QRTsolve");
	    sc->ve[i]=(c->ve[i]-s)/A->me[i][i];
	}
    }
    for (i = k; i >= 0; i--)
    {
	s = diag->ve[i]*sc->ve[i];
	for ( j = i+1; j < n; j++ )
	    s += A->me[j][i]*sc->ve[j];
	r_ii = fabs(A->me[i][i]);
	tmp_val = (r_ii*fabs(diag->ve[i]));
	beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	tmp_val = beta*s;
	sc->ve[i] -= tmp_val*diag->ve[i];
	for ( j = i+1; j < n; j++ )
	    sc->ve[j] -= tmp_val*A->me[j][i];
    }

    return sc;
}

/* QRcondest -- returns an estimate of the 2-norm condition number of the
		matrix factorised by QRfactor() or QRCPfactor()
	-- note that as Q does not affect the 2-norm condition number,
		it is not necessary to pass the diag, beta (or pivot) vectors
	-- generates a lower bound on the true condition number
	-- if the matrix is exactly singular, HUGE_VAL is returned
	-- note that QRcondest() is likely to be more reliable for
		matrices factored using QRCPfactor() */
#ifndef ANSI_C
double	QRcondest(QR)
MAT	*QR;
#else
double	QRcondest(const MAT *QR)
#endif
{
    STATIC	VEC	*y=VNULL;
    Real	norm1, norm2, sum, tmp1, tmp2;
    int		i, j, limit;

    if ( QR == MNULL )
	error(E_NULL,"QRcondest");

    limit = min(QR->m,QR->n);
    for ( i = 0; i < limit; i++ )
	if ( QR->me[i][i] == 0.0 )
	    return HUGE_VAL;

    y = v_resize(y,limit);
    MEM_STAT_REG(y,TYPE_VEC);
    /* use the trick for getting a unit vector y with ||R.y||_inf small
       from the LU condition estimator */
    for ( i = 0; i < limit; i++ )
    {
	sum = 0.0;
	for ( j = 0; j < i; j++ )
	    sum -= QR->me[j][i]*y->ve[j];
	sum -= (sum < 0.0) ? 1.0 : -1.0;
	y->ve[i] = sum / QR->me[i][i];
    }
    UTmlt(QR,y,y);

    /* now apply inverse power method to R^T.R */
    for ( i = 0; i < 3; i++ )
    {
	tmp1 = v_norm2(y);
	sv_mlt(1/tmp1,y,y);
	UTsolve(QR,y,y,0.0);
	tmp2 = v_norm2(y);
	sv_mlt(1/v_norm2(y),y,y);
	Usolve(QR,y,y,0.0);
    }
    /* now compute approximation for ||R^{-1}||_2 */
    norm1 = sqrt(tmp1)*sqrt(tmp2);

    /* now use complementary approach to compute approximation to ||R||_2 */
    for ( i = limit-1; i >= 0; i-- )
    {
	sum = 0.0;
	for ( j = i+1; j < limit; j++ )
	    sum += QR->me[i][j]*y->ve[j];
	y->ve[i] = (sum >= 0.0) ? 1.0 : -1.0;
	y->ve[i] = (QR->me[i][i] >= 0.0) ? y->ve[i] : - y->ve[i];
    }

    /* now apply power method to R^T.R */
    for ( i = 0; i < 3; i++ )
    {
	tmp1 = v_norm2(y);
	sv_mlt(1/tmp1,y,y);
	Umlt(QR,y,y);
	tmp2 = v_norm2(y);
	sv_mlt(1/tmp2,y,y);
	UTmlt(QR,y,y);
    }
    norm2 = sqrt(tmp1)*sqrt(tmp2);

    /* printf("QRcondest: norm1 = %g, norm2 = %g\n",norm1,norm2); */

#ifdef THREADSAFE
    V_FREE(y);
#endif

    return norm1*norm2;
}
