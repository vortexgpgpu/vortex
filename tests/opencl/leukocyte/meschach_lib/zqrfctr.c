
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

  Complex version
  
*/

static	char	rcsid[] = "$Id: zqrfctr.c,v 1.1 1994/01/13 04:21:22 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include	"zmatrix2.h" 


#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)


#define		sign(x)	((x) > 0.0 ? 1 : ((x) < 0.0 ? -1 : 0 ))

/* Note: The usual representation of a Householder transformation is taken
   to be:
   P = I - beta.u.u*
   where beta = 2/(u*.u) and u is called the Householder vector
   (u* is the conjugate transposed vector of u
*/

/* zQRfactor -- forms the QR factorisation of A
	-- factorisation stored in compact form as described above
	(not quite standard format) */
ZMAT	*zQRfactor(A,diag)
ZMAT	*A;
ZVEC	*diag;
{
    unsigned int	k,limit;
    Real	beta;
    STATIC	ZVEC	*tmp1=ZVNULL, *w=ZVNULL;
    
    if ( ! A || ! diag )
	error(E_NULL,"zQRfactor");
    limit = min(A->m,A->n);
    if ( diag->dim < limit )
	error(E_SIZES,"zQRfactor");
    
    tmp1 = zv_resize(tmp1,A->m);
    w    = zv_resize(w,   A->n);
    MEM_STAT_REG(tmp1,TYPE_ZVEC);
    MEM_STAT_REG(w,   TYPE_ZVEC);
    
    for ( k=0; k<limit; k++ )
    {
	/* get H/holder vector for the k-th column */
	zget_col(A,k,tmp1);
	zhhvec(tmp1,k,&beta,tmp1,&A->me[k][k]);
	diag->ve[k] = tmp1->ve[k];
	
	/* apply H/holder vector to remaining columns */
	tracecatch(_zhhtrcols(A,k,k+1,tmp1,beta,w),"zQRfactor");
    }

#ifdef	THREADSAFE
    ZV_FREE(tmp1);	ZV_FREE(w);
#endif

    return (A);
}

/* zQRCPfactor -- forms the QR factorisation of A with column pivoting
   -- factorisation stored in compact form as described above
   ( not quite standard format )				*/
ZMAT	*zQRCPfactor(A,diag,px)
ZMAT	*A;
ZVEC	*diag;
PERM	*px;
{
    unsigned int	i, i_max, j, k, limit;
    STATIC	ZVEC	*tmp1=ZVNULL, *tmp2=ZVNULL, *w=ZVNULL;
    STATIC	VEC	*gamma=VNULL;
    Real 	beta;
    Real	maxgamma, sum, tmp;
    complex	ztmp;
    
    if ( ! A || ! diag || ! px )
	error(E_NULL,"QRCPfactor");
    limit = min(A->m,A->n);
    if ( diag->dim < limit || px->size != A->n )
	error(E_SIZES,"QRCPfactor");
    
    tmp1 = zv_resize(tmp1,A->m);
    tmp2 = zv_resize(tmp2,A->m);
    gamma = v_resize(gamma,A->n);
    w    = zv_resize(w,A->n);
    MEM_STAT_REG(tmp1,TYPE_ZVEC);
    MEM_STAT_REG(tmp2,TYPE_ZVEC);
    MEM_STAT_REG(gamma,TYPE_VEC);
    MEM_STAT_REG(w,   TYPE_ZVEC);
    
    /* initialise gamma and px */
    for ( j=0; j<A->n; j++ )
    {
	px->pe[j] = j;
	sum = 0.0;
	for ( i=0; i<A->m; i++ )
	    sum += square(A->me[i][j].re) + square(A->me[i][j].im);
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
		ztmp = A->me[i][k];
		A->me[i][k] = A->me[i][i_max];
		A->me[i][i_max] = ztmp;
	    }
	}
	
	/* get H/holder vector for the k-th column */
	zget_col(A,k,tmp1);
	/* hhvec(tmp1,k,&beta->ve[k],tmp1,&A->me[k][k]); */
	zhhvec(tmp1,k,&beta,tmp1,&A->me[k][k]);
	diag->ve[k] = tmp1->ve[k];
	
	/* apply H/holder vector to remaining columns */
	_zhhtrcols(A,k,k+1,tmp1,beta,w);
	
	/* update gamma values */
	for ( j=k+1; j<A->n; j++ )
	    gamma->ve[j] -= square(A->me[k][j].re)+square(A->me[k][j].im);
    }

#ifdef	THREADSAFE
    ZV_FREE(tmp1);	ZV_FREE(tmp2);	V_FREE(gamma);	ZV_FREE(w);
#endif
    return (A);
}

/* zQsolve -- solves Qx = b, Q is an orthogonal matrix stored in compact
	form a la QRfactor()
	-- may be in-situ */
ZVEC	*_zQsolve(QR,diag,b,x,tmp)
ZMAT	*QR;
ZVEC	*diag, *b, *x, *tmp;
{
    unsigned int	dynamic;
    int		k, limit;
    Real	beta, r_ii, tmp_val;
    
    limit = min(QR->m,QR->n);
    dynamic = FALSE;
    if ( ! QR || ! diag || ! b )
	error(E_NULL,"_zQsolve");
    if ( diag->dim < limit || b->dim != QR->m )
	error(E_SIZES,"_zQsolve");
    x = zv_resize(x,QR->m);
    if ( tmp == ZVNULL )
	dynamic = TRUE;
    tmp = zv_resize(tmp,QR->m);
    
    /* apply H/holder transforms in normal order */
    x = zv_copy(b,x);
    for ( k = 0 ; k < limit ; k++ )
    {
	zget_col(QR,k,tmp);
	r_ii = zabs(tmp->ve[k]);
	tmp->ve[k] = diag->ve[k];
	tmp_val = (r_ii*zabs(diag->ve[k]));
	beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	/* hhtrvec(tmp,beta->ve[k],k,x,x); */
	zhhtrvec(tmp,beta,k,x,x);
    }
    
    if ( dynamic )
	ZV_FREE(tmp);
    
    return (x);
}

/* zmakeQ -- constructs orthogonal matrix from Householder vectors stored in
   compact QR form */
ZMAT	*zmakeQ(QR,diag,Qout)
ZMAT	*QR,*Qout;
ZVEC	*diag;
{
    STATIC	ZVEC	*tmp1=ZVNULL,*tmp2=ZVNULL;
    unsigned int	i, limit;
    Real	beta, r_ii, tmp_val;
    int	j;

    limit = min(QR->m,QR->n);
    if ( ! QR || ! diag )
	error(E_NULL,"zmakeQ");
    if ( diag->dim < limit )
	error(E_SIZES,"zmakeQ");
    Qout = zm_resize(Qout,QR->m,QR->m);

    tmp1 = zv_resize(tmp1,QR->m);	/* contains basis vec & columns of Q */
    tmp2 = zv_resize(tmp2,QR->m);	/* contains H/holder vectors */
    MEM_STAT_REG(tmp1,TYPE_ZVEC);
    MEM_STAT_REG(tmp2,TYPE_ZVEC);

    for ( i=0; i<QR->m ; i++ )
    {	/* get i-th column of Q */
	/* set up tmp1 as i-th basis vector */
	for ( j=0; j<QR->m ; j++ )
	    tmp1->ve[j].re = tmp1->ve[j].im = 0.0;
	tmp1->ve[i].re = 1.0;
	
	/* apply H/h transforms in reverse order */
	for ( j=limit-1; j>=0; j-- )
	{
	    zget_col(QR,j,tmp2);
	    r_ii = zabs(tmp2->ve[j]);
	    tmp2->ve[j] = diag->ve[j];
	    tmp_val = (r_ii*zabs(diag->ve[j]));
	    beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	    /* hhtrvec(tmp2,beta->ve[j],j,tmp1,tmp1); */
	    zhhtrvec(tmp2,beta,j,tmp1,tmp1);
	}
	
	/* insert into Q */
	zset_col(Qout,i,tmp1);
    }

#ifdef	THREADSAFE
    ZV_FREE(tmp1);	ZV_FREE(tmp2);
#endif

    return (Qout);
}

/* zmakeR -- constructs upper triangular matrix from QR (compact form)
	-- may be in-situ (all it does is zero the lower 1/2) */
ZMAT	*zmakeR(QR,Rout)
ZMAT	*QR,*Rout;
{
    unsigned int	i,j;
    
    if ( QR==ZMNULL )
	error(E_NULL,"zmakeR");
    Rout = zm_copy(QR,Rout);
    
    for ( i=1; i<QR->m; i++ )
	for ( j=0; j<QR->n && j<i; j++ )
	    Rout->me[i][j].re = Rout->me[i][j].im = 0.0;
    
    return (Rout);
}

/* zQRsolve -- solves the system Q.R.x=b where Q & R are stored in compact form
   -- returns x, which is created if necessary */
ZVEC	*zQRsolve(QR,diag,b,x)
ZMAT	*QR;
ZVEC	*diag, *b, *x;
{
    int	limit;
    STATIC	ZVEC	*tmp = ZVNULL;
    
    if ( ! QR || ! diag || ! b )
	error(E_NULL,"zQRsolve");
    limit = min(QR->m,QR->n);
    if ( diag->dim < limit || b->dim != QR->m )
	error(E_SIZES,"zQRsolve");
    tmp = zv_resize(tmp,limit);
    MEM_STAT_REG(tmp,TYPE_ZVEC);

    x = zv_resize(x,QR->n);
    _zQsolve(QR,diag,b,x,tmp);
    x = zUsolve(QR,x,x,0.0);
    x = zv_resize(x,QR->n);

#ifdef	THREADSAFE
    ZV_FREE(tmp);
#endif

    return x;
}

/* zQRAsolve -- solves the system (Q.R)*.x = b
	-- Q & R are stored in compact form
	-- returns x, which is created if necessary */
ZVEC	*zQRAsolve(QR,diag,b,x)
ZMAT	*QR;
ZVEC	*diag, *b, *x;
{
    int		j, limit;
    Real	beta, r_ii, tmp_val;
    STATIC	ZVEC	*tmp = ZVNULL;
    
    if ( ! QR || ! diag || ! b )
	error(E_NULL,"zQRAsolve");
    limit = min(QR->m,QR->n);
    if ( diag->dim < limit || b->dim != QR->n )
	error(E_SIZES,"zQRAsolve");

    x = zv_resize(x,QR->m);
    x = zUAsolve(QR,b,x,0.0);
    x = zv_resize(x,QR->m);

    tmp = zv_resize(tmp,x->dim);
    MEM_STAT_REG(tmp,TYPE_ZVEC);
    /*  printf("zQRAsolve: tmp->dim = %d, x->dim = %d\n", tmp->dim, x->dim); */
    
    /* apply H/h transforms in reverse order */
    for ( j=limit-1; j>=0; j-- )
    {
	zget_col(QR,j,tmp);
	tmp = zv_resize(tmp,QR->m);
	r_ii = zabs(tmp->ve[j]);
	tmp->ve[j] = diag->ve[j];
	tmp_val = (r_ii*zabs(diag->ve[j]));
	beta = ( tmp_val == 0.0 ) ? 0.0 : 1.0/tmp_val;
	zhhtrvec(tmp,beta,j,x,x);
    }

#ifdef	THREADSAFE
    ZV_FREE(tmp);
#endif

    return x;
}

/* zQRCPsolve -- solves A.x = b where A is factored by QRCPfactor()
   -- assumes that A is in the compact factored form */
ZVEC	*zQRCPsolve(QR,diag,pivot,b,x)
ZMAT	*QR;
ZVEC	*diag;
PERM	*pivot;
ZVEC	*b, *x;
{
    if ( ! QR || ! diag || ! pivot || ! b )
	error(E_NULL,"zQRCPsolve");
    if ( (QR->m > diag->dim && QR->n > diag->dim) || QR->n != pivot->size )
	error(E_SIZES,"zQRCPsolve");
    
    x = zQRsolve(QR,diag,b,x);
    x = pxinv_zvec(pivot,x,x);

    return x;
}

/* zUmlt -- compute out = upper_triang(U).x
	-- may be in situ */
ZVEC	*zUmlt(U,x,out)
ZMAT	*U;
ZVEC	*x, *out;
{
    int		i, limit;

    if ( U == ZMNULL || x == ZVNULL )
	error(E_NULL,"zUmlt");
    limit = min(U->m,U->n);
    if ( limit != x->dim )
	error(E_SIZES,"zUmlt");
    if ( out == ZVNULL || out->dim < limit )
	out = zv_resize(out,limit);

    for ( i = 0; i < limit; i++ )
	out->ve[i] = __zip__(&(x->ve[i]),&(U->me[i][i]),limit - i,Z_NOCONJ);
    return out;
}

/* zUAmlt -- returns out = upper_triang(U)^T.x */
ZVEC	*zUAmlt(U,x,out)
ZMAT	*U;
ZVEC	*x, *out;
{
    /* complex	sum; */
    complex	tmp;
    int		i, limit;

    if ( U == ZMNULL || x == ZVNULL )
	error(E_NULL,"zUAmlt");
    limit = min(U->m,U->n);
    if ( out == ZVNULL || out->dim < limit )
	out = zv_resize(out,limit);

    for ( i = limit-1; i >= 0; i-- )
    {
	tmp = x->ve[i];
	out->ve[i].re = out->ve[i].im = 0.0;
	__zmltadd__(&(out->ve[i]),&(U->me[i][i]),tmp,limit-i-1,Z_CONJ);
    }

    return out;
}


/* zQRcondest -- returns an estimate of the 2-norm condition number of the
		matrix factorised by QRfactor() or QRCPfactor()
	-- note that as Q does not affect the 2-norm condition number,
		it is not necessary to pass the diag, beta (or pivot) vectors
	-- generates a lower bound on the true condition number
	-- if the matrix is exactly singular, HUGE_VAL is returned
	-- note that QRcondest() is likely to be more reliable for
		matrices factored using QRCPfactor() */
double	zQRcondest(QR)
ZMAT	*QR;
{
    STATIC	ZVEC	*y=ZVNULL;
    Real	norm, norm1, norm2, tmp1, tmp2;
    complex	sum, tmp;
    int		i, j, limit;

    if ( QR == ZMNULL )
	error(E_NULL,"zQRcondest");

    limit = min(QR->m,QR->n);
    for ( i = 0; i < limit; i++ )
	/* if ( QR->me[i][i] == 0.0 ) */
	if ( is_zero(QR->me[i][i]) )
	    return HUGE_VAL;

    y = zv_resize(y,limit);
    MEM_STAT_REG(y,TYPE_ZVEC);
    /* use the trick for getting a unit vector y with ||R.y||_inf small
       from the LU condition estimator */
    for ( i = 0; i < limit; i++ )
    {
	sum.re = sum.im = 0.0;
	for ( j = 0; j < i; j++ )
	    /* sum -= QR->me[j][i]*y->ve[j]; */
	    sum = zsub(sum,zmlt(QR->me[j][i],y->ve[j]));
	/* sum -= (sum < 0.0) ? 1.0 : -1.0; */
	norm1 = zabs(sum);
	if ( norm1 == 0.0 )
	    sum.re = 1.0;
	else
	{
	    sum.re += sum.re / norm1;
	    sum.im += sum.im / norm1;
	}
	/* y->ve[i] = sum / QR->me[i][i]; */
	y->ve[i] = zdiv(sum,QR->me[i][i]);
    }
    zUAmlt(QR,y,y);

    /* now apply inverse power method to R*.R */
    for ( i = 0; i < 3; i++ )
    {
	tmp1 = zv_norm2(y);
	zv_mlt(zmake(1.0/tmp1,0.0),y,y);
	zUAsolve(QR,y,y,0.0);
	tmp2 = zv_norm2(y);
	zv_mlt(zmake(1.0/tmp2,0.0),y,y);
	zUsolve(QR,y,y,0.0);
    }
    /* now compute approximation for ||R^{-1}||_2 */
    norm1 = sqrt(tmp1)*sqrt(tmp2);

    /* now use complementary approach to compute approximation to ||R||_2 */
    for ( i = limit-1; i >= 0; i-- )
    {
	sum.re = sum.im = 0.0;
	for ( j = i+1; j < limit; j++ )
	    sum = zadd(sum,zmlt(QR->me[i][j],y->ve[j]));
	if ( is_zero(QR->me[i][i]) )
	    return HUGE_VAL;
	tmp = zdiv(sum,QR->me[i][i]);
	if ( is_zero(tmp) )
	{
	    y->ve[i].re = 1.0;
	    y->ve[i].im = 0.0;
	}
	else
	{
	    norm = zabs(tmp);
	    y->ve[i].re = sum.re / norm;
	    y->ve[i].im = sum.im / norm;
	}
	/* y->ve[i] = (sum >= 0.0) ? 1.0 : -1.0; */
	/* y->ve[i] = (QR->me[i][i] >= 0.0) ? y->ve[i] : - y->ve[i]; */
    }

    /* now apply power method to R*.R */
    for ( i = 0; i < 3; i++ )
    {
	tmp1 = zv_norm2(y);
	zv_mlt(zmake(1.0/tmp1,0.0),y,y);
	zUmlt(QR,y,y);
	tmp2 = zv_norm2(y);
	zv_mlt(zmake(1.0/tmp2,0.0),y,y);
	zUAmlt(QR,y,y);
    }
    norm2 = sqrt(tmp1)*sqrt(tmp2);

    /* printf("QRcondest: norm1 = %g, norm2 = %g\n",norm1,norm2); */

#ifdef	THREADSAFE
    ZV_FREE(y);
#endif

    return norm1*norm2;
}

