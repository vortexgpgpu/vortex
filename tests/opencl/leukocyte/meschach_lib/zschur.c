
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
	File containing routines for computing the Schur decomposition
	of a complex non-symmetric matrix
	See also: hessen.c
	Complex version
*/


#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"

static char rcsid[] = "$Id: zschur.c,v 1.4 1995/04/07 16:28:58 des Exp $";

#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
#define	b2s(t_or_f)	((t_or_f) ? "TRUE" : "FALSE")


/* zschur -- computes the Schur decomposition of the matrix A in situ
	-- optionally, gives Q matrix such that Q^*.A.Q is upper triangular
	-- returns upper triangular Schur matrix */
ZMAT	*zschur(A,Q)
ZMAT	*A, *Q;
{
    int		i, j, iter, k, k_min, k_max, k_tmp, n, split;
    Real	c;
    complex	det, discrim, lambda, lambda0, lambda1, s, sum, ztmp;
    complex	x, y;	/* for chasing algorithm */
    complex	**A_me;
    STATIC	ZVEC	*diag=ZVNULL;
    
    if ( ! A )
	error(E_NULL,"zschur");
    if ( A->m != A->n || ( Q && Q->m != Q->n ) )
	error(E_SQUARE,"zschur");
    if ( Q != ZMNULL && Q->m != A->m )
	error(E_SIZES,"zschur");
    n = A->n;
    diag = zv_resize(diag,A->n);
    MEM_STAT_REG(diag,TYPE_ZVEC);
    /* compute Hessenberg form */
    zHfactor(A,diag);
    
    /* save Q if necessary, and make A explicitly Hessenberg */
    zHQunpack(A,diag,Q,A);

    k_min = 0;	A_me = A->me;

    while ( k_min < n )
    {
	/* find k_max to suit:
	   submatrix k_min..k_max should be irreducible */
	k_max = n-1;
	for ( k = k_min; k < k_max; k++ )
	    if ( is_zero(A_me[k+1][k]) )
	    {	k_max = k;	break;	}

	if ( k_max <= k_min )
	{
	    k_min = k_max + 1;
	    continue;		/* outer loop */
	}

	/* now have r x r block with r >= 2:
	   apply Francis QR step until block splits */
	split = FALSE;		iter = 0;
	while ( ! split )
	{
	    complex	a00, a01, a10, a11;
	    iter++;
	    
	    /* set up Wilkinson/Francis complex shift */
	    /* use the smallest eigenvalue of the bottom 2 x 2 submatrix */
	    k_tmp = k_max - 1;

	    a00 = A_me[k_tmp][k_tmp];
	    a01 = A_me[k_tmp][k_max];
	    a10 = A_me[k_max][k_tmp];
	    a11 = A_me[k_max][k_max];
	    ztmp.re = 0.5*(a00.re - a11.re);
	    ztmp.im = 0.5*(a00.im - a11.im);
	    discrim = zsqrt(zadd(zmlt(ztmp,ztmp),zmlt(a01,a10)));
	    sum.re  = 0.5*(a00.re + a11.re);
	    sum.im  = 0.5*(a00.im + a11.im);
	    lambda0 = zadd(sum,discrim);
	    lambda1 = zsub(sum,discrim);
	    det = zsub(zmlt(a00,a11),zmlt(a01,a10)); 
	    
	    if ( is_zero(lambda0) && is_zero(lambda1) )
	      {                                                          
		lambda.re = lambda.im = 0.0;
	      } 
	    else if ( zabs(lambda0) > zabs(lambda1) )
		lambda = zdiv(det,lambda0);
	    else
		lambda = zdiv(det,lambda1);

	    /* perturb shift if convergence is slow */
	    if ( (iter % 10) == 0 )
	    {
		lambda.re += iter*0.02;
		lambda.im += iter*0.02;
	    }

	    /* set up Householder transformations */
	    k_tmp = k_min + 1;

	    x = zsub(A->me[k_min][k_min],lambda);
	    y = A->me[k_min+1][k_min];

	    /* use Givens' rotations to "chase" off-Hessenberg entry */
	    for ( k = k_min; k <= k_max-1; k++ )
	    {
		zgivens(x,y,&c,&s);
		zrot_cols(A,k,k+1,c,s,A);
		zrot_rows(A,k,k+1,c,s,A);
		if ( Q != ZMNULL )
		    zrot_cols(Q,k,k+1,c,s,Q);

		/* zero things that should be zero */
		if ( k > k_min )
		    A->me[k+1][k-1].re = A->me[k+1][k-1].im = 0.0;

		/* get next entry to chase along sub-diagonal */
		x = A->me[k+1][k];
		if ( k <= k_max - 2 )
		    y = A->me[k+2][k];
		else
		    y.re = y.im = 0.0;
	    }

	    for ( k = k_min; k <= k_max-2; k++ )
	    {
		/* zero appropriate sub-diagonals */
		A->me[k+2][k].re = A->me[k+2][k].im = 0.0;
	    }

	    /* test to see if matrix should split */
	    for ( k = k_min; k < k_max; k++ )
		if ( zabs(A_me[k+1][k]) < MACHEPS*
		    (zabs(A_me[k][k])+zabs(A_me[k+1][k+1])) )
		{
		    A_me[k+1][k].re = A_me[k+1][k].im = 0.0;
		    split = TRUE;
		}

	}
    }
    
    /* polish up A by zeroing strictly lower triangular elements
       and small sub-diagonal elements */
    for ( i = 0; i < A->m; i++ )
	for ( j = 0; j < i-1; j++ )
	    A_me[i][j].re = A_me[i][j].im = 0.0;
    for ( i = 0; i < A->m - 1; i++ )
	if ( zabs(A_me[i+1][i]) < MACHEPS*
	    (zabs(A_me[i][i])+zabs(A_me[i+1][i+1])) )
	    A_me[i+1][i].re = A_me[i+1][i].im = 0.0;

#ifdef	THREADSAFE
    ZV_FREE(diag);
#endif

    return A;
}


#if 0
/* schur_vecs -- returns eigenvectors computed from the real Schur
		decomposition of a matrix
	-- T is the block upper triangular Schur matrix
	-- Q is the orthognal matrix where A = Q.T.Q^T
	-- if Q is null, the eigenvectors of T are returned
	-- X_re is the real part of the matrix of eigenvectors,
		and X_im is the imaginary part of the matrix.
	-- X_re is returned */
MAT	*schur_vecs(T,Q,X_re,X_im)
MAT	*T, *Q, *X_re, *X_im;
{
	int	i, j, limit;
	Real	t11_re, t11_im, t12, t21, t22_re, t22_im;
	Real	l_re, l_im, det_re, det_im, invdet_re, invdet_im,
		val1_re, val1_im, val2_re, val2_im,
		tmp_val1_re, tmp_val1_im, tmp_val2_re, tmp_val2_im, **T_me;
	Real	sum, diff, discrim, magdet, norm, scale;
	STATIC VEC	*tmp1_re=VNULL, *tmp1_im=VNULL,
			*tmp2_re=VNULL, *tmp2_im=VNULL;

	if ( ! T || ! X_re )
	    error(E_NULL,"schur_vecs");
	if ( T->m != T->n || X_re->m != X_re->n ||
		( Q != MNULL && Q->m != Q->n ) ||
		( X_im != MNULL && X_im->m != X_im->n ) )
	    error(E_SQUARE,"schur_vecs");
	if ( T->m != X_re->m ||
		( Q != MNULL && T->m != Q->m ) ||
		( X_im != MNULL && T->m != X_im->m ) )
	    error(E_SIZES,"schur_vecs");

	tmp1_re = v_resize(tmp1_re,T->m);
	tmp1_im = v_resize(tmp1_im,T->m);
	tmp2_re = v_resize(tmp2_re,T->m);
	tmp2_im = v_resize(tmp2_im,T->m);
	MEM_STAT_REG(tmp1_re,TYPE_VEC);
	MEM_STAT_REG(tmp1_im,TYPE_VEC);
	MEM_STAT_REG(tmp2_re,TYPE_VEC);
	MEM_STAT_REG(tmp2_im,TYPE_VEC);

	T_me = T->me;
	i = 0;
	while ( i < T->m )
	{
	    if ( i+1 < T->m && T->me[i+1][i] != 0.0 )
	    {	/* complex eigenvalue */
		sum  = 0.5*(T_me[i][i]+T_me[i+1][i+1]);
		diff = 0.5*(T_me[i][i]-T_me[i+1][i+1]);
		discrim = diff*diff + T_me[i][i+1]*T_me[i+1][i];
		l_re = l_im = 0.0;
		if ( discrim < 0.0 )
		{	/* yes -- complex e-vals */
		    l_re = sum;
		    l_im = sqrt(-discrim);
		}
		else /* not correct Real Schur form */
		    error(E_RANGE,"schur_vecs");
	    }
	    else
	    {
		l_re = T_me[i][i];
		l_im = 0.0;
	    }

	    v_zero(tmp1_im);
	    v_rand(tmp1_re);
	    sv_mlt(MACHEPS,tmp1_re,tmp1_re);

	    /* solve (T-l.I)x = tmp1 */
	    limit = ( l_im != 0.0 ) ? i+1 : i;
	    /* printf("limit = %d\n",limit); */
	    for ( j = limit+1; j < T->m; j++ )
		tmp1_re->ve[j] = 0.0;
	    j = limit;
	    while ( j >= 0 )
	    {
		if ( j > 0 && T->me[j][j-1] != 0.0 )
		{   /* 2 x 2 diagonal block */
		    /* printf("checkpoint A\n"); */
		    val1_re = tmp1_re->ve[j-1] -
		      __ip__(&(tmp1_re->ve[j+1]),&(T->me[j-1][j+1]),limit-j);
		    /* printf("checkpoint B\n"); */
		    val1_im = tmp1_im->ve[j-1] -
		      __ip__(&(tmp1_im->ve[j+1]),&(T->me[j-1][j+1]),limit-j);
		    /* printf("checkpoint C\n"); */
		    val2_re = tmp1_re->ve[j] -
		      __ip__(&(tmp1_re->ve[j+1]),&(T->me[j][j+1]),limit-j);
		    /* printf("checkpoint D\n"); */
		    val2_im = tmp1_im->ve[j] -
		      __ip__(&(tmp1_im->ve[j+1]),&(T->me[j][j+1]),limit-j);
		    /* printf("checkpoint E\n"); */
		    
		    t11_re = T_me[j-1][j-1] - l_re;
		    t11_im = - l_im;
		    t22_re = T_me[j][j] - l_re;
		    t22_im = - l_im;
		    t12 = T_me[j-1][j];
		    t21 = T_me[j][j-1];

		    scale =  fabs(T_me[j-1][j-1]) + fabs(T_me[j][j]) +
			fabs(t12) + fabs(t21) + fabs(l_re) + fabs(l_im);

		    det_re = t11_re*t22_re - t11_im*t22_im - t12*t21;
		    det_im = t11_re*t22_im + t11_im*t22_re;
		    magdet = det_re*det_re+det_im*det_im;
		    if ( sqrt(magdet) < MACHEPS*scale )
		    {
		        det_re = MACHEPS*scale;
			magdet = det_re*det_re+det_im*det_im;
		    }
		    invdet_re =   det_re/magdet;
		    invdet_im = - det_im/magdet;
		    tmp_val1_re = t22_re*val1_re-t22_im*val1_im-t12*val2_re;
		    tmp_val1_im = t22_im*val1_re+t22_re*val1_im-t12*val2_im;
		    tmp_val2_re = t11_re*val2_re-t11_im*val2_im-t21*val1_re;
		    tmp_val2_im = t11_im*val2_re+t11_re*val2_im-t21*val1_im;
		    tmp1_re->ve[j-1] = invdet_re*tmp_val1_re -
		    		invdet_im*tmp_val1_im;
		    tmp1_im->ve[j-1] = invdet_im*tmp_val1_re +
		    		invdet_re*tmp_val1_im;
		    tmp1_re->ve[j]   = invdet_re*tmp_val2_re -
		    		invdet_im*tmp_val2_im;
		    tmp1_im->ve[j]   = invdet_im*tmp_val2_re +
		    		invdet_re*tmp_val2_im;
		    j -= 2;
	        }
	        else
		{
		    t11_re = T_me[j][j] - l_re;
		    t11_im = - l_im;
		    magdet = t11_re*t11_re + t11_im*t11_im;
		    scale = fabs(T_me[j][j]) + fabs(l_re);
		    if ( sqrt(magdet) < MACHEPS*scale )
		    {
		        t11_re = MACHEPS*scale;
			magdet = t11_re*t11_re + t11_im*t11_im;
		    }
		    invdet_re =   t11_re/magdet;
		    invdet_im = - t11_im/magdet;
		    /* printf("checkpoint F\n"); */
		    val1_re = tmp1_re->ve[j] -
		      __ip__(&(tmp1_re->ve[j+1]),&(T->me[j][j+1]),limit-j);
		    /* printf("checkpoint G\n"); */
		    val1_im = tmp1_im->ve[j] -
		      __ip__(&(tmp1_im->ve[j+1]),&(T->me[j][j+1]),limit-j);
		    /* printf("checkpoint H\n"); */
		    tmp1_re->ve[j] = invdet_re*val1_re - invdet_im*val1_im;
		    tmp1_im->ve[j] = invdet_im*val1_re + invdet_re*val1_im;
		    j -= 1;
		}
	    }

	    norm = v_norm_inf(tmp1_re) + v_norm_inf(tmp1_im);
	    sv_mlt(1/norm,tmp1_re,tmp1_re);
	    if ( l_im != 0.0 )
		sv_mlt(1/norm,tmp1_im,tmp1_im);
	    mv_mlt(Q,tmp1_re,tmp2_re);
	    if ( l_im != 0.0 )
		mv_mlt(Q,tmp1_im,tmp2_im);
	    if ( l_im != 0.0 )
		norm = sqrt(in_prod(tmp2_re,tmp2_re)+in_prod(tmp2_im,tmp2_im));
	    else
		norm = v_norm2(tmp2_re);
	    sv_mlt(1/norm,tmp2_re,tmp2_re);
	    if ( l_im != 0.0 )
		sv_mlt(1/norm,tmp2_im,tmp2_im);

	    if ( l_im != 0.0 )
	    {
		if ( ! X_im )
		error(E_NULL,"schur_vecs");
		set_col(X_re,i,tmp2_re);
		set_col(X_im,i,tmp2_im);
		sv_mlt(-1.0,tmp2_im,tmp2_im);
		set_col(X_re,i+1,tmp2_re);
		set_col(X_im,i+1,tmp2_im);
		i += 2;
	    }
	    else
	    {
		set_col(X_re,i,tmp2_re);
		if ( X_im != MNULL )
		    set_col(X_im,i,tmp1_im);	/* zero vector */
		i += 1;
	    }
	}

	return X_re;
}

#endif

