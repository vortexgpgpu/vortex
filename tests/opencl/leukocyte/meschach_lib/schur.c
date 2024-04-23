
/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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
	of a real non-symmetric matrix
	See also: hessen.c
*/

#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"


static char rcsid[] = "$Id: schur.c,v 1.7 1994/03/17 05:36:53 des Exp $";



#ifndef ANSI_C
static	void	hhldr3(x,y,z,nu1,beta,newval)
double	x, y, z;
Real	*nu1, *beta, *newval;
#else
static	void	hhldr3(double x, double y, double z,
		       Real *nu1, Real *beta, Real *newval)
#endif
{
	Real	alpha;

	if ( x >= 0.0 )
		alpha = sqrt(x*x+y*y+z*z);
	else
		alpha = -sqrt(x*x+y*y+z*z);
	*nu1 = x + alpha;
	*beta = 1.0/(alpha*(*nu1));
	*newval = alpha;
}

#ifndef ANSI_C
static	void	hhldr3cols(A,k,j0,beta,nu1,nu2,nu3)
MAT	*A;
int	k, j0;
double	beta, nu1, nu2, nu3;
#else
static	void	hhldr3cols(MAT *A, int k, int j0, double beta,
			   double nu1, double nu2, double nu3)
#endif
{
	Real	**A_me, ip, prod;
	int	j, n;

	if ( k < 0 || k+3 > A->m || j0 < 0 )
		error(E_BOUNDS,"hhldr3cols");
	A_me = A->me;		n = A->n;

	/* printf("hhldr3cols:(l.%d) j0 = %d, k = %d, A at 0x%lx, m = %d, n = %d\n",
	       __LINE__, j0, k, (long)A, A->m, A->n); */
	/* printf("hhldr3cols: A (dumped) =\n");	m_dump(stdout,A); */

	for ( j = j0; j < n; j++ )
	{
	    /*****	    
	    ip = nu1*A_me[k][j] + nu2*A_me[k+1][j] + nu3*A_me[k+2][j];
	    prod = ip*beta;
	    A_me[k][j]   -= prod*nu1;
	    A_me[k+1][j] -= prod*nu2;
	    A_me[k+2][j] -= prod*nu3;
	    *****/
	    /* printf("hhldr3cols: j = %d\n", j); */

	    ip = nu1*m_entry(A,k,j)+nu2*m_entry(A,k+1,j)+nu3*m_entry(A,k+2,j);
	    prod = ip*beta;
	    /*****
	    m_set_val(A,k  ,j,m_entry(A,k  ,j) - prod*nu1);
	    m_set_val(A,k+1,j,m_entry(A,k+1,j) - prod*nu2);
	    m_set_val(A,k+2,j,m_entry(A,k+2,j) - prod*nu3);
	    *****/
	    m_add_val(A,k  ,j,-prod*nu1);
	    m_add_val(A,k+1,j,-prod*nu2);
	    m_add_val(A,k+2,j,-prod*nu3);

	}
	/* printf("hhldr3cols:(l.%d) j0 = %d, k = %d, m = %d, n = %d\n",
	       __LINE__, j0, k, A->m, A->n); */
	/* putc('\n',stdout); */
}

#ifndef ANSI_C
static	void	hhldr3rows(A,k,i0,beta,nu1,nu2,nu3)
MAT	*A;
int	k, i0;
double	beta, nu1, nu2, nu3;
#else
static	void	hhldr3rows(MAT *A, int k, int i0, double beta,
			   double nu1, double nu2, double nu3)
#endif
{
	Real	**A_me, ip, prod;
	int	i, m;

	/* printf("hhldr3rows:(l.%d) A at 0x%lx\n", __LINE__, (long)A); */
	/* printf("hhldr3rows: k = %d\n", k); */
	if ( k < 0 || k+3 > A->n )
		error(E_BOUNDS,"hhldr3rows");
	A_me = A->me;		m = A->m;
	i0 = min(i0,m-1);

	for ( i = 0; i <= i0; i++ )
	{
	    /****
	    ip = nu1*A_me[i][k] + nu2*A_me[i][k+1] + nu3*A_me[i][k+2];
	    prod = ip*beta;
	    A_me[i][k]   -= prod*nu1;
	    A_me[i][k+1] -= prod*nu2;
	    A_me[i][k+2] -= prod*nu3;
	    ****/

	    ip = nu1*m_entry(A,i,k)+nu2*m_entry(A,i,k+1)+nu3*m_entry(A,i,k+2);
	    prod = ip*beta;
	    m_add_val(A,i,k  , - prod*nu1);
	    m_add_val(A,i,k+1, - prod*nu2);
	    m_add_val(A,i,k+2, - prod*nu3);

	}
}

/* schur -- computes the Schur decomposition of the matrix A in situ
	-- optionally, gives Q matrix such that Q^T.A.Q is upper triangular
	-- returns upper triangular Schur matrix */
#ifndef ANSI_C
MAT	*schur(A,Q)
MAT	*A, *Q;
#else
MAT	*schur(MAT *A, MAT *Q)
#endif
{
    int		i, j, iter, k, k_min, k_max, k_tmp, n, split;
    Real	beta2, c, discrim, dummy, nu1, s, t, tmp, x, y, z;
    Real	**A_me;
    Real	sqrt_macheps;
    STATIC	VEC	*diag=VNULL, *beta=VNULL;
    
    if ( ! A )
	error(E_NULL,"schur");
    if ( A->m != A->n || ( Q && Q->m != Q->n ) )
	error(E_SQUARE,"schur");
    if ( Q != MNULL && Q->m != A->m )
	error(E_SIZES,"schur");
    n = A->n;
    diag = v_resize(diag,A->n);
    beta = v_resize(beta,A->n);
    MEM_STAT_REG(diag,TYPE_VEC);
    MEM_STAT_REG(beta,TYPE_VEC);
    /* compute Hessenberg form */
    Hfactor(A,diag,beta);
    
    /* save Q if necessary */
    if ( Q )
	Q = makeHQ(A,diag,beta,Q);
    makeH(A,A);

    sqrt_macheps = sqrt(MACHEPS);

    k_min = 0;	A_me = A->me;

    while ( k_min < n )
    {
	Real	a00, a01, a10, a11;
	double	scale, t, numer, denom;

	/* find k_max to suit:
	   submatrix k_min..k_max should be irreducible */
	k_max = n-1;
	for ( k = k_min; k < k_max; k++ )
	    /* if ( A_me[k+1][k] == 0.0 ) */
	    if ( m_entry(A,k+1,k) == 0.0 )
	    {	k_max = k;	break;	}

	if ( k_max <= k_min )
	{
	    k_min = k_max + 1;
	    continue;		/* outer loop */
	}

	/* check to see if we have a 2 x 2 block
	   with complex eigenvalues */
	if ( k_max == k_min + 1 )
	{
	    /* tmp = A_me[k_min][k_min] - A_me[k_max][k_max]; */
	    a00 = m_entry(A,k_min,k_min);
	    a01 = m_entry(A,k_min,k_max);
	    a10 = m_entry(A,k_max,k_min);
	    a11 = m_entry(A,k_max,k_max);
	    tmp = a00 - a11;
	    /* discrim = tmp*tmp +
		4*A_me[k_min][k_max]*A_me[k_max][k_min]; */
	    discrim = tmp*tmp + 4*a01*a10;
	    if ( discrim < 0.0 )
	    {	/* yes -- e-vals are complex
		   -- put 2 x 2 block in form [a b; c a];
		   then eigenvalues have real part a & imag part sqrt(|bc|) */
		numer = - tmp;
		denom = ( a01+a10 >= 0.0 ) ?
		    (a01+a10) + sqrt((a01+a10)*(a01+a10)+tmp*tmp) :
		    (a01+a10) - sqrt((a01+a10)*(a01+a10)+tmp*tmp);
		if ( denom != 0.0 )
		{   /* t = s/c = numer/denom */
		    t = numer/denom;
		    scale = c = 1.0/sqrt(1+t*t);
		    s = c*t;
		}
		else
		{
		    c = 1.0;
		    s = 0.0;
		}
		rot_cols(A,k_min,k_max,c,s,A);
		rot_rows(A,k_min,k_max,c,s,A);
		if ( Q != MNULL )
		    rot_cols(Q,k_min,k_max,c,s,Q);
		k_min = k_max + 1;
		continue;
	    }
	    else /* discrim >= 0; i.e. block has two real eigenvalues */
	    {	/* no -- e-vals are not complex;
		   split 2 x 2 block and continue */
		/* s/c = numer/denom */
		numer = ( tmp >= 0.0 ) ?
		    - tmp - sqrt(discrim) : - tmp + sqrt(discrim);
		denom = 2*a01;
		if ( fabs(numer) < fabs(denom) )
		{   /* t = s/c = numer/denom */
		    t = numer/denom;
		    scale = c = 1.0/sqrt(1+t*t);
		    s = c*t;
		}
		else if ( numer != 0.0 )
		{   /* t = c/s = denom/numer */
		    t = denom/numer;
		    scale = 1.0/sqrt(1+t*t);
		    c = fabs(t)*scale;
		    s = ( t >= 0.0 ) ? scale : -scale;
		}
		else /* numer == denom == 0 */
		{
		    c = 0.0;
		    s = 1.0;
		}
		rot_cols(A,k_min,k_max,c,s,A);
		rot_rows(A,k_min,k_max,c,s,A);
		/* A->me[k_max][k_min] = 0.0; */
		if ( Q != MNULL )
		    rot_cols(Q,k_min,k_max,c,s,Q);
		k_min = k_max + 1;	/* go to next block */
		continue;
	    }
	}

	/* now have r x r block with r >= 2:
	   apply Francis QR step until block splits */
	split = FALSE;		iter = 0;
	while ( ! split )
	{
	    iter++;
	    
	    /* set up Wilkinson/Francis complex shift */
	    k_tmp = k_max - 1;

	    a00 = m_entry(A,k_tmp,k_tmp);
	    a01 = m_entry(A,k_tmp,k_max);
	    a10 = m_entry(A,k_max,k_tmp);
	    a11 = m_entry(A,k_max,k_max);

	    /* treat degenerate cases differently
	       -- if there are still no splits after five iterations
	          and the bottom 2 x 2 looks degenerate, force it to
		  split */
#ifdef DEBUG
	    printf("# schur: bottom 2 x 2 = [%lg, %lg; %lg, %lg]\n",
		   a00, a01, a10, a11);
#endif
	    if ( iter >= 5 &&
		 fabs(a00-a11) < sqrt_macheps*(fabs(a00)+fabs(a11)) &&
		 (fabs(a01) < sqrt_macheps*(fabs(a00)+fabs(a11)) ||
		  fabs(a10) < sqrt_macheps*(fabs(a00)+fabs(a11))) )
	    {
	      if ( fabs(a01) < sqrt_macheps*(fabs(a00)+fabs(a11)) )
		m_set_val(A,k_tmp,k_max,0.0);
	      if ( fabs(a10) < sqrt_macheps*(fabs(a00)+fabs(a11)) )
		{
		  m_set_val(A,k_max,k_tmp,0.0);
		  split = TRUE;
		  continue;
		}
	    }

	    s = a00 + a11;
	    t = a00*a11 - a01*a10;

	    /* break loop if a 2 x 2 complex block */
	    if ( k_max == k_min + 1 && s*s < 4.0*t )
	    {
		split = TRUE;
		continue;
	    }

	    /* perturb shift if convergence is slow */
	    if ( (iter % 10) == 0 )
	    {	s += iter*0.02;		t += iter*0.02;
	    }

	    /* set up Householder transformations */
	    k_tmp = k_min + 1;
	    /********************
	    x = A_me[k_min][k_min]*A_me[k_min][k_min] +
		A_me[k_min][k_tmp]*A_me[k_tmp][k_min] -
		    s*A_me[k_min][k_min] + t;
	    y = A_me[k_tmp][k_min]*
		(A_me[k_min][k_min]+A_me[k_tmp][k_tmp]-s);
	    if ( k_min + 2 <= k_max )
		z = A_me[k_tmp][k_min]*A_me[k_min+2][k_tmp];
	    else
		z = 0.0;
	    ********************/

	    a00 = m_entry(A,k_min,k_min);
	    a01 = m_entry(A,k_min,k_tmp);
	    a10 = m_entry(A,k_tmp,k_min);
	    a11 = m_entry(A,k_tmp,k_tmp);

	    /********************
	    a00 = A->me[k_min][k_min];
	    a01 = A->me[k_min][k_tmp];
	    a10 = A->me[k_tmp][k_min];
	    a11 = A->me[k_tmp][k_tmp];
	    ********************/
	    x = a00*a00 + a01*a10 - s*a00 + t;
	    y = a10*(a00+a11-s);
	    if ( k_min + 2 <= k_max )
		z = a10* /* m_entry(A,k_min+2,k_tmp) */ A->me[k_min+2][k_tmp];
	    else
		z = 0.0;

	    for ( k = k_min; k <= k_max-1; k++ )
	    {
		if ( k < k_max - 1 )
		{
		    hhldr3(x,y,z,&nu1,&beta2,&dummy);
		    tracecatch(hhldr3cols(A,k,max(k-1,0),  beta2,nu1,y,z),"schur");
		    tracecatch(hhldr3rows(A,k,min(n-1,k+3),beta2,nu1,y,z),"schur");
		    if ( Q != MNULL )
			hhldr3rows(Q,k,n-1,beta2,nu1,y,z);
		}
		else
		{
		    givens(x,y,&c,&s);
		    rot_cols(A,k,k+1,c,s,A);
		    rot_rows(A,k,k+1,c,s,A);
		    if ( Q )
			rot_cols(Q,k,k+1,c,s,Q);
		}
		/* if ( k >= 2 )
		    m_set_val(A,k,k-2,0.0); */
		/* x = A_me[k+1][k]; */
		x = m_entry(A,k+1,k);
		if ( k <= k_max - 2 )
		    /* y = A_me[k+2][k];*/
		    y = m_entry(A,k+2,k);
		else
		    y = 0.0;
		if ( k <= k_max - 3 )
		    /* z = A_me[k+3][k]; */
		    z = m_entry(A,k+3,k);
		else
		    z = 0.0;
	    }
	    /* if ( k_min > 0 )
		m_set_val(A,k_min,k_min-1,0.0);
	    if ( k_max < n - 1 )
		m_set_val(A,k_max+1,k_max,0.0); */
	    for ( k = k_min; k <= k_max-2; k++ )
	    {
		/* zero appropriate sub-diagonals */
		m_set_val(A,k+2,k,0.0);
		if ( k < k_max-2 )
		    m_set_val(A,k+3,k,0.0);
	    }

	    /* test to see if matrix should split */
	    for ( k = k_min; k < k_max; k++ )
		if ( fabs(A_me[k+1][k]) < MACHEPS*
		    (fabs(A_me[k][k])+fabs(A_me[k+1][k+1])) )
		{	A_me[k+1][k] = 0.0;	split = TRUE;	}
	}
    }
    
    /* polish up A by zeroing strictly lower triangular elements
       and small sub-diagonal elements */
    for ( i = 0; i < A->m; i++ )
	for ( j = 0; j < i-1; j++ )
	    A_me[i][j] = 0.0;
    for ( i = 0; i < A->m - 1; i++ )
	if ( fabs(A_me[i+1][i]) < MACHEPS*
	    (fabs(A_me[i][i])+fabs(A_me[i+1][i+1])) )
	    A_me[i+1][i] = 0.0;

#ifdef	THREADSAFE
    V_FREE(diag);	V_FREE(beta);
#endif

    return A;
}

/* schur_vals -- compute real & imaginary parts of eigenvalues
	-- assumes T contains a block upper triangular matrix
		as produced by schur()
	-- real parts stored in real_pt, imaginary parts in imag_pt */
#ifndef ANSI_C
void	schur_evals(T,real_pt,imag_pt)
MAT	*T;
VEC	*real_pt, *imag_pt;
#else
void	schur_evals(MAT *T, VEC *real_pt, VEC *imag_pt)
#endif
{
	int	i, n;
	Real	discrim, **T_me;
	Real	diff, sum, tmp;

	if ( ! T || ! real_pt || ! imag_pt )
		error(E_NULL,"schur_evals");
	if ( T->m != T->n )
		error(E_SQUARE,"schur_evals");
	n = T->n;	T_me = T->me;
	real_pt = v_resize(real_pt,(unsigned int)n);
	imag_pt = v_resize(imag_pt,(unsigned int)n);

	i = 0;
	while ( i < n )
	{
		if ( i < n-1 && T_me[i+1][i] != 0.0 )
		{   /* should be a complex eigenvalue */
		    sum  = 0.5*(T_me[i][i]+T_me[i+1][i+1]);
		    diff = 0.5*(T_me[i][i]-T_me[i+1][i+1]);
		    discrim = diff*diff + T_me[i][i+1]*T_me[i+1][i];
		    if ( discrim < 0.0 )
		    {	/* yes -- complex e-vals */
			real_pt->ve[i] = real_pt->ve[i+1] = sum;
			imag_pt->ve[i] = sqrt(-discrim);
			imag_pt->ve[i+1] = - imag_pt->ve[i];
		    }
		    else
		    {	/* no -- actually both real */
			tmp = sqrt(discrim);
			real_pt->ve[i]   = sum + tmp;
			real_pt->ve[i+1] = sum - tmp;
			imag_pt->ve[i]   = imag_pt->ve[i+1] = 0.0;
		    }
		    i += 2;
		}
		else
		{   /* real eigenvalue */
		    real_pt->ve[i] = T_me[i][i];
		    imag_pt->ve[i] = 0.0;
		    i++;
		}
	}
}

/* schur_vecs -- returns eigenvectors computed from the real Schur
		decomposition of a matrix
	-- T is the block upper triangular Schur matrix
	-- Q is the orthognal matrix where A = Q.T.Q^T
	-- if Q is null, the eigenvectors of T are returned
	-- X_re is the real part of the matrix of eigenvectors,
		and X_im is the imaginary part of the matrix.
	-- X_re is returned */
#ifndef ANSI_C
MAT	*schur_vecs(T,Q,X_re,X_im)
MAT	*T, *Q, *X_re, *X_im;
#else
MAT	*schur_vecs(MAT *T, MAT *Q, MAT *X_re, MAT *X_im)
#endif
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

#ifdef	THREADSAFE
	V_FREE(tmp1_re);	V_FREE(tmp1_im);
	V_FREE(tmp2_re);	V_FREE(tmp2_im);
#endif

	return X_re;
}

