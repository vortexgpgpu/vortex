
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
	This file contains a series of tests for the Meschach matrix
	library, parts 1 and 2
*/

static char rcsid[] = "$Id: torture.c,v 1.6 1994/08/25 15:22:11 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include	"matrix2.h"
#include        "matlab.h"

#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);

static char *test_err_list[] = {
   "unknown error",			/* 0 */
   "testing error messages",		/* 1 */
   "unexpected end-of-file"		/* 2 */
};


#define MAX_TEST_ERR   (sizeof(test_err_list)/sizeof(char *))

/* extern	int	malloc_chain_check(); */
/* #define MEMCHK() if ( malloc_chain_check(0) ) \
{ printf("Error in malloc chain: \"%s\", line %d\n", \
	 __FILE__, __LINE__); exit(0); } */
#define	MEMCHK() 

/* cmp_perm -- returns 1 if pi1 == pi2, 0 otherwise */
int	cmp_perm(pi1, pi2)
PERM	*pi1, *pi2;
{
    int		i;

    if ( ! pi1 || ! pi2 )
	error(E_NULL,"cmp_perm");
    if ( pi1->size != pi2->size )
	return 0;
    for ( i = 0; i < pi1->size; i++ )
	if ( pi1->pe[i] != pi2->pe[i] )
	    return 0;
    return 1;
}

/* px_rand -- generates sort-of random permutation */
PERM	*px_rand(pi)
PERM	*pi;
{
    int		i, j, k;

    if ( ! pi )
	error(E_NULL,"px_rand");

    for ( i = 0; i < 3*pi->size; i++ )
    {
	j = (rand() >> 8) % pi->size;
	k = (rand() >> 8) % pi->size;
	px_transp(pi,j,k);
    }

    return pi;
}

#define	SAVE_FILE	"asx5213a.mat"
#define	MATLAB_NAME	"alpha"
char	name[81] = MATLAB_NAME;

int main(argc, argv)
int	argc;
char	*argv[];
{
   VEC	*x = VNULL, *y = VNULL, *z = VNULL, *u = VNULL, *v = VNULL, 
        *w = VNULL;
   VEC	*diag = VNULL, *beta = VNULL;
   PERM	*pi1 = PNULL, *pi2 = PNULL, *pi3 = PNULL, *pivot = PNULL, 
        *blocks = PNULL;
   MAT	*A = MNULL, *B = MNULL, *C = MNULL, *D = MNULL, *Q = MNULL, 
        *U = MNULL;
   BAND *bA, *bB, *bC;
   Real	cond_est, s1, s2, s3;
   int	i, j, seed;
   FILE	*fp;
   char	*cp;


    mem_info_on(TRUE);

    setbuf(stdout,(char *)NULL);

    seed = 1111;
    if ( argc > 2 )
    {
	printf("usage: %s [seed]\n",argv[0]);
	exit(0);
    }
    else if ( argc == 2 )
	sscanf(argv[1], "%d", &seed);

    /* set seed for rand() */
    smrand(seed);

    mem_stat_mark(1);

    /* print version information */
    m_version();

    printf("# grep \"^Error\" the output for a listing of errors\n");
    printf("# Don't panic if you see \"Error\" appearing; \n");
    printf("# Also check the reported size of error\n");
    printf("# This program uses randomly generated problems and therefore\n");
    printf("# may occasionally produce ill-conditioned problems\n");
    printf("# Therefore check the size of the error compared with MACHEPS\n");
    printf("# If the error is within 1000*MACHEPS then don't worry\n");
    printf("# If you get an error of size 0.1 or larger there is \n");
    printf("# probably a bug in the code or the compilation procedure\n\n");
    printf("# seed = %d\n",seed);

    printf("# Check: MACHEPS = %g\n",MACHEPS);
    /* allocate, initialise, copy and resize operations */
    /* VEC */
    notice("vector initialise, copy & resize");
    x = v_get(12);
    y = v_get(15);
    z = v_get(12);
    v_rand(x);
    v_rand(y);
    z = v_copy(x,z);
    if ( v_norm2(v_sub(x,z,z)) >= MACHEPS )
	errmesg("VEC copy");
    v_copy(x,y);
    x = v_resize(x,10);
    y = v_resize(y,10);
    if ( v_norm2(v_sub(x,y,z)) >= MACHEPS )
	errmesg("VEC copy/resize");
    x = v_resize(x,15);
    y = v_resize(y,15);
    if ( v_norm2(v_sub(x,y,z)) >= MACHEPS )
	errmesg("VEC resize");

    /* MAT */
    notice("matrix initialise, copy & resize");
    A = m_get(8,5);
    B = m_get(3,9);
    C = m_get(8,5);
    m_rand(A);
    m_rand(B);
    C = m_copy(A,C);
    if ( m_norm_inf(m_sub(A,C,C)) >= MACHEPS )
	errmesg("MAT copy");
    m_copy(A,B);
    A = m_resize(A,3,5);
    B = m_resize(B,3,5);
    if ( m_norm_inf(m_sub(A,B,C)) >= MACHEPS )
	errmesg("MAT copy/resize");
    A = m_resize(A,10,10);
    B = m_resize(B,10,10);
    if ( m_norm_inf(m_sub(A,B,C)) >= MACHEPS )
	errmesg("MAT resize");

    MEMCHK();

    /* PERM */
    notice("permutation initialise, inverting & permuting vectors");
    pi1 = px_get(15);
    pi2 = px_get(12);
    px_rand(pi1);
    v_rand(x);
    px_vec(pi1,x,z);
    y = v_resize(y,x->dim);
    pxinv_vec(pi1,z,y);
    if ( v_norm2(v_sub(x,y,z)) >= MACHEPS )
	errmesg("PERMute vector");
    pi2 = px_inv(pi1,pi2);
    pi3 = px_mlt(pi1,pi2,PNULL);
    for ( i = 0; i < pi3->size; i++ )
	if ( pi3->pe[i] != i )
	    errmesg("PERM inverse/multiply");

    /* testing catch() etc */
    notice("error handling routines");
    catch(E_NULL,
	  catchall(v_add(VNULL,VNULL,VNULL);
		     errmesg("tracecatch() failure"),
		     printf("# tracecatch() caught error\n");
		     error(E_NULL,"main"));
	             errmesg("catch() failure"),
	  printf("# catch() caught E_NULL error\n"));

    /* testing attaching a new error list (error list 2) */

    notice("attaching error lists");
    printf("# IT IS NOT A REAL WARNING ... \n");
    err_list_attach(2,MAX_TEST_ERR,test_err_list,TRUE);
    if (!err_is_list_attached(2)) 
       errmesg("attaching the error list 2");
    ev_err(__FILE__,1,__LINE__,"main",2);
    err_list_free(2);
    if (err_is_list_attached(2)) 
       errmesg("detaching the error list 2");

    /* testing inner products and v_mltadd() etc */
    notice("inner products and linear combinations");
    u = v_get(x->dim);
    v_rand(u);
    v_rand(x);
    v_resize(y,x->dim);
    v_rand(y);
    v_mltadd(y,x,-in_prod(x,y)/in_prod(x,x),z);
    if ( fabs(in_prod(x,z)) >= MACHEPS*x->dim )
	errmesg("v_mltadd()/in_prod()");
    s1 = -in_prod(x,y)/(v_norm2(x)*v_norm2(x));
    sv_mlt(s1,x,u);
    v_add(y,u,u);
    if ( v_norm2(v_sub(u,z,u)) >= MACHEPS*x->dim )
	errmesg("sv_mlt()/v_norm2()");

#ifdef ANSI_C 
    v_linlist(u,x,s1,y,1.0,VNULL);
    if ( v_norm2(v_sub(u,z,u)) >= MACHEPS*x->dim )
	errmesg("v_linlist()");
#endif
#ifdef VARARGS
    v_linlist(u,x,s1,y,1.0,VNULL);
    if ( v_norm2(v_sub(u,z,u)) >= MACHEPS*x->dim )
	errmesg("v_linlist()");
#endif


    MEMCHK();

    /* vector norms */
    notice("vector norms");
    x = v_resize(x,12);
    v_rand(x);
    for ( i = 0; i < x->dim; i++ )
	if ( v_entry(x,i) >= 0.5 )
	    v_set_val(x,i,1.0);
        else
	    v_set_val(x,i,-1.0);
    s1 = v_norm1(x);
    s2 = v_norm2(x);	
    s3 = v_norm_inf(x);
    if ( fabs(s1 - x->dim) >= MACHEPS*x->dim ||
	 fabs(s2 - sqrt((Real)(x->dim))) >= MACHEPS*x->dim ||
	 fabs(s3 - 1.0) >= MACHEPS )
	errmesg("v_norm1/2/_inf()");

    /* test matrix multiply etc */
    notice("matrix multiply and invert");
    A = m_resize(A,10,10);
    B = m_resize(B,10,10);
    m_rand(A);
    m_inverse(A,B);
    m_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	m_set_val(C,i,i,m_entry(C,i,i)-1.0);
    if ( m_norm_inf(C) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("m_inverse()/m_mlt()");

    MEMCHK();

    /* ... and transposes */
    notice("transposes and transpose-multiplies");
    m_transp(A,A);	/* can do square matrices in situ */
    mtrm_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	m_set_val(C,i,i,m_entry(C,i,i)-1.0);
    if ( m_norm_inf(C) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("m_transp()/mtrm_mlt()");
    m_transp(A,A);
    m_transp(B,B);
    mmtr_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	m_set_val(C,i,i,m_entry(C,i,i)-1.0);
    if ( m_norm_inf(C) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("m_transp()/mmtr_mlt()");
    sm_mlt(3.71,B,B);
    mmtr_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	m_set_val(C,i,i,m_entry(C,i,i)-3.71);
    if ( m_norm_inf(C) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("sm_mlt()/mmtr_mlt()");
    m_transp(B,B);
    sm_mlt(1.0/3.71,B,B);

    MEMCHK();

    /* ... and matrix-vector multiplies */
    notice("matrix-vector multiplies");
    x = v_resize(x,A->n);
    y = v_resize(y,A->m);
    z = v_resize(z,A->m);
    u = v_resize(u,A->n);
    v_rand(x);
    v_rand(y);
    mv_mlt(A,x,z);
    s1 = in_prod(y,z);
    vm_mlt(A,y,u);
    s2 = in_prod(u,x);
    if ( fabs(s1 - s2) >= (MACHEPS*x->dim)*x->dim )
	errmesg("mv_mlt()/vm_mlt()");
    mv_mlt(B,z,u);
    if ( v_norm2(v_sub(u,x,u)) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("mv_mlt()/m_inverse()");

    MEMCHK();

    /* get/set row/col */
    notice("getting and setting rows and cols");
    x = v_resize(x,A->n);
    y = v_resize(y,B->m);
    x = get_row(A,3,x);
    y = get_col(B,3,y);
    if ( fabs(in_prod(x,y) - 1.0) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("get_row()/get_col()");
    sv_mlt(-1.0,x,x);
    sv_mlt(-1.0,y,y);
    set_row(A,3,x);
    set_col(B,3,y);
    m_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	m_set_val(C,i,i,m_entry(C,i,i)-1.0);
    if ( m_norm_inf(C) >= MACHEPS*m_norm_inf(A)*m_norm_inf(B)*5 )
	errmesg("set_row()/set_col()");

    MEMCHK();

    /* matrix norms */
    notice("matrix norms");
    A = m_resize(A,11,15);
    m_rand(A);
    s1 = m_norm_inf(A);
    B = m_transp(A,B);
    s2 = m_norm1(B);
    if ( fabs(s1 - s2) >= MACHEPS*A->m )
	errmesg("m_norm1()/m_norm_inf()");
    C = mtrm_mlt(A,A,C);
    s1 = 0.0;
    for ( i = 0; i < C->m && i < C->n; i++ )
	s1 += m_entry(C,i,i);
    if ( fabs(sqrt(s1) - m_norm_frob(A)) >= MACHEPS*A->m*A->n )
	errmesg("m_norm_frob");

    MEMCHK();
    
    /* permuting rows and columns */
    notice("permuting rows & cols");
    A = m_resize(A,11,15);
    B = m_resize(B,11,15);
    pi1 = px_resize(pi1,A->m);
    px_rand(pi1);
    x = v_resize(x,A->n);
    y = mv_mlt(A,x,y);
    px_rows(pi1,A,B);
    px_vec(pi1,y,z);
    mv_mlt(B,x,u);
    if ( v_norm2(v_sub(z,u,u)) >= MACHEPS*A->m )
	errmesg("px_rows()");
    pi1 = px_resize(pi1,A->n);
    px_rand(pi1);
    px_cols(pi1,A,B);
    pxinv_vec(pi1,x,z);
    mv_mlt(B,z,u);
    if ( v_norm2(v_sub(y,u,u)) >= MACHEPS*A->n )
	errmesg("px_cols()");

    MEMCHK();

    /* MATLAB save/load */
    notice("MATLAB save/load");
    A = m_resize(A,12,11);
    if ( (fp=fopen(SAVE_FILE,"w")) == (FILE *)NULL )
	printf("Cannot perform MATLAB save/load test\n");
    else
    {
	m_rand(A);
	m_save(fp, A, name);
	fclose(fp);
	if ( (fp=fopen(SAVE_FILE,"r")) == (FILE *)NULL )
	    printf("Cannot open save file \"%s\"\n",SAVE_FILE);
	else
	{
	    M_FREE(B);
	    B = m_load(fp,&cp);
	    if ( strcmp(name,cp) || m_norm1(m_sub(A,B,B)) >= MACHEPS*A->m )
		errmesg("mload()/m_save()");
	}
    }

    MEMCHK();

    /* Now, onto matrix factorisations */
    A = m_resize(A,10,10);
    B = m_resize(B,A->m,A->n);
    m_copy(A,B);
    x = v_resize(x,A->n);
    y = v_resize(y,A->m);
    z = v_resize(z,A->n);
    u = v_resize(u,A->m);
    v_rand(x);
    mv_mlt(B,x,y);
    z = v_copy(x,z);

    notice("LU factor/solve");
    pivot = px_get(A->m);
    LUfactor(A,pivot);
    tracecatch(LUsolve(A,pivot,y,x),"main");
    tracecatch(cond_est = LUcondest(A,pivot),"main");
    printf("# cond(A) approx= %g\n", cond_est);
    if ( v_norm2(v_sub(x,z,u)) >= MACHEPS*v_norm2(x)*cond_est)
    {
	errmesg("LUfactor()/LUsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(v_sub(x,z,u)), MACHEPS);
    }

    v_copy(y,x);
    tracecatch(LUsolve(A,pivot,x,x),"main");
    tracecatch(cond_est = LUcondest(A,pivot),"main");
    if ( v_norm2(v_sub(x,z,u)) >= MACHEPS*v_norm2(x)*cond_est)
    {
	errmesg("LUfactor()/LUsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(v_sub(x,z,u)), MACHEPS);
    }

    vm_mlt(B,z,y);
    v_copy(y,x);
    tracecatch(LUTsolve(A,pivot,x,x),"main");
    if ( v_norm2(v_sub(x,z,u)) >= MACHEPS*v_norm2(x)*cond_est)
    {
	errmesg("LUfactor()/LUTsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(v_sub(x,z,u)), MACHEPS);
    }
 
    MEMCHK();

    /* QR factorisation */
    m_copy(B,A);
    mv_mlt(B,z,y);
    notice("QR factor/solve:");
    diag = v_get(A->m);
    beta = v_get(A->m);
    QRfactor(A,diag);
    QRsolve(A,diag,y,x);
    if ( v_norm2(v_sub(x,z,u)) >= MACHEPS*v_norm2(x)*cond_est )
    {
	errmesg("QRfactor()/QRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(v_sub(x,z,u)), MACHEPS);
    }
    Q = m_get(A->m,A->m);
    makeQ(A,diag,Q);
    makeR(A,A);
    m_mlt(Q,A,C);
    m_sub(B,C,C);
    if ( m_norm1(C) >= MACHEPS*m_norm1(Q)*m_norm1(B) )
    {
	errmesg("QRfactor()/makeQ()/makeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(C), MACHEPS);
    }

    MEMCHK();

    /* now try with a non-square matrix */
    A = m_resize(A,15,7);
    m_rand(A);
    B = m_copy(A,B);
    diag = v_resize(diag,A->n);
    beta = v_resize(beta,A->n);
    x = v_resize(x,A->n);
    y = v_resize(y,A->m);
    v_rand(y);
    QRfactor(A,diag);
    x = QRsolve(A,diag,y,x);
    /* z is the residual vector */
    mv_mlt(B,x,z);	v_sub(z,y,z);
    /* check B^T.z = 0 */
    vm_mlt(B,z,u);
    if ( v_norm2(u) >= MACHEPS*m_norm1(B)*v_norm2(y) )
    {
	errmesg("QRfactor()/QRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(u), MACHEPS);
    }
    Q = m_resize(Q,A->m,A->m);
    makeQ(A,diag,Q);
    makeR(A,A);
    m_mlt(Q,A,C);
    m_sub(B,C,C);
    if ( m_norm1(C) >= MACHEPS*m_norm1(Q)*m_norm1(B) )
    {
	errmesg("QRfactor()/makeQ()/makeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(C), MACHEPS);
    }
    D = m_get(A->m,Q->m);
    mtrm_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q) )
    {
	errmesg("QRfactor()/makeQ()/makeR()");
	printf("# QR orthogonality error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* QRCP factorisation */
    m_copy(B,A);
    notice("QR factor/solve with column pivoting");
    pivot = px_resize(pivot,A->n);
    QRCPfactor(A,diag,pivot);
    z = v_resize(z,A->n);
    QRCPsolve(A,diag,pivot,y,z);
    /* pxinv_vec(pivot,z,x); */
    /* now compute residual (z) vector */
    mv_mlt(B,x,z);	v_sub(z,y,z);
    /* check B^T.z = 0 */
    vm_mlt(B,z,u);
    if ( v_norm2(u) >= MACHEPS*m_norm1(B)*v_norm2(y) )
    {
	errmesg("QRCPfactor()/QRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(u), MACHEPS);
    }

    Q = m_resize(Q,A->m,A->m);
    makeQ(A,diag,Q);
    makeR(A,A);
    m_mlt(Q,A,C);
    M_FREE(D);
    D = m_get(B->m,B->n);
    px_cols(pivot,C,D);
    m_sub(B,D,D);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm1(B) )
    {
	errmesg("QRCPfactor()/makeQ()/makeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* Cholesky and LDL^T factorisation */
    /* Use these for normal equations approach */
    notice("Cholesky factor/solve");
    mtrm_mlt(B,B,A);
    CHfactor(A);
    u = v_resize(u,B->n);
    vm_mlt(B,y,u);
    z = v_resize(z,B->n);
    CHsolve(A,u,z);
    v_sub(x,z,z);
    if ( v_norm2(z) >= MACHEPS*v_norm2(x)*100 )
    {
	errmesg("CHfactor()/CHsolve()");
	printf("# Cholesky solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }
    /* modified Cholesky factorisation should be identical with Cholesky
       factorisation provided the matrix is "sufficiently positive definite" */
    mtrm_mlt(B,B,C);
    MCHfactor(C,MACHEPS);
    m_sub(A,C,C);
    if ( m_norm1(C) >= MACHEPS*m_norm1(A) )
    {
	errmesg("MCHfactor()");
	printf("# Modified Cholesky error = %g [cf MACHEPS = %g]\n",
	       m_norm1(C), MACHEPS);
    }
    /* now test the LDL^T factorisation -- using a negative def. matrix */
    mtrm_mlt(B,B,A);
    sm_mlt(-1.0,A,A);
    m_copy(A,C);
    LDLfactor(A);
    LDLsolve(A,u,z);
    w = v_get(A->m);
    mv_mlt(C,z,w);
    v_sub(w,u,w);
    if ( v_norm2(w) >= MACHEPS*v_norm2(u)*m_norm1(C) )
    {
	errmesg("LDLfactor()/LDLsolve()");
	printf("# LDL^T residual = %g [cf MACHEPS = %g]\n",
	       v_norm2(w), MACHEPS);
    }
    v_add(x,z,z);
    if ( v_norm2(z) >= MACHEPS*v_norm2(x)*100 )
    {
	errmesg("LDLfactor()/LDLsolve()");
	printf("# LDL^T solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }

    MEMCHK();

    /* and now the Bunch-Kaufman-Parlett method */
    /* set up D to be an indefinite diagonal matrix */
    notice("Bunch-Kaufman-Parlett factor/solve");

    D = m_resize(D,B->m,B->m);
    m_zero(D);
    w = v_resize(w,B->m);
    v_rand(w);
    for ( i = 0; i < w->dim; i++ )
	if ( v_entry(w,i) >= 0.5 )
	    m_set_val(D,i,i,1.0);
	else
	    m_set_val(D,i,i,-1.0);
    /* set A <- B^T.D.B */
    C = m_resize(C,B->n,B->n);
    C = mtrm_mlt(B,D,C);
    A = m_mlt(C,B,A);
    C = m_resize(C,B->n,B->n);
    C = m_copy(A,C);
    /* ... and use BKPfactor() */
    blocks = px_get(A->m);
    pivot = px_resize(pivot,A->m);
    x = v_resize(x,A->m);
    y = v_resize(y,A->m);
    z = v_resize(z,A->m);
    v_rand(x);
    mv_mlt(A,x,y);
    BKPfactor(A,pivot,blocks);
    printf("# BKP pivot =\n");	px_output(pivot);
    printf("# BKP blocks =\n");	px_output(blocks);
    BKPsolve(A,pivot,blocks,y,z);
    /* compute & check residual */
    mv_mlt(C,z,w);
    v_sub(w,y,w);
    if ( v_norm2(w) >= MACHEPS*m_norm1(C)*v_norm2(z) )
    {
	errmesg("BKPfactor()/BKPsolve()");
	printf("# BKP residual size = %g [cf MACHEPS = %g]\n",
	       v_norm2(w), MACHEPS);
    }

    /* check update routines */
    /* check LDLupdate() first */
    notice("update L.D.L^T routine");
    A = mtrm_mlt(B,B,A);
    m_resize(C,A->m,A->n);
    C = m_copy(A,C);
    LDLfactor(A);
    s1 = 3.7;
    w = v_resize(w,A->m);
    v_rand(w);
    for ( i = 0; i < C->m; i++ )
	for ( j = 0; j < C->n; j++ )
	    m_set_val(C,i,j,m_entry(C,i,j)+s1*v_entry(w,i)*v_entry(w,j));
    LDLfactor(C);
    LDLupdate(A,w,s1);
    /* zero out strictly upper triangular parts of A and C */
    for ( i = 0; i < A->m; i++ )
	for ( j = i+1; j < A->n; j++ )
	{
	    m_set_val(A,i,j,0.0);
	    m_set_val(C,i,j,0.0);
	}
    if ( m_norm1(m_sub(A,C,C)) >= sqrt(MACHEPS)*m_norm1(A) )
    {
	errmesg("LDLupdate()");
	printf("# LDL update matrix error = %g [cf MACHEPS = %g]\n",
	       m_norm1(C), MACHEPS);
    }


    /* BAND MATRICES */

#define COL 40
#define UDIAG  5
#define LDIAG  2

   smrand(101);
   bA = bd_get(LDIAG,UDIAG,COL);
   bB = bd_get(LDIAG,UDIAG,COL);
   bC = bd_get(LDIAG,UDIAG,COL);
   A = m_resize(A,COL,COL);
   B = m_resize(B,COL,COL);
   pivot = px_resize(pivot,COL);
   x = v_resize(x,COL);
   w = v_resize(w,COL);
   z = v_resize(z,COL);

   m_rand(A); 
   /* generate band matrix */
   mat2band(A,LDIAG,UDIAG,bA);
   band2mat(bA,A);    /* now A is banded */
   bB = bd_copy(bA,bB); 

   v_rand(x);  
   mv_mlt(A,x,w);
   /* test of bd_mv_mlt */
   notice("bd_mv_mlt");
   bd_mv_mlt(bA,x,z);
   v_sub(z,w,z);
   if (v_norm2(z) > v_norm2(x)*sqrt(MACHEPS)) {
      errmesg("incorrect vector (bd_mv_mlt)");
      printf(" ||exact vector. - computed vector.|| = %g [MACHEPS = %g]\n",
             v_norm2(z),MACHEPS);
   }   

   z = v_copy(w,z);

   notice("band LU factorization");
   bdLUfactor(bA,pivot);

   /* pivot will be changed */
   bdLUsolve(bA,pivot,z,z);
   v_sub(x,z,z);
   if (v_norm2(z) > v_norm2(x)*sqrt(MACHEPS)) {
      errmesg("incorrect solution (band LU factorization)");
      printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",
	     v_norm2(z),MACHEPS);
   }

   /* solve transpose system */

   notice("band LU factorization for transpose system");
   m_transp(A,B);
   mv_mlt(B,x,w);

   bd_copy(bB,bA);
   bd_transp(bA,bA);  
   /* transposition in situ */
   bd_transp(bA,bB);
   bd_transp(bB,bB);

   bdLUfactor(bB,pivot);

   bdLUsolve(bB,pivot,w,z);
   v_sub(x,z,z);
   if (v_norm2(z) > v_norm2(x)*sqrt(MACHEPS)) {
      errmesg("incorrect solution (band transposed LU factorization)");
      printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",
	     v_norm2(z),MACHEPS);
   }


   /* Cholesky factorization */

   notice("band Choleski LDL' factorization");
   m_add(A,B,A);  /* symmetric matrix */
   for (i=0; i < COL; i++)     /* positive definite */
     A->me[i][i] += 2*LDIAG;   

   mat2band(A,LDIAG,LDIAG,bA);
   band2mat(bA,A);              /* corresponding matrix A */

   v_rand(x);
   mv_mlt(A,x,w);
   z = v_copy(w,z);
   
   bdLDLfactor(bA);

   z = bdLDLsolve(bA,z,z);
   v_sub(x,z,z);
   if (v_norm2(z) > v_norm2(x)*sqrt(MACHEPS)) {
      errmesg("incorrect solution (band LDL' factorization)");
      printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",
	     v_norm2(z),MACHEPS);
   }

   /* new bandwidths */
   m_rand(A);
   bA = bd_resize(bA,UDIAG,LDIAG,COL);
   bB = bd_resize(bB,UDIAG,LDIAG,COL);
   mat2band(A,UDIAG,LDIAG,bA);
   band2mat(bA,A);
   bd_copy(bA,bB);

   mv_mlt(A,x,w);

   notice("band LU factorization (resized)");
   bdLUfactor(bA,pivot);

   /* pivot will be changed */
   bdLUsolve(bA,pivot,w,z);
   v_sub(x,z,z);
   if (v_norm2(z) > v_norm2(x)*sqrt(MACHEPS)) {
      errmesg("incorrect solution (band LU factorization)");
      printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",
	     v_norm2(z),MACHEPS);
   }

   /* testing transposition */

   notice("band matrix transposition");
   m_zero(bA->mat);
   bd_copy(bB,bA);
   m_zero(bB->mat);
   bd_copy(bA,bB);

   bd_transp(bB,bB);
   bd_transp(bB,bB);

   m_zero(bC->mat);
   bd_copy(bB,bC);

   m_sub(bA->mat,bC->mat,bC->mat);
   if (m_norm_inf(bC->mat) > MACHEPS*bC->mat->n) {
      errmesg("band transposition");
      printf(" difference ||A - (A')'|| = %g\n",m_norm_inf(bC->mat));
   }
 
   bd_free(bA);
   bd_free(bB);
   bd_free(bC);


    MEMCHK();

    /* now check QRupdate() routine */
    notice("update QR routine");

    B = m_resize(B,15,7);
    A = m_resize(A,B->m,B->n);
    m_copy(B,A);
    diag = v_resize(diag,A->n);
    beta = v_resize(beta,A->n);
    QRfactor(A,diag);
    Q = m_resize(Q,A->m,A->m);
    makeQ(A,diag,Q);
    makeR(A,A);
    m_resize(C,A->m,A->n);
    w = v_resize(w,A->m);
    v = v_resize(v,A->n);
    u = v_resize(u,A->m);
    v_rand(w);
    v_rand(v);
    vm_mlt(Q,w,u);
    QRupdate(Q,A,u,v);
    m_mlt(Q,A,C);
    for ( i = 0; i < B->m; i++ )
	for ( j = 0; j < B->n; j++ )
	    m_set_val(B,i,j,m_entry(B,i,j)+v_entry(w,i)*v_entry(v,j));
    m_sub(B,C,C);
    if ( m_norm1(C) >= MACHEPS*m_norm1(A)*m_norm1(Q)*2 )
    {
	errmesg("QRupdate()");
	printf("# Reconstruction error in QR update = %g [cf MACHEPS = %g]\n",
	       m_norm1(C), MACHEPS);
    }
    m_resize(D,Q->m,Q->n);
    mtrm_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= 10*MACHEPS*m_norm1(Q)*m_norm_inf(Q) )
    {
	errmesg("QRupdate()");
	printf("# QR update orthogonality error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    /* Now check eigenvalue/SVD routines */
    notice("eigenvalue and SVD routines");
    A = m_resize(A,11,11);
    B = m_resize(B,A->m,A->n);
    C = m_resize(C,A->m,A->n);
    D = m_resize(D,A->m,A->n);
    Q = m_resize(Q,A->m,A->n);

    m_rand(A);
    /* A <- A + A^T  for symmetric case */
    m_add(A,m_transp(A,C),A);
    u = v_resize(u,A->m);
    u = symmeig(A,Q,u);
    m_zero(B);
    for ( i = 0; i < B->m; i++ )
	m_set_val(B,i,i,v_entry(u,i));
    m_mlt(Q,B,C);
    mmtr_mlt(C,Q,D);
    m_sub(A,D,D);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q)*v_norm_inf(u)*3 )
    {
	errmesg("symmeig()");
	printf("# Reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }
    mtrm_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q)*3 )
    {
	errmesg("symmeig()");
	printf("# symmeig() orthogonality error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* now test (real) Schur decomposition */
    /* m_copy(A,B); */
    M_FREE(A);
    A = m_get(11,11);
    m_rand(A);
    B = m_copy(A,B);
    MEMCHK();

    B = schur(B,Q);
    MEMCHK();

    m_mlt(Q,B,C);
    mmtr_mlt(C,Q,D);
    MEMCHK();
    m_sub(A,D,D);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q)*m_norm1(B)*5 )
    {
	errmesg("schur()");
	printf("# Schur reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    /* orthogonality check */
    mmtr_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q)*10 )
    {
	errmesg("schur()");
	printf("# Schur orthogonality error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* now test SVD */
    A = m_resize(A,11,7);
    m_rand(A);
    U = m_get(A->n,A->n);
    Q = m_resize(Q,A->m,A->m);
    u = v_resize(u,max(A->m,A->n));
    svd(A,Q,U,u);
    /* check reconstruction of A */
    D = m_resize(D,A->m,A->n);
    C = m_resize(C,A->m,A->n);
    m_zero(D);
    for ( i = 0; i < min(A->m,A->n); i++ )
	m_set_val(D,i,i,v_entry(u,i));
    mtrm_mlt(Q,D,C);
    m_mlt(C,U,D);
    m_sub(A,D,D);
    if ( m_norm1(D) >= MACHEPS*m_norm1(U)*m_norm_inf(Q)*m_norm1(A) )
    {
	errmesg("svd()");
	printf("# SVD reconstruction error = %g [cf MACHEPS = %g]\n",
	       m_norm1(D), MACHEPS);
    }
    /* check orthogonality of Q and U */
    D = m_resize(D,Q->n,Q->n);
    mtrm_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= MACHEPS*m_norm1(Q)*m_norm_inf(Q)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (Q) = %g [cf MACHEPS = %g\n",
	       m_norm1(D), MACHEPS);
    }
    D = m_resize(D,U->n,U->n);
    mtrm_mlt(U,U,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( m_norm1(D) >= MACHEPS*m_norm1(U)*m_norm_inf(U)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (U) = %g [cf MACHEPS = %g\n",
	       m_norm1(D), MACHEPS);
    }
    for ( i = 0; i < u->dim; i++ )
	if ( v_entry(u,i) < 0 || (i < u->dim-1 &&
				  v_entry(u,i+1) > v_entry(u,i)) )
	    break;
    if ( i < u->dim )
    {
	errmesg("svd()");
	printf("# SVD sorting error\n");
    }


    /* test of long vectors */
    notice("Long vectors");
    x = v_resize(x,100000);
    y = v_resize(y,100000);
    z = v_resize(z,100000);
    v_rand(x);
    v_rand(y);
    v_mltadd(x,y,3.0,z);
    sv_mlt(1.0/3.0,z,z);
    v_mltadd(z,x,-1.0/3.0,z);
    v_sub(z,y,x);
    if (v_norm2(x) >= MACHEPS*(x->dim)) {
       errmesg("long vectors");
       printf(" norm = %g\n",v_norm2(x));
    }

    mem_stat_free(1);

    MEMCHK();

    /**************************************************
    VEC		*x, *y, *z, *u, *v, *w;
    VEC		*diag, *beta;
    PERM	*pi1, *pi2, *pi3, *pivot, *blocks;
    MAT		*A, *B, *C, *D, *Q, *U;
    **************************************************/
    V_FREE(x);		V_FREE(y);	V_FREE(z);
    V_FREE(u);		V_FREE(v);	V_FREE(w);
    V_FREE(diag);	V_FREE(beta);
    PX_FREE(pi1);	PX_FREE(pi2);	PX_FREE(pi3);
    PX_FREE(pivot);	PX_FREE(blocks);
    M_FREE(A);		M_FREE(B);	M_FREE(C);
    M_FREE(D);		M_FREE(Q);	M_FREE(U);

    MEMCHK();
    printf("# Finished torture test\n");
    mem_info();

    return 0;
}


