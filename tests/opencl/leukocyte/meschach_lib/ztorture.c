
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
	This file contains a series of tests for the Meschach matrix
	library, complex routines
*/

static char rcsid[] = "$Id: $";

#include	<stdio.h>
#include	<math.h>
#include 	"zmatrix2.h"
#include        "matlab.h"


#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);

/* extern	int	malloc_chain_check(); */
/* #define MEMCHK() if ( malloc_chain_check(0) ) \
{ printf("Error in malloc chain: \"%s\", line %d\n", \
	 __FILE__, __LINE__); exit(0); } */
#define	MEMCHK()

#define	checkpt()	printf("At line %d in file \"%s\"\n",__LINE__,__FILE__)

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

void	main(argc, argv)
int	argc;
char	*argv[];
{
    ZVEC 	*x = ZVNULL, *y = ZVNULL, *z = ZVNULL, *u = ZVNULL;
    ZVEC	*diag = ZVNULL;
    PERM	*pi1 = PNULL, *pi2 = PNULL, *pivot = PNULL;
    ZMAT	*A = ZMNULL, *B = ZMNULL, *C = ZMNULL, *D = ZMNULL,
	*Q = ZMNULL;
    complex	ONE;
    complex	z1, z2, z3;
    Real	cond_est, s1, s2, s3;
    int		i, seed;
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

    /* print out version information */
    m_version();

    printf("# Meschach Complex numbers & vectors torture test\n\n");
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

    printf("\n");

    mem_stat_mark(1);

    notice("complex arithmetic & special functions");

    ONE = zmake(1.0,0.0);
    printf("# ONE = ");	z_output(ONE);
    z1.re = mrand();	z1.im = mrand();
    z2.re = mrand();	z2.im = mrand();
    z3 = zadd(z1,z2);
    if ( fabs(z1.re+z2.re-z3.re) + fabs(z1.im+z2.im-z3.im) > 10*MACHEPS )
	errmesg("zadd");
    z3 = zsub(z1,z2);
    if ( fabs(z1.re-z2.re-z3.re) + fabs(z1.im-z2.im-z3.im) > 10*MACHEPS )
	errmesg("zadd");
    z3 = zmlt(z1,z2);
    if ( fabs(z1.re*z2.re - z1.im*z2.im - z3.re) +
	 fabs(z1.im*z2.re + z1.re*z2.im - z3.im) > 10*MACHEPS )
	errmesg("zmlt");
    s1 = zabs(z1);
    if ( fabs(s1*s1 - (z1.re*z1.re+z1.im*z1.im)) > 10*MACHEPS )
	errmesg("zabs");
    if ( zabs(zsub(z1,zmlt(z2,zdiv(z1,z2)))) > 10*MACHEPS ||
	 zabs(zsub(ONE,zdiv(z1,zmlt(z2,zdiv(z1,z2))))) > 10*MACHEPS )
	errmesg("zdiv");

    z3 = zsqrt(z1);
    if ( zabs(zsub(z1,zmlt(z3,z3))) > 10*MACHEPS )
	errmesg("zsqrt");
    if ( zabs(zsub(z1,zlog(zexp(z1)))) > 10*MACHEPS )
	errmesg("zexp/zlog");
    

    printf("# Check: MACHEPS = %g\n",MACHEPS);
    /* allocate, initialise, copy and resize operations */
    /* ZVEC */
    notice("vector initialise, copy & resize");
    x = zv_get(12);
    y = zv_get(15);
    z = zv_get(12);
    zv_rand(x);
    zv_rand(y);
    z = zv_copy(x,z);
    if ( zv_norm2(zv_sub(x,z,z)) >= MACHEPS )
	errmesg("ZVEC copy");
    zv_copy(x,y);
    x = zv_resize(x,10);
    y = zv_resize(y,10);
    if ( zv_norm2(zv_sub(x,y,z)) >= MACHEPS )
	errmesg("ZVEC copy/resize");
    x = zv_resize(x,15);
    y = zv_resize(y,15);
    if ( zv_norm2(zv_sub(x,y,z)) >= MACHEPS )
	errmesg("VZEC resize");

    /* ZMAT */
    notice("matrix initialise, copy & resize");
    A = zm_get(8,5);
    B = zm_get(3,9);
    C = zm_get(8,5);
    zm_rand(A);
    zm_rand(B);
    C = zm_copy(A,C);
    if ( zm_norm_inf(zm_sub(A,C,C)) >= MACHEPS )
	errmesg("ZMAT copy");
    zm_copy(A,B);
    A = zm_resize(A,3,5);
    B = zm_resize(B,3,5);
    if ( zm_norm_inf(zm_sub(A,B,C)) >= MACHEPS )
	errmesg("ZMAT copy/resize");
    A = zm_resize(A,10,10);
    B = zm_resize(B,10,10);
    if ( zm_norm_inf(zm_sub(A,B,C)) >= MACHEPS )
	errmesg("ZMAT resize");

    MEMCHK();

    /* PERM */
    notice("permutation initialise, inverting & permuting vectors");
    pi1 = px_get(15);
    pi2 = px_get(12);
    px_rand(pi1);
    zv_rand(x);
    px_zvec(pi1,x,z);
    y = zv_resize(y,x->dim);
    pxinv_zvec(pi1,z,y);
    if ( zv_norm2(zv_sub(x,y,z)) >= MACHEPS )
	errmesg("PERMute vector");

    /* testing catch() etc */
    notice("error handling routines");
    catch(E_NULL,
	  catchall(zv_add(ZVNULL,ZVNULL,ZVNULL);
		     errmesg("tracecatch() failure"),
		     printf("# tracecatch() caught error\n");
		     error(E_NULL,"main"));
	             errmesg("catch() failure"),
	  printf("# catch() caught E_NULL error\n"));

    /* testing inner products and v_mltadd() etc */
    notice("inner products and linear combinations");
    u = zv_get(x->dim);
    zv_rand(u);
    zv_rand(x);
    zv_resize(y,x->dim);
    zv_rand(y);
    zv_mltadd(y,x,zneg(zdiv(zin_prod(x,y),zin_prod(x,x))),z);
    if ( zabs(zin_prod(x,z)) >= 5*MACHEPS*x->dim )
    {
	errmesg("zv_mltadd()/zin_prod()");
	printf("# error norm = %g\n", zabs(zin_prod(x,z)));
    }

    z1 = zneg(zdiv(zin_prod(x,y),zmake(zv_norm2(x)*zv_norm2(x),0.0)));
    zv_mlt(z1,x,u);
    zv_add(y,u,u);
    if ( zv_norm2(zv_sub(u,z,u)) >= MACHEPS*x->dim )
    {
	errmesg("zv_mlt()/zv_norm2()");
	printf("# error norm = %g\n", zv_norm2(u));
    }

#ifdef ANSI_C
    zv_linlist(u,x,z1,y,ONE,VNULL);
    if ( zv_norm2(zv_sub(u,z,u)) >= MACHEPS*x->dim )
	errmesg("zv_linlist()");
#endif
#ifdef VARARGS
    zv_linlist(u,x,z1,y,ONE,VNULL);
    if ( zv_norm2(zv_sub(u,z,u)) >= MACHEPS*x->dim )
	errmesg("zv_linlist()");
#endif

    MEMCHK();

    /* vector norms */
    notice("vector norms");
    x = zv_resize(x,12);
    zv_rand(x);
    for ( i = 0; i < x->dim; i++ )
	if ( zabs(zv_entry(x,i)) >= 0.7 )
	    zv_set_val(x,i,ONE);
        else
	    zv_set_val(x,i,zneg(ONE));
    s1 = zv_norm1(x);
    s2 = zv_norm2(x);	
    s3 = zv_norm_inf(x);
    if ( fabs(s1 - x->dim) >= MACHEPS*x->dim ||
	 fabs(s2 - sqrt((double)(x->dim))) >= MACHEPS*x->dim ||
	 fabs(s3 - 1.0) >= MACHEPS )
	errmesg("zv_norm1/2/_inf()");

    /* test matrix multiply etc */
    notice("matrix multiply and invert");
    A = zm_resize(A,10,10);
    B = zm_resize(B,10,10);
    zm_rand(A);
    zm_inverse(A,B);
    zm_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	zm_sub_val(C,i,i,ONE);
    if ( zm_norm_inf(C) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zm_inverse()/zm_mlt()");

    MEMCHK();

    /* ... and adjoints */
    notice("adjoints and adjoint-multiplies");
    zm_adjoint(A,A);	/* can do square matrices in situ */
    zmam_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	zm_set_val(C,i,i,zsub(zm_entry(C,i,i),ONE));
    if ( zm_norm_inf(C) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zm_adjoint()/zmam_mlt()");
    zm_adjoint(A,A);
    zm_adjoint(B,B);
    zmma_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	zm_set_val(C,i,i,zsub(zm_entry(C,i,i),ONE));
    if ( zm_norm_inf(C) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zm_adjoint()/zmma_mlt()");
    zsm_mlt(zmake(3.71,2.753),B,B);
    zmma_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	zm_set_val(C,i,i,zsub(zm_entry(C,i,i),zmake(3.71,-2.753)));
    if ( zm_norm_inf(C) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("szm_mlt()/zmma_mlt()");
    zm_adjoint(B,B);
    zsm_mlt(zdiv(ONE,zmake(3.71,-2.753)),B,B);

    MEMCHK();

    /* ... and matrix-vector multiplies */
    notice("matrix-vector multiplies");
    x = zv_resize(x,A->n);
    y = zv_resize(y,A->m);
    z = zv_resize(z,A->m);
    u = zv_resize(u,A->n);
    zv_rand(x);
    zv_rand(y);
    zmv_mlt(A,x,z);
    z1 = zin_prod(y,z);
    zvm_mlt(A,y,u);
    z2 = zin_prod(u,x);
    if ( zabs(zsub(z1,z2)) >= (MACHEPS*x->dim)*x->dim )
    {
	errmesg("zmv_mlt()/zvm_mlt()");
	printf("# difference between inner products is %g\n",
	       zabs(zsub(z1,z2)));
    }
    zmv_mlt(B,z,u);
    if ( zv_norm2(zv_sub(u,x,u)) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zmv_mlt()/zvm_mlt()");

    MEMCHK();

    /* get/set row/col */
    notice("getting and setting rows and cols");
    x = zv_resize(x,A->n);
    y = zv_resize(y,B->m);
    x = zget_row(A,3,x);
    y = zget_col(B,3,y);
    if ( zabs(zsub(_zin_prod(x,y,0,Z_NOCONJ),ONE)) >=
	MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zget_row()/zget_col()");
    zv_mlt(zmake(-1.0,0.0),x,x);
    zv_mlt(zmake(-1.0,0.0),y,y);
    zset_row(A,3,x);
    zset_col(B,3,y);
    zm_mlt(A,B,C);
    for ( i = 0; i < C->m; i++ )
	zm_set_val(C,i,i,zsub(zm_entry(C,i,i),ONE));
    if ( zm_norm_inf(C) >= MACHEPS*zm_norm_inf(A)*zm_norm_inf(B)*5 )
	errmesg("zset_row()/zset_col()");

    MEMCHK();

    /* matrix norms */
    notice("matrix norms");
    A = zm_resize(A,11,15);
    zm_rand(A);
    s1 = zm_norm_inf(A);
    B = zm_adjoint(A,B);
    s2 = zm_norm1(B);
    if ( fabs(s1 - s2) >= MACHEPS*A->m )
	errmesg("zm_norm1()/zm_norm_inf()");
    C = zmam_mlt(A,A,C);
    z1.re = z1.im = 0.0;
    for ( i = 0; i < C->m && i < C->n; i++ )
	z1 = zadd(z1,zm_entry(C,i,i));
    if ( fabs(sqrt(z1.re) - zm_norm_frob(A)) >= MACHEPS*A->m*A->n )
	errmesg("zm_norm_frob");

    MEMCHK();
    
    /* permuting rows and columns */
    /******************************
    notice("permuting rows & cols");
    A = zm_resize(A,11,15);
    B = zm_resize(B,11,15);
    pi1 = px_resize(pi1,A->m);
    px_rand(pi1);
    x = zv_resize(x,A->n);
    y = zmv_mlt(A,x,y);
    px_rows(pi1,A,B);
    px_zvec(pi1,y,z);
    zmv_mlt(B,x,u);
    if ( zv_norm2(zv_sub(z,u,u)) >= MACHEPS*A->m )
	errmesg("px_rows()");
    pi1 = px_resize(pi1,A->n);
    px_rand(pi1);
    px_cols(pi1,A,B);
    pxinv_zvec(pi1,x,z);
    zmv_mlt(B,z,u);
    if ( zv_norm2(zv_sub(y,u,u)) >= MACHEPS*A->n )
	errmesg("px_cols()");
    ******************************/

    MEMCHK();

    /* MATLAB save/load */
    notice("MATLAB save/load");
    A = zm_resize(A,12,11);
    if ( (fp=fopen(SAVE_FILE,"w")) == (FILE *)NULL )
	printf("Cannot perform MATLAB save/load test\n");
    else
    {
	zm_rand(A);
	zm_save(fp, A, name);
	fclose(fp);
	if ( (fp=fopen(SAVE_FILE,"r")) == (FILE *)NULL )
	    printf("Cannot open save file \"%s\"\n",SAVE_FILE);
	else
	{
	    ZM_FREE(B);
	    B = zm_load(fp,&cp);
	    if ( strcmp(name,cp) || zm_norm1(zm_sub(A,B,C)) >=
		 MACHEPS*A->m )
	    {
		errmesg("zm_load()/zm_save()");
		printf("# orig. name = %s, restored name = %s\n", name, cp);
		printf("# orig. A =\n");	zm_output(A);
		printf("# restored A =\n");	zm_output(B);
	    }
	}
    }

    MEMCHK();

    /* Now, onto matrix factorisations */
    A = zm_resize(A,10,10);
    B = zm_resize(B,A->m,A->n);
    zm_copy(A,B);
    x = zv_resize(x,A->n);
    y = zv_resize(y,A->m);
    z = zv_resize(z,A->n);
    u = zv_resize(u,A->m);
    zv_rand(x);
    zmv_mlt(B,x,y);
    z = zv_copy(x,z);

    notice("LU factor/solve");
    pivot = px_get(A->m);
    zLUfactor(A,pivot);
    tracecatch(zLUsolve(A,pivot,y,x),"main");
    tracecatch(cond_est = zLUcondest(A,pivot),"main");
    printf("# cond(A) approx= %g\n", cond_est);
    if ( zv_norm2(zv_sub(x,z,u)) >= MACHEPS*zv_norm2(x)*cond_est)
    {
	errmesg("zLUfactor()/zLUsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(zv_sub(x,z,u)), MACHEPS);
    }


    zv_copy(y,x);
    tracecatch(zLUsolve(A,pivot,x,x),"main");
    tracecatch(cond_est = zLUcondest(A,pivot),"main");
    if ( zv_norm2(zv_sub(x,z,u)) >= MACHEPS*zv_norm2(x)*cond_est)
    {
	errmesg("zLUfactor()/zLUsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(zv_sub(x,z,u)), MACHEPS);
    }

    zvm_mlt(B,z,y);
    zv_copy(y,x);
    tracecatch(zLUAsolve(A,pivot,x,x),"main");
    if ( zv_norm2(zv_sub(x,z,u)) >= MACHEPS*zv_norm2(x)*cond_est)
    {
	errmesg("zLUfactor()/zLUAsolve()");
	printf("# LU solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(zv_sub(x,z,u)), MACHEPS);
    }

    MEMCHK();

    /* QR factorisation */
    zm_copy(B,A);
    zmv_mlt(B,z,y);
    notice("QR factor/solve:");
    diag = zv_get(A->m);
    zQRfactor(A,diag);
    zQRsolve(A,diag,y,x);
    if ( zv_norm2(zv_sub(x,z,u)) >= MACHEPS*zv_norm2(x)*cond_est )
    {
	errmesg("zQRfactor()/zQRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(zv_sub(x,z,u)), MACHEPS);
    }
    printf("# QR cond(A) approx= %g\n", zQRcondest(A));
    Q = zm_get(A->m,A->m);
    zmakeQ(A,diag,Q);
    zmakeR(A,A);
    zm_mlt(Q,A,C);
    zm_sub(B,C,C);
    if ( zm_norm1(C) >= MACHEPS*zm_norm1(Q)*zm_norm1(B) )
    {
	errmesg("zQRfactor()/zmakeQ()/zmakeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(C), MACHEPS);
    }

    MEMCHK();

    /* now try with a non-square matrix */
    A = zm_resize(A,15,7);
    zm_rand(A);
    B = zm_copy(A,B);
    diag = zv_resize(diag,A->n);
    x = zv_resize(x,A->n);
    y = zv_resize(y,A->m);
    zv_rand(y);
    zQRfactor(A,diag);
    x = zQRsolve(A,diag,y,x);
    /* z is the residual vector */
    zmv_mlt(B,x,z);	zv_sub(z,y,z);
    /* check B*.z = 0 */
    zvm_mlt(B,z,u);
    if ( zv_norm2(u) >= 100*MACHEPS*zm_norm1(B)*zv_norm2(y) )
    {
	errmesg("zQRfactor()/zQRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(u), MACHEPS);
    }
    Q = zm_resize(Q,A->m,A->m);
    zmakeQ(A,diag,Q);
    zmakeR(A,A);
    zm_mlt(Q,A,C);
    zm_sub(B,C,C);
    if ( zm_norm1(C) >= MACHEPS*zm_norm1(Q)*zm_norm1(B) )
    {
	errmesg("zQRfactor()/zmakeQ()/zmakeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(C), MACHEPS);
    }
    D = zm_get(A->m,Q->m);
    zmam_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	zm_set_val(D,i,i,zsub(zm_entry(D,i,i),ONE));
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm_inf(Q) )
    {
	errmesg("QRfactor()/makeQ()/makeR()");
	printf("# QR orthogonality error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* QRCP factorisation */
    zm_copy(B,A);
    notice("QR factor/solve with column pivoting");
    pivot = px_resize(pivot,A->n);
    zQRCPfactor(A,diag,pivot);
    z = zv_resize(z,A->n);
    zQRCPsolve(A,diag,pivot,y,z);
    /* pxinv_zvec(pivot,z,x); */
    /* now compute residual (z) vector */
    zmv_mlt(B,x,z);	zv_sub(z,y,z);
    /* check B^T.z = 0 */
    zvm_mlt(B,z,u);
    if ( zv_norm2(u) >= MACHEPS*zm_norm1(B)*zv_norm2(y) )
    {
	errmesg("QRCPfactor()/QRsolve()");
	printf("# QR solution error = %g [cf MACHEPS = %g]\n",
	       zv_norm2(u), MACHEPS);
    }

    Q = zm_resize(Q,A->m,A->m);
    zmakeQ(A,diag,Q);
    zmakeR(A,A);
    zm_mlt(Q,A,C);
    ZM_FREE(D);
    D = zm_get(B->m,B->n);
    /******************************
    px_cols(pivot,C,D);
    zm_sub(B,D,D);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm1(B) )
    {
	errmesg("QRCPfactor()/makeQ()/makeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }
    ******************************/

    /* Now check eigenvalue/SVD routines */
    notice("complex Schur routines");
    A = zm_resize(A,11,11);
    B = zm_resize(B,A->m,A->n);
    C = zm_resize(C,A->m,A->n);
    D = zm_resize(D,A->m,A->n);
    Q = zm_resize(Q,A->m,A->n);

    MEMCHK();

    /* now test complex Schur decomposition */
    /* zm_copy(A,B); */
    ZM_FREE(A);
    A = zm_get(11,11);
    zm_rand(A);
    B = zm_copy(A,B);
    MEMCHK();

    B = zschur(B,Q);
    checkpt();

    zm_mlt(Q,B,C);
    zmma_mlt(C,Q,D);
    MEMCHK();
    zm_sub(A,D,D);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm_inf(Q)*zm_norm1(B)*5 )
    {
	errmesg("zschur()");
	printf("# Schur reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }

    /* orthogonality check */
    zmma_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	zm_set_val(D,i,i,zsub(zm_entry(D,i,i),ONE));
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm_inf(Q)*10 )
    {
	errmesg("zschur()");
	printf("# Schur orthogonality error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }

    MEMCHK();

    /* now test SVD */
    /******************************
    A = zm_resize(A,11,7);
    zm_rand(A);
    U = zm_get(A->n,A->n);
    Q = zm_resize(Q,A->m,A->m);
    u = zv_resize(u,max(A->m,A->n));
    svd(A,Q,U,u);
    ******************************/
    /* check reconstruction of A */
    /******************************
    D = zm_resize(D,A->m,A->n);
    C = zm_resize(C,A->m,A->n);
    zm_zero(D);
    for ( i = 0; i < min(A->m,A->n); i++ )
	zm_set_val(D,i,i,v_entry(u,i));
    zmam_mlt(Q,D,C);
    zm_mlt(C,U,D);
    zm_sub(A,D,D);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(U)*zm_norm_inf(Q)*zm_norm1(A) )
    {
	errmesg("svd()");
	printf("# SVD reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }
    ******************************/
    /* check orthogonality of Q and U */
    /******************************
    D = zm_resize(D,Q->n,Q->n);
    zmam_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm_inf(Q)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (Q) = %g [cf MACHEPS = %g\n",
	       zm_norm1(D), MACHEPS);
    }
    D = zm_resize(D,U->n,U->n);
    zmam_mlt(U,U,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(U)*zm_norm_inf(U)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (U) = %g [cf MACHEPS = %g\n",
	       zm_norm1(D), MACHEPS);
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
    ******************************/

    ZV_FREE(x);	ZV_FREE(y);	ZV_FREE(z);
    ZV_FREE(u);	ZV_FREE(diag);
    PX_FREE(pi1);	PX_FREE(pi2);	PX_FREE(pivot);
    ZM_FREE(A);	ZM_FREE(B);	ZM_FREE(C);
    ZM_FREE(D);	ZM_FREE(Q);

    mem_stat_free(1);

    MEMCHK();
    printf("# Finished torture test for complex numbers/vectors/matrices\n");
    mem_info();
}

