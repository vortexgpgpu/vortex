
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
	This file contains tests for the sparse matrix part of Meschach
*/

#include	<stdio.h>
#include	<math.h>
#include	"matrix2.h"
#include	"sparse2.h"
#include        "iter.h"

#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);

/* for iterative methods */

#if REAL == DOUBLE
#define	EPS	1e-7
#elif REAL == FLOAT
#define EPS   1e-3
#endif

int	chk_col_accessSPT(A)
SPMAT	*A;
{
    int		i, j, nxt_idx, nxt_row, scan_cnt, total_cnt;
    SPROW	*r;
    row_elt	*e;

    if ( ! A )
	error(E_NULL,"chk_col_accessSPT");
    if ( ! A->flag_col )
	return FALSE;

    /* scan down each column, counting the number of entries met */
    scan_cnt = 0;
    for ( j = 0; j < A->n; j++ )
    {
	i = -1;
	nxt_idx = A->start_idx[j];
	nxt_row = A->start_row[j];
	while ( nxt_row >= 0 && nxt_idx >= 0 && nxt_row > i )
	{
	    i = nxt_row;
	    r = &(A->row[i]);
	    e = &(r->elt[nxt_idx]);
	    nxt_idx = e->nxt_idx;
	    nxt_row = e->nxt_row;
	    scan_cnt++;
	}
    }

    total_cnt = 0;
    for ( i = 0; i < A->m; i++ )
	total_cnt += A->row[i].len;
    if ( total_cnt != scan_cnt )
	return FALSE;
    else
	return TRUE;
}


void	main(argc, argv)
int	argc;
char	*argv[];
{
    VEC		*x, *y, *z, *u, *v;
    Real	s1, s2;
    PERM	*pivot;
    SPMAT	*A, *B, *C;
    SPMAT       *B1, *C1;
    SPROW	*r;
    int		i, j, k, deg, seed, m, m_old, n, n_old;


    mem_info_on(TRUE);

    setbuf(stdout, (char *)NULL);
    /* get seed if in argument list */
    if ( argc == 1 )
	seed = 1111;
    else if ( argc == 2 && sscanf(argv[1],"%d",&seed) == 1 )
	;
    else
    {
	printf("usage: %s [seed]\n", argv[0]);
	exit(0);
    }
    srand(seed);

    /* set up two random sparse matrices */
    m = 120;
    n = 100;
    deg = 8;
    notice("allocating sparse matrices");
    A = sp_get(m,n,deg);
    B = sp_get(m,n,deg);
    notice("setting and getting matrix entries");
    for ( k = 0; k < m*deg; k++ )
    {
	i = (rand() >> 8) % m;
	j = (rand() >> 8) % n;
	sp_set_val(A,i,j,rand()/((Real)MAX_RAND));
	i = (rand() >> 8) % m;
	j = (rand() >> 8) % n;
	sp_set_val(B,i,j,rand()/((Real)MAX_RAND));
    }
    for ( k = 0; k < 10; k++ )
    {
	s1 = rand()/((Real)MAX_RAND);
	i = (rand() >> 8) % m;
	j = (rand() >> 8) % n;
	sp_set_val(A,i,j,s1);
	s2 = sp_get_val(A,i,j);
	if ( fabs(s1 - s2) >= MACHEPS )
	    break;
    }
    if ( k < 10 )
	errmesg("sp_set_val()/sp_get_val()");

    /* test copy routines */
    notice("copy routines");
    x = v_get(n);
    y = v_get(m);
    z = v_get(m);
    /* first copy routine */
    C = sp_copy(A);
    for ( k = 0; k < 100; k++ )
    {
	v_rand(x);
	sp_mv_mlt(A,x,y);
	sp_mv_mlt(C,x,z);
	if ( v_norm_inf(v_sub(y,z,z)) >= MACHEPS*deg*m )
	    break;
    }
    if ( k < 100 )
    {
	errmesg("sp_copy()/sp_mv_mlt()");
	printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",
	       v_norm_inf(z), MACHEPS);
    }
    /* second copy routine
       -- note that A & B have different sparsity patterns */

    mem_stat_mark(1);
    sp_copy2(A,B);
    mem_stat_free(1);
    for ( k = 0; k < 10; k++ )
    {
	v_rand(x);
	sp_mv_mlt(A,x,y);
	sp_mv_mlt(B,x,z);
	if ( v_norm_inf(v_sub(y,z,z)) >= MACHEPS*deg*m )
	    break;
    }
    if ( k < 10 )
    {
	errmesg("sp_copy2()/sp_mv_mlt()");
	printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",
	       v_norm_inf(z), MACHEPS);
    }

    /* now check compacting routine */
    notice("compacting routine");
    sp_compact(B,0.0);
    for ( k = 0; k < 10; k++ )
    {
	v_rand(x);
	sp_mv_mlt(A,x,y);
	sp_mv_mlt(B,x,z);
	if ( v_norm_inf(v_sub(y,z,z)) >= MACHEPS*deg*m )
	    break;
    }
    if ( k < 10 )
    {
	errmesg("sp_compact()");
	printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",
	       v_norm_inf(z), MACHEPS);
    }
    for ( i = 0; i < B->m; i++ )
    {
	r = &(B->row[i]);
	for ( j = 0; j < r->len; j++ )
	    if ( r->elt[j].val == 0.0 )
		break;
    }
    if ( i < B->m )
    {
	errmesg("sp_compact()");
	printf("# Zero entry in compacted matrix\n");
    }

    /* check column access paths */
    notice("resizing and access paths");
    m_old = A->m-1;
    n_old = A->n-1;
    A = sp_resize(A,A->m+10,A->n+10);
    for ( k = 0 ; k < 20; k++ )
    {
	i = m_old + ((rand() >> 8) % 10);
	j = n_old + ((rand() >> 8) % 10);
	s1 = rand()/((Real)MAX_RAND);
	sp_set_val(A,i,j,s1);
	if ( fabs(s1 - sp_get_val(A,i,j)) >= MACHEPS )
	    break;
    }
    if ( k < 20 )
	errmesg("sp_resize()");
    sp_col_access(A);
    if ( ! chk_col_accessSPT(A) )
    {
	errmesg("sp_col_access()");
    }
    sp_diag_access(A);
    for ( i = 0; i < A->m; i++ )
    {
	r = &(A->row[i]);
	if ( r->diag != sprow_idx(r,i) )
	    break;
    }
    if ( i < A->m )
    {
	errmesg("sp_diag_access()");
    }

    /* test both sp_mv_mlt() and sp_vm_mlt() */
    x = v_resize(x,B->n);
    y = v_resize(y,B->m);
    u = v_get(B->m);
    v = v_get(B->n);
    for ( k = 0; k < 10; k++ )
    {
	v_rand(x);
	v_rand(y);
	sp_mv_mlt(B,x,u);
	sp_vm_mlt(B,y,v);
	if ( fabs(in_prod(x,v) - in_prod(y,u)) >=
	    MACHEPS*v_norm2(x)*v_norm2(u)*5 )
	    break;
    }
    if ( k < 10 )
    {
	errmesg("sp_mv_mlt()/sp_vm_mlt()");
	printf("# Error in inner products = %g [cf MACHEPS = %g]\n",
	       fabs(in_prod(x,v) - in_prod(y,u)), MACHEPS);
    }

    SP_FREE(A);
    SP_FREE(B);
    SP_FREE(C);

    /* now test Cholesky and LU factorise and solve */
    notice("sparse Cholesky factorise/solve");
    A = iter_gen_sym(120,8);
    B = sp_copy(A);
    spCHfactor(A);
    x = v_resize(x,A->m);
    y = v_resize(y,A->m);
    v_rand(x);
    sp_mv_mlt(B,x,y);
    z = v_resize(z,A->m);
    spCHsolve(A,y,z);
    v = v_resize(v,A->m);
    sp_mv_mlt(B,z,v);
    /* compute residual */
    v_sub(y,v,v);
    if ( v_norm2(v) >= MACHEPS*v_norm2(y)*10 )
    {
	errmesg("spCHfactor()/spCHsolve()");
	printf("# Sparse Cholesky residual = %g [cf MACHEPS = %g]\n",
	       v_norm2(v), MACHEPS);
    }
    /* compute error in solution */
    v_sub(x,z,z);
    if ( v_norm2(z) > MACHEPS*v_norm2(x)*10 )
    {
	errmesg("spCHfactor()/spCHsolve()");
	printf("# Solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }

    /* now test symbolic and incomplete factorisation */
    SP_FREE(A);
    A = sp_copy(B);
    
    mem_stat_mark(2);
    spCHsymb(A);
    mem_stat_mark(2);

    spICHfactor(A);
    spCHsolve(A,y,z);
    v = v_resize(v,A->m);
    sp_mv_mlt(B,z,v);
    /* compute residual */
    v_sub(y,v,v);
    if ( v_norm2(v) >= MACHEPS*v_norm2(y)*5 )
    {
	errmesg("spCHsymb()/spICHfactor()");
	printf("# Sparse Cholesky residual = %g [cf MACHEPS = %g]\n",
	       v_norm2(v), MACHEPS);
    }
    /* compute error in solution */
    v_sub(x,z,z);
    if ( v_norm2(z) > MACHEPS*v_norm2(x)*10 )
    {
	errmesg("spCHsymb()/spICHfactor()");
	printf("# Solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }

    /* now test sparse LU factorisation */
    notice("sparse LU factorise/solve");
    SP_FREE(A);
    SP_FREE(B);
    A = iter_gen_nonsym(100,100,8,1.0);

    B = sp_copy(A);
    x = v_resize(x,A->n);
    z = v_resize(z,A->n);
    y = v_resize(y,A->m);
    v = v_resize(v,A->m);

    v_rand(x);
    sp_mv_mlt(B,x,y);
    pivot = px_get(A->m);

    mem_stat_mark(3);
    spLUfactor(A,pivot,0.1);
    spLUsolve(A,pivot,y,z);
    mem_stat_free(3);
    sp_mv_mlt(B,z,v);

    /* compute residual */
    v_sub(y,v,v);
    if ( v_norm2(v) >= MACHEPS*v_norm2(y)*A->m )
    {
	errmesg("spLUfactor()/spLUsolve()");
	printf("# Sparse LU residual = %g [cf MACHEPS = %g]\n",
	       v_norm2(v), MACHEPS);
    }
    /* compute error in solution */
    v_sub(x,z,z);
    if ( v_norm2(z) > MACHEPS*v_norm2(x)*100*A->m )
    {
	errmesg("spLUfactor()/spLUsolve()");
	printf("# Sparse LU solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }

    /* now check spLUTsolve */
    mem_stat_mark(4);
    sp_vm_mlt(B,x,y);
    spLUTsolve(A,pivot,y,z);
    sp_vm_mlt(B,z,v);
    mem_stat_free(4);

    /* compute residual */
    v_sub(y,v,v);
    if ( v_norm2(v) >= MACHEPS*v_norm2(y)*A->m )
    {
	errmesg("spLUTsolve()");
	printf("# Sparse LU residual = %g [cf MACHEPS = %g]\n",
	       v_norm2(v), MACHEPS);
    }
    /* compute error in solution */
    v_sub(x,z,z);
    if ( v_norm2(z) > MACHEPS*v_norm2(x)*100*A->m )
    {
	errmesg("spLUTsolve()");
	printf("# Sparse LU solution error = %g [cf MACHEPS = %g]\n",
	       v_norm2(z), MACHEPS);
    }

    /* algebraic operations */
    notice("addition,subtraction and multiplying by a number");
    SP_FREE(A);
    SP_FREE(B);

    m = 120;
    n = 120;
    deg = 5;
    A = sp_get(m,n,deg);
    B = sp_get(m,n,deg);
    C = sp_get(m,n,deg);
    C1 = sp_get(m,n,deg);

    for ( k = 0; k < m*deg; k++ )
    {
	i = (rand() >> 8) % m;
	j = (rand() >> 8) % n;
	sp_set_val(A,i,j,rand()/((Real)MAX_RAND));
	i = (rand() >> 8) % m;
	j = (rand() >> 8) % n;
	sp_set_val(B,i,j,rand()/((Real)MAX_RAND));
    }
    
    s1 = mrand(); 
    B1 = sp_copy(B);

    mem_stat_mark(1);
    sp_smlt(B,s1,C);
    sp_add(A,C,C1);
    sp_sub(C1,A,C);
    sp_smlt(C,-1.0/s1,C);
    sp_add(C,B1,C);

    s2 = 0.0;
    for (k=0; k < C->m; k++) {
       r = &(C->row[k]);
       for (j=0; j < r->len; j++) {
	  if (s2 < fabs(r->elt[j].val)) 
	    s2 = fabs(r->elt[j].val);
       }
    }

    if (s2 > MACHEPS*A->m) {
       errmesg("add, sub, mlt sparse matrices (args not in situ)\n");
       printf(" difference = %g [MACEPS = %g]\n",s2,MACHEPS);
    }

    sp_mltadd(A,B1,s1,C1);
    sp_sub(C1,A,A);
    sp_smlt(A,1.0/s1,C1);
    sp_sub(C1,B1,C1);
    mem_stat_free(1);

    s2 = 0.0;
    for (k=0; k < C1->m; k++) {
       r = &(C1->row[k]);
       for (j=0; j < r->len; j++) {
	  if (s2 < fabs(r->elt[j].val)) 
	    s2 = fabs(r->elt[j].val);
       }
    }

    if (s2 > MACHEPS*A->m) {
       errmesg("add, sub, mlt sparse matrices (args not in situ)\n");
       printf(" difference = %g [MACEPS = %g]\n",s2,MACHEPS);
    }

    V_FREE(x);
    V_FREE(y);    
    V_FREE(z);
    V_FREE(u);
    V_FREE(v);  
    PX_FREE(pivot);
    SP_FREE(A);
    SP_FREE(B);
    SP_FREE(C);
    SP_FREE(B1);
    SP_FREE(C1);

    printf("# Done testing (%s)\n",argv[0]);
    mem_info();
}
    




