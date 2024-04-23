
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
	Sparse LU factorisation
	See also: sparse.[ch] etc for details about sparse matrices
*/

#include	<stdio.h>
#include	<math.h>
#include        "sparse2.h"



/* Macro for speedup */
/* #define	sprow_idx2(r,c,hint)	\
   ( ( (hint) >= 0 && (r)->elt[hint].col == (c)) ? hint : sprow_idx((r),(c)) ) */


/* spLUfactor -- sparse LU factorisation with pivoting
	-- uses partial pivoting and Markowitz criterion
			|a[p][k]| >= alpha * max_i |a[i][k]|
	-- creates fill-in as needed
	-- in situ factorisation */
#ifndef ANSI_C
SPMAT	*spLUfactor(A,px,alpha)
SPMAT	*A;
PERM	*px;
double	alpha;
#else
SPMAT	*spLUfactor(SPMAT *A, PERM *px, double alpha)
#endif
{
	int	i, best_i, k, idx, len, best_len, m, n;
	SPROW	*r, *r_piv, tmp_row;
	STATIC	SPROW	*merge = (SPROW *)NULL;
	Real	max_val, tmp;
	STATIC VEC	*col_vals=VNULL;

	if ( ! A || ! px )
		error(E_NULL,"spLUfctr");
	if ( alpha <= 0.0 || alpha > 1.0 )
		error(E_RANGE,"alpha in spLUfctr");
	if ( px->size <= A->m )
		px = px_resize(px,A->m);
	px_ident(px);
	col_vals = v_resize(col_vals,A->m);
	MEM_STAT_REG(col_vals,TYPE_VEC);

	m = A->m;	n = A->n;
	if ( ! A->flag_col )
		sp_col_access(A);
	if ( ! A->flag_diag )
		sp_diag_access(A);
	A->flag_col = A->flag_diag = FALSE;
	if ( ! merge ) {
	   merge = sprow_get(20);
	   MEM_STAT_REG(merge,TYPE_SPROW);
	}

	for ( k = 0; k < n; k++ )
	{
	    /* find pivot row/element for partial pivoting */

	    /* get first row with a non-zero entry in the k-th column */
	    max_val = 0.0;
	    for ( i = k; i < m; i++ )
	    {
		r = &(A->row[i]);
		idx = sprow_idx(r,k);
		if ( idx < 0 )
		    tmp = 0.0;
		else
		    tmp = r->elt[idx].val;
		if ( fabs(tmp) > max_val )
		    max_val = fabs(tmp);
		col_vals->ve[i] = tmp;
	    }

	    if ( max_val == 0.0 )
		continue;

	    best_len = n+1;	/* only if no possibilities */
	    best_i = -1;
	    for ( i = k; i < m; i++ )
	    {
		tmp = fabs(col_vals->ve[i]);
		if ( tmp == 0.0 )
		    continue;
		if ( tmp >= alpha*max_val )
		{
		    r = &(A->row[i]);
		    idx = sprow_idx(r,k);
		    len = (r->len) - idx;
		    if ( len < best_len )
		    {
			best_len = len;
			best_i = i;
		    }
		}
	    }

	    /* swap row #best_i with row #k */
	    MEM_COPY(&(A->row[best_i]),&tmp_row,sizeof(SPROW));
	    MEM_COPY(&(A->row[k]),&(A->row[best_i]),sizeof(SPROW));
	    MEM_COPY(&tmp_row,&(A->row[k]),sizeof(SPROW));
	    /* swap col_vals entries */
	    tmp = col_vals->ve[best_i];
	    col_vals->ve[best_i] = col_vals->ve[k];
	    col_vals->ve[k] = tmp;
	    px_transp(px,k,best_i);

	    r_piv = &(A->row[k]);
	    for ( i = k+1; i < n; i++ )
	    {
		/* compute and set multiplier */
		tmp = col_vals->ve[i]/col_vals->ve[k];
		if ( tmp != 0.0 )
		    sp_set_val(A,i,k,tmp);
		else
		    continue;

		/* perform row operations */
		merge->len = 0;
		r = &(A->row[i]);
		sprow_mltadd(r,r_piv,-tmp,k+1,merge,TYPE_SPROW);
		idx = sprow_idx(r,k+1);
		if ( idx < 0 )
		    idx = -(idx+2);
		/* see if r needs expanding */
		if ( r->maxlen < idx + merge->len )
		    sprow_xpd(r,idx+merge->len,TYPE_SPMAT);
		r->len = idx+merge->len;
		MEM_COPY((char *)(merge->elt),(char *)&(r->elt[idx]),
			merge->len*sizeof(row_elt));
	    }
	}
#ifdef	THREADSAFE
	sprow_free(merge);	V_FREE(col_vals);
#endif

	return A;
}

/* spLUsolve -- solve A.x = b using factored matrix A from spLUfactor()
	-- returns x
	-- may not be in-situ */
#ifndef ANSI_C
VEC	*spLUsolve(A,pivot,b,x)
SPMAT	*A;
PERM	*pivot;
VEC	*b, *x;
#else
VEC	*spLUsolve(const SPMAT *A, PERM *pivot, const VEC *b, VEC *x)
#endif
{
	int	i, idx, len, lim;
	Real	sum, *x_ve;
	SPROW	*r;
	row_elt	*elt;

	if ( ! A || ! b )
	    error(E_NULL,"spLUsolve");
	if ( (pivot != PNULL && A->m != pivot->size) || A->m != b->dim )
	    error(E_SIZES,"spLUsolve");
	if ( ! x || x->dim != A->n )
	    x = v_resize(x,A->n);

	if ( pivot != PNULL )
	    x = px_vec(pivot,b,x);
	else
	    x = v_copy(b,x);

	x_ve = x->ve;
	lim = min(A->m,A->n);
	for ( i = 0; i < lim; i++ )
	{
	    sum = x_ve[i];
	    r = &(A->row[i]);
	    len = r->len;
	    elt = r->elt;
	    for ( idx = 0; idx < len && elt->col < i; idx++, elt++ )
		sum -= elt->val*x_ve[elt->col];
	    x_ve[i] = sum;
	}

	for ( i = lim-1; i >= 0; i-- )
	{
	    sum = x_ve[i];
	    r = &(A->row[i]);
	    len = r->len;
	    elt = &(r->elt[len-1]);
	    for ( idx = len-1; idx >= 0 && elt->col > i; idx--, elt-- )
		sum -= elt->val*x_ve[elt->col];
	    if ( idx < 0 || elt->col != i || elt->val == 0.0 )
		error(E_SING,"spLUsolve");
	    x_ve[i] = sum/elt->val;
	}

	return x;
}

/* spLUTsolve -- solve A.x = b using factored matrix A from spLUfactor()
	-- returns x
	-- may not be in-situ */
#ifndef ANSI_C
VEC	*spLUTsolve(A,pivot,b,x)
SPMAT	*A;
PERM	*pivot;
VEC	*b, *x;
#else
VEC	*spLUTsolve(SPMAT *A, PERM *pivot, const VEC *b, VEC *x)
#endif
{
	int	i, idx, lim, rownum;
	Real	sum, *tmp_ve;
	/* SPROW	*r; */
	row_elt	*elt;
	STATIC VEC	*tmp=VNULL;

	if ( ! A || ! b )
	    error(E_NULL,"spLUTsolve");
	if ( (pivot != PNULL && A->m != pivot->size) || A->m != b->dim )
	    error(E_SIZES,"spLUTsolve");
	tmp = v_copy(b,tmp);
	MEM_STAT_REG(tmp,TYPE_VEC);

	if ( ! A->flag_col )
	    sp_col_access(A);
	if ( ! A->flag_diag )
	    sp_diag_access(A);

	lim = min(A->m,A->n);
	tmp_ve = tmp->ve;
	/* solve U^T.tmp = b */
	for ( i = 0; i < lim; i++ )
	{
	    sum = tmp_ve[i];
	    rownum = A->start_row[i];
	    idx    = A->start_idx[i];
	    if ( rownum < 0 || idx < 0 )
		error(E_SING,"spLUTsolve");
	    while ( rownum < i && rownum >= 0 && idx >= 0 )
	    {
		elt = &(A->row[rownum].elt[idx]);
		sum -= elt->val*tmp_ve[rownum];
		rownum = elt->nxt_row;
		idx    = elt->nxt_idx;
	    }
	    if ( rownum != i )
		error(E_SING,"spLUTsolve");
	    elt = &(A->row[rownum].elt[idx]);
	    if ( elt->val == 0.0 )
		error(E_SING,"spLUTsolve");
	    tmp_ve[i] = sum/elt->val;
	}

	/* now solve L^T.tmp = (old) tmp */
	for ( i = lim-1; i >= 0; i-- )
	{
	    sum = tmp_ve[i];
	    rownum = i;
	    idx    = A->row[rownum].diag;
	    if ( idx < 0 )
		error(E_NULL,"spLUTsolve");
	    elt = &(A->row[rownum].elt[idx]);
	    rownum = elt->nxt_row;
	    idx    = elt->nxt_idx;
	    while ( rownum < lim && rownum >= 0 && idx >= 0 )
	    {
		elt = &(A->row[rownum].elt[idx]);
		sum -= elt->val*tmp_ve[rownum];
		rownum = elt->nxt_row;
		idx    = elt->nxt_idx;
	    }
	    tmp_ve[i] = sum;
	}

	if ( pivot != PNULL )
	    x = pxinv_vec(pivot,tmp,x);
	else
	    x = v_copy(tmp,x);

#ifdef	THREADSAFE
	V_FREE(tmp);
#endif

	return x;
}

/* spILUfactor -- sparse modified incomplete LU factorisation with
						no pivoting
	-- all pivot entries are ensured to be >= alpha in magnitude
	-- setting alpha = 0 gives incomplete LU factorisation
	-- no fill-in is generated
	-- in situ factorisation */
#ifndef ANSI_C
SPMAT	*spILUfactor(A,alpha)
SPMAT	*A;
double	alpha;
#else
SPMAT	*spILUfactor(SPMAT *A, double alpha)
#endif
{
    int		i, k, idx, idx_piv, m, n, old_idx, old_idx_piv;
    SPROW	*r, *r_piv;
    Real	piv_val, tmp;
    
    /* printf("spILUfactor: entered\n"); */
    if ( ! A )
	error(E_NULL,"spILUfactor");
    if ( alpha < 0.0 )
	error(E_RANGE,"[alpha] in spILUfactor");
    
    m = A->m;	n = A->n;
    sp_diag_access(A);
    sp_col_access(A);
    
    for ( k = 0; k < n; k++ )
    {
	/* printf("spILUfactor(l.%d): checkpoint A: k = %d\n",__LINE__,k); */
	/* printf("spILUfactor(l.%d): A =\n", __LINE__); */
	/* sp_output(A); */
	r_piv = &(A->row[k]);
	idx_piv = r_piv->diag;
	if ( idx_piv < 0 )
	{
	    sprow_set_val(r_piv,k,alpha);
	    idx_piv = sprow_idx(r_piv,k);
	}
	/* printf("spILUfactor: checkpoint B\n"); */
	if ( idx_piv < 0 )
	    error(E_BOUNDS,"spILUfactor");
	old_idx_piv = idx_piv;
	piv_val = r_piv->elt[idx_piv].val;
	/* printf("spILUfactor: checkpoint C\n"); */
	if ( fabs(piv_val) < alpha )
	    piv_val = ( piv_val < 0.0 ) ? -alpha : alpha;
	if ( piv_val == 0.0 )	/* alpha == 0.0 too! */
	    error(E_SING,"spILUfactor");

	/* go to next row with a non-zero in this column */
	i = r_piv->elt[idx_piv].nxt_row;
	old_idx = idx = r_piv->elt[idx_piv].nxt_idx;
	while ( i >= k )
	{
	    /* printf("spILUfactor: checkpoint D: i = %d\n",i); */
	    /* perform row operations */
	    r = &(A->row[i]);
	    /* idx = sprow_idx(r,k); */
	    /* printf("spLUfactor(l.%d) i = %d, idx = %d\n",
		   __LINE__, i, idx); */
	    if ( idx < 0 )
	    {
		idx = r->elt[old_idx].nxt_idx;
		i = r->elt[old_idx].nxt_row;
		continue;
	    }
	    /* printf("spILUfactor: checkpoint E\n"); */
	    /* compute and set multiplier */
	    r->elt[idx].val = tmp = r->elt[idx].val/piv_val;
	    /* printf("spILUfactor: piv_val = %g, multiplier = %g\n",
		   piv_val, tmp); */
	    /* printf("spLUfactor(l.%d) multiplier = %g\n", __LINE__, tmp); */
	    if ( tmp == 0.0 )
	    {
		idx = r->elt[old_idx].nxt_idx;
		i = r->elt[old_idx].nxt_row;
		continue;
	    }
	    /* idx = sprow_idx(r,k+1); */
	    /* if ( idx < 0 )
		idx = -(idx+2); */
	    idx_piv++;	idx++;	/* now look beyond the multiplier entry */
	    /* printf("spILUfactor: checkpoint F: idx = %d, idx_piv = %d\n",
		   idx, idx_piv); */
	    while ( idx_piv < r_piv->len && idx < r->len )
	    {
		/* printf("spILUfactor: checkpoint G: idx = %d, idx_piv = %d\n",
		       idx, idx_piv); */
		if ( r_piv->elt[idx_piv].col < r->elt[idx].col )
		    idx_piv++;
		else if ( r_piv->elt[idx_piv].col > r->elt[idx].col )
		    idx++;
		else /* column numbers match */
		{
		    /* printf("spILUfactor(l.%d) subtract %g times the ",
			   __LINE__, tmp); */
		    /* printf("(%d,%d) entry to the (%d,%d) entry\n",
			   k, r_piv->elt[idx_piv].col,
			   i, r->elt[idx].col); */
		    r->elt[idx].val -= tmp*r_piv->elt[idx_piv].val;
		    idx++;	idx_piv++;
		}
	    }

	    /* bump to next row with a non-zero in column k */
	    /* printf("spILUfactor(l.%d) column = %d, row[%d] =\n",
		   __LINE__, r->elt[old_idx].col, i); */
	    /* sprow_foutput(stdout,r); */
	    i = r->elt[old_idx].nxt_row;
	    old_idx = idx = r->elt[old_idx].nxt_idx;
	    /* printf("spILUfactor(l.%d) i = %d, idx = %d\n", __LINE__, i, idx); */
	    /* and restore idx_piv to index of pivot entry */
	    idx_piv = old_idx_piv;
	}
    }
    /* printf("spILUfactor: exiting\n"); */
    return A;
}
