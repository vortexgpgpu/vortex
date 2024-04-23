
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
	Sparse Cholesky factorisation code
	To be used with sparse.h, sparse.c etc

*/

static char	rcsid[] = "$Id: spchfctr.c,v 1.5 1996/08/20 19:45:33 stewart Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "sparse2.h"


#ifndef MALLOCDECL
#ifndef ANSI_C
extern	char	*calloc(), *realloc();
#endif
#endif



/* sprow_ip -- finds the (partial) inner product of a pair of sparse rows
	-- uses a "merging" approach & assumes column ordered rows
	-- row indices for inner product are all < lim */
#ifndef ANSI_C
static double	sprow_ip(row1, row2, lim)
SPROW	*row1, *row2;
int	lim;
#else
static double	sprow_ip(const SPROW *row1, const SPROW *row2, int lim)
#endif
{
	int			idx1, idx2, len1, len2, tmp;
	register row_elt	*elts1, *elts2;
	register Real		sum;

	elts1 = row1->elt;	elts2 = row2->elt;
	len1 = row1->len;	len2 = row2->len;

	sum = 0.0;

	if ( len1 <= 0 || len2 <= 0 )
		return 0.0;
	if ( elts1->col >= lim || elts2->col >= lim )
		return 0.0;

	/* use sprow_idx() to speed up inner product where one row is
		much longer than the other */
	idx1 = idx2 = 0;
	if ( len1 > 2*len2 )
	{
		idx1 = sprow_idx(row1,elts2->col);
		idx1 = (idx1 < 0) ? -(idx1+2) : idx1;
		if ( idx1 < 0 )
			error(E_UNKNOWN,"sprow_ip");
		len1 -= idx1;
	}
	else if ( len2 > 2*len1 )
	{
		idx2 = sprow_idx(row2,elts1->col);
		idx2 = (idx2 < 0) ? -(idx2+2) : idx2;
		if ( idx2 < 0 )
			error(E_UNKNOWN,"sprow_ip");
		len2 -= idx2;
	}
	if ( len1 <= 0 || len2 <= 0 )
		return 0.0;

	elts1 = &(elts1[idx1]);		elts2 = &(elts2[idx2]);


	for ( ; ; )	/* forever do... */
	{
		if ( (tmp=elts1->col-elts2->col) < 0 )
		{
		    len1--;		elts1++;
		    if ( ! len1 || elts1->col >= lim )
			break;
		}
		else if ( tmp > 0 )
		{
		    len2--;		elts2++;
		    if ( ! len2 || elts2->col >= lim )
			break;
		}
		else
		{
		    sum += elts1->val * elts2->val;
		    len1--;		elts1++;
		    len2--;		elts2++;
		    if ( ! len1 || ! len2 ||
				elts1->col >= lim || elts2->col >= lim )
			break;
		}
	}

	return sum;
}

/* sprow_sqr -- returns same as sprow_ip(row, row, lim) */
#ifndef ANSI_C
static double	sprow_sqr(row, lim)
SPROW	*row;
int	lim;
#else
static double	sprow_sqr(const SPROW *row, int lim)
#endif
{
	register	row_elt	*elts;
	int		idx, len;
	register	Real	sum, tmp;

	sum = 0.0;
	elts = row->elt;	len = row->len;
	for ( idx = 0; idx < len; idx++, elts++ )
	{
		if ( elts->col >= lim )
			break;
		tmp = elts->val;
		sum += tmp*tmp;
	}

	return sum;
}

static	int	*scan_row = (int *)NULL, *scan_idx = (int *)NULL,
			*col_list = (int *)NULL;
static	int	scan_len = 0;

/* set_scan -- expand scan_row and scan_idx arrays
	-- return new length */
#ifndef ANSI_C
int	set_scan(new_len)
int	new_len;
#else
int	set_scan(int new_len)
#endif
{
	if ( new_len <= scan_len )
		return scan_len;
	if ( new_len <= scan_len+5 )
		new_len += 5;

	/* update scan_len */
        scan_len = new_len;

	if ( ! scan_row || ! scan_idx || ! col_list )
	{
		scan_row = (int *)calloc(new_len,sizeof(int));
		scan_idx = (int *)calloc(new_len,sizeof(int));
		col_list = (int *)calloc(new_len,sizeof(int));
	}
	else
	{
		scan_row = (int *)realloc((char *)scan_row,new_len*sizeof(int));
		scan_idx = (int *)realloc((char *)scan_idx,new_len*sizeof(int));
		col_list = (int *)realloc((char *)col_list,new_len*sizeof(int));
	}

	if ( ! scan_row || ! scan_idx || ! col_list )
		error(E_MEM,"set_scan");
	return new_len;
}

/* spCHfactor -- sparse Cholesky factorisation
	-- only the lower triangular part of A (incl. diagonal) is used */
#ifndef ANSI_C
SPMAT	*spCHfactor(A)
SPMAT	*A;
#else
SPMAT	*spCHfactor(SPMAT *A)
#endif
{
	register 	int	i;
	int	idx, k, m, minim, n, num_scan, diag_idx, tmp1;
	Real	pivot, tmp2;
	SPROW	*r_piv, *r_op;
	row_elt	*elt_piv, *elt_op, *old_elt;

	if ( A == SMNULL )
		error(E_NULL,"spCHfactor");
	if ( A->m != A->n )
		error(E_SQUARE,"spCHfactor");

	/* set up access paths if not already done so */
	sp_col_access(A);
	sp_diag_access(A);

	/* printf("spCHfactor() -- checkpoint 1\n"); */
	m = A->m;	n = A->n;
	for ( k = 0; k < m; k++ )
	{
		r_piv = &(A->row[k]);
		if ( r_piv->len > scan_len )
			set_scan(r_piv->len);
		elt_piv = r_piv->elt;
		diag_idx = sprow_idx2(r_piv,k,r_piv->diag);
		if ( diag_idx < 0 )
			error(E_POSDEF,"spCHfactor");
		old_elt = &(elt_piv[diag_idx]);
		for ( i = 0; i < r_piv->len; i++ )
		{
			if ( elt_piv[i].col > k )
				break;
			col_list[i] = elt_piv[i].col;
			scan_row[i] = elt_piv[i].nxt_row;
			scan_idx[i] = elt_piv[i].nxt_idx;
		}
		/* printf("spCHfactor() -- checkpoint 2\n"); */
		num_scan = i;	/* number of actual entries in scan_row etc. */
		/* printf("num_scan = %d\n",num_scan); */

		/* set diagonal entry of Cholesky factor */
		tmp2 = elt_piv[diag_idx].val - sprow_sqr(r_piv,k);
		if ( tmp2 <= 0.0 )
			error(E_POSDEF,"spCHfactor");
		elt_piv[diag_idx].val = pivot = sqrt(tmp2);

		/* now set the k-th column of the Cholesky factors */
		/* printf("k = %d\n",k); */
		for ( ; ; )	/* forever do... */
		{
		    /* printf("spCHfactor() -- checkpoint 3\n"); */
		    /* find next row where something (non-trivial) happens
			i.e. find min(scan_row) */
		    /* printf("scan_row: "); */
		    minim = n;
		    for ( i = 0; i < num_scan; i++ )
		    {
			tmp1 = scan_row[i];
			/* printf("%d ",tmp1); */
			minim = ( tmp1 >= 0 && tmp1 < minim ) ? tmp1 : minim;
		    }
		    /* printf("minim = %d\n",minim); */
		    /* printf("col_list: "); */
		    /*  for ( i = 0; i < num_scan; i++ ) */
			/*  printf("%d ",col_list[i]); */
		    /*  printf("\n"); */

		    if ( minim >= n )
			break;	/* nothing more to do for this column */
		    r_op = &(A->row[minim]);
		    elt_op = r_op->elt;

		    /* set next entry in column k of Cholesky factors */
		    idx = sprow_idx2(r_op,k,scan_idx[num_scan-1]);
		    if ( idx < 0 )
		    {	/* fill-in */
			sp_set_val(A,minim,k,
					-sprow_ip(r_piv,r_op,k)/pivot);
			/* in case a realloc() has occurred... */
			elt_op = r_op->elt;
			/* now set up column access path again */
			idx = sprow_idx2(r_op,k,-(idx+2));
			tmp1 = old_elt->nxt_row;
			old_elt->nxt_row = minim;
			r_op->elt[idx].nxt_row = tmp1;
			tmp1 = old_elt->nxt_idx;
			old_elt->nxt_idx = idx;
			r_op->elt[idx].nxt_idx = tmp1;
		    }
		    else
		        elt_op[idx].val = (elt_op[idx].val -
				sprow_ip(r_piv,r_op,k))/pivot;

		    /* printf("spCHfactor() -- checkpoint 4\n"); */

		    /* remember current element in column k for column chain */
		    idx = sprow_idx2(r_op,k,idx);
		    old_elt = &(r_op->elt[idx]);

		    /* update scan_row */
		    /* printf("spCHfactor() -- checkpoint 5\n"); */
		    /* printf("minim = %d\n",minim); */
		    for ( i = 0; i < num_scan; i++ )
		    {
			if ( scan_row[i] != minim )
				continue;
			idx = sprow_idx2(r_op,col_list[i],scan_idx[i]);
			if ( idx < 0 )
			{	scan_row[i] = -1;	continue;	}
			scan_row[i] = elt_op[idx].nxt_row;
			scan_idx[i] = elt_op[idx].nxt_idx;
			/* printf("scan_row[%d] = %d\n",i,scan_row[i]); */
			/* printf("scan_idx[%d] = %d\n",i,scan_idx[i]); */
		    }
			
		}
	    /* printf("spCHfactor() -- checkpoint 6\n"); */
	    /* sp_dump(stdout,A); */
	    /* printf("\n\n\n"); */
	}

	return A;
}

/* spCHsolve -- solve L.L^T.out=b where L is a sparse matrix,
	-- out, b dense vectors
	-- returns out; operation may be in-situ */
#ifndef ANSI_C
VEC	*spCHsolve(L,b,out)
SPMAT	*L;
VEC	*b, *out;
#else
VEC	*spCHsolve(SPMAT *L, const VEC *b, VEC *out)
#endif
{
	int	i, j_idx, n, scan_idx, scan_row;
	SPROW	*row;
	row_elt	*elt;
	Real	diag_val, sum, *out_ve;

	if ( L == SMNULL || b == VNULL )
		error(E_NULL,"spCHsolve");
	if ( L->m != L->n )
		error(E_SQUARE,"spCHsolve");
	if ( b->dim != L->m )
		error(E_SIZES,"spCHsolve");

	if ( ! L->flag_col )
		sp_col_access(L);
	if ( ! L->flag_diag )
		sp_diag_access(L);

	out = v_copy(b,out);
	out_ve = out->ve;

	/* forward substitution: solve L.x=b for x */
	n = L->n;
	for ( i = 0; i < n; i++ )
	{
		sum = out_ve[i];
		row = &(L->row[i]);
		elt = row->elt;
		for ( j_idx = 0; j_idx < row->len; j_idx++, elt++ )
		{
		    if ( elt->col >= i )
			break;
		    sum -= elt->val*out_ve[elt->col];
		}
		if ( row->diag >= 0 )
		    out_ve[i] = sum/(row->elt[row->diag].val);
		else
		    error(E_SING,"spCHsolve");
	}

	/* backward substitution: solve L^T.out = x for out */
	for ( i = n-1; i >= 0; i-- )
	{
		sum = out_ve[i];
		row = &(L->row[i]);
		/* Note that row->diag >= 0 by above loop */
		elt = &(row->elt[row->diag]);
		diag_val = elt->val;

		/* scan down column */
		scan_idx = elt->nxt_idx;
		scan_row = elt->nxt_row;
		while ( scan_row >= 0 /* && scan_idx >= 0 */ )
		{
		    row = &(L->row[scan_row]);
		    elt = &(row->elt[scan_idx]);
		    sum -= elt->val*out_ve[scan_row];
		    scan_idx = elt->nxt_idx;
		    scan_row = elt->nxt_row;
		}
		out_ve[i] = sum/diag_val;
	}

	return out;
}

/* spICHfactor -- sparse Incomplete Cholesky factorisation
	-- does a Cholesky factorisation assuming NO FILL-IN
	-- as for spCHfactor(), only the lower triangular part of A is used */
#ifndef ANSI_C
SPMAT	*spICHfactor(A)
SPMAT	*A;
#else
SPMAT	*spICHfactor(SPMAT *A)
#endif
{
	int	k, m, n, nxt_row, nxt_idx, diag_idx;
	Real	pivot, tmp2;
	SPROW	*r_piv, *r_op;
	row_elt	*elt_piv, *elt_op;

	if ( A == SMNULL )
		error(E_NULL,"spICHfactor");
	if ( A->m != A->n )
		error(E_SQUARE,"spICHfactor");

	/* set up access paths if not already done so */
	if ( ! A->flag_col )
		sp_col_access(A);
	if ( ! A->flag_diag )
		sp_diag_access(A);

	m = A->m;	n = A->n;
	for ( k = 0; k < m; k++ )
	{
		r_piv = &(A->row[k]);

		diag_idx = r_piv->diag;
		if ( diag_idx < 0 )
			error(E_POSDEF,"spICHfactor");

		elt_piv = r_piv->elt;

		/* set diagonal entry of Cholesky factor */
		tmp2 = elt_piv[diag_idx].val - sprow_sqr(r_piv,k);
		if ( tmp2 <= 0.0 )
			error(E_POSDEF,"spICHfactor");
		elt_piv[diag_idx].val = pivot = sqrt(tmp2);

		/* find next row where something (non-trivial) happens */
		nxt_row = elt_piv[diag_idx].nxt_row;
		nxt_idx = elt_piv[diag_idx].nxt_idx;

		/* now set the k-th column of the Cholesky factors */
		while ( nxt_row >= 0 && nxt_idx >= 0 )
		{
		    /* nxt_row and nxt_idx give next next row (& index)
			of the entry to be modified */
		    r_op = &(A->row[nxt_row]);
		    elt_op = r_op->elt;
		    elt_op[nxt_idx].val = (elt_op[nxt_idx].val -
				sprow_ip(r_piv,r_op,k))/pivot;

		    nxt_row = elt_op[nxt_idx].nxt_row;
		    nxt_idx = elt_op[nxt_idx].nxt_idx;
		}
	}

	return A;
}


/* spCHsymb -- symbolic sparse Cholesky factorisation
	-- does NOT do any floating point arithmetic; just sets up the structure
	-- only the lower triangular part of A (incl. diagonal) is used */
#ifndef ANSI_C
SPMAT	*spCHsymb(A)
SPMAT	*A;
#else
SPMAT	*spCHsymb(SPMAT *A)
#endif
{
	register 	int	i;
	int	idx, k, m, minim, n, num_scan, diag_idx, tmp1;
	SPROW	*r_piv, *r_op;
	row_elt	*elt_piv, *elt_op, *old_elt;

	if ( A == SMNULL )
		error(E_NULL,"spCHsymb");
	if ( A->m != A->n )
		error(E_SQUARE,"spCHsymb");

	/* set up access paths if not already done so */
	if ( ! A->flag_col )
		sp_col_access(A);
	if ( ! A->flag_diag )
		sp_diag_access(A);

	/* printf("spCHsymb() -- checkpoint 1\n"); */
	m = A->m;	n = A->n;
	for ( k = 0; k < m; k++ )
	{
		r_piv = &(A->row[k]);
		if ( r_piv->len > scan_len )
			set_scan(r_piv->len);
		elt_piv = r_piv->elt;
		diag_idx = sprow_idx2(r_piv,k,r_piv->diag);
		if ( diag_idx < 0 )
			error(E_POSDEF,"spCHsymb");
		old_elt = &(elt_piv[diag_idx]);
		for ( i = 0; i < r_piv->len; i++ )
		{
			if ( elt_piv[i].col > k )
				break;
			col_list[i] = elt_piv[i].col;
			scan_row[i] = elt_piv[i].nxt_row;
			scan_idx[i] = elt_piv[i].nxt_idx;
		}
		/* printf("spCHsymb() -- checkpoint 2\n"); */
		num_scan = i;	/* number of actual entries in scan_row etc. */
		/* printf("num_scan = %d\n",num_scan); */

		/* now set the k-th column of the Cholesky factors */
		/* printf("k = %d\n",k); */
		for ( ; ; )	/* forever do... */
		{
		    /* printf("spCHsymb() -- checkpoint 3\n"); */
		    /* find next row where something (non-trivial) happens
			i.e. find min(scan_row) */
		    minim = n;
		    for ( i = 0; i < num_scan; i++ )
		    {
			tmp1 = scan_row[i];
			/* printf("%d ",tmp1); */
			minim = ( tmp1 >= 0 && tmp1 < minim ) ? tmp1 : minim;
		    }

		    if ( minim >= n )
			break;	/* nothing more to do for this column */
		    r_op = &(A->row[minim]);
		    elt_op = r_op->elt;

		    /* set next entry in column k of Cholesky factors */
		    idx = sprow_idx2(r_op,k,scan_idx[num_scan-1]);
		    if ( idx < 0 )
		    {	/* fill-in */
			sp_set_val(A,minim,k,0.0);
			/* in case a realloc() has occurred... */
			elt_op = r_op->elt;
			/* now set up column access path again */
			idx = sprow_idx2(r_op,k,-(idx+2));
			tmp1 = old_elt->nxt_row;
			old_elt->nxt_row = minim;
			r_op->elt[idx].nxt_row = tmp1;
			tmp1 = old_elt->nxt_idx;
			old_elt->nxt_idx = idx;
			r_op->elt[idx].nxt_idx = tmp1;
		    }

		    /* printf("spCHsymb() -- checkpoint 4\n"); */

		    /* remember current element in column k for column chain */
		    idx = sprow_idx2(r_op,k,idx);
		    old_elt = &(r_op->elt[idx]);

		    /* update scan_row */
		    /* printf("spCHsymb() -- checkpoint 5\n"); */
		    /* printf("minim = %d\n",minim); */
		    for ( i = 0; i < num_scan; i++ )
		    {
			if ( scan_row[i] != minim )
				continue;
			idx = sprow_idx2(r_op,col_list[i],scan_idx[i]);
			if ( idx < 0 )
			{	scan_row[i] = -1;	continue;	}
			scan_row[i] = elt_op[idx].nxt_row;
			scan_idx[i] = elt_op[idx].nxt_idx;
			/* printf("scan_row[%d] = %d\n",i,scan_row[i]); */
			/* printf("scan_idx[%d] = %d\n",i,scan_idx[i]); */
		    }
			
		}
	    /* printf("spCHsymb() -- checkpoint 6\n"); */
	}

	return A;
}

/* comp_AAT -- compute A.A^T where A is a given sparse matrix */
#ifndef ANSI_C
SPMAT	*comp_AAT(A)
SPMAT	*A;
#else
SPMAT	*comp_AAT(SPMAT *A)
#endif
{
	SPMAT	*AAT;
	SPROW	*r, *r2;
	row_elt	*elts, *elts2;
	int	i, idx, idx2, j, m, minim, n, num_scan, tmp1;
	Real	ip;

	if ( ! A )
		error(E_NULL,"comp_AAT");
	m = A->m;	n = A->n;

	/* set up column access paths */
	if ( ! A->flag_col )
		sp_col_access(A);

	AAT = sp_get(m,m,10);

	for ( i = 0; i < m; i++ )
	{
		/* initialisation */
		r = &(A->row[i]);
		elts = r->elt;

		/* set up scan lists for this row */
		if ( r->len > scan_len )
		    set_scan(r->len);
		for ( j = 0; j < r->len; j++ )
		{
		    col_list[j] = elts[j].col;
		    scan_row[j] = elts[j].nxt_row;
		    scan_idx[j] = elts[j].nxt_idx;
		}
		num_scan = r->len;

		/* scan down the rows for next non-zero not
			associated with a diagonal entry */
		for ( ; ; )
		{
		    minim = m;
		    for ( idx = 0; idx < num_scan; idx++ )
		    {
			tmp1 = scan_row[idx];
			minim = ( tmp1 >= 0 && tmp1 < minim ) ? tmp1 : minim;
		    }
		    if ( minim >= m )
		 	break;
		    r2 = &(A->row[minim]);
		    if ( minim > i )
		    {
			ip = sprow_ip(r,r2,n);
		        sp_set_val(AAT,minim,i,ip);
		        sp_set_val(AAT,i,minim,ip);
		    }
		    /* update scan entries */
		    elts2 = r2->elt;
		    for ( idx = 0; idx < num_scan; idx++ )
		    {
			if ( scan_row[idx] != minim || scan_idx[idx] < 0 )
			    continue;
			idx2 = scan_idx[idx];
			scan_row[idx] = elts2[idx2].nxt_row;
			scan_idx[idx] = elts2[idx2].nxt_idx;
		    }
		}

		/* set the diagonal entry */
		sp_set_val(AAT,i,i,sprow_sqr(r,n));
	}

	return AAT;
}

