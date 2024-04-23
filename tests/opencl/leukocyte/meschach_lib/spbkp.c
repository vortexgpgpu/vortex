
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
  Sparse matrix Bunch--Kaufman--Parlett factorisation and solve
  Radical revision started Thu 05th Nov 1992, 09:36:12 AM
  to use Karen George's suggestion of leaving the the row elements unordered
  Radical revision completed Mon 07th Dec 1992, 10:59:57 AM
*/

static	char	rcsid[] = "$Id: spbkp.c,v 1.6 1996/08/20 19:53:10 stewart Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "sparse2.h"


#ifdef MALLOCDECL
#include <malloc.h>
#endif

#define alpha	0.6403882032022076 /* = (1+sqrt(17))/8 */


#define	btos(x)	((x) ? "TRUE" : "FALSE")

/* assume no use of sqr() uses side-effects */
#define	sqr(x)	((x)*(x))

/* unord_get_idx -- returns index (encoded if entry not allocated)
	of the element of row r with column j
	-- uses linear search */
#ifndef ANSI_C
int	unord_get_idx(r,j)
SPROW	*r;
int	j;
#else
int	unord_get_idx(SPROW *r, int j)
#endif
{
    int		idx;
    row_elt	*e;

    if ( ! r || ! r->elt )
	error(E_NULL,"unord_get_idx");
    for ( idx = 0, e = r->elt; idx < r->len; idx++, e++ )
	if ( e->col == j )
	    break;
    if ( idx >= r->len )
	return -(r->len+2);
    else
	return idx;
}

/* unord_get_val -- returns value of the (i,j) entry of A
	-- same assumptions as unord_get_idx() */
#ifndef ANSI_C
double	unord_get_val(A,i,j)
SPMAT	*A;
int	i, j;
#else
double	unord_get_val(SPMAT *A, int i, int j)
#endif
{
    SPROW	*r;
    int		idx;

    if ( ! A )
	error(E_NULL,"unord_get_val");
    if ( i < 0 || i >= A->m || j < 0 || j >= A->n )
	error(E_BOUNDS,"unord_get_val");

    r = &(A->row[i]);
    idx = unord_get_idx(r,j);
    if ( idx < 0 )
	return 0.0;
    else
	return r->elt[idx].val;
}

	    
/* bkp_swap_elt -- swaps the (i,j) with the (k,l) entry of sparse matrix
	-- either or both of the entries may be unallocated */
#ifndef ANSI_C
static SPMAT	*bkp_swap_elt(A,i1,j1,idx1,i2,j2,idx2)
SPMAT	*A;
int	i1, j1, idx1, i2, j2, idx2;
#else
static SPMAT	*bkp_swap_elt(SPMAT *A, int i1, int j1, 
			      int idx1, int i2, int j2, int idx2)
#endif
{
    int		tmp_row, tmp_idx;
    SPROW	*r1, *r2;
    row_elt	*e1, *e2;
    Real	tmp;

    if ( ! A )
	error(E_NULL,"bkp_swap_elt");

    if ( i1 < 0 || j1 < 0 || i2 < 0 || j2 < 0 ||
	 i1 >= A->m || j1 >= A->n || i2 >= A->m || j2 >= A->n )
    {
	error(E_BOUNDS,"bkp_swap_elt");
    }

    if ( i1 == i2 && j1 == j2 )
	return A;
    if ( idx1 < 0 && idx2 < 0 )	/* neither allocated */
	return A;

    r1 = &(A->row[i1]);		r2 = &(A->row[i2]);
    /* if ( idx1 >= r1->len || idx2 >= r2->len )
	error(E_BOUNDS,"bkp_swap_elt"); */
    if ( idx1 < 0 )	/* assume not allocated */
    {
	idx1 = r1->len;
	if ( idx1 >= r1->maxlen )
	{    tracecatch(sprow_xpd(r1,2*r1->maxlen+1,TYPE_SPMAT),
			"bkp_swap_elt");	}
	r1->len = idx1+1;
	r1->elt[idx1].col = j1;
	r1->elt[idx1].val = 0.0;
	/* now patch up column access path */
	tmp_row = -1;	tmp_idx = j1;
	chase_col(A,j1,&tmp_row,&tmp_idx,i1-1);

	if ( tmp_row < 0 )
	{
	    r1->elt[idx1].nxt_row = A->start_row[j1];
	    r1->elt[idx1].nxt_idx = A->start_idx[j1];
	    A->start_row[j1] = i1;
	    A->start_idx[j1] = idx1;
	}
	else
	{
	    row_elt	*tmp_e;

	    tmp_e = &(A->row[tmp_row].elt[tmp_idx]);
	    r1->elt[idx1].nxt_row = tmp_e->nxt_row;
	    r1->elt[idx1].nxt_idx = tmp_e->nxt_idx;
	    tmp_e->nxt_row = i1;
	    tmp_e->nxt_idx = idx1;
	}
    }
    else if ( r1->elt[idx1].col != j1 )
	error(E_INTERN,"bkp_swap_elt");
    if ( idx2 < 0 )
    {
	idx2 = r2->len;
	if ( idx2 >= r2->maxlen )
	{    tracecatch(sprow_xpd(r2,2*r2->maxlen+1,TYPE_SPMAT),
			"bkp_swap_elt");	}

	r2->len = idx2+1;
	r2->elt[idx2].col = j2;
	r2->elt[idx2].val = 0.0;
	/* now patch up column access path */
	tmp_row = -1;	tmp_idx = j2;
	chase_col(A,j2,&tmp_row,&tmp_idx,i2-1);
	if ( tmp_row < 0 )
	{
	    r2->elt[idx2].nxt_row = A->start_row[j2];
	    r2->elt[idx2].nxt_idx = A->start_idx[j2];
	    A->start_row[j2] = i2;
	    A->start_idx[j2] = idx2;
	}
	else
	{
	    row_elt	*tmp_e;

	    tmp_e = &(A->row[tmp_row].elt[tmp_idx]);
	    r2->elt[idx2].nxt_row = tmp_e->nxt_row;
	    r2->elt[idx2].nxt_idx = tmp_e->nxt_idx;
	    tmp_e->nxt_row = i2;
	    tmp_e->nxt_idx = idx2;
	}
    }
    else if ( r2->elt[idx2].col != j2 )
	error(E_INTERN,"bkp_swap_elt");

    e1 = &(r1->elt[idx1]);	e2 = &(r2->elt[idx2]);

    tmp = e1->val;
    e1->val = e2->val;
    e2->val = tmp;

    return A;
}

/* bkp_bump_col -- bumps row and idx to next entry in column j */
#ifndef ANSI_C
row_elt	*bkp_bump_col(A, j, row, idx)
SPMAT	*A;
int	j, *row, *idx;
#else
row_elt	*bkp_bump_col(SPMAT *A, int j, int *row, int *idx)
#endif
{
    SPROW	*r;
    row_elt	*e;

    if ( *row < 0 )
    {
	*row = A->start_row[j];
	*idx = A->start_idx[j];
    }
    else
    {
	r = &(A->row[*row]);
	e = &(r->elt[*idx]);
	if ( e->col != j )
	    error(E_INTERN,"bkp_bump_col");
	*row = e->nxt_row;
	*idx = e->nxt_idx;
    }
    if ( *row < 0 )
	return (row_elt *)NULL;
    else
	return &(A->row[*row].elt[*idx]);
}

/* bkp_interchange -- swap rows/cols i and j (symmetric pivot)
	-- uses just the upper triangular part */
#ifndef ANSI_C
SPMAT	*bkp_interchange(A, i1, i2)
SPMAT	*A;
int	i1, i2;
#else
SPMAT	*bkp_interchange(SPMAT *A, int i1, int i2)
#endif
{
    int		tmp_row, tmp_idx;
    int		row1, row2, idx1, idx2, tmp_row1, tmp_idx1, tmp_row2, tmp_idx2;
    SPROW	*r1, *r2;
    row_elt	*e1, *e2;
    IVEC	*done_list = IVNULL;

    if ( ! A )
	error(E_NULL,"bkp_interchange");
    if ( i1 < 0 || i1 >= A->n || i2 < 0 || i2 >= A->n )
	error(E_BOUNDS,"bkp_interchange");
    if ( A->m != A->n )
	error(E_SQUARE,"bkp_interchange");

    if ( i1 == i2 )
	return A;
    if ( i1 > i2 )
    {	tmp_idx = i1;	i1 = i2;	i2 = tmp_idx;	}

    done_list = iv_resize(done_list,A->n);
    for ( tmp_idx = 0; tmp_idx < A->n; tmp_idx++ )
	done_list->ive[tmp_idx] = FALSE;
    row1 = -1;		idx1 = i1;
    row2 = -1;		idx2 = i2;
    e1 = bkp_bump_col(A,i1,&row1,&idx1);
    e2 = bkp_bump_col(A,i2,&row2,&idx2);

    while ( (row1 >= 0 && row1 < i1) || (row2 >= 0 && row2 < i1) )
	/* Note: "row2 < i1" not "row2 < i2" as we must stop before the
	   "knee bend" */
    {
	if ( row1 >= 0 && row1 < i1 && ( row1 < row2 || row2 < 0 ) )
	{
	    tmp_row1 = row1;	tmp_idx1 = idx1;
	    e1 = bkp_bump_col(A,i1,&tmp_row1,&tmp_idx1);
	    if ( ! done_list->ive[row1] )
	    {
		if ( row1 == row2 )
		    bkp_swap_elt(A,row1,i1,idx1,row1,i2,idx2);
		else
		    bkp_swap_elt(A,row1,i1,idx1,row1,i2,-1);
		done_list->ive[row1] = TRUE;
	    }
	    row1 = tmp_row1;	idx1 = tmp_idx1;
	}
	else if ( row2 >= 0 && row2 < i1 && ( row2 < row1 || row1 < 0 ) )
	{
	    tmp_row2 = row2;	tmp_idx2 = idx2;
	    e2 = bkp_bump_col(A,i2,&tmp_row2,&tmp_idx2);
	    if ( ! done_list->ive[row2] )
	    {
		if ( row1 == row2 )
		    bkp_swap_elt(A,row2,i1,idx1,row2,i2,idx2);
		else
		    bkp_swap_elt(A,row2,i1,-1,row2,i2,idx2);
		done_list->ive[row2] = TRUE;
	    }
	    row2 = tmp_row2;	idx2 = tmp_idx2;
	}
	else if ( row1 == row2 )
	{
	    tmp_row1 = row1;	tmp_idx1 = idx1;
	    e1 = bkp_bump_col(A,i1,&tmp_row1,&tmp_idx1);
	    tmp_row2 = row2;	tmp_idx2 = idx2;
	    e2 = bkp_bump_col(A,i2,&tmp_row2,&tmp_idx2);
	    if ( ! done_list->ive[row1] )
	    {
		bkp_swap_elt(A,row1,i1,idx1,row2,i2,idx2);
		done_list->ive[row1] = TRUE;
	    }
	    row1 = tmp_row1;	idx1 = tmp_idx1;
	    row2 = tmp_row2;	idx2 = tmp_idx2;
	}
    }

    /* ensure we are **past** the first knee */
    while ( row2 >= 0 && row2 <= i1 )
	e2 = bkp_bump_col(A,i2,&row2,&idx2);

    /* at/after 1st "knee bend" */
    r1 = &(A->row[i1]);
    idx1 = 0;
    e1 = &(r1->elt[idx1]);
    while ( row2 >= 0 && row2 < i2 )
    {
	/* used for update of e2 at end of loop */
	tmp_row = row2;	tmp_idx = idx2;
	if ( ! done_list->ive[row2] )
	{
	    r2 = &(A->row[row2]);
	    bkp_bump_col(A,i2,&tmp_row,&tmp_idx);
	    done_list->ive[row2] = TRUE;
	    tmp_idx1 = unord_get_idx(r1,row2);
	    tracecatch(bkp_swap_elt(A,row2,i2,idx2,i1,row2,tmp_idx1),
		       "bkp_interchange");
	}

	/* update e1 and e2 */
	row2 = tmp_row;	idx2 = tmp_idx;
	e2 = ( row2 >= 0 ) ? &(A->row[row2].elt[idx2]) : (row_elt *)NULL;
    }

    idx1 = 0;
    e1 = r1->elt;
    while ( idx1 < r1->len )
    {
	if ( e1->col >= i2 || e1->col <= i1 )
	{
	    idx1++;
	    e1++;
	    continue;
	}
	if ( ! done_list->ive[e1->col] )
	{
	    tmp_idx2 = unord_get_idx(&(A->row[e1->col]),i2);
	    tracecatch(bkp_swap_elt(A,i1,e1->col,idx1,e1->col,i2,tmp_idx2),
		       "bkp_interchange");
	    done_list->ive[e1->col] = TRUE;
	}
	idx1++;
	e1++;
    }

    /* at/after 2nd "knee bend" */
    idx1 = 0;
    e1 = &(r1->elt[idx1]);
    r2 = &(A->row[i2]);
    idx2 = 0;
    e2 = &(r2->elt[idx2]);
    while ( idx1 < r1->len )
    {
	if ( e1->col <= i2 )
	{
	    idx1++;	e1++;
	    continue;
	}
	if ( ! done_list->ive[e1->col] )
	{
	    tmp_idx2 = unord_get_idx(r2,e1->col);
	    tracecatch(bkp_swap_elt(A,i1,e1->col,idx1,i2,e1->col,tmp_idx2),
		       "bkp_interchange");
	    done_list->ive[e1->col] = TRUE;
	}
	idx1++;	e1++;
    }

    idx2 = 0;	e2 = r2->elt;
    while ( idx2 < r2->len )
    {
	if ( e2->col <= i2 )
	{
	    idx2++;	e2++;
	    continue;
	}
	if ( ! done_list->ive[e2->col] )
	{
	    tmp_idx1 = unord_get_idx(r1,e2->col);
	    tracecatch(bkp_swap_elt(A,i2,e2->col,idx2,i1,e2->col,tmp_idx1),
		       "bkp_interchange");
	    done_list->ive[e2->col] = TRUE;
	}
	idx2++;	e2++;
    }

    /* now interchange the digonal entries! */
    idx1 = unord_get_idx(&(A->row[i1]),i1);
    idx2 = unord_get_idx(&(A->row[i2]),i2);
    if ( idx1 >= 0 || idx2 >= 0 )
    {
	tracecatch(bkp_swap_elt(A,i1,i1,idx1,i2,i2,idx2),
		   "bkp_interchange");
    }

    return A;
}


/* iv_min -- returns minimum of an integer vector
   -- sets index to the position in iv if index != NULL */
#ifndef ANSI_C
int	iv_min(iv,index)
IVEC	*iv;
int	*index;
#else
int	iv_min(IVEC *iv, int *index)
#endif
{
    int		i, i_min, min_val, tmp;
    
    if ( ! iv ) 
	error(E_NULL,"iv_min");
    if ( iv->dim <= 0 )
	error(E_SIZES,"iv_min");
    i_min = 0;
    min_val = iv->ive[0];
    for ( i = 1; i < iv->dim; i++ )
    {
	tmp = iv->ive[i];
	if ( tmp < min_val )
	{
	    min_val = tmp;
	    i_min = i;
	}
    }
    
    if ( index != (int *)NULL )
	*index = i_min;
    
    return min_val;
}

/* max_row_col -- returns max { |A[j][k]| : k >= i, k != j, k != l } given j
	using symmetry and only the upper triangular part of A */
#ifndef ANSI_C
static double max_row_col(A,i,j,l)
SPMAT	*A;
int	i, j, l;
#else
static double max_row_col(SPMAT *A, int i,int j, int l)
#endif
{
    int		row_num, idx;
    SPROW	*r;
    row_elt	*e;
    Real	max_val, tmp;

    if ( ! A )
	error(E_NULL,"max_row_col");
    if ( i < 0 || i > A->n || j < 0 || j >= A->n )
	error(E_BOUNDS,"max_row_col");

    max_val = 0.0;

    idx = unord_get_idx(&(A->row[i]),j);
    if ( idx < 0 )
    {
	row_num = -1;	idx = j;
	e = chase_past(A,j,&row_num,&idx,i);
    }
    else
    {
	row_num = i;
	e = &(A->row[i].elt[idx]);
    }
    while ( row_num >= 0 && row_num < j )
    {
	if ( row_num != l )
	{
	    tmp = fabs(e->val);
	    if ( tmp > max_val )
		max_val = tmp;
	}
	e = bump_col(A,j,&row_num,&idx);
    }
    r = &(A->row[j]);
    for ( idx = 0, e = r->elt; idx < r->len; idx++, e++ )
    {
	if ( e->col > j && e->col != l )
	{
	    tmp = fabs(e->val);
	    if ( tmp > max_val )
		max_val = tmp;
	}
    }

    return max_val;
}

/* nonzeros -- counts non-zeros in A */
#ifndef ANSI_C
static int	nonzeros(A)
SPMAT	*A;
#else
static int	nonzeros(const SPMAT *A)
#endif
{
    int		cnt, i;

    if ( ! A )
	return 0;
    cnt = 0;
    for ( i = 0; i < A->m; i++ )
	cnt += A->row[i].len;

    return cnt;
}

/* chk_col_access -- for spBKPfactor()
	-- checks that column access path is OK */
#ifndef ANSI_C
int	chk_col_access(A)
SPMAT	*A;
#else
int	chk_col_access(const SPMAT *A)
#endif
{
    int		cnt_nz, j, row, idx;
    SPROW	*r;
    row_elt	*e;

    if ( ! A )
	error(E_NULL,"chk_col_access");

    /* count nonzeros as we go down columns */
    cnt_nz = 0;
    for ( j = 0; j < A->n; j++ )
    {
	row = A->start_row[j];
	idx = A->start_idx[j];
	while ( row >= 0 )
	{
	    if ( row >= A->m || idx < 0 )
		return FALSE;
	    r = &(A->row[row]);
	    if ( idx >= r->len )
		return FALSE;
	    e = &(r->elt[idx]);
	    if ( e->nxt_row >= 0 && e->nxt_row <= row )
		return FALSE;
	    row = e->nxt_row;
	    idx = e->nxt_idx;
	    cnt_nz++;
	}
    }

    if ( cnt_nz != nonzeros(A) )
	return FALSE;
    else
	return TRUE;
}

/* col_cmp -- compare two columns -- for sorting rows using qsort() */
#ifndef ANSI_C
static int	col_cmp(e1,e2)
row_elt	*e1, *e2;
#else
static int	col_cmp(const row_elt *e1, const row_elt *e2)
#endif
{
    return e1->col - e2->col;
}

/* spBKPfactor -- sparse Bunch-Kaufman-Parlett factorisation of A in-situ
   -- A is factored into the form P'AP = MDM' where 
   P is a permutation matrix, M lower triangular and D is block
   diagonal with blocks of size 1 or 2
   -- P is stored in pivot; blocks[i]==i iff D[i][i] is a block */
#ifndef ANSI_C
SPMAT	*spBKPfactor(A,pivot,blocks,tol)
SPMAT	*A;
PERM	*pivot, *blocks;
double	tol;
#else
SPMAT	*spBKPfactor(SPMAT *A, PERM *pivot, PERM *blocks, double tol)
#endif
{
    int		i, j, k, l, n, onebyone, r;
    int		idx, idx1, idx_piv;
    int		row_num;
    int		best_deg, best_j, best_l, best_cost, mark_cost, deg, deg_j,
			deg_l, ignore_deg;
    int		list_idx, list_idx2, old_list_idx;
    SPROW	*row, *r_piv, *r1_piv;
    row_elt	*e, *e1;
    Real	aii, aip1, aip1i;
    Real	det, max_j, max_l, s, t;
    STATIC IVEC	*scan_row = IVNULL, *scan_idx = IVNULL, *col_list = IVNULL,
		*tmp_iv = IVNULL;
    STATIC IVEC *deg_list = IVNULL;
    STATIC IVEC	*orig_idx = IVNULL, *orig1_idx = IVNULL;
    STATIC PERM	*order = PNULL;

    if ( ! A || ! pivot || ! blocks )
	error(E_NULL,"spBKPfactor");
    if ( A->m != A->n )
	error(E_SQUARE,"spBKPfactor");
    if ( A->m != pivot->size || pivot->size != blocks->size )
	error(E_SIZES,"spBKPfactor");
    if ( tol <= 0.0 || tol > 1.0 )
	error(E_RANGE,"spBKPfactor");
    
    n = A->n;
    
    px_ident(pivot);	px_ident(blocks);
    sp_col_access(A);	sp_diag_access(A);
    ignore_deg = FALSE;

    deg_list = iv_resize(deg_list,n);
    if ( order != NULL )
      px_ident(order);
    order = px_resize(order,n);
    MEM_STAT_REG(deg_list,TYPE_IVEC);
    MEM_STAT_REG(order,TYPE_PERM);

    scan_row = iv_resize(scan_row,5);
    scan_idx = iv_resize(scan_idx,5);
    col_list = iv_resize(col_list,5);
    orig_idx = iv_resize(orig_idx,5);
    orig_idx = iv_resize(orig1_idx,5);
    orig_idx = iv_resize(tmp_iv,5);
    MEM_STAT_REG(scan_row,TYPE_IVEC);
    MEM_STAT_REG(scan_idx,TYPE_IVEC);
    MEM_STAT_REG(col_list,TYPE_IVEC);
    MEM_STAT_REG(orig_idx,TYPE_IVEC);
    MEM_STAT_REG(orig1_idx,TYPE_IVEC);
    MEM_STAT_REG(tmp_iv,TYPE_IVEC);

    for ( i = 0; i < n-1; i = onebyone ? i+1 : i+2 )
    {
	/* now we want to use a Markowitz-style selection rule for
	   determining which rows to swap and whether to use
	   1x1 or 2x2 pivoting */

	/* get list of degrees of nodes */
	deg_list = iv_resize(deg_list,n-i);
	if ( ! ignore_deg )
	    for ( j = i; j < n; j++ )
		deg_list->ive[j-i] = 0;
	else
	{
	    for ( j = i; j < n; j++ )
		deg_list->ive[j-i] = 1;
	    if ( i < n )
		deg_list->ive[0] = 0;
	}
	order = px_resize(order,n-i);
	px_ident(order);

	if ( ! ignore_deg )
	{
	    for ( j = i; j < n; j++ )
	    {
		/* idx = sprow_idx(&(A->row[j]),j+1); */
		/* idx = fixindex(idx); */
		idx = 0;
		row = &(A->row[j]);
		e = &(row->elt[idx]);
		/* deg_list->ive[j-i] += row->len - idx; */
		for ( ; idx < row->len; idx++, e++ )
		    if ( e->col >= i )
			deg_list->ive[e->col - i]++;
	    }
	    /* now deg_list[k] == degree of node k+i */
	    
	    /* now sort them into increasing order */
	    iv_sort(deg_list,order);
	    /* now deg_list[idx] == degree of node i+order[idx] */
	}

	/* now we can chase through the nodes in order of increasing
	   degree, picking out the ones that satisfy our stability
	   criterion */
	list_idx = 0;	r = -1;
	best_j = best_l = -1;
	for ( deg = 0; deg <= n; deg++ )
	{
	    Real	ajj, all, ajl;

	    if ( list_idx >= deg_list->dim )
		break;	/* That's all folks! */
	    old_list_idx = list_idx;
	    while ( list_idx < deg_list->dim &&
		    deg_list->ive[list_idx] <= deg )
	    {
		j = i+order->pe[list_idx];
		if ( j < i )
		    continue;
		/* can we use row/col j for a 1 x 1 pivot? */
		/* find max_j = max_{k>=i} {|A[k][j]|,|A[j][k]|} */
		ajj = fabs(unord_get_val(A,j,j));
		if ( ajj == 0.0 )
		{
		    list_idx++;
		    continue;	/* can't use this for 1 x 1 pivot */
		}

		max_j = max_row_col(A,i,j,-1);
		if ( ajj >= tol/* *alpha */ *max_j )
		{
		    onebyone = TRUE;
		    best_j = j;
		    best_deg = deg_list->ive[list_idx];
		    break;
		}
		list_idx++;
	    }
	    if ( best_j >= 0 )
		break;
	    best_cost = 2*n;	/* > any possible Markowitz cost (bound) */
	    best_j = best_l = -1;
	    list_idx = old_list_idx;
	    while ( list_idx < deg_list->dim &&
		    deg_list->ive[list_idx] <= deg )
	    {
		j = i+order->pe[list_idx];
		ajj = fabs(unord_get_val(A,j,j));
		for ( list_idx2 = 0; list_idx2 < list_idx; list_idx2++ )
		{
		    deg_j = deg;
		    deg_l = deg_list->ive[list_idx2];
		    l = i+order->pe[list_idx2];
		    if ( l < i )
			continue;
		    /* try using rows/cols (j,l) for a 2 x 2 pivot block */
		    all = fabs(unord_get_val(A,l,l));
		    ajl = ( j > l ) ? fabs(unord_get_val(A,l,j)) :
					   fabs(unord_get_val(A,j,l));
		    det = fabs(ajj*all - ajl*ajl);
		    if ( det == 0.0 )
			continue;
		    max_j = max_row_col(A,i,j,l);
		    max_l = max_row_col(A,i,l,j);
		    if ( tol*(all*max_j+ajl*max_l) < det &&
			 tol*(ajl*max_j+ajj*max_l) < det )
		    {
			/* acceptably stable 2 x 2 pivot */
			/* this is actually an overestimate of the
			   Markowitz cost for choosing (j,l) */
			mark_cost = (ajj == 0.0) ?
			    ((all == 0.0) ? deg_j+deg_l : deg_j+2*deg_l) :
				((all == 0.0) ? 2*deg_j+deg_l :
				 2*(deg_j+deg_l));
			if ( mark_cost < best_cost )
			{
			    onebyone = FALSE;
			    best_cost = mark_cost;
			    best_j = j;
			    best_l = l;
			    best_deg = deg_j;
			}
		    }
		}
		list_idx++;
	    }
	    if ( best_j >= 0 )
		break;
	}

	if ( best_deg > (int)floor(0.8*(n-i)) )
	    ignore_deg = TRUE;

	/* now do actual interchanges */
	if ( best_j >= 0 && onebyone )
	{
	    bkp_interchange(A,i,best_j);
	    px_transp(pivot,i,best_j);
	}
	else if ( best_j >= 0 && best_l >= 0 && ! onebyone )
	{
	    if ( best_j == i || best_j == i+1 )
	    {
		if ( best_l == i || best_l == i+1 )
		{
		    /* no pivoting, but must update blocks permutation */
		    px_transp(blocks,i,i+1);
		    goto dopivot;
		}
		bkp_interchange(A,(best_j == i) ? i+1 : i,best_l);
		px_transp(pivot,(best_j == i) ? i+1 : i,best_l);
	    }
	    else if ( best_l == i || best_l == i+1 )
	    {
		bkp_interchange(A,(best_l == i) ? i+1 : i,best_j);
		px_transp(pivot,(best_l == i) ? i+1 : i,best_j);
	    }
	    else /* best_j & best_l outside i, i+1 */
	    {
		if ( i != best_j )
		{
		    bkp_interchange(A,i,best_j);
		    px_transp(pivot,i,best_j);
		}
		if ( i+1 != best_l )
		{
		    bkp_interchange(A,i+1,best_l);
		    px_transp(pivot,i+1,best_l);
		}
	    }
	}
	else	/* can't pivot &/or nothing to pivot */
	    continue;

	/* update blocks permutation */
	if ( ! onebyone )
	    px_transp(blocks,i,i+1);

	dopivot:
	if ( onebyone )
	{
	    int		idx_j, idx_k, s_idx, s_idx2;
	    row_elt	*e_ij, *e_ik;

	    r_piv = &(A->row[i]);
	    idx_piv = unord_get_idx(r_piv,i);
	    /* if idx_piv < 0 then aii == 0 and no pivoting can be done;
	       -- this means that we should continue to the next iteration */
	    if ( idx_piv < 0 )
		continue;
	    aii = r_piv->elt[idx_piv].val;
	    if ( aii == 0.0 )
		continue;

	    /* for ( j = i+1; j < n; j++ )  { ... pivot step ... } */
	    /* initialise scan_... etc for the 1 x 1 pivot */
	    scan_row = iv_resize(scan_row,r_piv->len);
	    scan_idx = iv_resize(scan_idx,r_piv->len);
	    col_list = iv_resize(col_list,r_piv->len);
	    orig_idx = iv_resize(orig_idx,r_piv->len);
	    row_num = i;	s_idx = idx = 0;
	    e = &(r_piv->elt[idx]);
	    for ( idx = 0; idx < r_piv->len; idx++, e++ )
	    {
		if ( e->col < i )
		    continue;
		scan_row->ive[s_idx] = i;
		scan_idx->ive[s_idx] = idx;
		orig_idx->ive[s_idx] = idx;
		col_list->ive[s_idx] = e->col;
		s_idx++;
	    }
	    scan_row = iv_resize(scan_row,s_idx);
	    scan_idx = iv_resize(scan_idx,s_idx);
	    col_list = iv_resize(col_list,s_idx);
	    orig_idx = iv_resize(orig_idx,s_idx);

	    order = px_resize(order,scan_row->dim);
	    px_ident(order);
	    iv_sort(col_list,order);

	    tmp_iv = iv_resize(tmp_iv,scan_row->dim);
	    for ( idx = 0; idx < order->size; idx++ )
		tmp_iv->ive[idx] = scan_idx->ive[order->pe[idx]];
	    iv_copy(tmp_iv,scan_idx);
	    for ( idx = 0; idx < order->size; idx++ )
		tmp_iv->ive[idx] = scan_row->ive[order->pe[idx]];
	    iv_copy(tmp_iv,scan_row);
	    for ( idx = 0; idx < scan_row->dim; idx++ )
		tmp_iv->ive[idx] = orig_idx->ive[order->pe[idx]];
	    iv_copy(tmp_iv,orig_idx);

	    /* now do actual pivot */
	    /* for ( j = i+1; j < n-1; j++ ) .... */

	    for ( s_idx = 0; s_idx < scan_row->dim; s_idx++ )
	    {
		idx_j = orig_idx->ive[s_idx];
		if ( idx_j < 0 )
		    error(E_INTERN,"spBKPfactor");
		e_ij = &(r_piv->elt[idx_j]);
		j = e_ij->col;
		if ( j < i+1 )
		    continue;
		scan_to(A,scan_row,scan_idx,col_list,j);

		/* compute multiplier */
		t = e_ij->val / aii;

		/* for ( k = j; k < n; k++ ) { .... update A[j][k] .... } */
		/* this is the row in which pivoting is done */
		row = &(A->row[j]);
		for ( s_idx2 = s_idx; s_idx2 < scan_row->dim; s_idx2++ )
		{
		    idx_k = orig_idx->ive[s_idx2];
		    e_ik = &(r_piv->elt[idx_k]);
		    k = e_ik->col;
		    /* k >= j since col_list has been sorted */

		    if ( scan_row->ive[s_idx2] == j )
		    {	/* no fill-in -- can be done directly */
			idx = scan_idx->ive[s_idx2];
			/* idx = sprow_idx2(row,k,idx); */
			row->elt[idx].val -= t*e_ik->val;
		    }
		    else
		    {	/* fill-in -- insert entry & patch column */
			int	old_row, old_idx;
			row_elt	*old_e, *new_e;

			old_row = scan_row->ive[s_idx2];
			old_idx = scan_idx->ive[s_idx2];
			/* old_idx = sprow_idx2(&(A->row[old_row]),k,old_idx); */

			if ( old_idx < 0 )
			    error(E_INTERN,"spBKPfactor");
			/* idx = sprow_idx(row,k); */
			/* idx = fixindex(idx); */
			idx = row->len;

			/* sprow_set_val(row,k,-t*e_ik->val); */
			if ( row->len >= row->maxlen )
			{ tracecatch(sprow_xpd(row,2*row->maxlen+1,TYPE_SPMAT),
				     "spBKPfactor");		}

			row->len = idx+1;

			new_e = &(row->elt[idx]);
			new_e->val = -t*e_ik->val;
			new_e->col = k;

			old_e = &(A->row[old_row].elt[old_idx]);
			new_e->nxt_row = old_e->nxt_row;
			new_e->nxt_idx = old_e->nxt_idx;
			old_e->nxt_row = j;
			old_e->nxt_idx = idx;
		    }
		}
		e_ij->val = t;
	    }
	}
	else /* onebyone == FALSE */
	{	/* do 2 x 2 pivot */
	    int	idx_k, idx1_k, s_idx, s_idx2;
	    int	old_col;
	    row_elt	*e_tmp;

	    r_piv = &(A->row[i]);
	    idx_piv = unord_get_idx(r_piv,i);
	    aii = aip1i = 0.0;
	    e_tmp = r_piv->elt;
	    for ( idx_piv = 0; idx_piv < r_piv->len; idx_piv++, e_tmp++ )
		if ( e_tmp->col == i )
		    aii = e_tmp->val;
	        else if ( e_tmp->col == i+1 )
		    aip1i = e_tmp->val;

	    r1_piv = &(A->row[i+1]);
	    e_tmp = r1_piv->elt;
	    aip1 = unord_get_val(A,i+1,i+1);
	    det = aii*aip1 - aip1i*aip1i;	/* Must have det < 0 */
	    if ( aii == 0.0 && aip1i == 0.0 )
	    {
		/* error(E_RANGE,"spBKPfactor"); */
		onebyone = TRUE;
		continue;	/* cannot pivot */
	    }

	    if ( det == 0.0 )
	    {
		if ( aii != 0.0 )
		    error(E_RANGE,"spBKPfactor");
		onebyone = TRUE;
		continue;	/* cannot pivot */
	    }
	    aip1i = aip1i/det;
	    aii = aii/det;
	    aip1 = aip1/det;
	    
	    /* initialise scan_... etc for the 2 x 2 pivot */
	    s_idx = r_piv->len + r1_piv->len;
	    scan_row = iv_resize(scan_row,s_idx);
	    scan_idx = iv_resize(scan_idx,s_idx);
	    col_list = iv_resize(col_list,s_idx);
	    orig_idx = iv_resize(orig_idx,s_idx);
	    orig1_idx = iv_resize(orig1_idx,s_idx);

	    e = r_piv->elt;
	    for ( idx = 0; idx < r_piv->len; idx++, e++ )
	    {
		scan_row->ive[idx] = i;
		scan_idx->ive[idx] = idx;
		col_list->ive[idx] = e->col;
		orig_idx->ive[idx] = idx;
		orig1_idx->ive[idx] = -1;
	    }
	    e = r_piv->elt;
	    e1 = r1_piv->elt;
	    for ( idx = 0; idx < r1_piv->len; idx++, e1++ )
	    {
		scan_row->ive[idx+r_piv->len] = i+1;
		scan_idx->ive[idx+r_piv->len] = idx;
		col_list->ive[idx+r_piv->len] = e1->col;
		orig_idx->ive[idx+r_piv->len] = -1;
		orig1_idx->ive[idx+r_piv->len] = idx;
	    }

	    e1 = r1_piv->elt;
	    order = px_resize(order,scan_row->dim);
	    px_ident(order);
	    iv_sort(col_list,order);
	    tmp_iv = iv_resize(tmp_iv,scan_row->dim);
	    for ( idx = 0; idx < order->size; idx++ )
		tmp_iv->ive[idx] = scan_idx->ive[order->pe[idx]];
	    iv_copy(tmp_iv,scan_idx);
	    for ( idx = 0; idx < order->size; idx++ )
		tmp_iv->ive[idx] = scan_row->ive[order->pe[idx]];
	    iv_copy(tmp_iv,scan_row);
	    for ( idx = 0; idx < scan_row->dim; idx++ )
		tmp_iv->ive[idx] = orig_idx->ive[order->pe[idx]];
	    iv_copy(tmp_iv,orig_idx);
	    for ( idx = 0; idx < scan_row->dim; idx++ )
		tmp_iv->ive[idx] = orig1_idx->ive[order->pe[idx]];
	    iv_copy(tmp_iv,orig1_idx);

	    s_idx = 0;
	    old_col = -1;
	    for ( idx = 0; idx < scan_row->dim; idx++ )
	    {
		if ( col_list->ive[idx] == old_col )
		{
		    if ( scan_row->ive[idx] == i )
		    {
			scan_row->ive[s_idx-1] = scan_row->ive[idx];
			scan_idx->ive[s_idx-1] = scan_idx->ive[idx];
			col_list->ive[s_idx-1] = col_list->ive[idx];
			orig_idx->ive[s_idx-1] = orig_idx->ive[idx];
			orig1_idx->ive[s_idx-1] = orig1_idx->ive[idx-1];
		    }
		    else if ( idx > 0 )
		    {
			scan_row->ive[s_idx-1] = scan_row->ive[idx-1];
			scan_idx->ive[s_idx-1] = scan_idx->ive[idx-1];
			col_list->ive[s_idx-1] = col_list->ive[idx-1];
			orig_idx->ive[s_idx-1] = orig_idx->ive[idx-1];
			orig1_idx->ive[s_idx-1] = orig1_idx->ive[idx];
		    }
		}
		else
		{
		    scan_row->ive[s_idx] = scan_row->ive[idx];
		    scan_idx->ive[s_idx] = scan_idx->ive[idx];
		    col_list->ive[s_idx] = col_list->ive[idx];
		    orig_idx->ive[s_idx] = orig_idx->ive[idx];
		    orig1_idx->ive[s_idx] = orig1_idx->ive[idx];
		    s_idx++;
		}
		old_col = col_list->ive[idx];
	    }
	    scan_row = iv_resize(scan_row,s_idx);
	    scan_idx = iv_resize(scan_idx,s_idx);
	    col_list = iv_resize(col_list,s_idx);
	    orig_idx = iv_resize(orig_idx,s_idx);
	    orig1_idx = iv_resize(orig1_idx,s_idx);

	    /* for ( j = i+2; j < n; j++ )  { .... row operation .... } */
	    for ( s_idx = 0; s_idx < scan_row->dim; s_idx++ )
	    {
		int	idx_piv, idx1_piv;
		Real	aip1j, aij, aik, aip1k;
		row_elt	*e_ik, *e_ip1k;

		j = col_list->ive[s_idx];
		if ( j < i+2 )
		    continue;
		tracecatch(scan_to(A,scan_row,scan_idx,col_list,j),
			   "spBKPfactor");

		idx_piv = orig_idx->ive[s_idx];
		aij = ( idx_piv < 0 ) ? 0.0 : r_piv->elt[idx_piv].val;
		/* aij = ( s_idx < r_piv->len ) ? r_piv->elt[s_idx].val :
		    0.0; */
		/* aij   = sp_get_val(A,i,j); */
		idx1_piv = orig1_idx->ive[s_idx];
		aip1j = ( idx1_piv < 0 ) ? 0.0 : r1_piv->elt[idx1_piv].val;
		/* aip1j = ( s_idx < r_piv->len ) ? 0.0 :
		    r1_piv->elt[s_idx-r_piv->len].val; */
		/* aip1j = sp_get_val(A,i+1,j); */
		s = - aip1i*aip1j + aip1*aij;
		t = - aip1i*aij + aii*aip1j;

		/* for ( k = j; k < n; k++ )  { .... update entry .... } */
		row = &(A->row[j]);
		/* set idx_k and idx1_k indices */
		s_idx2 = s_idx;
		k = col_list->ive[s_idx2];
		idx_k = orig_idx->ive[s_idx2];
		idx1_k = orig1_idx->ive[s_idx2];

		while ( s_idx2 < scan_row->dim )
		{
		    k = col_list->ive[s_idx2];
		    idx_k = orig_idx->ive[s_idx2];
		    idx1_k = orig1_idx->ive[s_idx2];
		    e_ik = ( idx_k < 0 ) ? (row_elt *)NULL :
			&(r_piv->elt[idx_k]);
		    e_ip1k = ( idx1_k < 0 ) ? (row_elt *)NULL :
			&(r1_piv->elt[idx1_k]);
		    aik = ( idx_k >= 0 ) ? e_ik->val : 0.0;
		    aip1k = ( idx1_k >= 0 ) ? e_ip1k->val : 0.0;
		    if ( scan_row->ive[s_idx2] == j )
		    {	/* no fill-in */
			row = &(A->row[j]);
			/* idx = sprow_idx(row,k); */
			idx = scan_idx->ive[s_idx2];
			if ( idx < 0 )
			    error(E_INTERN,"spBKPfactor");
			row->elt[idx].val -= s*aik + t*aip1k;
		    }
		    else
		    {	/* fill-in -- insert entry & patch column */
			Real	tmp;
			int	old_row, old_idx;
			row_elt	*old_e, *new_e;

			tmp = - s*aik - t*aip1k;
			if ( tmp != 0.0 )
			{
			    row = &(A->row[j]);
			    old_row = scan_row->ive[s_idx2];
			    old_idx = scan_idx->ive[s_idx2];

			    idx = row->len;
			    if ( row->len >= row->maxlen )
			    {  tracecatch(sprow_xpd(row,2*row->maxlen+1,
						    TYPE_SPMAT),
					   "spBKPfactor");	    }

			    row->len = idx + 1;
			    /* idx = sprow_idx(row,k); */
			    new_e = &(row->elt[idx]);
			    new_e->val = tmp;
			    new_e->col = k;

			    if ( old_row < 0 )
				error(E_INTERN,"spBKPfactor");
			    /* old_idx = sprow_idx2(&(A->row[old_row]),
						  k,old_idx); */
			    old_e = &(A->row[old_row].elt[old_idx]);
			    new_e->nxt_row = old_e->nxt_row;
			    new_e->nxt_idx = old_e->nxt_idx;
			    old_e->nxt_row = j;
			    old_e->nxt_idx = idx;
			}
		    }

		    /* update idx_k, idx1_k, s_idx2 etc */
		    s_idx2++;
		}

		/* store multipliers -- may involve fill-in (!) */
		/* idx = sprow_idx(r_piv,j); */
		idx = orig_idx->ive[s_idx];
		if ( idx >= 0 )
		{
		    r_piv->elt[idx].val = s;
		}
		else if ( s != 0.0 )
		{
		    int		old_row, old_idx;
		    row_elt	*new_e, *old_e;

		    old_row = -1;	old_idx = j;

		    if ( i > 0 )
		    {
			tracecatch(chase_col(A,j,&old_row,&old_idx,i-1),
				   "spBKPfactor");
		    }
		    /* sprow_set_val(r_piv,j,s); */
		    idx = r_piv->len;
		    if ( r_piv->len >= r_piv->maxlen )
		    {	tracecatch(sprow_xpd(r_piv,2*r_piv->maxlen+1,
					     TYPE_SPMAT),
				   "spBKPfactor");		    }

		    r_piv->len = idx + 1;
		    /* idx = sprow_idx(r_piv,j); */
		    /* if ( idx < 0 )
			error(E_INTERN,"spBKPfactor"); */
		    new_e = &(r_piv->elt[idx]);
		    new_e->val = s;
		    new_e->col = j;
		    if ( old_row < 0 )
		    {
			new_e->nxt_row = A->start_row[j];
			new_e->nxt_idx = A->start_idx[j];
			A->start_row[j] = i;
			A->start_idx[j] = idx;
		    }
		    else
		    {
			/* old_idx = sprow_idx2(&(A->row[old_row]),j,old_idx);*/
			if ( old_idx < 0 )
			    error(E_INTERN,"spBKPfactor");
			old_e = &(A->row[old_row].elt[old_idx]);
			new_e->nxt_row = old_e->nxt_row;
			new_e->nxt_idx = old_e->nxt_idx;
			old_e->nxt_row = i;
			old_e->nxt_idx = idx;
		    }
		}
		/* idx1 = sprow_idx(r1_piv,j); */
		idx1 = orig1_idx->ive[s_idx];
		if ( idx1 >= 0 )
		{
		    r1_piv->elt[idx1].val = t;
		}
		else if ( t != 0.0 )
		{
		    int		old_row, old_idx;
		    row_elt	*new_e, *old_e;

		    old_row = -1;	old_idx = j;
		    tracecatch(chase_col(A,j,&old_row,&old_idx,i),
			       "spBKPfactor");
		    /* sprow_set_val(r1_piv,j,t); */
		    idx1 = r1_piv->len;
		    if ( r1_piv->len >= r1_piv->maxlen )
		    {	tracecatch(sprow_xpd(r1_piv,2*r1_piv->maxlen+1,
					     TYPE_SPMAT),
				   "spBKPfactor");		    }

		    r1_piv->len = idx1 + 1;
		    /* idx1 = sprow_idx(r1_piv,j); */
		    /* if ( idx < 0 )
			error(E_INTERN,"spBKPfactor"); */
		    new_e = &(r1_piv->elt[idx1]);
		    new_e->val = t;
		    new_e->col = j;
		    if ( idx1 < 0 )
			error(E_INTERN,"spBKPfactor");
		    new_e = &(r1_piv->elt[idx1]);
		    if ( old_row < 0 )
		    {
			new_e->nxt_row = A->start_row[j];
			new_e->nxt_idx = A->start_idx[j];
			A->start_row[j] = i+1;
			A->start_idx[j] = idx1;
		    }
		    else
		    {
			old_idx = sprow_idx2(&(A->row[old_row]),j,old_idx);
			if ( old_idx < 0 )
			    error(E_INTERN,"spBKPfactor");
			old_e = &(A->row[old_row].elt[old_idx]);
			new_e->nxt_row = old_e->nxt_row;
			new_e->nxt_idx = old_e->nxt_idx;
			old_e->nxt_row = i+1;
			old_e->nxt_idx = idx1;
		    }
		}
	    }
	}
    }

    /* now sort the rows arrays */
    for ( i = 0; i < A->m; i++ )
	qsort(A->row[i].elt,A->row[i].len,sizeof(row_elt),(int(*)())col_cmp);
    A->flag_col = A->flag_diag = FALSE;

#ifdef	THREADSAFE
    IV_FREE(scan_row);	IV_FREE(scan_idx);	IV_FREE(col_list);
    IV_FREE(tmp_iv);	IV_FREE(deg_list);	IV_FREE(orig_idx);
    IV_FREE(orig1_idx);	PX_FREE(order);
#endif
    return A;
}

/* spBKPsolve -- solves A.x = b where A has been factored a la BKPfactor()
   -- returns x, which is created if NULL */
#ifndef ANSI_C
VEC	*spBKPsolve(A,pivot,block,b,x)
SPMAT	*A;
PERM	*pivot, *block;
VEC	*b, *x;
#else
VEC	*spBKPsolve(SPMAT *A, PERM *pivot, PERM *block,
		    const VEC *b, VEC *x)
#endif
{
    STATIC VEC	*tmp=VNULL;	/* dummy storage needed */
    int		i /* , j */, n, onebyone;
    int		row_num, idx;
    Real	a11, a12, a22, b1, b2, det, sum, *tmp_ve, tmp_diag;
    SPROW	*r;
    row_elt	*e;
    
    if ( ! A || ! pivot || ! block || ! b )
	error(E_NULL,"spBKPsolve");
    if ( A->m != A->n )
	error(E_SQUARE,"spBKPsolve");
    n = A->n;
    if ( b->dim != n || pivot->size != n || block->size != n )
	error(E_SIZES,"spBKPsolve");
    x = v_resize(x,n);
    tmp = v_resize(tmp,n);
    MEM_STAT_REG(tmp,TYPE_VEC);
    
    tmp_ve = tmp->ve;

    if ( ! A->flag_col )
	sp_col_access(A);

    px_vec(pivot,b,tmp);
    /* printf("# BKPsolve: effect of pivot: tmp =\n");	v_output(tmp); */

    /* solve for lower triangular part */
    for ( i = 0; i < n; i++ )
    {
	sum = tmp_ve[i];
	if ( block->pe[i] < i )
	{
	    /* for ( j = 0; j < i-1; j++ )
		  sum -= A_me[j][i]*tmp_ve[j]; */
	    row_num = -1;	idx = i;
	    e = bump_col(A,i,&row_num,&idx);
	    while ( row_num >= 0 && row_num < i-1 )
	    {
		sum -= e->val*tmp_ve[row_num];
		e = bump_col(A,i,&row_num,&idx);
	    }
	}
	else
	{
	    /* for ( j = 0; j < i; j++ )
	          sum -= A_me[j][i]*tmp_ve[j]; */
	    row_num = -1; idx = i;
	    e = bump_col(A,i,&row_num,&idx);
	    while ( row_num >= 0 && row_num < i )
	    {
		sum -= e->val*tmp_ve[row_num];
		e = bump_col(A,i,&row_num,&idx);
	    }
	}
	tmp_ve[i] = sum;
    }

    /* printf("# BKPsolve: solving L part: tmp =\n");	v_output(tmp); */
    /* solve for diagonal part */
    for ( i = 0; i < n; i = onebyone ? i+1 : i+2 )
    {
	onebyone = ( block->pe[i] == i );
	if ( onebyone )
	{
	    /* tmp_ve[i] /= A_me[i][i]; */
	    tmp_diag = sp_get_val(A,i,i);
	    if ( tmp_diag == 0.0 )
		error(E_SING,"spBKPsolve");
	    tmp_ve[i] /= tmp_diag;
	}
	else
	{
	    a11 = sp_get_val(A,i,i);
	    a22 = sp_get_val(A,i+1,i+1);
	    a12 = sp_get_val(A,i,i+1);
	    b1 = tmp_ve[i];
	    b2 = tmp_ve[i+1];
	    det = a11*a22-a12*a12;	/* < 0 : see BKPfactor() */
	    if ( det == 0.0 )
		error(E_SING,"BKPsolve");
	    det = 1/det;
	    tmp_ve[i]   = det*(a22*b1-a12*b2);
	    tmp_ve[i+1] = det*(a11*b2-a12*b1);
	}
    }

    /* printf("# BKPsolve: solving D part: tmp =\n");	v_output(tmp); */
    /* solve for transpose of lower triangular part */
    for ( i = n-2; i >= 0; i-- )
    {
	sum = tmp_ve[i];
	if ( block->pe[i] > i )
	{
	    /* onebyone is false */
	    /* for ( j = i+2; j < n; j++ )
		  sum -= A_me[i][j]*tmp_ve[j]; */
	    if ( i+2 >= n )
		continue;
	    r = &(A->row[i]);
	    idx = sprow_idx(r,i+2);
	    idx = fixindex(idx);
	    e = &(r->elt[idx]);
	    for ( ; idx < r->len; idx++, e++ )
		sum -= e->val*tmp_ve[e->col];
	}
	else /* onebyone */
	{
	    /* for ( j = i+1; j < n; j++ )
		  sum -= A_me[i][j]*tmp_ve[j]; */
	    r = &(A->row[i]);
	    idx = sprow_idx(r,i+1);
	    idx = fixindex(idx);
	    e = &(r->elt[idx]);
	    for ( ; idx < r->len; idx++, e++ )
		sum -= e->val*tmp_ve[e->col];
	}
	tmp_ve[i] = sum;
    }

    /* printf("# BKPsolve: solving L^T part: tmp =\n");v_output(tmp); */
    /* and do final permutation */
    x = pxinv_vec(pivot,tmp,x);

#ifdef	THREADSAFE
    V_FREE(tmp);
#endif

    return x;
}



