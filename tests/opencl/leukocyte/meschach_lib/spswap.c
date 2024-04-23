
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
	Sparse matrix swap and permutation routines
	Modified Mon 09th Nov 1992, 08:50:54 PM
	to use Karen George's suggestion to use unordered rows
*/

static	char	rcsid[] = "$Id: spswap.c,v 1.3 1994/01/13 05:44:43 des Exp $";

#include	<stdio.h>
#include	<math.h>
#include        "sparse2.h"


#define	btos(x)	((x) ? "TRUE" : "FALSE")

/* scan_to -- updates scan (int) vectors to point to the last row in each
	column with row # <= max_row, if any */
#ifndef ANSI_C
void	scan_to(A, scan_row, scan_idx, col_list, max_row)
SPMAT	*A;
IVEC	*scan_row, *scan_idx, *col_list;
int	max_row;
#else
void	scan_to(SPMAT *A, IVEC *scan_row, IVEC *scan_idx, IVEC *col_list, 
		int max_row)
#endif
{
    int		col, idx, j_idx, row_num;
    SPROW	*r;
    row_elt	*e;

    if ( ! A || ! scan_row || ! scan_idx || ! col_list )
	error(E_NULL,"scan_to");
    if ( scan_row->dim != scan_idx->dim || scan_idx->dim != col_list->dim )
	error(E_SIZES,"scan_to");

    if ( max_row < 0 )
	return;

    if ( ! A->flag_col )
	sp_col_access(A);

    for ( j_idx = 0; j_idx < scan_row->dim; j_idx++ )
    {
	row_num = scan_row->ive[j_idx];
	idx = scan_idx->ive[j_idx];
	col = col_list->ive[j_idx];

	if ( col < 0 || col >= A->n )
	    error(E_BOUNDS,"scan_to");
	if ( row_num < 0 )
	{
	    idx = col;
	    continue;
	}
	r = &(A->row[row_num]);
	if ( idx < 0 )
	    error(E_INTERN,"scan_to");
	e = &(r->elt[idx]);
	if ( e->col != col )
	    error(E_INTERN,"scan_to");
	if ( idx < 0 )
	{
	    printf("scan_to: row_num = %d, idx = %d, col = %d\n",
		   row_num, idx, col);
	    error(E_INTERN,"scan_to");
	}
	/* if ( e->nxt_row <= max_row )
	    chase_col(A, col, &row_num, &idx, max_row); */
	while ( e->nxt_row >= 0 && e->nxt_row <= max_row )
	{
	    row_num = e->nxt_row;
	    idx = e->nxt_idx;
	    e = &(A->row[row_num].elt[idx]);
	}
	    
	/* printf("scan_to: computed j_idx = %d, row_num = %d, idx = %d\n",
	       j_idx, row_num, idx); */
	scan_row->ive[j_idx] = row_num;
	scan_idx->ive[j_idx] = idx;
    }
}

/* patch_col -- patches column access paths for fill-in */
#ifndef ANSI_C
void patch_col(A, col, old_row, old_idx, row_num, idx)
SPMAT	*A;
int	col, old_row, old_idx, row_num, idx;
#else
void patch_col(SPMAT *A, int col, int old_row, int old_idx, int row_num, 
	       int idx)
#endif
{
    SPROW	*r;
    row_elt	*e;
    
    if ( old_row >= 0 )
    {
	r = &(A->row[old_row]);
	old_idx = sprow_idx2(r,col,old_idx);
	e = &(r->elt[old_idx]);
	e->nxt_row = row_num;
	e->nxt_idx = idx;
    }
    else
    {
	A->start_row[col] = row_num;
	A->start_idx[col] = idx;
    }
}

/* chase_col -- chases column access path in column col, starting with
   row_num and idx, to find last row # in this column <= max_row
   -- row_num is returned; idx is also set by this routine
   -- assumes that the column access paths (possibly without the
   nxt_idx fields) are set up */
#ifndef ANSI_C
row_elt *chase_col(A, col, row_num, idx, max_row)
SPMAT	*A;
int	col, *row_num, *idx, max_row;
#else
row_elt *chase_col(const SPMAT *A, int col, int *row_num, int *idx, 
		   int max_row)
#endif
{
    int		old_idx, old_row, tmp_idx, tmp_row;
    SPROW	*r;
    row_elt	*e;
    
    if ( col < 0 || col >= A->n )
	error(E_BOUNDS,"chase_col");
    tmp_row = *row_num;
    if ( tmp_row < 0 )
    {
	if ( A->start_row[col] > max_row )
	{
	    tmp_row = -1;
	    tmp_idx = col;
	    return (row_elt *)NULL;
	}
	else
	{
	    tmp_row = A->start_row[col];
	    tmp_idx = A->start_idx[col];
	}
    }
    else
	tmp_idx = *idx;
    
    old_row = tmp_row;
    old_idx = tmp_idx;
    while ( tmp_row >= 0 && tmp_row < max_row )
    {
	r = &(A->row[tmp_row]);
	/* tmp_idx = sprow_idx2(r,col,tmp_idx); */
	if ( tmp_idx < 0 || tmp_idx >= r->len ||
	     r->elt[tmp_idx].col != col )
	{
#ifdef DEBUG
	    printf("chase_col:error: col = %d, row # = %d, idx = %d\n",
		   col, tmp_row, tmp_idx);
	    printf("chase_col:error: old_row = %d, old_idx = %d\n",
		   old_row, old_idx);
	    printf("chase_col:error: A =\n");
	    sp_dump(stdout,A);
#endif
	    error(E_INTERN,"chase_col");
	}
	e = &(r->elt[tmp_idx]);
	old_row = tmp_row;
	old_idx = tmp_idx;
	tmp_row = e->nxt_row;
	tmp_idx = e->nxt_idx;
    }
    if ( old_row > max_row )
    {
	old_row = -1;
	old_idx = col;
	e = (row_elt *)NULL;
    }
    else if ( tmp_row <= max_row && tmp_row >= 0 )
    {
	old_row = tmp_row;
	old_idx = tmp_idx;
    }

    *row_num = old_row;
    if ( old_row >= 0 )
	*idx = old_idx;
    else
	*idx = col;

    return e;
}

/* chase_past -- as for chase_col except that we want the first
	row whose row # >= min_row; -1 indicates no such row */
#ifndef ANSI_C
row_elt *chase_past(A, col, row_num, idx, min_row)
SPMAT	*A;
int	col, *row_num, *idx, min_row;
#else
row_elt *chase_past(const SPMAT *A, int col, int *row_num, int *idx, 
		    int min_row)
#endif
{
    SPROW	*r;
    row_elt	*e;
    int		tmp_idx, tmp_row;

    tmp_row = *row_num;
    tmp_idx = *idx;
    chase_col(A,col,&tmp_row,&tmp_idx,min_row);
    if ( tmp_row < 0 )	/* use A->start_row[..] etc. */
    {
	if ( A->start_row[col] < 0 )
	    tmp_row = -1;
	else
	{
	    tmp_row = A->start_row[col];
	    tmp_idx = A->start_idx[col];
	}
    }
    else if ( tmp_row < min_row )
    {
	r = &(A->row[tmp_row]);
	if ( tmp_idx < 0 || tmp_idx >= r->len ||
	     r->elt[tmp_idx].col != col )
	    error(E_INTERN,"chase_past");
	tmp_row = r->elt[tmp_idx].nxt_row;
	tmp_idx = r->elt[tmp_idx].nxt_idx;
    }

    *row_num = tmp_row;
    *idx = tmp_idx;
    if ( tmp_row < 0 )
	e = (row_elt *)NULL;
    else
    {
	if ( tmp_idx < 0 || tmp_idx >= A->row[tmp_row].len ||
	     A->row[tmp_row].elt[tmp_idx].col != col )
	    error(E_INTERN,"bump_col");
	e = &(A->row[tmp_row].elt[tmp_idx]);
    }

    return e;
}

/* bump_col -- move along to next nonzero entry in column col after row_num
	-- update row_num and idx */
#ifndef ANSI_C
row_elt *bump_col(A, col, row_num, idx)
SPMAT	*A;
int	col, *row_num, *idx;
#else
row_elt *bump_col(const SPMAT *A, int col, int *row_num, int *idx)
#endif
{
    SPROW	*r;
    row_elt	*e;
    int		tmp_row, tmp_idx;

    tmp_row = *row_num;
    tmp_idx = *idx;
    /* printf("bump_col: col = %d, row# = %d, idx = %d\n",
	   col, *row_num, *idx); */
    if ( tmp_row < 0 )
    {
	tmp_row = A->start_row[col];
	tmp_idx = A->start_idx[col];
    }
    else
    {
	r = &(A->row[tmp_row]);
	if ( tmp_idx < 0 || tmp_idx >= r->len ||
	     r->elt[tmp_idx].col != col )
	    error(E_INTERN,"bump_col");
	e = &(r->elt[tmp_idx]);
	tmp_row = e->nxt_row;
	tmp_idx = e->nxt_idx;
    }
    if ( tmp_row < 0 )
    {
	e = (row_elt *)NULL;
	tmp_idx = col;
    }
    else
    {
	if ( tmp_idx < 0 || tmp_idx >= A->row[tmp_row].len ||
	     A->row[tmp_row].elt[tmp_idx].col != col )
	    error(E_INTERN,"bump_col");
	e = &(A->row[tmp_row].elt[tmp_idx]);
    }
    *row_num = tmp_row;
    *idx = tmp_idx;

    return e;
}


