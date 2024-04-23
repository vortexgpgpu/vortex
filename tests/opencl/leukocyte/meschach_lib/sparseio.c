
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
	This file has the routines for sparse matrix input/output
	It works in conjunction with sparse.c, sparse.h etc
*/

#include        <stdio.h>
#include        "sparse.h"

static char rcsid[] = "$Id: sparseio.c,v 1.4 1994/01/13 05:34:25 des Exp $";



/* local variables */
static char line[MAXLINE];

/* sp_foutput -- output sparse matrix A to file/stream fp */
#ifndef ANSI_C
void    sp_foutput(fp,A)
FILE    *fp;
SPMAT  *A;
#else
void    sp_foutput(FILE *fp, const SPMAT *A)
#endif
{
	int     i, j_idx, m /* , n */;
	SPROW  *rows;
	row_elt *elts;

	fprintf(fp,"SparseMatrix: ");
	if ( A == SMNULL )
	{
		fprintf(fp,"*** NULL ***\n");
		error(E_NULL,"sp_foutput");    return;
	}
	fprintf(fp,"%d by %d\n",A->m,A->n);
	m = A->m;       /* n = A->n; */
	if ( ! (rows=A->row) )
	{
		fprintf(fp,"*** NULL rows ***\n");
		error(E_NULL,"sp_foutput");    return;
	}

	for ( i = 0; i < m; i++ )
	{
		fprintf(fp,"row %d: ",i);
		if ( ! (elts=rows[i].elt) )
		{
			fprintf(fp,"*** NULL element list ***\n");
			continue;
		}
		for ( j_idx = 0; j_idx < rows[i].len; j_idx++ )
		{
			fprintf(fp,"%d:%-20.15g ",elts[j_idx].col,
							elts[j_idx].val);
			if ( j_idx % 3 == 2 && j_idx != rows[i].len-1 )
				fprintf(fp,"\n     ");
		}
		fprintf(fp,"\n");
	}
	fprintf(fp,"#\n");	/* to stop looking beyond for next entry */
}

/* sp_foutput2 -- print out sparse matrix **as a dense matrix**
	-- see output format used in matrix.h etc */
/******************************************************************
void    sp_foutput2(fp,A)
FILE    *fp;
SPMAT  *A;
{
	int     cnt, i, j, j_idx;
	SPROW  *r;
	row_elt *elt;

	if ( A == SMNULL )
	{
		fprintf(fp,"Matrix: *** NULL ***\n");
		return;
	}
	fprintf(fp,"Matrix: %d by %d\n",A->m,A->n);
	for ( i = 0; i < A->m; i++ )
	{
		fprintf(fp,"row %d:",i);
		r = &(A->row[i]);
		elt = r->elt;
		cnt = j = j_idx = 0;
		while ( j_idx < r->len || j < A->n )
		{
			if ( j_idx >= r->len )
				fprintf(fp,"%14.9g ",0.0);
			else if ( j < elt[j_idx].col )
				fprintf(fp,"%14.9g ",0.0);
			else
				fprintf(fp,"%14.9g ",elt[j_idx++].val);
			if ( cnt++ % 4 == 3 )
				fprintf(fp,"\n");
			j++;
		}
		fprintf(fp,"\n");
	}
}
******************************************************************/

/* sp_dump -- prints ALL relevant information about the sparse matrix A */
#ifndef ANSI_C
void    sp_dump(fp,A)
FILE    *fp;
SPMAT  *A;
#else
void    sp_dump(FILE *fp, const SPMAT *A)
#endif
{
	int     i, j, j_idx;
	SPROW  *rows;
	row_elt *elts;

	fprintf(fp,"SparseMatrix dump:\n");
	if ( ! A )
	{       fprintf(fp,"*** NULL ***\n");   return; }
	fprintf(fp,"Matrix at 0x%lx\n",(long)A);
	fprintf(fp,"Dimensions: %d by %d\n",A->m,A->n);
	fprintf(fp,"MaxDimensions: %d by %d\n",A->max_m,A->max_n);
	fprintf(fp,"flag_col = %d, flag_diag = %d\n",A->flag_col,A->flag_diag);
	fprintf(fp,"start_row @ 0x%lx:\n",(long)(A->start_row));
	for ( j = 0; j < A->n; j++ )
	{
		fprintf(fp,"%d ",A->start_row[j]);
		if ( j % 10 == 9 )
			fprintf(fp,"\n");
	}
	fprintf(fp,"\n");
	fprintf(fp,"start_idx @ 0x%lx:\n",(long)(A->start_idx));
	for ( j = 0; j < A->n; j++ )
	{
		fprintf(fp,"%d ",A->start_idx[j]);
		if ( j % 10 == 9 )
			fprintf(fp,"\n");
	}
	fprintf(fp,"\n");
	fprintf(fp,"Rows @ 0x%lx:\n",(long)(A->row));
	if ( ! A->row )
	{       fprintf(fp,"*** NULL row ***\n");       return; }
	rows = A->row;
	for ( i = 0; i < A->m; i++ )
	{
		fprintf(fp,"row %d: len = %d, maxlen = %d, diag idx = %d\n",
			i,rows[i].len,rows[i].maxlen,rows[i].diag);
		fprintf(fp,"element list @ 0x%lx\n",(long)(rows[i].elt));
		if ( ! rows[i].elt )
		{
			fprintf(fp,"*** NULL element list ***\n");
			continue;
		}
		elts = rows[i].elt;
		for ( j_idx = 0; j_idx < rows[i].len; j_idx++, elts++ )
		    fprintf(fp,"Col: %d, Val: %g, nxt_row = %d, nxt_idx = %d\n",
			elts->col,elts->val,elts->nxt_row,elts->nxt_idx);
		fprintf(fp,"\n");
	}
}

#define MINSCRATCH      100

/* sp_finput -- input sparse matrix from stream/file fp
	-- uses friendly input routine if fp is a tty
	-- uses format identical to output format otherwise */
#ifndef ANSI_C
SPMAT  *sp_finput(fp)
FILE    *fp;
#else
SPMAT  *sp_finput(FILE *fp)
#endif
{
	int     i, len, ret_val;
	int     col, curr_col, m, n, tmp, tty;
	Real  val;
	SPMAT  *A;
	SPROW  *rows;

	static row_elt *scratch;
	static int	scratch_len = 0;

	if ( ! scratch )
	  {
	    scratch = NEW_A(MINSCRATCH,row_elt);
	    if ( scratch == NULL )
	      error(E_MEM,"sp_finput");
	    scratch_len = MINSCRATCH;
	  }

	for ( i = 0; i < scratch_len; i++ )
	  scratch[i].nxt_row = scratch[i].nxt_idx = -1;

	tty = isatty(fileno(fp));

	if ( tty )
	{
		fprintf(stderr,"SparseMatrix: ");
		do {
			fprintf(stderr,"input rows cols: ");
			if ( ! fgets(line,MAXLINE,fp) )
			    error(E_INPUT,"sp_finput");
		} while ( sscanf(line,"%u %u",&m,&n) != 2 );
		A = sp_get(m,n,5);
		rows = A->row;

		for ( i = 0; i < m; i++ )
		{
		    /* get a row... */
		    fprintf(stderr,"Row %d:\n",i);
		    fprintf(stderr,"Enter <col> <val> or 'e' to end row\n");
		    curr_col = -1;

		    len = 0;
		    for ( ; ; )  /* forever do... */
		      {
		      /* if we need more scratch space, let's get it!
		       -- using amortized doubling */
		      if ( len >= scratch_len )
			{
			  scratch = RENEW(scratch,2*scratch_len,row_elt);
			  if ( ! scratch )
			    error(E_MEM,"sp_finput");
			  scratch_len = 2*scratch_len;
			}
			do {  /* get an entry... */
			    fprintf(stderr,"Entry %d: ",len);
			    if ( ! fgets(line,MAXLINE,fp) )
				error(E_INPUT,"sp_finput");
			    if ( *line == 'e' || *line == 'E' )
				break;
#if REAL == DOUBLE
			} while ( sscanf(line,"%u %lf",&col,&val) != 2 ||
#elif REAL == FLOAT
			} while ( sscanf(line,"%u %f",&col,&val) != 2 ||
#endif
				    col >= n || col <= curr_col );

			if ( *line == 'e' || *line == 'E' )
			    break;

			scratch[len].col = col;
			scratch[len].val = val;
			curr_col = col;

			len++;
		    }

		    /* Note: len = # elements in row */
		    if ( len > 5 )
		     {
			if (mem_info_is_on()) {
			   mem_bytes(TYPE_SPMAT,
					   A->row[i].maxlen*sizeof(row_elt),
					   len*sizeof(row_elt));  
			}

			rows[i].elt = (row_elt *)realloc((char *)rows[i].elt,
							 len*sizeof(row_elt));
			rows[i].maxlen = len;
		    }
		    MEM_COPY(scratch,rows[i].elt,len*sizeof(row_elt));
		    rows[i].len  = len;
		    rows[i].diag = sprow_idx(&(rows[i]),i);
		}
	}
	else /* not tty */
	{
	        ret_val = 0;
		skipjunk(fp);
		fscanf(fp,"SparseMatrix:");
		skipjunk(fp);
		if ( (ret_val=fscanf(fp,"%u by %u",&m,&n)) != 2 )
		    error((ret_val == EOF) ? E_EOF : E_FORMAT,"sp_finput");
		A = sp_get(m,n,5);

		/* initialise start_row */
		for ( i = 0; i < A->n; i++ )
			A->start_row[i] = -1;

		rows = A->row;
		for ( i = 0; i < m; i++ )
		{
		    /* printf("Reading row # %d\n",i); */
		    rows[i].diag = -1;
		    skipjunk(fp);
		    if ( (ret_val=fscanf(fp,"row %d :",&tmp)) != 1 ||
			 tmp != i )
			error((ret_val == EOF) ? E_EOF : E_FORMAT,
			      "sp_finput");
		    curr_col = -1;
		    len = 0;
		    for ( ; ; )  /* forever do... */
		      {
		      /* if we need more scratch space, let's get it!
		       -- using amortized doubling */
		      if ( len >= scratch_len )
			{
			  scratch = RENEW(scratch,2*scratch_len,row_elt);
			  if ( ! scratch )
			    error(E_MEM,"sp_finput");
			  scratch_len = 2*scratch_len;
			}
#if REAL == DOUBLE
			if ( (ret_val=fscanf(fp,"%u : %lf",&col,&val)) != 2 )
#elif REAL == FLOAT
			if ( (ret_val=fscanf(fp,"%u : %f",&col,&val)) != 2 )
#endif
			    break;
			if ( col <= curr_col || col >= n )
			    error(E_FORMAT,"sp_finput");
			scratch[len].col = col;
			scratch[len].val = val;

			len++;
		    }
		    if ( ret_val == EOF )
			error(E_EOF,"sp_finput");

		    if ( len > rows[i].maxlen )
		    {
			rows[i].elt = (row_elt *)realloc((char *)rows[i].elt,
							len*sizeof(row_elt));
			rows[i].maxlen = len;
		    }
		    MEM_COPY(scratch,rows[i].elt,len*sizeof(row_elt));
		    rows[i].len  = len;
		    /* printf("Have read row # %d\n",i); */
		    rows[i].diag = sprow_idx(&(rows[i]),i);
		    /* printf("Have set diag index for row # %d\n",i); */
		}
	}

	return A;
}

