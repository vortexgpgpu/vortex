
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
  Sparse matrix package
  See also: sparse.h, matrix.h
  */

#include	<stdio.h>
#include	<math.h>
#include        <stdlib.h>
#include	"sparse.h"


static char	rcsid[] = "$Id: sparse.c,v 1.10 1994/03/08 05:46:07 des Exp $";

#define	MINROWLEN	10



/* sp_get_val -- returns the (i,j) entry of the sparse matrix A */
#ifndef ANSI_C
double	sp_get_val(A,i,j)
SPMAT	*A;
int	i, j;
#else
double	sp_get_val(const SPMAT *A, int i, int j)
#endif
{
   SPROW	*r;
   int	idx;
   
   if ( A == SMNULL )
     error(E_NULL,"sp_get_val");
   if ( i < 0 || i >= A->m || j < 0 || j >= A->n )
     error(E_SIZES,"sp_get_val");
   
   r = A->row+i;
   idx = sprow_idx(r,j);
   if ( idx < 0 )
     return 0.0;
   /* else */
   return r->elt[idx].val;
}

/* sp_set_val -- sets the (i,j) entry of the sparse matrix A */
#ifndef ANSI_C
double	sp_set_val(A,i,j,val)
SPMAT	*A;
int	i, j;
double	val;
#else
double	sp_set_val(SPMAT *A, int i, int j, double val)
#endif
{
   SPROW	*r;
   int	idx, idx2, new_len;
   
   if ( A == SMNULL )
     error(E_NULL,"sp_set_val");
   if ( i < 0 || i >= A->m || j < 0 || j >= A->n )
     error(E_SIZES,"sp_set_val");
   
   r = A->row+i;
   idx = sprow_idx(r,j);
   /* printf("sp_set_val: idx = %d\n",idx); */
   if ( idx >= 0 )
   {	r->elt[idx].val = val;	return val;	}
   /* else */ if ( idx < -1 )
   {
      /* Note: this destroys the column & diag access paths */
      A->flag_col = A->flag_diag = FALSE;
      /* shift & insert new value */
      idx = -(idx+2);	/* this is the intended insertion index */
      if ( r->len >= r->maxlen )
      {
	 r->len = r->maxlen;
	 new_len = max(2*r->maxlen+1,5);
	 if (mem_info_is_on()) {
	    mem_bytes(TYPE_SPMAT,A->row[i].maxlen*sizeof(row_elt),
			    new_len*sizeof(row_elt));
	 }

	 r->elt = RENEW(r->elt,new_len,row_elt);
	 if ( ! r->elt )	/* can't allocate */
	   error(E_MEM,"sp_set_val");
	 r->maxlen = 2*r->maxlen+1;
      }
      for ( idx2 = r->len-1; idx2 >= idx; idx2-- )
	MEM_COPY((char *)(&(r->elt[idx2])),
		 (char *)(&(r->elt[idx2+1])),sizeof(row_elt));
      /************************************************************
	if ( idx < r->len )
	MEM_COPY((char *)(&(r->elt[idx])),(char *)(&(r->elt[idx+1])),
	(r->len-idx)*sizeof(row_elt));
	************************************************************/
      r->len++;
      r->elt[idx].col = j;
      return r->elt[idx].val = val;
   }
   /* else -- idx == -1, error in index/matrix! */
   return 0.0;
}

/* sp_mv_mlt -- sparse matrix/dense vector multiply
   -- result is in out, which is returned unless out==NULL on entry
   --  if out==NULL on entry then the result vector is created */
#ifndef ANSI_C
VEC	*sp_mv_mlt(A,x,out)
SPMAT	*A;
VEC	*x, *out;
#else
VEC	*sp_mv_mlt(const SPMAT *A, const VEC *x, VEC *out)
#endif
{
   int	i, j_idx, m, n, max_idx;
   Real	sum, *x_ve;
   SPROW	*r;
   row_elt	*elts;
   
   if ( ! A || ! x )
     error(E_NULL,"sp_mv_mlt");
   if ( x->dim != A->n )
     error(E_SIZES,"sp_mv_mlt");
   if ( ! out || out->dim < A->m )
     out = v_resize(out,A->m);
   if ( out == x )
     error(E_INSITU,"sp_mv_mlt");
   m = A->m;	n = A->n;
   x_ve = x->ve;
   
   for ( i = 0; i < m; i++ )
   {
      sum = 0.0;
      r = &(A->row[i]);
      max_idx = r->len;
      elts    = r->elt;
      for ( j_idx = 0; j_idx < max_idx; j_idx++, elts++ )
	sum += elts->val*x_ve[elts->col];
      out->ve[i] = sum;
   }
   return out;
}

/* sp_vm_mlt -- sparse matrix/dense vector multiply from left
   -- result is in out, which is returned unless out==NULL on entry
   -- if out==NULL on entry then result vector is created & returned */
#ifndef ANSI_C
VEC	*sp_vm_mlt(A,x,out)
SPMAT	*A;
VEC	*x, *out;
#else
VEC	*sp_vm_mlt(const SPMAT *A, const VEC *x, VEC *out)
#endif
{
   int	i, j_idx, m, n, max_idx;
   Real	tmp, *x_ve, *out_ve;
   SPROW	*r;
   row_elt	*elts;
   
   if ( ! A || ! x )
     error(E_NULL,"sp_vm_mlt");
   if ( x->dim != A->m )
     error(E_SIZES,"sp_vm_mlt");
   if ( ! out || out->dim < A->n )
     out = v_resize(out,A->n);
   if ( out == x )
     error(E_INSITU,"sp_vm_mlt");
   
   m = A->m;	n = A->n;
   v_zero(out);
   x_ve = x->ve;	out_ve = out->ve;
   
   for ( i = 0; i < m; i++ )
   {
      r = A->row+i;
      max_idx = r->len;
      elts    = r->elt;
      tmp = x_ve[i];
      for ( j_idx = 0; j_idx < max_idx; j_idx++, elts++ )
	out_ve[elts->col] += elts->val*tmp;
   }
   
   return out;
}


/* sp_get -- get sparse matrix
   -- len is number of elements available for each row without
   allocating further memory */
#ifndef ANSI_C
SPMAT	*sp_get(m,n,maxlen)
int	m, n, maxlen;
#else
SPMAT	*sp_get(int m, int n, int maxlen)
#endif
{
   SPMAT	*A;
   SPROW	*rows;
   int	i;
   
   if ( m < 0 || n < 0 )
     error(E_NEG,"sp_get");

   maxlen = max(maxlen,1);
   
   A = NEW(SPMAT);
   if ( ! A )		/* can't allocate */
     error(E_MEM,"sp_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,sizeof(SPMAT));
      mem_numvar(TYPE_SPMAT,1);
   }
   /* fprintf(stderr,"Have SPMAT structure\n"); */
   
   A->row = rows = NEW_A(m,SPROW);
   if ( ! A->row )		/* can't allocate */
     error(E_MEM,"sp_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,m*sizeof(SPROW));
   }
   /* fprintf(stderr,"Have row structure array\n"); */
   
   A->start_row = NEW_A(n,int);
   A->start_idx = NEW_A(n,int);
   if ( ! A->start_row || ! A->start_idx )	/* can't allocate */
     error(E_MEM,"sp_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,2*n*sizeof(int));
   }
   for ( i = 0; i < n; i++ )
     A->start_row[i] = A->start_idx[i] = -1;
   /* fprintf(stderr,"Have start_row array\n"); */
   
   A->m = A->max_m = m;
   A->n = A->max_n = n;
   
   for ( i = 0; i < m; i++, rows++ )
   {
      rows->elt = NEW_A(maxlen,row_elt);
      if ( ! rows->elt )
	error(E_MEM,"sp_get");
      else if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,0,maxlen*sizeof(row_elt));
      }
      /* fprintf(stderr,"Have row %d element array\n",i); */
      rows->len = 0;
      rows->maxlen = maxlen;
      rows->diag = -1;
   }
   
   return A;
}


/* sp_free -- frees up the memory for a sparse matrix */
#ifndef ANSI_C
int	sp_free(A)
SPMAT	*A;
#else
int	sp_free(SPMAT *A)
#endif
{
   SPROW	*r;
   int	i;
   
   if ( ! A )
     return -1;
   if ( A->start_row != (int *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,A->max_n*sizeof(int),0);
      }
      free((char *)(A->start_row));
   }
   if ( A->start_idx != (int *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,A->max_n*sizeof(int),0);
      }
      
      free((char *)(A->start_idx));
   }
   if ( ! A->row )
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,sizeof(SPMAT),0);
	 mem_numvar(TYPE_SPMAT,-1);
      }
      
      free((char *)A);
      return 0;
   }
   for ( i = 0; i < A->m; i++ )
   {
      r = &(A->row[i]);
      if ( r->elt != (row_elt *)NULL ) {
	 if (mem_info_is_on()) {
	    mem_bytes(TYPE_SPMAT,A->row[i].maxlen*sizeof(row_elt),0);
	 }
	 free((char *)(r->elt));
      }
   }
   
   if (mem_info_is_on()) {
      if (A->row) 
	mem_bytes(TYPE_SPMAT,A->max_m*sizeof(SPROW),0);
      mem_bytes(TYPE_SPMAT,sizeof(SPMAT),0);
      mem_numvar(TYPE_SPMAT,-1);
   }
   
   free((char *)(A->row));
   free((char *)A);

   return 0;
}


/* sp_copy -- constructs a copy of a given matrix
   -- note that the max_len fields (etc) are no larger in the copy
   than necessary
   -- result is returned */
#ifndef ANSI_C
SPMAT	*sp_copy(A)
SPMAT	*A;
#else
SPMAT	*sp_copy(const SPMAT *A)
#endif
{
   SPMAT	*out;
   SPROW	*row1, *row2;
   int	i;
   
   if ( A == SMNULL )
     error(E_NULL,"sp_copy");
   if ( ! (out=NEW(SPMAT)) )
     error(E_MEM,"sp_copy");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,sizeof(SPMAT));
      mem_numvar(TYPE_SPMAT,1);
   }
   out->m = out->max_m = A->m;	out->n = out->max_n = A->n;
   
   /* set up rows */
   if ( ! (out->row=NEW_A(A->m,SPROW)) )
     error(E_MEM,"sp_copy");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,A->m*sizeof(SPROW));
   }
   for ( i = 0; i < A->m; i++ )
   {
      row1 = &(A->row[i]);
      row2 = &(out->row[i]);
      if ( ! (row2->elt=NEW_A(max(row1->len,3),row_elt)) )
	error(E_MEM,"sp_copy");
      else if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,0,max(row1->len,3)*sizeof(row_elt));
      }
      row2->len = row1->len;
      row2->maxlen = max(row1->len,3);
      row2->diag = row1->diag;
      MEM_COPY((char *)(row1->elt),(char *)(row2->elt),
	       row1->len*sizeof(row_elt));
   }
   
   /* set up start arrays -- for column access */
   if ( ! (out->start_idx=NEW_A(A->n,int)) ||
       ! (out->start_row=NEW_A(A->n,int)) )
     error(E_MEM,"sp_copy");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_SPMAT,0,2*A->n*sizeof(int));
   }
   MEM_COPY((char *)(A->start_idx),(char *)(out->start_idx),
	    A->n*sizeof(int));
   MEM_COPY((char *)(A->start_row),(char *)(out->start_row),
	    A->n*sizeof(int));
   
   return out;
}

/* sp_col_access -- set column access path; i.e. nxt_row, nxt_idx fields
   -- returns A */
#ifndef ANSI_C
SPMAT	*sp_col_access(A)
SPMAT	*A;
#else
SPMAT	*sp_col_access(SPMAT *A)
#endif
{
   int	i, j, j_idx, len, m, n;
   SPROW	*row;
   row_elt	*r_elt;
   int	*start_row, *start_idx;
   
   if ( A == SMNULL )
     error(E_NULL,"sp_col_access");
   
   m = A->m;	n = A->n;
   
   /* initialise start_row and start_idx */
   start_row = A->start_row;	start_idx = A->start_idx;
   for ( j = 0; j < n; j++ )
   {	*start_row++ = -1;	*start_idx++ = -1;	}
   
   start_row = A->start_row;	start_idx = A->start_idx;
   
   /* now work UP the rows, setting nxt_row, nxt_idx fields */
   for ( i = m-1; i >= 0; i-- )
   {
      row = &(A->row[i]);
      r_elt = row->elt;
      len   = row->len;
      for ( j_idx = 0; j_idx < len; j_idx++, r_elt++ )
      {
	 j = r_elt->col;
	 r_elt->nxt_row = start_row[j];
	 r_elt->nxt_idx = start_idx[j];
	 start_row[j] = i;
	 start_idx[j] = j_idx;
      }
   }
   
   A->flag_col = TRUE;
   return A;
}

/* sp_diag_access -- set diagonal access path(s) */
#ifndef ANSI_C
SPMAT	*sp_diag_access(A)
SPMAT	*A;
#else
SPMAT	*sp_diag_access(SPMAT *A)
#endif
{
   int	i, m;
   SPROW	*row;
   
   if ( A == SMNULL )
     error(E_NULL,"sp_diag_access");
   
   m = A->m;
   
   row = A->row;
   for ( i = 0; i < m; i++, row++ )
     row->diag = sprow_idx(row,i);
   
   A->flag_diag = TRUE;
   
   return A;
}

/* sp_m2dense -- convert a sparse matrix to a dense one */
#ifndef ANSI_C
MAT	*sp_m2dense(A,out)
SPMAT	*A;
MAT	*out;
#else
MAT	*sp_m2dense(const SPMAT *A, MAT *out)
#endif
{
   int	i, j_idx;
   SPROW	*row;
   row_elt	*elt;
   
   if ( ! A )
     error(E_NULL,"sp_m2dense");
   if ( ! out || out->m < A->m || out->n < A->n )
     out = m_get(A->m,A->n);
   
   m_zero(out);
   for ( i = 0; i < A->m; i++ )
   {
      row = &(A->row[i]);
      elt = row->elt;
      for ( j_idx = 0; j_idx < row->len; j_idx++, elt++ )
	out->me[i][elt->col] = elt->val;
   }
   
   return out;
}


/*  C = A+B, can be in situ */
#ifndef ANSI_C
SPMAT *sp_add(A,B,C)
SPMAT *A, *B, *C;
#else
SPMAT *sp_add(const SPMAT *A, const SPMAT *B, SPMAT *C)
#endif
{
   int i, in_situ;
   SPROW *rc;
   STATIC SPROW *tmp = NULL;

   if ( ! A || ! B )
     error(E_NULL,"sp_add");
   if ( A->m != B->m || A->n != B->n )
     error(E_SIZES,"sp_add");
   if (C == A || C == B)
     in_situ = TRUE;
   else in_situ = FALSE;

   if ( ! C )
     C = sp_get(A->m,A->n,5);
   else {
      if ( C->m != A->m || C->n != A->n  )
	error(E_SIZES,"sp_add");
      if (!in_situ) sp_zero(C);
   }

   if (tmp == (SPROW *)NULL && in_situ) {
      tmp = sprow_get(MINROWLEN);
      MEM_STAT_REG(tmp,TYPE_SPROW);
   }

   if (in_situ)
     for (i=0; i < A->m; i++) {
	rc = &(C->row[i]);
	sprow_add(&(A->row[i]),&(B->row[i]),0,tmp,TYPE_SPROW);
	sprow_resize(rc,tmp->len,TYPE_SPMAT);
	MEM_COPY(tmp->elt,rc->elt,tmp->len*sizeof(row_elt));
	rc->len = tmp->len;
     }
   else
     for (i=0; i < A->m; i++) {
	sprow_add(&(A->row[i]),&(B->row[i]),0,&(C->row[i]),TYPE_SPMAT);
     }

   C->flag_col = C->flag_diag = FALSE;

#ifdef	THREADSAFE
   sprow_free(tmp);
#endif

   return C;
}

/*  C = A-B, cannot be in situ */
#ifndef ANSI_C
SPMAT *sp_sub(A,B,C)
SPMAT *A, *B, *C;
#else
SPMAT *sp_sub(const SPMAT *A, const SPMAT *B, SPMAT *C)
#endif
{
   int i, in_situ;
   SPROW *rc;
   STATIC SPROW *tmp = NULL;
   
   if ( ! A || ! B )
     error(E_NULL,"sp_sub");
   if ( A->m != B->m || A->n != B->n )
     error(E_SIZES,"sp_sub");
   if (C == A || C == B)
     in_situ = TRUE;
   else in_situ = FALSE;

   if ( ! C )
     C = sp_get(A->m,A->n,5);
   else {
      if ( C->m != A->m || C->n != A->n  )
	error(E_SIZES,"sp_sub");
      if (!in_situ) sp_zero(C);
   }

   if (tmp == (SPROW *)NULL && in_situ) {
      tmp = sprow_get(MINROWLEN);
      MEM_STAT_REG(tmp,TYPE_SPROW);
   }

   if (in_situ)
     for (i=0; i < A->m; i++) {
	rc = &(C->row[i]);
	sprow_sub(&(A->row[i]),&(B->row[i]),0,tmp,TYPE_SPROW);
	sprow_resize(rc,tmp->len,TYPE_SPMAT);
	MEM_COPY(tmp->elt,rc->elt,tmp->len*sizeof(row_elt));
	rc->len = tmp->len;
     }
   else
     for (i=0; i < A->m; i++) {
	sprow_sub(&(A->row[i]),&(B->row[i]),0,&(C->row[i]),TYPE_SPMAT);
     }

   C->flag_col = C->flag_diag = FALSE;

#ifdef	THREADSAFE
   sprow_free(tmp);
#endif

   return C;
}

/*  C = A+alpha*B, cannot be in situ */
#ifndef ANSI_C
SPMAT *sp_mltadd(A,B,alpha,C)
SPMAT *A, *B, *C;
double alpha;
#else
SPMAT *sp_mltadd(const SPMAT *A, const SPMAT *B, double alpha, SPMAT *C)
#endif
{
   int i, in_situ;
   SPROW *rc;
   STATIC SPROW *tmp = NULL;

   if ( ! A || ! B )
     error(E_NULL,"sp_mltadd");
   if ( A->m != B->m || A->n != B->n )
     error(E_SIZES,"sp_mltadd");
   if (C == A || C == B)
     in_situ = TRUE;
   else in_situ = FALSE;

   if ( ! C )
     C = sp_get(A->m,A->n,5);
   else {
      if ( C->m != A->m || C->n != A->n  )
	error(E_SIZES,"sp_mltadd");
      if (!in_situ) sp_zero(C);
   }

   if (tmp == (SPROW *)NULL && in_situ) {
      tmp = sprow_get(MINROWLEN);
      MEM_STAT_REG(tmp,TYPE_SPROW);
   }

   if (in_situ)
     for (i=0; i < A->m; i++) {
	rc = &(C->row[i]);
	sprow_mltadd(&(A->row[i]),&(B->row[i]),alpha,0,tmp,TYPE_SPROW);
	sprow_resize(rc,tmp->len,TYPE_SPMAT);
	MEM_COPY(tmp->elt,rc->elt,tmp->len*sizeof(row_elt));
	rc->len = tmp->len;
     }
   else
     for (i=0; i < A->m; i++) {
	sprow_mltadd(&(A->row[i]),&(B->row[i]),alpha,0,
		     &(C->row[i]),TYPE_SPMAT);
     }
   
   C->flag_col = C->flag_diag = FALSE;

#ifdef	THREADSAFE
   sprow_free(tmp);
#endif
   
   return C;
}



/*  B = alpha*A, can be in situ */
#ifndef ANSI_C
SPMAT *sp_smlt(A,alpha,B)
SPMAT *A, *B;
double alpha;
#else
SPMAT *sp_smlt(const SPMAT *A, double alpha, SPMAT *B)
#endif
{
   int i;

   if ( ! A )
     error(E_NULL,"sp_smlt");
   if ( ! B )
     B = sp_get(A->m,A->n,5);
   else
     if ( A->m != B->m || A->n != B->n )
       error(E_SIZES,"sp_smlt");

   for (i=0; i < A->m; i++) {
      sprow_smlt(&(A->row[i]),alpha,0,&(B->row[i]),TYPE_SPMAT);
   }
   return B;
}



/* sp_zero -- zero all the (represented) elements of a sparse matrix */
#ifndef ANSI_C
SPMAT	*sp_zero(A)
SPMAT	*A;
#else
SPMAT	*sp_zero(SPMAT *A)
#endif
{
   int	i, idx, len;
   row_elt	*elt;
   
   if ( ! A )
     error(E_NULL,"sp_zero");
   
   for ( i = 0; i < A->m; i++ )
   {
      elt = A->row[i].elt;
      len = A->row[i].len;
      for ( idx = 0; idx < len; idx++ )
	(*elt++).val = 0.0;
   }
   
   return A;
}

/* sp_copy2 -- copy sparse matrix (type 2) 
   -- keeps structure of the OUT matrix */
#ifndef ANSI_C
SPMAT	*sp_copy2(A,OUT)
SPMAT	*A, *OUT;
#else
SPMAT	*sp_copy2(const SPMAT *A, SPMAT *OUT)
#endif
{
   int	i /* , idx, len1, len2 */;
   SPROW	*r1, *r2;
   STATIC SPROW	*scratch = (SPROW *)NULL;
   /* row_elt	*e1, *e2; */
   
   if ( ! A )
     error(E_NULL,"sp_copy2");
   if ( ! OUT )
     OUT = sp_get(A->m,A->n,10);
   if ( ! scratch ) {
      scratch = sprow_xpd(scratch,MINROWLEN,TYPE_SPROW);
      MEM_STAT_REG(scratch,TYPE_SPROW);
   }

   if ( OUT->m < A->m )
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,A->max_m*sizeof(SPROW),
		      A->m*sizeof(SPROW));
      }

      OUT->row = RENEW(OUT->row,A->m,SPROW);
      if ( ! OUT->row )
	error(E_MEM,"sp_copy2");
      
      for ( i = OUT->m; i < A->m; i++ )
      {
	 OUT->row[i].elt = NEW_A(MINROWLEN,row_elt);
	 if ( ! OUT->row[i].elt )
	   error(E_MEM,"sp_copy2");
	 else if (mem_info_is_on()) {
	    mem_bytes(TYPE_SPMAT,0,MINROWLEN*sizeof(row_elt));
	 }
	 
	 OUT->row[i].maxlen = MINROWLEN;
	 OUT->row[i].len = 0;
      }
      OUT->m = A->m;
   }
   
   OUT->flag_col = OUT->flag_diag = FALSE;
   /* sp_zero(OUT); */

   for ( i = 0; i < A->m; i++ )
   {
      r1 = &(A->row[i]);	r2 = &(OUT->row[i]);
      sprow_copy(r1,r2,scratch,TYPE_SPROW);
      if ( r2->maxlen < scratch->len )
	sprow_xpd(r2,scratch->len,TYPE_SPMAT);
      MEM_COPY((char *)(scratch->elt),(char *)(r2->elt),
	       scratch->len*sizeof(row_elt));
      r2->len = scratch->len;
      /*******************************************************
	e1 = r1->elt;		e2 = r2->elt;
	len1 = r1->len;		len2 = r2->len;
	for ( idx = 0; idx < len2; idx++, e2++ )
	e2->val = 0.0;
	for ( idx = 0; idx < len1; idx++, e1++ )
	sprow_set_val(r2,e1->col,e1->val);
	*******************************************************/
   }

   sp_col_access(OUT);

#ifdef	THREADSAFE
   sprow_free(scratch);
#endif

   return OUT;
}

/* sp_resize -- resize a sparse matrix
   -- don't destroying any contents if possible
   -- returns resized matrix */
#ifndef ANSI_C
SPMAT	*sp_resize(A,m,n)
SPMAT	*A;
int	m, n;
#else
SPMAT	*sp_resize(SPMAT *A, int m, int n)
#endif
{
   int	i, len;
   SPROW	*r;
   
   if (m < 0 || n < 0)
     error(E_NEG,"sp_resize");

   if ( ! A )
     return sp_get(m,n,10);

   if (m == A->m && n == A->n)
     return A;

   if ( m <= A->max_m )
   {
      for ( i = A->m; i < m; i++ )
	A->row[i].len = 0;
      A->m = m;
   }
   else
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_SPMAT,A->max_m*sizeof(SPROW),
			 m*sizeof(SPROW));
      }

      A->row = RENEW(A->row,(unsigned)m,SPROW);
      if ( ! A->row )
	error(E_MEM,"sp_resize");
      for ( i = A->m; i < m; i++ )
      {
	 if ( ! (A->row[i].elt = NEW_A(MINROWLEN,row_elt)) )
	   error(E_MEM,"sp_resize");
	 else if (mem_info_is_on()) {
	    mem_bytes(TYPE_SPMAT,0,MINROWLEN*sizeof(row_elt));
	 }
	 A->row[i].len = 0;	A->row[i].maxlen = MINROWLEN;
      }
      A->m = A->max_m = m;
   }

   /* update number of rows */
   A->n = n;

   /* do we need to increase the size of start_idx[] and start_row[] ? */
   if ( n > A->max_n )
   {	/* only have to update the start_idx & start_row arrays */
      if (mem_info_is_on())
      {
	  mem_bytes(TYPE_SPMAT,2*A->max_n*sizeof(int),
		    2*n*sizeof(int));
      }

      A->start_row = RENEW(A->start_row,(unsigned)n,int);
      A->start_idx = RENEW(A->start_idx,(unsigned)n,int);
      if ( ! A->start_row || ! A->start_idx )
	error(E_MEM,"sp_resize");
      A->max_n = n;	/* ...and update max_n */

      return A;
   }

   if ( n <= A->n )
       /* make sure that all rows are truncated just before column n */
       for ( i = 0; i < A->m; i++ )
       {
	   r = &(A->row[i]);
	   len = sprow_idx(r,n);
	   if ( len < 0 )
	       len = -(len+2);
	   if ( len < 0 )
	       error(E_MEM,"sp_resize");
	   r->len = len;
       }
   
   return A;
}


/* sp_compact -- removes zeros and near-zeros from a sparse matrix */
#ifndef ANSI_C
SPMAT	*sp_compact(A,tol)
SPMAT	*A;
double	tol;
#else
SPMAT	*sp_compact(SPMAT *A, double tol)
#endif
{
   int	i, idx1, idx2;
   SPROW	*r;
   row_elt	*elt1, *elt2;
   
   if (  ! A )
     error(E_NULL,"sp_compact");
   if ( tol < 0.0 )
     error(E_RANGE,"sp_compact");
   
   A->flag_col = A->flag_diag = FALSE;
   
   for ( i = 0; i < A->m; i++ )
   {
      r = &(A->row[i]);
      elt1 = elt2 = r->elt;
      idx1 = idx2 = 0;
      while ( idx1 < r->len )
      {
	 /* printf("# sp_compact: idx1 = %d, idx2 = %d\n",idx1,idx2); */
	 if ( fabs(elt1->val) <= tol )
	 {	idx1++;	elt1++;	continue;	}
	 if ( elt1 != elt2 )
	   MEM_COPY(elt1,elt2,sizeof(row_elt));
	 idx1++;	elt1++;
	 idx2++;	elt2++;
      }
      r->len = idx2;
   }
   
   return A;
}

/* sp_mlt (C) Copyright David Stewart and Fabrizio Novalis <novalis@mars.elet.polimi.it> */
/* sp_mlt -- computes out = A*B and returns out */
SPMAT   *sp_mlt(const SPMAT *A, const SPMAT *B, SPMAT *out)
{
  int     i, j, k, idx, cp;
  SPROW   *rA, *rB, *rout, *rtemp;
  double  valA;

  if ( ! A || ! B )
    error(E_NULL,"sp_mlt");
  if ( A->n != B->m )
    error(E_SIZES,"sp_mlt");
  out = sp_resize(out,A->m,B->n);
  sp_zero(out);
  rtemp = sprow_get(B->n);
  for ( i = 0; i < A->m; i++ ) /* per ogni riga */
    {
      rtemp = sprow_resize(rtemp,0,TYPE_SPROW);
      rA = &(A->row[i]);
      rout = &(out->row[i]);
      for ( idx = 0; idx < rA->len; idx++ ) /* per ogni elemento != 0
					       della riga corrente */
	{
	  j = rA->elt[idx].col;
	  valA = rA->elt[idx].val;
	  rB = &(B->row[j]);
	  sprow_mltadd(rtemp,rB,valA,0,rout,TYPE_SPMAT);

	  for ( cp = 0; cp < rout->len; cp++ )
	    {
	      rtemp->elt[cp].col = rout->elt[cp].col;
	      rtemp->elt[cp].val = rout->elt[cp].val;
	    }
	  rtemp->len=rout->len;
	}
    }
  return out;
}

/* varying number of arguments */

#ifdef ANSI_C

/* To allocate memory to many arguments. 
   The function should be called:
   sp_get_vars(m,n,deg,&x,&y,&z,...,NULL);
   where 
     int m,n,deg;
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     m x n is the dimension of matrices x,y,z,...
     returned value is equal to the number of allocated variables
*/

int sp_get_vars(int m,int n,int deg,...) 
{
   va_list ap;
   int i=0;
   SPMAT **par;
   
   va_start(ap, deg);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      *par = sp_get(m,n,deg);
      i++;
   } 

   va_end(ap);
   return i;
}


/* To resize memory for many arguments. 
   The function should be called:
   sp_resize_vars(m,n,&x,&y,&z,...,NULL);
   where 
     int m,n;
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     m X n is the resized dimension of matrices x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
*/
  
int sp_resize_vars(int m,int n,...) 
{
   va_list ap;
   int i=0;
   SPMAT **par;
   
   va_start(ap, n);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      *par = sp_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}

/* To deallocate memory for many arguments. 
   The function should be called:
   sp_free_vars(&x,&y,&z,...,NULL);
   where 
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
*/

int sp_free_vars(SPMAT **va,...)
{
   va_list ap;
   int i=1;
   SPMAT **par;
   
   sp_free(*va);
   *va = (SPMAT *) NULL;
   va_start(ap, va);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      sp_free(*par); 
      *par = (SPMAT *)NULL;
      i++;
   } 

   va_end(ap);
   return i;
}


#elif VARARGS

/* To allocate memory to many arguments. 
   The function should be called:
   sp_get_vars(m,n,deg,&x,&y,&z,...,NULL);
   where 
     int m,n,deg;
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     m x n is the dimension of matrices x,y,z,...
     returned value is equal to the number of allocated variables
*/

int sp_get_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, m, n, deg;
   SPMAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   deg = va_arg(ap,int);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      *par = sp_get(m,n,deg);
      i++;
   } 

   va_end(ap);
   return i;
}


/* To resize memory for many arguments. 
   The function should be called:
   sp_resize_vars(m,n,&x,&y,&z,...,NULL);
   where 
     int m,n;
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     m X n is the resized dimension of matrices x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
*/

int sp_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, m, n;
   SPMAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      *par = sp_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}



/* To deallocate memory for many arguments. 
   The function should be called:
   sp_free_vars(&x,&y,&z,...,NULL);
   where 
     SPMAT *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
*/

int sp_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   SPMAT **par;
   
   va_start(ap);
   while (par = va_arg(ap,SPMAT **)) {   /* NULL ends the list*/
      sp_free(*par); 
      *par = (SPMAT *)NULL;
      i++;
   } 

   va_end(ap);
   return i;
}



#endif

