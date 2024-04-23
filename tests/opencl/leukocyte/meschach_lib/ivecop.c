
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


/* ivecop.c  */

#include	<stdio.h>
#include 	"matrix.h"

static	char	rcsid[] = "$Id: ivecop.c,v 1.6 1996/08/20 18:19:21 stewart Exp $";

static char    line[MAXLINE];



/* iv_get -- get integer vector -- see also memory.c */
#ifndef ANSI_C
IVEC	*iv_get(dim)
int	dim;
#else
IVEC	*iv_get(int dim)
#endif
{
   IVEC	*iv;
   /* unsigned int	i; */
   
   if (dim < 0)
     error(E_NEG,"iv_get");

   if ((iv=NEW(IVEC)) == IVNULL )
     error(E_MEM,"iv_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_IVEC,0,sizeof(IVEC));
      mem_numvar(TYPE_IVEC,1);
   }
   
   iv->dim = iv->max_dim = dim;
   if ((iv->ive = NEW_A(dim,int)) == (int *)NULL )
     error(E_MEM,"iv_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_IVEC,0,dim*sizeof(int));
   }
   
   return (iv);
}

/* iv_free -- returns iv & asoociated memory back to memory heap */
#ifndef ANSI_C
int	iv_free(iv)
IVEC	*iv;
#else
int	iv_free(IVEC *iv)
#endif
{
   if ( iv==IVNULL || iv->dim > MAXDIM )
     /* don't trust it */
     return (-1);
   
   if ( iv->ive == (int *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_IVEC,sizeof(IVEC),0);
	 mem_numvar(TYPE_IVEC,-1);
      }
      free((char *)iv);
   }
   else
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_IVEC,sizeof(IVEC)+iv->max_dim*sizeof(int),0);
	 mem_numvar(TYPE_IVEC,-1);
      }	
      free((char *)iv->ive);
      free((char *)iv);
   }
   
   return (0);
}

/* iv_resize -- returns the IVEC with dimension new_dim
   -- iv is set to the zero vector */
#ifndef ANSI_C
IVEC	*iv_resize(iv,new_dim)
IVEC	*iv;
int	new_dim;
#else
IVEC	*iv_resize(IVEC *iv, int new_dim)
#endif
{
   int	i;
   
   if (new_dim < 0)
     error(E_NEG,"iv_resize");

   if ( ! iv )
     return iv_get(new_dim);
   
   if (new_dim == iv->dim)
     return iv;

   if ( new_dim > iv->max_dim )
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_IVEC,iv->max_dim*sizeof(int),
		      new_dim*sizeof(int));
      }
      iv->ive = RENEW(iv->ive,new_dim,int);
      if ( ! iv->ive )
	error(E_MEM,"iv_resize");
      iv->max_dim = new_dim;
   }
   if ( iv->dim <= new_dim )
     for ( i = iv->dim; i < new_dim; i++ )
       iv->ive[i] = 0;
   iv->dim = new_dim;
   
   return iv;
}

/* iv_copy -- copy integer vector in to out
   -- out created/resized if necessary */
#ifndef ANSI_C
IVEC	*iv_copy(in,out)
IVEC	*in, *out;
#else
IVEC	*iv_copy(const IVEC *in, IVEC *out)
#endif
{
   int		i;
   
   if ( ! in )
     error(E_NULL,"iv_copy");
   out = iv_resize(out,in->dim);
   for ( i = 0; i < in->dim; i++ )
     out->ive[i] = in->ive[i];
   
   return out;
}

/* iv_move -- move selected pieces of an IVEC
	-- moves the length dim0 subvector with initial index i0
	   to the corresponding subvector of out with initial index i1
	-- out is resized if necessary */
#ifndef ANSI_C
IVEC	*iv_move(in,i0,dim0,out,i1)
IVEC	*in, *out;
int	i0, dim0, i1;
#else
IVEC	*iv_move(const IVEC *in, int i0, int dim0, IVEC *out, int i1)
#endif
{
    if ( ! in )
	error(E_NULL,"iv_move");
    if ( i0 < 0 || dim0 < 0 || i1 < 0 ||
	 i0+dim0 > in->dim )
	error(E_BOUNDS,"iv_move");

    if ( (! out) || i1+dim0 > out->dim )
	out = iv_resize(out,i1+dim0);

    MEM_COPY(&(in->ive[i0]),&(out->ive[i1]),dim0*sizeof(int));

    return out;
}

/* iv_add -- integer vector addition -- may be in-situ */
#ifndef ANSI_C
IVEC	*iv_add(iv1,iv2,out)
IVEC	*iv1,*iv2,*out;
#else
IVEC	*iv_add(const IVEC *iv1, const IVEC *iv2, IVEC *out)
#endif
{
   unsigned int	i;
   int	*out_ive, *iv1_ive, *iv2_ive;
   
   if ( iv1==IVNULL || iv2==IVNULL )
     error(E_NULL,"iv_add");
   if ( iv1->dim != iv2->dim )
     error(E_SIZES,"iv_add");
   if ( out==IVNULL || out->dim != iv1->dim )
     out = iv_resize(out,iv1->dim);
   
   out_ive = out->ive;
   iv1_ive = iv1->ive;
   iv2_ive = iv2->ive;
   
   for ( i = 0; i < iv1->dim; i++ )
     out_ive[i] = iv1_ive[i] + iv2_ive[i];
   
   return (out);
}



/* iv_sub -- integer vector addition -- may be in-situ */
#ifndef ANSI_C
IVEC	*iv_sub(iv1,iv2,out)
IVEC	*iv1,*iv2,*out;
#else
IVEC	*iv_sub(const IVEC *iv1, const IVEC *iv2, IVEC *out)
#endif
{
   unsigned int	i;
   int	*out_ive, *iv1_ive, *iv2_ive;
   
   if ( iv1==IVNULL || iv2==IVNULL )
     error(E_NULL,"iv_sub");
   if ( iv1->dim != iv2->dim )
     error(E_SIZES,"iv_sub");
   if ( out==IVNULL || out->dim != iv1->dim )
     out = iv_resize(out,iv1->dim);
   
   out_ive = out->ive;
   iv1_ive = iv1->ive;
   iv2_ive = iv2->ive;
   
   for ( i = 0; i < iv1->dim; i++ )
     out_ive[i] = iv1_ive[i] - iv2_ive[i];
   
   return (out);
}

#define	MAX_STACK	60


/* iv_sort -- sorts vector x, and generates permutation that gives the order
   of the components; x = [1.3, 3.7, 0.5] -> [0.5, 1.3, 3.7] and
   the permutation is order = [2, 0, 1].
   -- if order is NULL on entry then it is ignored
   -- the sorted vector x is returned */
#ifndef ANSI_C
IVEC	*iv_sort(x, order)
IVEC	*x;
PERM	*order;
#else
IVEC	*iv_sort(IVEC *x, PERM *order)
#endif
{
   int		*x_ive, tmp, v;
   /* int		*order_pe; */
   int		dim, i, j, l, r, tmp_i;
   int		stack[MAX_STACK], sp;
   
   if ( ! x )
     error(E_NULL,"iv_sort");
   if ( order != PNULL && order->size != x->dim )
     order = px_resize(order, x->dim);
   
   x_ive = x->ive;
   dim = x->dim;
   if ( order != PNULL )
     px_ident(order);
   
   if ( dim <= 1 )
     return x;
   
   /* using quicksort algorithm in Sedgewick,
      "Algorithms in C", Ch. 9, pp. 118--122 (1990) */
   sp = 0;
   l = 0;	r = dim-1;	v = x_ive[0];
   for ( ; ; )
   {
      while ( r > l )
      {
	 /* "i = partition(x_ive,l,r);" */
	 v = x_ive[r];
	 i = l-1;
	 j = r;
	 for ( ; ; )
	 {
	    while ( x_ive[++i] < v )
	      ;
	    --j;
	    while ( x_ive[j] > v && j != 0 )
	      --j;
	    if ( i >= j )	break;
	    
	    tmp = x_ive[i];
	    x_ive[i] = x_ive[j];
	    x_ive[j] = tmp;
	    if ( order != PNULL )
	    {
	       tmp_i = order->pe[i];
	       order->pe[i] = order->pe[j];
	       order->pe[j] = tmp_i;
	    }
	 }
	 tmp = x_ive[i];
	 x_ive[i] = x_ive[r];
	 x_ive[r] = tmp;
	 if ( order != PNULL )
	 {
	    tmp_i = order->pe[i];
	    order->pe[i] = order->pe[r];
	    order->pe[r] = tmp_i;
	 }
	 
	 if ( i-l > r-i )
	 {   stack[sp++] = l;   stack[sp++] = i-1;   l = i+1;   }
	 else
	 {   stack[sp++] = i+1;   stack[sp++] = r;   r = i-1;   }
      }
      
      /* recursion elimination */
      if ( sp == 0 )
	break;
      r = stack[--sp];
      l = stack[--sp];
   }
   
   return x;
}
