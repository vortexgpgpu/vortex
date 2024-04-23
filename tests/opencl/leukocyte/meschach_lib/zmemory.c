
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


/* Memory allocation and de-allocation for complex matrices and vectors */

#include	<stdio.h>
#include	"zmatrix.h"

static	char	rcsid[] = "$Id: zmemory.c,v 1.2 1994/04/05 02:13:14 des Exp $";



/* zv_zero -- zeros all entries of a complex vector
   -- uses __zzero__() */
#ifndef ANSI_C
ZVEC	*zv_zero(x)
ZVEC	*x;
#else
ZVEC	*zv_zero(ZVEC *x)
#endif
{
   if ( ! x )
     error(E_NULL,"zv_zero");
   __zzero__(x->ve,x->dim);
   
   return x;
}

/* zm_zero -- zeros all entries of a complex matrix
   -- uses __zzero__() */
#ifndef ANSI_C
ZMAT	*zm_zero(A)
ZMAT	*A;
#else
ZMAT	*zm_zero(ZMAT *A)
#endif
{
   int		i;
   
   if ( ! A )
     error(E_NULL,"zm_zero");
   for ( i = 0; i < A->m; i++ )
     __zzero__(A->me[i],A->n);
   
   return A;
}

/* zm_get -- gets an mxn complex matrix (in ZMAT form) */
#ifndef ANSI_C
ZMAT	*zm_get(m,n)
int	m,n;
#else
ZMAT	*zm_get(int m, int n)
#endif
{
   ZMAT	*matrix;
   unsigned int	i;
   
   if (m < 0 || n < 0)
     error(E_NEG,"zm_get");

   if ((matrix=NEW(ZMAT)) == (ZMAT *)NULL )
     error(E_MEM,"zm_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ZMAT,0,sizeof(ZMAT));
      mem_numvar(TYPE_ZMAT,1);
   }
   
   matrix->m = m;		matrix->n = matrix->max_n = n;
   matrix->max_m = m;	matrix->max_size = m*n;
#ifndef SEGMENTED
   if ((matrix->base = NEW_A(m*n,complex)) == (complex *)NULL )
   {
      free(matrix);
      error(E_MEM,"zm_get");
   }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ZMAT,0,m*n*sizeof(complex));
   }
#else
   matrix->base = (complex *)NULL;
#endif
   if ((matrix->me = (complex **)calloc(m,sizeof(complex *))) == 
       (complex **)NULL )
   {	free(matrix->base);	free(matrix);
	error(E_MEM,"zm_get");
     }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ZMAT,0,m*sizeof(complex *));
   }
#ifndef SEGMENTED
   /* set up pointers */
   for ( i=0; i<m; i++ )
     matrix->me[i] = &(matrix->base[i*n]);
#else
   for ( i = 0; i < m; i++ )
     if ( (matrix->me[i]=NEW_A(n,complex)) == (complex *)NULL )
       error(E_MEM,"zm_get");
     else if (mem_info_is_on()) {
	mem_bytes(TYPE_ZMAT,0,n*sizeof(complex));
     }
#endif
   
   return (matrix);
}


/* zv_get -- gets a ZVEC of dimension 'dim'
   -- Note: initialized to zero */
#ifndef ANSI_C
ZVEC	*zv_get(size)
int	size;
#else
ZVEC	*zv_get(int size)
#endif
{
   ZVEC	*vector;

   if (size < 0)
     error(E_NEG,"zv_get");

   if ((vector=NEW(ZVEC)) == (ZVEC *)NULL )
     error(E_MEM,"zv_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ZVEC,0,sizeof(ZVEC));
      mem_numvar(TYPE_ZVEC,1);
   }
   vector->dim = vector->max_dim = size;
   if ((vector->ve=NEW_A(size,complex)) == (complex *)NULL )
   {
      free(vector);
      error(E_MEM,"zv_get");
   }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_ZVEC,0,size*sizeof(complex));
   }
   return (vector);
}

/* zm_free -- returns ZMAT & asoociated memory back to memory heap */
#ifndef ANSI_C
int	zm_free(mat)
ZMAT	*mat;
#else
int	zm_free(ZMAT *mat)
#endif
{
#ifdef SEGMENTED
   int	i;
#endif
   
   if ( mat==(ZMAT *)NULL || (int)(mat->m) < 0 ||
       (int)(mat->n) < 0 )
     /* don't trust it */
     return (-1);
   
#ifndef SEGMENTED
   if ( mat->base != (complex *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZMAT,mat->max_m*mat->max_n*sizeof(complex),0);
      }	   
      free((char *)(mat->base));
   }
#else
   for ( i = 0; i < mat->max_m; i++ )
     if ( mat->me[i] != (complex *)NULL ) {
	if (mem_info_is_on()) {
	   mem_bytes(TYPE_ZMAT,mat->max_n*sizeof(complex),0);
	}
	free((char *)(mat->me[i]));
     }
#endif
   if ( mat->me != (complex **)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZMAT,mat->max_m*sizeof(complex *),0);
      }	   
      free((char *)(mat->me));
   }
   
   if (mem_info_is_on()) {
      mem_bytes(TYPE_ZMAT,sizeof(ZMAT),0);
      mem_numvar(TYPE_ZMAT,-1);
   }
   free((char *)mat);
   
   return (0);
}


/* zv_free -- returns ZVEC & asoociated memory back to memory heap */
#ifndef ANSI_C
int	zv_free(vec)
ZVEC	*vec;
#else
int	zv_free(ZVEC *vec)
#endif
{
   if ( vec==(ZVEC *)NULL || (int)(vec->dim) < 0 )
     /* don't trust it */
     return (-1);
   
   if ( vec->ve == (complex *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZVEC,sizeof(ZVEC),0);
	 mem_numvar(TYPE_ZVEC,-1);
      }
      free((char *)vec);
   }
   else
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZVEC,vec->max_dim*sizeof(complex)+
		      sizeof(ZVEC),0);
	 mem_numvar(TYPE_ZVEC,-1);
      }
      
      free((char *)vec->ve);
      free((char *)vec);
   }
   
   return (0);
}


/* zm_resize -- returns the matrix A of size new_m x new_n; A is zeroed
   -- if A == NULL on entry then the effect is equivalent to m_get() */
#ifndef ANSI_C
ZMAT	*zm_resize(A,new_m,new_n)
ZMAT	*A;
int	new_m, new_n;
#else
ZMAT	*zm_resize(ZMAT *A, int new_m, int new_n)
#endif
{
   unsigned int	i, new_max_m, new_max_n, new_size, old_m, old_n;
   
   if (new_m < 0 || new_n < 0)
     error(E_NEG,"zm_resize");

   if ( ! A )
     return zm_get(new_m,new_n);
   
   if (new_m == A->m && new_n == A->n)
     return A;

   old_m = A->m;	old_n = A->n;
   if ( new_m > A->max_m )
   {	/* re-allocate A->me */
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZMAT,A->max_m*sizeof(complex *),
		      new_m*sizeof(complex *));
      }

      A->me = RENEW(A->me,new_m,complex *);
      if ( ! A->me )
	error(E_MEM,"zm_resize");
   }
   new_max_m = max(new_m,A->max_m);
   new_max_n = max(new_n,A->max_n);
   
#ifndef SEGMENTED
   new_size = new_max_m*new_max_n;
   if ( new_size > A->max_size )
   {	/* re-allocate A->base */
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_ZMAT,A->max_m*A->max_n*sizeof(complex),
		new_size*sizeof(complex));      
      }

      A->base = RENEW(A->base,new_size,complex);
      if ( ! A->base )
	error(E_MEM,"zm_resize");
      A->max_size = new_size;
   }
   
   /* now set up A->me[i] */
   for ( i = 0; i < new_m; i++ )
     A->me[i] = &(A->base[i*new_n]);
   
   /* now shift data in matrix */
   if ( old_n > new_n )
   {
      for ( i = 1; i < min(old_m,new_m); i++ )
	MEM_COPY((char *)&(A->base[i*old_n]),
		 (char *)&(A->base[i*new_n]),
		 sizeof(complex)*new_n);
   }
   else if ( old_n < new_n )
   {
      for ( i = min(old_m,new_m)-1; i > 0; i-- )
      {   /* copy & then zero extra space */
	 MEM_COPY((char *)&(A->base[i*old_n]),
		  (char *)&(A->base[i*new_n]),
		  sizeof(complex)*old_n);
	 __zzero__(&(A->base[i*new_n+old_n]),(new_n-old_n));
      }
      __zzero__(&(A->base[old_n]),(new_n-old_n));
      A->max_n = new_n;
   }
   /* zero out the new rows.. */
   for ( i = old_m; i < new_m; i++ )
     __zzero__(&(A->base[i*new_n]),new_n);
#else
   if ( A->max_n < new_n )
   {
      complex	*tmp;
      
      for ( i = 0; i < A->max_m; i++ )
      {
	 if (mem_info_is_on()) {
	    mem_bytes(TYPE_ZMAT,A->max_n*sizeof(complex),
			 new_max_n*sizeof(complex));
	 }

	 if ( (tmp = RENEW(A->me[i],new_max_n,complex)) == NULL )
	   error(E_MEM,"zm_resize");
	 else {
	    A->me[i] = tmp;
	 }
      }
      for ( i = A->max_m; i < new_max_m; i++ )
      {
	 if ( (tmp = NEW_A(new_max_n,complex)) == NULL )
	   error(E_MEM,"zm_resize");
	 else {
	    A->me[i] = tmp;
	    if (mem_info_is_on()) {
	       mem_bytes(TYPE_ZMAT,0,new_max_n*sizeof(complex));
	    }
	 }
      }
   }
   else if ( A->max_m < new_m )
   {
      for ( i = A->max_m; i < new_m; i++ )
	if ( (A->me[i] = NEW_A(new_max_n,complex)) == NULL )
	  error(E_MEM,"zm_resize");
	else if (mem_info_is_on()) {
	   mem_bytes(TYPE_ZMAT,0,new_max_n*sizeof(complex));
	}
      
   }
   
   if ( old_n < new_n )
   {
      for ( i = 0; i < old_m; i++ )
	__zzero__(&(A->me[i][old_n]),new_n-old_n);
   }
   
   /* zero out the new rows.. */
   for ( i = old_m; i < new_m; i++ )
     __zzero__(A->me[i],new_n);
#endif
   
   A->max_m = new_max_m;
   A->max_n = new_max_n;
   A->max_size = A->max_m*A->max_n;
   A->m = new_m;	A->n = new_n;
   
   return A;
}


/* zv_resize -- returns the (complex) vector x with dim new_dim
   -- x is set to the zero vector */
#ifndef ANSI_C
ZVEC	*zv_resize(x,new_dim)
ZVEC	*x;
int	new_dim;
#else
ZVEC	*zv_resize(ZVEC *x, int new_dim)
#endif
{
   if (new_dim < 0)
     error(E_NEG,"zv_resize");

   if ( ! x )
     return zv_get(new_dim);

   if (new_dim == x->dim)
     return x;

   if ( x->max_dim == 0 )	/* assume that it's from sub_zvec */
     return zv_get(new_dim);
   
   if ( new_dim > x->max_dim )
   {
      if (mem_info_is_on()) { 
	 mem_bytes(TYPE_ZVEC,x->max_dim*sizeof(complex),
		      new_dim*sizeof(complex));
      }

      x->ve = RENEW(x->ve,new_dim,complex);
      if ( ! x->ve )
	error(E_MEM,"zv_resize");
      x->max_dim = new_dim;
   }
   
   if ( new_dim > x->dim )
     __zzero__(&(x->ve[x->dim]),new_dim - x->dim);
   x->dim = new_dim;
   
   return x;
}


/* varying arguments */

#ifdef ANSI_C

#include <stdarg.h>


/* To allocate memory to many arguments. 
   The function should be called:
   zv_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/

int zv_get_vars(int dim,...) 
{
   va_list ap;
   int i=0;
   ZVEC **par;
   
   va_start(ap, dim);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      *par = zv_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}



int zm_get_vars(int m,int n,...) 
{
   va_list ap;
   int i=0;
   ZMAT **par;
   
   va_start(ap, n);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      *par = zm_get(m,n);
      i++;
   } 

   va_end(ap);
   return i;
}



/* To resize memory for many arguments. 
   The function should be called:
   v_resize_vars(new_dim,&x,&y,&z,...,NULL);
   where 
     int new_dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int zv_resize_vars(int new_dim,...)
{
   va_list ap;
   int i=0;
   ZVEC **par;
   
   va_start(ap, new_dim);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      *par = zv_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}



int zm_resize_vars(int m,int n,...) 
{
   va_list ap;
   int i=0;
   ZMAT **par;
   
   va_start(ap, n);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      *par = zm_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}


/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/

int zv_free_vars(ZVEC **pv,...)
{
   va_list ap;
   int i=1;
   ZVEC **par;
   
   zv_free(*pv);
   *pv = ZVNULL;
   va_start(ap, pv);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      zv_free(*par); 
      *par = ZVNULL;
      i++;
   } 

   va_end(ap);
   return i;
}



int zm_free_vars(ZMAT **va,...)
{
   va_list ap;
   int i=1;
   ZMAT **par;
   
   zm_free(*va);
   *va = ZMNULL;
   va_start(ap, va);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      zm_free(*par); 
      *par = ZMNULL;
      i++;
   } 

   va_end(ap);
   return i;
}



#elif VARARGS

#include <varargs.h>

/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/

int zv_get_vars(va_alist) va_dcl
{
   va_list ap;
   int dim,i=0;
   ZVEC **par;
   
   va_start(ap);
   dim = va_arg(ap,int);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      *par = zv_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}



int zm_get_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, n, m;
   ZMAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      *par = zm_get(m,n);
      i++;
   } 

   va_end(ap);
   return i;
}



/* To resize memory for many arguments. 
   The function should be called:
   v_resize_vars(new_dim,&x,&y,&z,...,NULL);
   where 
     int new_dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int zv_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, new_dim;
   ZVEC **par;
   
   va_start(ap);
   new_dim = va_arg(ap,int);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      *par = zv_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}


int zm_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, m, n;
   ZMAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      *par = zm_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}



/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/

int zv_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   ZVEC **par;
   
   va_start(ap);
   while (par = va_arg(ap,ZVEC **)) {   /* NULL ends the list*/
      zv_free(*par); 
      *par = ZVNULL;
      i++;
   } 

   va_end(ap);
   return i;
}



int zm_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   ZMAT **par;
   
   va_start(ap);
   while (par = va_arg(ap,ZMAT **)) {   /* NULL ends the list*/
      zm_free(*par); 
      *par = ZMNULL;
      i++;
   } 

   va_end(ap);
   return i;
}


#endif

