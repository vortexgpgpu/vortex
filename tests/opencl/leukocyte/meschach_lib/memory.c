
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


/* memory.c 1.3 11/25/87 */

#include 	"matrix.h"


static	char	rcsid[] = "$Id: memory.c,v 1.13 1994/04/05 02:10:37 des Exp $";

/* m_get -- gets an mxn matrix (in MAT form) by dynamic memory allocation
	-- normally ALL matrices should be obtained this way
	-- if either m or n is negative this will raise an error
	-- note that 0 x n and m x 0 matrices can be created */
#ifndef ANSI_C
MAT	*m_get(m,n)
int	m,n;
#else
MAT	*m_get(int m, int n)
#endif
{
   MAT	*matrix;
   int	i;
   
   if (m < 0 || n < 0)
     error(E_NEG,"m_get");

   if ((matrix=NEW(MAT)) == (MAT *)NULL )
     error(E_MEM,"m_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_MAT,0,sizeof(MAT));
      mem_numvar(TYPE_MAT,1);
   }
   
   matrix->m = m;		matrix->n = matrix->max_n = n;
   matrix->max_m = m;	matrix->max_size = m*n;
#ifndef SEGMENTED
   if ((matrix->base = NEW_A(m*n,Real)) == (Real *)NULL )
   {
      free(matrix);
      error(E_MEM,"m_get");
   }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_MAT,0,m*n*sizeof(Real));
   }
#else
   matrix->base = (Real *)NULL;
#endif
   if ((matrix->me = (Real **)calloc(m,sizeof(Real *))) == 
       (Real **)NULL )
   {	free(matrix->base);	free(matrix);
	error(E_MEM,"m_get");
     }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_MAT,0,m*sizeof(Real *));
   }
   
#ifndef SEGMENTED
   /* set up pointers */
   for ( i=0; i<m; i++ )
     matrix->me[i] = &(matrix->base[i*n]);
#else
   for ( i = 0; i < m; i++ )
     if ( (matrix->me[i]=NEW_A(n,Real)) == (Real *)NULL )
       error(E_MEM,"m_get");
     else if (mem_info_is_on()) {
	mem_bytes(TYPE_MAT,0,n*sizeof(Real));
       }
#endif
   
   return (matrix);
}


/* px_get -- gets a PERM of given 'size' by dynamic memory allocation
	-- Note: initialized to the identity permutation
	-- the permutation is on the set {0,1,2,...,size-1} */
#ifndef ANSI_C
PERM	*px_get(size)
int	size;
#else
PERM	*px_get(int size)
#endif
{
   PERM	*permute;
   int	i;

   if (size < 0)
     error(E_NEG,"px_get");

   if ((permute=NEW(PERM)) == (PERM *)NULL )
     error(E_MEM,"px_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_PERM,0,sizeof(PERM));
      mem_numvar(TYPE_PERM,1);
   }
   
   permute->size = permute->max_size = size;
   if ((permute->pe = NEW_A(size,unsigned int)) == (unsigned int *)NULL )
     error(E_MEM,"px_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_PERM,0,size*sizeof(unsigned int));
   }
   
   for ( i=0; i<size; i++ )
     permute->pe[i] = i;
   
   return (permute);
}

/* v_get -- gets a VEC of dimension 'size'
   -- Note: initialized to zero */
#ifndef ANSI_C
VEC	*v_get(size)
int	size;
#else
VEC	*v_get(int size)
#endif
{
   VEC	*vector;
   
   if (size < 0)
     error(E_NEG,"v_get");

   if ((vector=NEW(VEC)) == (VEC *)NULL )
     error(E_MEM,"v_get");
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_VEC,0,sizeof(VEC));
      mem_numvar(TYPE_VEC,1);
   }
   
   vector->dim = vector->max_dim = size;
   if ((vector->ve=NEW_A(size,Real)) == (Real *)NULL )
   {
      free(vector);
      error(E_MEM,"v_get");
   }
   else if (mem_info_is_on()) {
      mem_bytes(TYPE_VEC,0,size*sizeof(Real));
   }
   
   return (vector);
}

/* m_free -- returns MAT & asoociated memory back to memory heap */
#ifndef ANSI_C
int	m_free(mat)
MAT	*mat;
#else
int	m_free(MAT *mat)
#endif
{
#ifdef SEGMENTED
   int	i;
#endif
   
   if ( mat==(MAT *)NULL || (int)(mat->m) < 0 ||
       (int)(mat->n) < 0 )
     /* don't trust it */
     return (-1);
   
#ifndef SEGMENTED
   if ( mat->base != (Real *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_MAT,mat->max_m*mat->max_n*sizeof(Real),0);
      }
      free((char *)(mat->base));
   }
#else
   for ( i = 0; i < mat->max_m; i++ )
     if ( mat->me[i] != (Real *)NULL ) {
	if (mem_info_is_on()) {
	   mem_bytes(TYPE_MAT,mat->max_n*sizeof(Real),0);
	}
	free((char *)(mat->me[i]));
     }
#endif
   if ( mat->me != (Real **)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_MAT,mat->max_m*sizeof(Real *),0);
      }
      free((char *)(mat->me));
   }
   
   if (mem_info_is_on()) {
      mem_bytes(TYPE_MAT,sizeof(MAT),0);
      mem_numvar(TYPE_MAT,-1);
   }
   free((char *)mat);
   
   return (0);
}



/* px_free -- returns PERM & asoociated memory back to memory heap */
#ifndef ANSI_C
int	px_free(px)
PERM	*px;
#else
int	px_free(PERM *px)
#endif
{
   if ( px==(PERM *)NULL || (int)(px->size) < 0 )
     /* don't trust it */
     return (-1);
   
   if ( px->pe == (unsigned int *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_PERM,sizeof(PERM),0);
	 mem_numvar(TYPE_PERM,-1);
      }      
      free((char *)px);
   }
   else
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_PERM,sizeof(PERM)+px->max_size*sizeof(unsigned int),0);
	 mem_numvar(TYPE_PERM,-1);
      }
      free((char *)px->pe);
      free((char *)px);
   }
   
   return (0);
}



/* v_free -- returns VEC & asoociated memory back to memory heap */
#ifndef ANSI_C
int	v_free(vec)
VEC	*vec;
#else
int	v_free(VEC *vec)
#endif
{
   if ( vec==(VEC *)NULL || (int)(vec->dim) < 0 )
     /* don't trust it */
     return (-1);
   
   if ( vec->ve == (Real *)NULL ) {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_VEC,sizeof(VEC),0);
	 mem_numvar(TYPE_VEC,-1);
      }
      free((char *)vec);
   }
   else
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_VEC,sizeof(VEC)+vec->max_dim*sizeof(Real),0);
	 mem_numvar(TYPE_VEC,-1);
      }
      free((char *)vec->ve);
      free((char *)vec);
   }
   
   return (0);
}



/* m_resize -- returns the matrix A of size new_m x new_n; A is zeroed
   -- if A == NULL on entry then the effect is equivalent to m_get() */
#ifndef ANSI_C
MAT	*m_resize(A,new_m,new_n)
MAT	*A;
int	new_m, new_n;
#else
MAT	*m_resize(MAT *A,int new_m, int new_n)
#endif
{
   int	i;
   int	new_max_m, new_max_n, new_size, old_m, old_n;
   
   if (new_m < 0 || new_n < 0)
     error(E_NEG,"m_resize");

   if ( ! A )
     return m_get(new_m,new_n);

   /* nothing was changed */
   if (new_m == A->m && new_n == A->n)
     return A;

   old_m = A->m;	old_n = A->n;
   if ( new_m > A->max_m )
   {	/* re-allocate A->me */
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_MAT,A->max_m*sizeof(Real *),
		      new_m*sizeof(Real *));
      }

      A->me = RENEW(A->me,new_m,Real *);
      if ( ! A->me )
	error(E_MEM,"m_resize");
   }
   new_max_m = max(new_m,A->max_m);
   new_max_n = max(new_n,A->max_n);
   
#ifndef SEGMENTED
   new_size = new_max_m*new_max_n;
   if ( new_size > A->max_size )
   {	/* re-allocate A->base */
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_MAT,A->max_m*A->max_n*sizeof(Real),
		      new_size*sizeof(Real));
      }

      A->base = RENEW(A->base,new_size,Real);
      if ( ! A->base )
	error(E_MEM,"m_resize");
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
		 sizeof(Real)*new_n);
   }
   else if ( old_n < new_n )
   {
      for ( i = (int)(min(old_m,new_m))-1; i > 0; i-- )
      {   /* copy & then zero extra space */
	 MEM_COPY((char *)&(A->base[i*old_n]),
		  (char *)&(A->base[i*new_n]),
		  sizeof(Real)*old_n);
	 __zero__(&(A->base[i*new_n+old_n]),(new_n-old_n));
      }
      __zero__(&(A->base[old_n]),(new_n-old_n));
      A->max_n = new_n;
   }
   /* zero out the new rows.. */
   for ( i = old_m; i < new_m; i++ )
     __zero__(&(A->base[i*new_n]),new_n);
#else
   if ( A->max_n < new_n )
   {
      Real	*tmp;
      
      for ( i = 0; i < A->max_m; i++ )
      {
	 if (mem_info_is_on()) {
	    mem_bytes(TYPE_MAT,A->max_n*sizeof(Real),
			 new_max_n*sizeof(Real));
	 }	

	 if ( (tmp = RENEW(A->me[i],new_max_n,Real)) == NULL )
	   error(E_MEM,"m_resize");
	 else {	
	    A->me[i] = tmp;
	 }
      }
      for ( i = A->max_m; i < new_max_m; i++ )
      {
	 if ( (tmp = NEW_A(new_max_n,Real)) == NULL )
	   error(E_MEM,"m_resize");
	 else {
	    A->me[i] = tmp;

	    if (mem_info_is_on()) {
	       mem_bytes(TYPE_MAT,0,new_max_n*sizeof(Real));
	    }	    
	 }
      }
   }
   else if ( A->max_m < new_m )
   {
      for ( i = A->max_m; i < new_m; i++ ) 
	if ( (A->me[i] = NEW_A(new_max_n,Real)) == NULL )
	  error(E_MEM,"m_resize");
	else if (mem_info_is_on()) {
	   mem_bytes(TYPE_MAT,0,new_max_n*sizeof(Real));
	}
      
   }
   
   if ( old_n < new_n )
   {
      for ( i = 0; i < old_m; i++ )
	__zero__(&(A->me[i][old_n]),new_n-old_n);
   }
   
   /* zero out the new rows.. */
   for ( i = old_m; i < new_m; i++ )
     __zero__(A->me[i],new_n);
#endif
   
   A->max_m = new_max_m;
   A->max_n = new_max_n;
   A->max_size = A->max_m*A->max_n;
   A->m = new_m;	A->n = new_n;
   
   return A;
}

/* px_resize -- returns the permutation px with size new_size
   -- px is set to the identity permutation */
#ifndef ANSI_C
PERM	*px_resize(px,new_size)
PERM	*px;
int	new_size;
#else
PERM	*px_resize(PERM *px, int new_size)
#endif
{
   int	i;
   
   if (new_size < 0)
     error(E_NEG,"px_resize");

   if ( ! px )
     return px_get(new_size);
   
   /* nothing is changed */
   if (new_size == px->size)
     return px;

   if ( new_size > px->max_size )
   {
      if (mem_info_is_on()) {
	 mem_bytes(TYPE_PERM,px->max_size*sizeof(unsigned int),
		      new_size*sizeof(unsigned int));
      }
      px->pe = RENEW(px->pe,new_size,unsigned int);
      if ( ! px->pe )
	error(E_MEM,"px_resize");
      px->max_size = new_size;
   }
   if ( px->size <= new_size )
     /* extend permutation */
     for ( i = px->size; i < new_size; i++ )
       px->pe[i] = i;
   else
     for ( i = 0; i < new_size; i++ )
       px->pe[i] = i;
   
   px->size = new_size;
   
   return px;
}

/* v_resize -- returns the vector x with dim new_dim
   -- x is set to the zero vector */
#ifndef ANSI_C
VEC	*v_resize(x,new_dim)
VEC	*x;
int	new_dim;
#else
VEC	*v_resize(VEC *x, int new_dim)
#endif
{
   
   if (new_dim < 0)
     error(E_NEG,"v_resize");

   if ( ! x )
     return v_get(new_dim);

   /* nothing is changed */
   if (new_dim == x->dim)
     return x;

   if ( x->max_dim == 0 )	/* assume that it's from sub_vec */
     return v_get(new_dim);
   
   if ( new_dim > x->max_dim )
   {
      if (mem_info_is_on()) { 
	 mem_bytes(TYPE_VEC,x->max_dim*sizeof(Real),
			 new_dim*sizeof(Real));
      }

      x->ve = RENEW(x->ve,new_dim,Real);
      if ( ! x->ve )
	error(E_MEM,"v_resize");
      x->max_dim = new_dim;
   }
   
   if ( new_dim > x->dim )
     __zero__(&(x->ve[x->dim]),new_dim - x->dim);
   x->dim = new_dim;
   
   return x;
}




/* Varying number of arguments */
/* other functions of this type are in sparse.c and zmemory.c */



#ifdef ANSI_C


/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/

int v_get_vars(int dim,...) 
{
   va_list ap;
   int i=0;
   VEC **par;
   
   va_start(ap, dim);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      *par = v_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}


int iv_get_vars(int dim,...) 
{
   va_list ap;
   int i=0;
   IVEC **par;
   
   va_start(ap, dim);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      *par = iv_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}

int m_get_vars(int m,int n,...) 
{
   va_list ap;
   int i=0;
   MAT **par;
   
   va_start(ap, n);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      *par = m_get(m,n);
      i++;
   } 

   va_end(ap);
   return i;
}

int px_get_vars(int dim,...) 
{
   va_list ap;
   int i=0;
   PERM **par;
   
   va_start(ap, dim);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      *par = px_get(dim);
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
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int v_resize_vars(int new_dim,...)
{
   va_list ap;
   int i=0;
   VEC **par;
   
   va_start(ap, new_dim);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      *par = v_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}



int iv_resize_vars(int new_dim,...) 
{
   va_list ap;
   int i=0;
   IVEC **par;
   
   va_start(ap, new_dim);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      *par = iv_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}

int m_resize_vars(int m,int n,...) 
{
   va_list ap;
   int i=0;
   MAT **par;
   
   va_start(ap, n);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      *par = m_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}


int px_resize_vars(int new_dim,...) 
{
   va_list ap;
   int i=0;
   PERM **par;
   
   va_start(ap, new_dim);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      *par = px_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}

/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/


int v_free_vars(VEC **pv,...)
{
   va_list ap;
   int i=1;
   VEC **par;
   
   v_free(*pv);
   *pv = VNULL;
   va_start(ap, pv);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      v_free(*par); 
      *par = VNULL;
      i++;
   } 

   va_end(ap);
   return i;
}


int iv_free_vars(IVEC **ipv,...)
{
   va_list ap;
   int i=1;
   IVEC **par;
   
   iv_free(*ipv);
   *ipv = IVNULL;
   va_start(ap, ipv);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      iv_free(*par); 
      *par = IVNULL;
      i++;
   } 

   va_end(ap);
   return i;
}


int px_free_vars(PERM **vpx,...)
{
   va_list ap;
   int i=1;
   PERM **par;
   
   px_free(*vpx);
   *vpx = PNULL;
   va_start(ap, vpx);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      px_free(*par); 
      *par = PNULL;
      i++;
   } 

   va_end(ap);
   return i;
}

int m_free_vars(MAT **va,...)
{
   va_list ap;
   int i=1;
   MAT **par;
   
   m_free(*va);
   *va = MNULL;
   va_start(ap, va);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      m_free(*par); 
      *par = MNULL;
      i++;
   } 

   va_end(ap);
   return i;
}


#elif VARARGS
/* old varargs is used */



/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,VNULL);
   where 
     int dim;
     VEC *x, *y, *z,...;
     The last argument should be VNULL ! 
     dim is the length of vectors x,y,z,...
*/

int v_get_vars(va_alist) va_dcl
{
   va_list ap;
   int dim,i=0;
   VEC **par;
   
   va_start(ap);
   dim = va_arg(ap,int);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      *par = v_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}


int iv_get_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, dim;
   IVEC **par;
   
   va_start(ap);
   dim = va_arg(ap,int);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      *par = iv_get(dim);
      i++;
   } 

   va_end(ap);
   return i;
}

int m_get_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, n, m;
   MAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      *par = m_get(m,n);
      i++;
   } 

   va_end(ap);
   return i;
}



int px_get_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, dim;
   PERM **par;
   
   va_start(ap);
   dim = va_arg(ap,int);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      *par = px_get(dim);
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
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int v_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, new_dim;
   VEC **par;
   
   va_start(ap);
   new_dim = va_arg(ap,int);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      *par = v_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}



int iv_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, new_dim;
   IVEC **par;
   
   va_start(ap);
   new_dim = va_arg(ap,int);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      *par = iv_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}

int m_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, m, n;
   MAT **par;
   
   va_start(ap);
   m = va_arg(ap,int);
   n = va_arg(ap,int);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      *par = m_resize(*par,m,n);
      i++;
   } 

   va_end(ap);
   return i;
}

int px_resize_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0, new_dim;
   PERM **par;
   
   va_start(ap);
   new_dim = va_arg(ap,int);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      *par = px_resize(*par,new_dim);
      i++;
   } 

   va_end(ap);
   return i;
}


/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/


int v_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   VEC **par;
   
   va_start(ap);
   while (par = va_arg(ap,VEC **)) {   /* NULL ends the list*/
      v_free(*par); 
      *par = VNULL;
      i++;
   } 

   va_end(ap);
   return i;
}



int iv_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   IVEC **par;
   
   va_start(ap);
   while (par = va_arg(ap,IVEC **)) {   /* NULL ends the list*/
      iv_free(*par); 
      *par = IVNULL;
      i++;
   } 

   va_end(ap);
   return i;
}


int px_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   PERM **par;
   
   va_start(ap);
   while (par = va_arg(ap,PERM **)) {   /* NULL ends the list*/
      px_free(*par); 
      *par = PNULL;
      i++;
   } 

   va_end(ap);
   return i;
}

int m_free_vars(va_alist) va_dcl
{
   va_list ap;
   int i=0;
   MAT **par;
   
   va_start(ap);
   while (par = va_arg(ap,MAT **)) {   /* NULL ends the list*/
      m_free(*par); 
      *par = MNULL;
      i++;
   } 

   va_end(ap);
   return i;
}



#endif /* VARARGS */
  

