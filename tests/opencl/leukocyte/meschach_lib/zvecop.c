
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


#include	<stdio.h>
#include	"matrix.h"
#include	"zmatrix.h"
static	char	rcsid[] = "$Id: zvecop.c,v 1.3 1997/10/07 16:13:54 stewart Exp stewart $";



/* _zin_prod -- inner product of two vectors from i0 downwards
	-- flag != 0 means compute sum_i a[i]*.b[i];
	-- flag == 0 means compute sum_i a[i].b[i] */
#ifndef ANSI_C
complex	_zin_prod(a,b,i0,flag)
ZVEC	*a,*b;
unsigned int	i0, flag;
#else
complex	_zin_prod(const ZVEC *a, const ZVEC *b, 
		  unsigned int i0, unsigned int flag)
#endif
{
	unsigned int	limit;

	if ( a==ZVNULL || b==ZVNULL )
		error(E_NULL,"_zin_prod");
	limit = min(a->dim,b->dim);
	if ( i0 > limit )
		error(E_BOUNDS,"_zin_prod");

	return __zip__(&(a->ve[i0]),&(b->ve[i0]),(int)(limit-i0),flag);
}

/* zv_mlt -- scalar-vector multiply -- may be in-situ */
#ifndef ANSI_C
ZVEC	*zv_mlt(scalar,vector,out)
complex	scalar;
ZVEC	*vector,*out;
#else
ZVEC	*zv_mlt(complex scalar, const ZVEC *vector, ZVEC *out)
#endif
{
	/* unsigned int	dim, i; */
	/* complex	*out_ve, *vec_ve; */

	if ( vector==ZVNULL )
		error(E_NULL,"zv_mlt");
	if ( out==ZVNULL || out->dim != vector->dim )
		out = zv_resize(out,vector->dim);
	if ( scalar.re == 0.0 && scalar.im == 0.0 )
		return zv_zero(out);
	if ( scalar.re == 1.0 && scalar.im == 0.0 )
		return zv_copy(vector,out);

	__zmlt__(vector->ve,scalar,out->ve,(int)(vector->dim));

	return (out);
}

/* zv_add -- vector addition -- may be in-situ */
#ifndef ANSI_C
ZVEC	*zv_add(vec1,vec2,out)
ZVEC	*vec1,*vec2,*out;
#else
ZVEC	*zv_add(const ZVEC *vec1, const ZVEC *vec2, ZVEC *out)
#endif
{
	unsigned int	dim;

	if ( vec1==ZVNULL || vec2==ZVNULL )
		error(E_NULL,"zv_add");
	if ( vec1->dim != vec2->dim )
		error(E_SIZES,"zv_add");
	if ( out==ZVNULL || out->dim != vec1->dim )
		out = zv_resize(out,vec1->dim);
	dim = vec1->dim;
	__zadd__(vec1->ve,vec2->ve,out->ve,(int)dim);

	return (out);
}

/* zv_mltadd -- scalar/vector multiplication and addition
		-- out = v1 + scale.v2		*/
#ifndef ANSI_C
ZVEC	*zv_mltadd(v1,v2,scale,out)
ZVEC	*v1,*v2,*out;
complex	scale;
#else
ZVEC	*zv_mltadd(const ZVEC *v1, const ZVEC *v2, complex scale, ZVEC *out)
#endif
{
	/* register unsigned int	dim, i; */
	/* complex	*out_ve, *v1_ve, *v2_ve; */

	if ( v1==ZVNULL || v2==ZVNULL )
		error(E_NULL,"zv_mltadd");
	if ( v1->dim != v2->dim )
		error(E_SIZES,"zv_mltadd");
	if ( scale.re == 0.0 && scale.im == 0.0 )
		return zv_copy(v1,out);
	if ( scale.re == 1.0 && scale.im == 0.0 )
		return zv_add(v1,v2,out);

	if ( v2 != out )
	{
	    tracecatch(out = zv_copy(v1,out),"zv_mltadd");

	    /* dim = v1->dim; */
	    __zmltadd__(out->ve,v2->ve,scale,(int)(v1->dim),0);
	}
	else
	{
	    tracecatch(out = zv_mlt(scale,v2,out),"zv_mltadd");
	    out = zv_add(v1,out,out);
	}

	return (out);
}

/* zv_sub -- vector subtraction -- may be in-situ */
#ifndef ANSI_C
ZVEC	*zv_sub(vec1,vec2,out)
ZVEC	*vec1,*vec2,*out;
#else
ZVEC	*zv_sub(const ZVEC *vec1, const ZVEC *vec2, ZVEC *out)
#endif
{
	/* unsigned int	i, dim; */
	/* complex	*out_ve, *vec1_ve, *vec2_ve; */

	if ( vec1==ZVNULL || vec2==ZVNULL )
		error(E_NULL,"zv_sub");
	if ( vec1->dim != vec2->dim )
		error(E_SIZES,"zv_sub");
	if ( out==ZVNULL || out->dim != vec1->dim )
		out = zv_resize(out,vec1->dim);

	__zsub__(vec1->ve,vec2->ve,out->ve,(int)(vec1->dim));

	return (out);
}

/* zv_map -- maps function f over components of x: out[i] = f(x[i])
	-- _zv_map sets out[i] = f(x[i],params) */
#ifndef ANSI_C
ZVEC	*zv_map(f,x,out)
#ifdef PROTOYPES_IN_STRUCT
complex	(*f)(complex);
#else
complex (*f)();
#endif
ZVEC	*x, *out;
#else
ZVEC	*zv_map(complex (*f)(complex), const ZVEC *x, ZVEC *out)
#endif
{
	complex	*x_ve, *out_ve;
	int	i, dim;

	if ( ! x || ! f )
		error(E_NULL,"zv_map");
	if ( ! out || out->dim != x->dim )
		out = zv_resize(out,x->dim);

	dim = x->dim;	x_ve = x->ve;	out_ve = out->ve;
	for ( i = 0; i < dim; i++ )
		out_ve[i] = (*f)(x_ve[i]);

	return out;
}

#ifndef ANSI_C
ZVEC	*_zv_map(f,params,x,out)
#ifdef PROTOTYPES_IN_STRUCT
complex	(*f)(void *,complex);
#else
complex	(*f)();
#endif
ZVEC	*x, *out;
void	*params;
#else
ZVEC	*_zv_map(complex (*f)(void *,complex), void *params,
		 const ZVEC *x, ZVEC *out)
#endif
{
	complex	*x_ve, *out_ve;
	int	i, dim;

	if ( ! x || ! f )
		error(E_NULL,"_zv_map");
	if ( ! out || out->dim != x->dim )
		out = zv_resize(out,x->dim);

	dim = x->dim;	x_ve = x->ve;	out_ve = out->ve;
	for ( i = 0; i < dim; i++ )
		out_ve[i] = (*f)(params,x_ve[i]);

	return out;
}

/* zv_lincomb -- returns sum_i a[i].v[i], a[i] real, v[i] vectors */
#ifndef ANSI_C
ZVEC	*zv_lincomb(n,v,a,out)
int	n;	/* number of a's and v's */
complex	a[];
ZVEC	*v[], *out;
#else
ZVEC	*zv_lincomb(int n, const ZVEC *v[], const complex a[], ZVEC *out)
#endif
{
	int	i;

	if ( ! a || ! v )
		error(E_NULL,"zv_lincomb");
	if ( n <= 0 )
		return ZVNULL;

	for ( i = 1; i < n; i++ )
		if ( out == v[i] )
		    error(E_INSITU,"zv_lincomb");

	out = zv_mlt(a[0],v[0],out);
	for ( i = 1; i < n; i++ )
	{
		if ( ! v[i] )
			error(E_NULL,"zv_lincomb");
		if ( v[i]->dim != out->dim )
			error(E_SIZES,"zv_lincomb");
		out = zv_mltadd(out,v[i],a[i],out);
	}

	return out;
}


#ifdef ANSI_C


/* zv_linlist -- linear combinations taken from a list of arguments;
   calling:
      zv_linlist(out,v1,a1,v2,a2,...,vn,an,NULL);
   where vi are vectors (ZVEC *) and ai are numbers (complex)
*/

ZVEC	*zv_linlist(ZVEC *out,ZVEC *v1,complex a1,...)
{
   va_list ap;
   ZVEC *par;
   complex a_par;

   if ( ! v1 )
     return ZVNULL;
   
   va_start(ap, a1);
   out = zv_mlt(a1,v1,out);
   
   while (par = va_arg(ap,ZVEC *)) {   /* NULL ends the list*/
      a_par = va_arg(ap,complex);
      if (a_par.re == 0.0 && a_par.im == 0.0) continue;
      if ( out == par )		
	error(E_INSITU,"zv_linlist");
      if ( out->dim != par->dim )	
	error(E_SIZES,"zv_linlist");

      if (a_par.re == 1.0 && a_par.im == 0.0)
	out = zv_add(out,par,out);
      else if (a_par.re == -1.0 && a_par.im == 0.0)
	out = zv_sub(out,par,out);
      else
	out = zv_mltadd(out,par,a_par,out); 
   } 
   
   va_end(ap);
   return out;
}


#elif VARARGS

/* zv_linlist -- linear combinations taken from a list of arguments;
   calling:
      zv_linlist(out,v1,a1,v2,a2,...,vn,an,NULL);
   where vi are vectors (ZVEC *) and ai are numbers (complex)
*/
ZVEC  *zv_linlist(va_alist) va_dcl
{
   va_list ap;
   ZVEC *par, *out;
   complex a_par;

   va_start(ap);
   out = va_arg(ap,ZVEC *);
   par = va_arg(ap,ZVEC *);
   if ( ! par ) {
      va_end(ap);
      return ZVNULL;
   }
   
   a_par = va_arg(ap,complex);
   out = zv_mlt(a_par,par,out);
   
   while (par = va_arg(ap,ZVEC *)) {   /* NULL ends the list*/
      a_par = va_arg(ap,complex);
      if (a_par.re == 0.0 && a_par.im == 0.0) continue;
      if ( out == par )		
	error(E_INSITU,"zv_linlist");
      if ( out->dim != par->dim )	
	error(E_SIZES,"zv_linlist");

      if (a_par.re == 1.0 && a_par.im == 0.0)
	out = zv_add(out,par,out);
      else if (a_par.re == -1.0 && a_par.im == 0.0)
	out = zv_sub(out,par,out);
      else
	out = zv_mltadd(out,par,a_par,out); 
   } 
   
   va_end(ap);
   return out;
}


#endif



/* zv_star -- computes componentwise (Hadamard) product of x1 and x2
	-- result out is returned */
#ifndef ANSI_C
ZVEC	*zv_star(x1, x2, out)
ZVEC	*x1, *x2, *out;
#else
ZVEC	*zv_star(const ZVEC *x1, const ZVEC *x2, ZVEC *out)
#endif
{
    int		i;
    Real	t_re, t_im;

    if ( ! x1 || ! x2 )
	error(E_NULL,"zv_star");
    if ( x1->dim != x2->dim )
	error(E_SIZES,"zv_star");
    out = zv_resize(out,x1->dim);

    for ( i = 0; i < x1->dim; i++ )
    {
	/* out->ve[i] = x1->ve[i] * x2->ve[i]; */
	t_re = x1->ve[i].re*x2->ve[i].re - x1->ve[i].im*x2->ve[i].im;
	t_im = x1->ve[i].re*x2->ve[i].im + x1->ve[i].im*x2->ve[i].re;
	out->ve[i].re = t_re;
	out->ve[i].im = t_im;
    }

    return out;
}

/* zv_slash -- computes componentwise ratio of x2 and x1
	-- out[i] = x2[i] / x1[i]
	-- if x1[i] == 0 for some i, then raise E_SING error
	-- result out is returned */
#ifndef ANSI_C
ZVEC	*zv_slash(x1, x2, out)
ZVEC	*x1, *x2, *out;
#else
ZVEC	*zv_slash(const ZVEC *x1, const ZVEC *x2, ZVEC *out)
#endif
{
    int		i;
    Real	r2, t_re, t_im;
    complex	tmp;

    if ( ! x1 || ! x2 )
	error(E_NULL,"zv_slash");
    if ( x1->dim != x2->dim )
	error(E_SIZES,"zv_slash");
    out = zv_resize(out,x1->dim);

    for ( i = 0; i < x1->dim; i++ )
    {
	r2 = x1->ve[i].re*x1->ve[i].re + x1->ve[i].im*x1->ve[i].im;
	if ( r2 == 0.0 )
	    error(E_SING,"zv_slash");
	tmp.re =   x1->ve[i].re / r2;
	tmp.im = - x1->ve[i].im / r2;
	t_re = tmp.re*x2->ve[i].re - tmp.im*x2->ve[i].im;
	t_im = tmp.re*x2->ve[i].im + tmp.im*x2->ve[i].re;
	out->ve[i].re = t_re;
	out->ve[i].im = t_im;
    }

    return out;
}

/* zv_sum -- returns sum of entries of a vector */
#ifndef ANSI_C
complex	zv_sum(x)
ZVEC	*x;
#else
complex	zv_sum(const ZVEC *x)
#endif
{
    int		i;
    complex	sum;

    if ( ! x )
	error(E_NULL,"zv_sum");

    sum.re = sum.im = 0.0;
    for ( i = 0; i < x->dim; i++ )
    {
	sum.re += x->ve[i].re;
	sum.im += x->ve[i].im;
    }

    return sum;
}

/* px_zvec -- permute vector */
#ifndef ANSI_C
ZVEC	*px_zvec(px,vector,out)
PERM	*px;
ZVEC	*vector,*out;
#else
ZVEC	*px_zvec(PERM *px, ZVEC *vector, ZVEC *out)
#endif
{
    unsigned int	old_i, i, size, start;
    complex	tmp;
    
    if ( px==PNULL || vector==ZVNULL )
	error(E_NULL,"px_zvec");
    if ( px->size > vector->dim )
	error(E_SIZES,"px_zvec");
    if ( out==ZVNULL || out->dim < vector->dim )
	out = zv_resize(out,vector->dim);
    
    size = px->size;
    if ( size == 0 )
	return zv_copy(vector,out);
    
    if ( out != vector )
    {
	for ( i=0; i<size; i++ )
	    if ( px->pe[i] >= size )
		error(E_BOUNDS,"px_vec");
	    else
		out->ve[i] = vector->ve[px->pe[i]];
    }
    else
    {	/* in situ algorithm */
	start = 0;
	while ( start < size )
	{
	    old_i = start;
	    i = px->pe[old_i];
	    if ( i >= size )
	    {
		start++;
		continue;
	    }
	    tmp = vector->ve[start];
	    while ( TRUE )
	    {
		vector->ve[old_i] = vector->ve[i];
		px->pe[old_i] = i+size;
		old_i = i;
		i = px->pe[old_i];
		if ( i >= size )
		    break;
		if ( i == start )
		{
		    vector->ve[old_i] = tmp;
		    px->pe[old_i] = i+size;
		    break;
		}
	    }
	    start++;
	}
	
	for ( i = 0; i < size; i++ )
	    if ( px->pe[i] < size )
		error(E_BOUNDS,"px_vec");
	    else
		px->pe[i] = px->pe[i]-size;
    }
    
    return out;
}

/* pxinv_zvec -- apply the inverse of px to x, returning the result in out
		-- may NOT be in situ */
#ifndef ANSI_C
ZVEC	*pxinv_zvec(px,x,out)
PERM	*px;
ZVEC	*x, *out;
#else
ZVEC	*pxinv_zvec(PERM *px, ZVEC *x, ZVEC *out)
#endif
{
    unsigned int	i, size;
    
    if ( ! px || ! x )
	error(E_NULL,"pxinv_zvec");
    if ( px->size > x->dim )
	error(E_SIZES,"pxinv_zvec");
    if ( ! out || out->dim < x->dim )
	out = zv_resize(out,x->dim);
    
    size = px->size;
    if ( size == 0 )
	return zv_copy(x,out);
    if ( out != x )
    {
	for ( i=0; i<size; i++ )
	    if ( px->pe[i] >= size )
		error(E_BOUNDS,"pxinv_vec");
	    else
		out->ve[px->pe[i]] = x->ve[i];
    }
    else
    {	/* in situ algorithm --- cheat's way out */
	px_inv(px,px);
	px_zvec(px,x,out);
	px_inv(px,px);
    }
    
    
    return out;
}

/* zv_rand -- randomise a complex vector; uniform in [0,1)+[0,1)*i */
#ifndef ANSI_C
ZVEC	*zv_rand(x)
ZVEC	*x;
#else
ZVEC	*zv_rand(ZVEC *x)
#endif
{
    if ( ! x )
	error(E_NULL,"zv_rand");

    mrandlist((Real *)(x->ve),2*x->dim);

    return x;
}
