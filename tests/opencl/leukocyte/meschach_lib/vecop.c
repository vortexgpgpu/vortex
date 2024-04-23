
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


/* vecop.c 1.3 8/18/87 */

#include	<stdio.h>
#include	"matrix.h"

static	char	rcsid[] = "$Id: vecop.c,v 1.5 1996/08/20 18:18:10 stewart Exp $";


/* _in_prod -- inner product of two vectors from i0 downwards
	-- that is, returns a(i0:dim)^T.b(i0:dim) */
#ifndef ANSI_C
double	_in_prod(a,b,i0)
VEC	*a,*b;
unsigned int	i0;
#else
double	_in_prod(const VEC *a, const VEC *b, unsigned int i0)
#endif
{
	unsigned int	limit;
	/* Real	*a_v, *b_v; */
	/* register Real	sum; */

	if ( a==(VEC *)NULL || b==(VEC *)NULL )
		error(E_NULL,"_in_prod");
	limit = min(a->dim,b->dim);
	if ( i0 > limit )
		error(E_BOUNDS,"_in_prod");

	return __ip__(&(a->ve[i0]),&(b->ve[i0]),(int)(limit-i0));
	/*****************************************
	a_v = &(a->ve[i0]);		b_v = &(b->ve[i0]);
	for ( i=i0; i<limit; i++ )
		sum += a_v[i]*b_v[i];
		sum += (*a_v++)*(*b_v++);

	return (double)sum;
	******************************************/
}

/* sv_mlt -- scalar-vector multiply -- out <- scalar*vector 
	-- may be in-situ */
#ifndef ANSI_C
VEC	*sv_mlt(scalar,vector,out)
double	scalar;
VEC	*vector,*out;
#else
VEC	*sv_mlt(double scalar, const VEC *vector, VEC *out)
#endif
{
	/* unsigned int	dim, i; */
	/* Real	*out_ve, *vec_ve; */

	if ( vector==(VEC *)NULL )
		error(E_NULL,"sv_mlt");
	if ( out==(VEC *)NULL || out->dim != vector->dim )
		out = v_resize(out,vector->dim);
	if ( scalar == 0.0 )
		return v_zero(out);
	if ( scalar == 1.0 )
		return v_copy(vector,out);

	__smlt__(vector->ve,(double)scalar,out->ve,(int)(vector->dim));
	/**************************************************
	dim = vector->dim;
	out_ve = out->ve;	vec_ve = vector->ve;
	for ( i=0; i<dim; i++ )
		out->ve[i] = scalar*vector->ve[i];
		(*out_ve++) = scalar*(*vec_ve++);
	**************************************************/
	return (out);
}

/* v_add -- vector addition -- out <- v1+v2 -- may be in-situ */
#ifndef ANSI_C
VEC	*v_add(vec1,vec2,out)
VEC	*vec1,*vec2,*out;
#else
VEC	*v_add(const VEC *vec1, const VEC *vec2, VEC *out)
#endif
{
	unsigned int	dim;
	/* Real	*out_ve, *vec1_ve, *vec2_ve; */

	if ( vec1==(VEC *)NULL || vec2==(VEC *)NULL )
		error(E_NULL,"v_add");
	if ( vec1->dim != vec2->dim )
		error(E_SIZES,"v_add");
	if ( out==(VEC *)NULL || out->dim != vec1->dim )
		out = v_resize(out,vec1->dim);
	dim = vec1->dim;
	__add__(vec1->ve,vec2->ve,out->ve,(int)dim);
	/************************************************************
	out_ve = out->ve;	vec1_ve = vec1->ve;	vec2_ve = vec2->ve;
	for ( i=0; i<dim; i++ )
		out->ve[i] = vec1->ve[i]+vec2->ve[i];
		(*out_ve++) = (*vec1_ve++) + (*vec2_ve++);
	************************************************************/

	return (out);
}

/* v_mltadd -- scalar/vector multiplication and addition
		-- out = v1 + scale.v2		*/
#ifndef ANSI_C
VEC	*v_mltadd(v1,v2,scale,out)
VEC	*v1,*v2,*out;
double	scale;
#else
VEC	*v_mltadd(const VEC *v1, const VEC *v2, double scale, VEC *out)
#endif
{
	/* register unsigned int	dim, i; */
	/* Real	*out_ve, *v1_ve, *v2_ve; */

	if ( v1==(VEC *)NULL || v2==(VEC *)NULL )
		error(E_NULL,"v_mltadd");
	if ( v1->dim != v2->dim )
		error(E_SIZES,"v_mltadd");
	if ( scale == 0.0 )
		return v_copy(v1,out);
	if ( scale == 1.0 )
		return v_add(v1,v2,out);

	if ( v2 != out )
	{
	    tracecatch(out = v_copy(v1,out),"v_mltadd");

	    /* dim = v1->dim; */
	    __mltadd__(out->ve,v2->ve,scale,(int)(v1->dim));
	}
	else
	{
	    tracecatch(out = sv_mlt(scale,v2,out),"v_mltadd");
	    out = v_add(v1,out,out);
	}
	/************************************************************
	out_ve = out->ve;	v1_ve = v1->ve;		v2_ve = v2->ve;
	for ( i=0; i < dim ; i++ )
		out->ve[i] = v1->ve[i] + scale*v2->ve[i];
		(*out_ve++) = (*v1_ve++) + scale*(*v2_ve++);
	************************************************************/

	return (out);
}

/* v_sub -- vector subtraction -- may be in-situ */
#ifndef ANSI_C
VEC	*v_sub(vec1,vec2,out)
VEC	*vec1,*vec2,*out;
#else
VEC	*v_sub(const VEC *vec1, const VEC *vec2, VEC *out)
#endif
{
	/* unsigned int	i, dim; */
	/* Real	*out_ve, *vec1_ve, *vec2_ve; */

	if ( vec1==(VEC *)NULL || vec2==(VEC *)NULL )
		error(E_NULL,"v_sub");
	if ( vec1->dim != vec2->dim )
		error(E_SIZES,"v_sub");
	if ( out==(VEC *)NULL || out->dim != vec1->dim )
		out = v_resize(out,vec1->dim);

	__sub__(vec1->ve,vec2->ve,out->ve,(int)(vec1->dim));
	/************************************************************
	dim = vec1->dim;
	out_ve = out->ve;	vec1_ve = vec1->ve;	vec2_ve = vec2->ve;
	for ( i=0; i<dim; i++ )
		out->ve[i] = vec1->ve[i]-vec2->ve[i];
		(*out_ve++) = (*vec1_ve++) - (*vec2_ve++);
	************************************************************/

	return (out);
}

/* v_map -- maps function f over components of x: out[i] = f(x[i])
	-- v_map sets out[i] = f(params,x[i]) */
#ifndef ANSI_C
VEC	*v_map(f,x,out)
double	(*f)();
VEC	*x, *out;
#else
#ifdef PROTOTYPES_IN_STRUCT
VEC	*v_map(double (*f)(double), const VEC *x, VEC *out)
#else
VEC	*v_map(double (*f)(), const VEC *x, VEC *out)
#endif
#endif
{
	Real	*x_ve, *out_ve;
	int	i, dim;

	if ( ! x || ! f )
		error(E_NULL,"v_map");
	if ( ! out || out->dim != x->dim )
		out = v_resize(out,x->dim);

	dim = x->dim;	x_ve = x->ve;	out_ve = out->ve;
	for ( i = 0; i < dim; i++ )
		*out_ve++ = (*f)(*x_ve++);

	return out;
}

/* _v_map -- sets out[i] <- f(params, x[i]), i = 0, 1, .., dim-1 */
#ifndef ANSI_C
VEC	*_v_map(f,params,x,out)
double	(*f)();
void	*params;
VEC	*x, *out;
#else
#ifdef PROTOTYPES_IN_STRUCT
VEC	*_v_map(double (*f)(void *,double), void *params, const VEC *x, VEC *out)
#else
VEC	*_v_map(double (*f)(), void *params, const VEC *x, VEC *out)
#endif
#endif
{
	Real	*x_ve, *out_ve;
	int	i, dim;

	if ( ! x || ! f )
		error(E_NULL,"_v_map");
	if ( ! out || out->dim != x->dim )
		out = v_resize(out,x->dim);

	dim = x->dim;	x_ve = x->ve;	out_ve = out->ve;
	for ( i = 0; i < dim; i++ )
		*out_ve++ = (*f)(params,*x_ve++);

	return out;
}

/* v_lincomb -- returns sum_i a[i].v[i], a[i] real, v[i] vectors */
#ifndef ANSI_C
VEC	*v_lincomb(n,v,a,out)
int	n;	/* number of a's and v's */
Real	a[];
VEC	*v[], *out;
#else
VEC	*v_lincomb(int n, const VEC *v[], const Real a[], VEC *out)
#endif
{
	int	i;

	if ( ! a || ! v )
		error(E_NULL,"v_lincomb");
	if ( n <= 0 )
		return VNULL;

	for ( i = 1; i < n; i++ )
		if ( out == v[i] )
		    error(E_INSITU,"v_lincomb");

	out = sv_mlt(a[0],v[0],out);
	for ( i = 1; i < n; i++ )
	{
		if ( ! v[i] )
			error(E_NULL,"v_lincomb");
		if ( v[i]->dim != out->dim )
			error(E_SIZES,"v_lincomb");
		out = v_mltadd(out,v[i],a[i],out);
	}

	return out;
}



#ifdef ANSI_C

/* v_linlist -- linear combinations taken from a list of arguments;
   calling:
      v_linlist(out,v1,a1,v2,a2,...,vn,an,NULL);
   where vi are vectors (VEC *) and ai are numbers (double)
*/
VEC  *v_linlist(VEC *out,VEC *v1,double a1,...)
{
   va_list ap;
   VEC *par;
   double a_par;

   if ( ! v1 )
     return VNULL;
   
   va_start(ap, a1);
   out = sv_mlt(a1,v1,out);
   
   while (par = va_arg(ap,VEC *)) {   /* NULL ends the list*/
      a_par = va_arg(ap,double);
      if (a_par == 0.0) continue;
      if ( out == par )		
	error(E_INSITU,"v_linlist");
      if ( out->dim != par->dim )	
	error(E_SIZES,"v_linlist");

      if (a_par == 1.0)
	out = v_add(out,par,out);
      else if (a_par == -1.0)
	out = v_sub(out,par,out);
      else
	out = v_mltadd(out,par,a_par,out); 
   } 
   
   va_end(ap);
   return out;
}
 
#elif VARARGS


/* v_linlist -- linear combinations taken from a list of arguments;
   calling:
      v_linlist(out,v1,a1,v2,a2,...,vn,an,NULL);
   where vi are vectors (VEC *) and ai are numbers (double)
*/
VEC  *v_linlist(va_alist) va_dcl
{
   va_list ap;
   VEC *par, *out;
   double a_par;

   va_start(ap);
   out = va_arg(ap,VEC *);
   par = va_arg(ap,VEC *);
   if ( ! par ) {
      va_end(ap);
      return VNULL;
   }
   
   a_par = va_arg(ap,double);
   out = sv_mlt(a_par,par,out);
   
   while (par = va_arg(ap,VEC *)) {   /* NULL ends the list*/
      a_par = va_arg(ap,double);
      if (a_par == 0.0) continue;
      if ( out == par )		
	error(E_INSITU,"v_linlist");
      if ( out->dim != par->dim )	
	error(E_SIZES,"v_linlist");

      if (a_par == 1.0)
	out = v_add(out,par,out);
      else if (a_par == -1.0)
	out = v_sub(out,par,out);
      else
	out = v_mltadd(out,par,a_par,out); 
   } 
   
   va_end(ap);
   return out;
}

#endif
  




/* v_star -- computes componentwise (Hadamard) product of x1 and x2
	-- result out is returned */
#ifndef ANSI_C
VEC	*v_star(x1, x2, out)
VEC	*x1, *x2, *out;
#else
VEC	*v_star(const VEC *x1, const VEC *x2, VEC *out)
#endif
{
    int		i;

    if ( ! x1 || ! x2 )
	error(E_NULL,"v_star");
    if ( x1->dim != x2->dim )
	error(E_SIZES,"v_star");
    out = v_resize(out,x1->dim);

    for ( i = 0; i < x1->dim; i++ )
	out->ve[i] = x1->ve[i] * x2->ve[i];

    return out;
}

/* v_slash -- computes componentwise ratio of x2 and x1
	-- out[i] = x2[i] / x1[i]
	-- if x1[i] == 0 for some i, then raise E_SING error
	-- result out is returned */
#ifndef ANSI_C
VEC	*v_slash(x1, x2, out)
VEC	*x1, *x2, *out;
#else
VEC	*v_slash(const VEC *x1, const VEC *x2, VEC *out)
#endif
{
    int		i;
    Real	tmp;

    if ( ! x1 || ! x2 )
	error(E_NULL,"v_slash");
    if ( x1->dim != x2->dim )
	error(E_SIZES,"v_slash");
    out = v_resize(out,x1->dim);

    for ( i = 0; i < x1->dim; i++ )
    {
	tmp = x1->ve[i];
	if ( tmp == 0.0 )
	    error(E_SING,"v_slash");
	out->ve[i] = x2->ve[i] / tmp;
    }

    return out;
}

/* v_min -- computes minimum component of x, which is returned
	-- also sets min_idx to the index of this minimum */
#ifndef ANSI_C
double	v_min(x, min_idx)
VEC	*x;
int	*min_idx;
#else
double	v_min(const VEC *x, int *min_idx)
#endif
{
    int		i, i_min;
    Real	min_val, tmp;

    if ( ! x )
	error(E_NULL,"v_min");
    if ( x->dim <= 0 )
	error(E_SIZES,"v_min");
    i_min = 0;
    min_val = x->ve[0];
    for ( i = 1; i < x->dim; i++ )
    {
	tmp = x->ve[i];
	if ( tmp < min_val )
	{
	    min_val = tmp;
	    i_min = i;
	}
    }

    if ( min_idx != NULL )
	*min_idx = i_min;
    return min_val;
}

/* v_max -- computes maximum component of x, which is returned
	-- also sets max_idx to the index of this maximum */
#ifndef ANSI_C
double	v_max(x, max_idx)
VEC	*x;
int	*max_idx;
#else
double	v_max(const VEC *x, int *max_idx)
#endif
{
    int		i, i_max;
    Real	max_val, tmp;

    if ( ! x )
	error(E_NULL,"v_max");
    if ( x->dim <= 0 )
	error(E_SIZES,"v_max");
    i_max = 0;
    max_val = x->ve[0];
    for ( i = 1; i < x->dim; i++ )
    {
	tmp = x->ve[i];
	if ( tmp > max_val )
	{
	    max_val = tmp;
	    i_max = i;
	}
    }

    if ( max_idx != NULL )
	*max_idx = i_max;
    return max_val;
}

#define	MAX_STACK	60


/* v_sort -- sorts vector x, and generates permutation that gives the order
	of the components; x = [1.3, 3.7, 0.5] -> [0.5, 1.3, 3.7] and
	the permutation is order = [2, 0, 1].
	-- if order is NULL on entry then it is ignored
	-- the sorted vector x is returned */
#ifndef ANSI_C
VEC	*v_sort(x, order)
VEC	*x;
PERM	*order;
#else
VEC	*v_sort(VEC *x, PERM *order)
#endif
{
    Real	*x_ve, tmp, v;
    /* int		*order_pe; */
    int		dim, i, j, l, r, tmp_i;
    int		stack[MAX_STACK], sp;

    if ( ! x )
	error(E_NULL,"v_sort");
    if ( order != PNULL && order->size != x->dim )
	order = px_resize(order, x->dim);

    x_ve = x->ve;
    dim = x->dim;
    if ( order != PNULL )
	px_ident(order);

    if ( dim <= 1 )
	return x;

    /* using quicksort algorithm in Sedgewick,
       "Algorithms in C", Ch. 9, pp. 118--122 (1990) */
    sp = 0;
    l = 0;	r = dim-1;	v = x_ve[0];
    for ( ; ; )
    {
	while ( r > l )
	{
	    /* "i = partition(x_ve,l,r);" */
	    v = x_ve[r];
	    i = l-1;
	    j = r;
	    for ( ; ; )
	    {
		while ( x_ve[++i] < v )
		    ;
		--j;
		while ( x_ve[j] > v && j != 0 )
		    --j;
		if ( i >= j )	break;
		
		tmp = x_ve[i];
		x_ve[i] = x_ve[j];
		x_ve[j] = tmp;
		if ( order != PNULL )
		{
		    tmp_i = order->pe[i];
		    order->pe[i] = order->pe[j];
		    order->pe[j] = tmp_i;
		}
	    }
	    tmp = x_ve[i];
	    x_ve[i] = x_ve[r];
	    x_ve[r] = tmp;
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

/* v_sum -- returns sum of entries of a vector */
#ifndef ANSI_C
double	v_sum(x)
VEC	*x;
#else
double	v_sum(const VEC *x)
#endif
{
    int		i;
    Real	sum;

    if ( ! x )
	error(E_NULL,"v_sum");

    sum = 0.0;
    for ( i = 0; i < x->dim; i++ )
	sum += x->ve[i];

    return sum;
}

/* v_conv -- computes convolution product of two vectors */
#ifndef ANSI_C
VEC	*v_conv(x1, x2, out)
VEC	*x1, *x2, *out;
#else
VEC	*v_conv(const VEC *x1, const VEC *x2, VEC *out)
#endif
{
    int		i;

    if ( ! x1 || ! x2 )
	error(E_NULL,"v_conv");
    if ( x1 == out || x2 == out )
	error(E_INSITU,"v_conv");
    if ( x1->dim == 0 || x2->dim == 0 )
	return out = v_resize(out,0);

    out = v_resize(out,x1->dim + x2->dim - 1);
    v_zero(out);
    for ( i = 0; i < x1->dim; i++ )
	__mltadd__(&(out->ve[i]),x2->ve,x1->ve[i],x2->dim);

    return out;
}

/* v_pconv -- computes a periodic convolution product
	-- the period is the dimension of x2 */
#ifndef ANSI_C
VEC	*v_pconv(x1, x2, out)
VEC	*x1, *x2, *out;
#else
VEC	*v_pconv(const VEC *x1, const VEC *x2, VEC *out)
#endif
{
    int		i;

    if ( ! x1 || ! x2 )
	error(E_NULL,"v_pconv");
    if ( x1 == out || x2 == out )
	error(E_INSITU,"v_pconv");
    out = v_resize(out,x2->dim);
    if ( x2->dim == 0 )
	return out;

    v_zero(out);
    for ( i = 0; i < x1->dim; i++ )
    {
	__mltadd__(&(out->ve[i]),x2->ve,x1->ve[i],x2->dim - i);
	if ( i > 0 )
	    __mltadd__(out->ve,&(x2->ve[x2->dim - i]),x1->ve[i],i);
    }

    return out;
}
