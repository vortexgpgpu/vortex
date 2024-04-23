
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
	Elementary functions for complex numbers
	-- if not already defined
*/

#include	<math.h>
#include	"zmatrix.h"

static char rcsid[] = "$Id: zfunc.c,v 1.3 1995/04/07 16:27:25 des Exp $";

#ifndef COMPLEX_H

#ifndef zmake
/* zmake -- create complex number real + i*imag */
complex	zmake(real,imag)
double	real, imag;
{
    complex	w;	/* == real + i*imag */

    w.re = real;	w.im = imag;
    return w;
}
#endif

#ifndef zneg
/* zneg -- returns negative of z */
complex	zneg(z)
complex	z;
{
    z.re = - z.re;
    z.im = - z.im;

    return z;
}
#endif

#ifndef zabs
/* zabs -- returns |z| */
double	zabs(z)
complex	z;
{
    Real	x, y, tmp;
    int		x_expt, y_expt;

    /* Note: we must ensure that overflow does not occur! */
    x = ( z.re >= 0.0 ) ? z.re : -z.re;
    y = ( z.im >= 0.0 ) ? z.im : -z.im;
    if ( x < y )
    {
	tmp = x;
	x = y;
	y = tmp;
    }
    if ( x == 0.0 ) /* then y == 0.0 as well */
	return 0.0;
    x = frexp(x,&x_expt);
    y = frexp(y,&y_expt);
    y = ldexp(y,y_expt-x_expt);
    tmp = sqrt(x*x+y*y);

    return ldexp(tmp,x_expt);
}
#endif

#ifndef zadd
/* zadd -- returns z1+z2 */
complex zadd(z1,z2)
complex	z1, z2;
{
    complex z;

    z.re = z1.re + z2.re;
    z.im = z1.im + z2.im;

    return z;
}
#endif

#ifndef zsub
/* zsub -- returns z1-z2 */
complex zsub(z1,z2)
complex	z1, z2;
{
    complex z;

    z.re = z1.re - z2.re;
    z.im = z1.im - z2.im;

    return z;
}
#endif

#ifndef zmlt
/* zmlt -- returns z1*z2 */
complex	zmlt(z1,z2)
complex	z1, z2;
{
    complex z;

    z.re = z1.re * z2.re - z1.im * z2.im;
    z.im = z1.re * z2.im + z1.im * z2.re;

    return z;
}
#endif

#ifndef zinv
/* zmlt -- returns 1/z */
complex	zinv(z)
complex	z;
{
    Real	x, y, tmp;
    int		x_expt, y_expt;

    if ( z.re == 0.0 && z.im == 0.0 )
	error(E_SING,"zinv");
    /* Note: we must ensure that overflow does not occur! */
    x = ( z.re >= 0.0 ) ? z.re : -z.re;
    y = ( z.im >= 0.0 ) ? z.im : -z.im;
    if ( x < y )
    {
	tmp = x;
	x = y;
	y = tmp;
    }
    x = frexp(x,&x_expt);
    y = frexp(y,&y_expt);
    y = ldexp(y,y_expt-x_expt);

    tmp = 1.0/(x*x + y*y);
    z.re =  z.re*tmp*ldexp(1.0,-2*x_expt);
    z.im = -z.im*tmp*ldexp(1.0,-2*x_expt);

    return z;
}
#endif

#ifndef zdiv
/* zdiv -- returns z1/z2 */
complex	zdiv(z1,z2)
complex	z1, z2;
{
    return zmlt(z1,zinv(z2));
}
#endif

#ifndef zsqrt
/* zsqrt -- returns sqrt(z); uses branch with Re sqrt(z) >= 0 */
complex	zsqrt(z)
complex	z;
{
    complex	w;	/* == sqrt(z) at end */
    Real	alpha;
    
    alpha = sqrt(0.5*(fabs(z.re) + zabs(z)));
    if (alpha!=0) 
      {
    	if (z.re>=0.0)
	  {
	    w.re = alpha;
	    w.im = z.im / (2.0*alpha);
	  }
    	else
	  {
	    w.re = fabs(z.im)/(2.0*alpha);
	    w.im = ( z.im >= 0 ) ? alpha : - alpha;
	  }
      }
    else
      w.re = w.im = 0.0;

    return w;
}
#endif

#ifndef	zexp
/* zexp -- returns exp(z) */
complex	zexp(z)
complex	z;
{
    complex	w;	/* == exp(z) at end */
    Real	r;

    r = exp(z.re);
    w.re = r*cos(z.im);
    w.im = r*sin(z.im);

    return w;
}
#endif

#ifndef	zlog
/* zlog -- returns log(z); uses principal branch with -pi <= Im log(z) <= pi */
complex	zlog(z)
complex	z;
{
    complex	w;	/* == log(z) at end */

    w.re = log(zabs(z));
    w.im = atan2(z.im,z.re);

    return w;
}
#endif

#ifndef zconj
complex	zconj(z)
complex	z;
{
    complex	w;	/* == conj(z) */

    w.re =   z.re;
    w.im = - z.im;
    return w;
}
#endif

#endif

