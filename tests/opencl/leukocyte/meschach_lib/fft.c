
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
	Fast Fourier Transform routine
	Loosely based on the Fortran routine in Rabiner & Gold's
	"Digital Signal Processing"
*/

static char rcsid[] = "$Id: fft.c,v 1.4 1996/08/20 14:21:05 stewart Exp $";

#include        <stdio.h>
#include        <math.h>
#include        "matrix.h"
#include        "matrix2.h"


/* fft -- d.i.t. fast Fourier transform 
        -- radix-2 FFT only
        -- vector extended to a power of 2 */
#ifndef ANSI_C
void    fft(x_re,x_im)
VEC     *x_re, *x_im;
#else
void    fft(VEC *x_re, VEC *x_im)
#endif
{
    int         i, ip, j, k, li, n, length;
    Real      *xr, *xi;
    Real	theta, pi = 3.1415926535897932384;
    Real      w_re, w_im, u_re, u_im, t_re, t_im;
    Real      tmp, tmpr, tmpi;

    if ( ! x_re || ! x_im )
        error(E_NULL,"fft");
    if ( x_re->dim != x_im->dim )
        error(E_SIZES,"fft");

    n = 1;
    while ( x_re->dim > n )
        n *= 2;
    x_re = v_resize(x_re,n);
    x_im = v_resize(x_im,n);
    /*  printf("# fft: x_re =\n");  v_output(x_re); */
    /*  printf("# fft: x_im =\n");  v_output(x_im); */
    xr   = x_re->ve;
    xi   = x_im->ve;

    /* Decimation in time (DIT) algorithm */
    j = 0;
    for ( i = 0; i < n-1; i++ )
    {
        if ( i < j )
        {
            tmp = xr[i];
            xr[i] = xr[j];
            xr[j] = tmp;
            tmp = xi[i];
            xi[i] = xi[j];
            xi[j] = tmp;
        }
        k = n / 2;
        while ( k <= j )
        {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    /* Actual FFT */
    for ( li = 1; li < n; li *= 2 )
    {
        length = 2*li;
        theta  = pi/li;
        u_re   = 1.0;
        u_im   = 0.0;
        if ( li == 1 )
        {
            w_re = -1.0;
            w_im =  0.0;
        }
        else if ( li == 2 )
        {
            w_re =  0.0;
            w_im =  1.0;
        }
        else
        {
            w_re = cos(theta);
            w_im = sin(theta);
        }
        for ( j = 0; j < li; j++ )
        {
            for ( i =  j; i < n; i += length )
            {
                ip = i + li;
                /* step 1 */
                t_re = xr[ip]*u_re - xi[ip]*u_im;
                t_im = xr[ip]*u_im + xi[ip]*u_re;
                /* step 2 */
                xr[ip] = xr[i]  - t_re;
                xi[ip] = xi[i]  - t_im;
                /* step 3 */
                xr[i] += t_re;
                xi[i] += t_im;
            }
            tmpr = u_re*w_re - u_im*w_im;
            tmpi = u_im*w_re + u_re*w_im;
            u_re = tmpr;
            u_im = tmpi;
        }
    }
}

/* ifft -- inverse FFT using the same interface as fft() */
#ifndef ANSI_C
void	ifft(x_re,x_im)
VEC	*x_re, *x_im;
#else
void	ifft(VEC *x_re, VEC *x_im)
#endif
{
    /* we just use complex conjugates */

    sv_mlt(-1.0,x_im,x_im);
    fft(x_re,x_im);
    sv_mlt(-1.0/((double)(x_re->dim)),x_im,x_im);
    sv_mlt( 1.0/((double)(x_re->dim)),x_re,x_re);
}
