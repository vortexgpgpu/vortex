
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

double	dclean(x)
double	x;
{
    static double	y;
    y = x;
    return y;	/* prevents optimisation */
}

main()
{
    static double	deps, deps1, dtmp;

    deps = 1.0;
    while ( dclean(1.0+deps) > 1.0 )
	deps = 0.5*deps;

    printf("%g\n", 2.0*deps);
}
