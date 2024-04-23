
/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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
  This file contains basic routines which are used by the functions
  in meschach.a etc.
  These are the routines that should be modified in order to take
  full advantage of specialised architectures (pipelining, vector
  processors etc).
  */

static	char	*rcsid = "$Id: machine.c,v 1.4 1994/01/13 05:28:56 des Exp $";

#include	"machine.h"

/* __ip__ -- inner product */
#ifndef ANSI_C
double	__ip__(dp1,dp2,len)
register Real	*dp1, *dp2;
int	len;
#else
double	__ip__(const Real *dp1, const Real *dp2, int len)
#endif
{
#ifdef VUNROLL
    register int	len4;
    register Real	sum1, sum2, sum3;
#endif
    register int	i;
    register Real     sum;

    sum = 0.0;
#ifdef VUNROLL
    sum1 = sum2 = sum3 = 0.0;
    
    len4 = len / 4;
    len  = len % 4;
    
    for ( i = 0; i < len4; i++ )
    {
	sum  += dp1[4*i]*dp2[4*i];
	sum1 += dp1[4*i+1]*dp2[4*i+1];
	sum2 += dp1[4*i+2]*dp2[4*i+2];
	sum3 += dp1[4*i+3]*dp2[4*i+3];
    }
    sum  += sum1 + sum2 + sum3;
    dp1 += 4*len4;	dp2 += 4*len4;
#endif
    
    for ( i = 0; i < len; i++ )
	sum  += dp1[i]*dp2[i];
    
    return sum;
}

/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
#ifndef ANSI_C
void	__mltadd__(dp1,dp2,s,len)
register Real	*dp1, *dp2;
register double s;
register int	len;
#else
void	__mltadd__(Real *dp1, const Real *dp2, double s, int len)
#endif
{
    register int	i;
#ifdef VUNROLL
    register int        len4;
    
    len4 = len / 4;
    len  = len % 4;
    for ( i = 0; i < len4; i++ )
    {
	dp1[4*i]   += s*dp2[4*i];
	dp1[4*i+1] += s*dp2[4*i+1];
	dp1[4*i+2] += s*dp2[4*i+2];
	dp1[4*i+3] += s*dp2[4*i+3];
    }
    dp1 += 4*len4;	dp2 += 4*len4;
#endif
    
    for ( i = 0; i < len; i++ )
	dp1[i] += s*dp2[i];
}

/* __smlt__ scalar multiply array c.f. sv_mlt() */
#ifndef ANSI_C
void	__smlt__(dp,s,out,len)
register Real	*dp, *out;
register double s;
register int	len;
#else
void	__smlt__(const Real *dp, double s, Real *out, int len)
#endif
{
    register int	i;
    for ( i = 0; i < len; i++ )
	out[i] = s*dp[i];
}

/* __add__ -- add arrays c.f. v_add() */
#ifndef ANSI_C
void	__add__(dp1,dp2,out,len)
register Real	*dp1, *dp2, *out;
register int	len;
#else
void	__add__(const Real *dp1, const Real *dp2, Real *out, int len)
#endif
{
    register int	i;
    for ( i = 0; i < len; i++ )
	out[i] = dp1[i] + dp2[i];
}

/* __sub__ -- subtract arrays c.f. v_sub() */
#ifndef ANSI_C
void	__sub__(dp1,dp2,out,len)
register Real	*dp1, *dp2, *out;
register int	len;
#else
void	__sub__(const Real *dp1, const Real *dp2, Real *out, int len)
#endif
{
    register int	i;
    for ( i = 0; i < len; i++ )
	out[i] = dp1[i] - dp2[i];
}

/* __zero__ -- zeros an array of floating point numbers */
#ifndef ANSI_C
void	__zero__(dp,len)
register Real	*dp;
register int	len;
#else
void	__zero__(Real *dp, int len)
#endif
{
#ifdef CHAR0ISDBL0
    /* if a floating point zero is equivalent to a string of nulls */
    MEM_ZERO((char *)dp,len*sizeof(Real));
#else
    /* else, need to zero the array entry by entry */
    int	i;
    for ( i = 0; i < len; i++ )
	dp[i] = 0.0;
#endif
}

