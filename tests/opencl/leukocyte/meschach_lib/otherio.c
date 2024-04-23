
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
	File for doing assorted I/O operations not invlolving
	MAT/VEC/PERM objects
*/
static	char	rcsid[] = "$Id: otherio.c,v 1.2 1994/01/13 05:34:52 des Exp $";

#include	<stdio.h>
#include	<ctype.h>
#include	"matrix.h"



/* scratch area -- enough for a single line */
static	char	scratch[MAXLINE+1];

/* default value for fy_or_n */
static	int	y_n_dflt = TRUE;

/* fy_or_n -- yes-or-no to question is string s
	-- question written to stderr, input from fp 
	-- if fp is NOT a tty then return y_n_dflt */
#ifndef ANSI_C
int	fy_or_n(fp,s)
FILE	*fp;
char	*s;
#else
int	fy_or_n(FILE *fp, const char *s)
#endif
{
	char	*cp;

	if ( ! isatty(fileno(fp)) )
		return y_n_dflt;

	for ( ; ; )
	{
		fprintf(stderr,"%s (y/n) ? ",s);
		if ( fgets(scratch,MAXLINE,fp)==NULL )
			error(E_INPUT,"fy_or_n");
		cp = scratch;
		while ( isspace(*cp) )
			cp++;
		if ( *cp == 'y' || *cp == 'Y' )
			return TRUE;
		if ( *cp == 'n' || *cp == 'N' )
			return FALSE;
		fprintf(stderr,"Please reply with 'y' or 'Y' for yes ");
		fprintf(stderr,"and 'n' or 'N' for no.\n");
	}
}

/* yn_dflt -- sets the value of y_n_dflt to val */
#ifndef ANSI_C
int	yn_dflt(val)
int	val;
#else
int	yn_dflt(int val)
#endif
{	return y_n_dflt = val;		}

/* fin_int -- return integer read from file/stream fp
	-- prompt s on stderr if fp is a tty
	-- check that x lies between low and high: re-prompt if
		fp is a tty, error exit otherwise
	-- ignore check if low > high		*/
#ifndef ANSI_C
int	fin_int(fp,s,low,high)
FILE	*fp;
char	*s;
int	low, high;
#else
int	fin_int(FILE *fp, const char *s, int low, int high)
#endif
{
	int	retcode, x;

	if ( ! isatty(fileno(fp)) )
	{
		skipjunk(fp);
		if ( (retcode=fscanf(fp,"%d",&x)) == EOF )
			error(E_INPUT,"fin_int");
		if ( retcode <= 0 )
			error(E_FORMAT,"fin_int");
		if ( low <= high && ( x < low || x > high ) )
			error(E_BOUNDS,"fin_int");
		return x;
	}

	for ( ; ; )
	{
		fprintf(stderr,"%s: ",s);
		if ( fgets(scratch,MAXLINE,stdin)==NULL )
			error(E_INPUT,"fin_int");
		retcode = sscanf(scratch,"%d",&x);
		if ( ( retcode==1 && low > high ) ||
					( x >= low && x <= high ) )
			return x;
		fprintf(stderr,"Please type an integer in range [%d,%d].\n",
							low,high);
	}
}


/* fin_double -- return double read from file/stream fp
	-- prompt s on stderr if fp is a tty
	-- check that x lies between low and high: re-prompt if
		fp is a tty, error exit otherwise
	-- ignore check if low > high		*/
#ifndef ANSI_C
double	fin_double(fp,s,low,high)
FILE	*fp;
char	*s;
double	low, high;
#else
double	fin_double(FILE *fp, const char *s, double low, double high)
#endif
{
	Real	retcode, x;

	if ( ! isatty(fileno(fp)) )
	{
		skipjunk(fp);
#if REAL == DOUBLE
		if ( (retcode=fscanf(fp,"%lf",&x)) == EOF )
#elif REAL == FLOAT
		if ( (retcode=fscanf(fp,"%f",&x)) == EOF )
#endif
			error(E_INPUT,"fin_double");
		if ( retcode <= 0 )
			error(E_FORMAT,"fin_double");
		if ( low <= high && ( x < low || x > high ) )
			error(E_BOUNDS,"fin_double");
		return (double)x;
	}

	for ( ; ; )
	{
		fprintf(stderr,"%s: ",s);
		if ( fgets(scratch,MAXLINE,stdin)==NULL )
			error(E_INPUT,"fin_double");
#if REAL == DOUBLE
		retcode = sscanf(scratch,"%lf",&x);
#elif REAL == FLOAT 
		retcode = sscanf(scratch,"%f",&x);
#endif
		if ( ( retcode==1 && low > high ) ||
					( x >= low && x <= high ) )
			return (double)x;
		fprintf(stderr,"Please type an double in range [%g,%g].\n",
							low,high);
	}
}


