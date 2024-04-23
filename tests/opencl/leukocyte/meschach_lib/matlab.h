
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


/* matlab.h -- Header file for matlab.c, spmatlab.c and zmatlab.c
   for save/load formats */

#ifndef MATLAB_DEF

#define	MATLAB_DEF

/* structure required by MATLAB */
typedef struct {
	long    type;   /* matrix type */
	long    m;      /* # rows */
	long    n;      /* # cols */
	long    imag;   /* is complex? */
	long    namlen; /* length of variable name */
		} matlab;

/* macros for matrix storage type */
#define INTEL   0       /* for 80x87 format */
#define PC      INTEL
#define MOTOROLA        1       /* 6888x format */
#define SUN     MOTOROLA
#define APOLLO  MOTOROLA
#define MAC     MOTOROLA
#define VAX_D   2
#define VAX_G   3

#define COL_ORDER       0
#define ROW_ORDER       1

#define DOUBLE_PREC  0       /* double precision */
#define SINGLE_PREC  1       /* single precision */
#define INT_32  2       /* 32 bit integers (signed) */
#define INT_16  3       /* 16 bit integers (signed) */
#define INT_16u 4       /* 16 bit integers (unsigned) */
/* end of macros for matrix storage type */

#ifndef MACH_ID
#define MACH_ID         MOTOROLA
#endif

#define ORDER           COL_ORDER

#if REAL == DOUBLE
#define PRECISION       DOUBLE_PREC
#elif REAL == FLOAT
#define PRECISION  	SINGLE_PREC
#endif


/* prototypes */

#ifdef ANSI_C

MAT *m_save(FILE *,MAT *,const char *);
MAT *m_load(FILE *,char **);
VEC *v_save(FILE *,VEC *,const char *);
double d_save(FILE *,double,const char *);

#else

extern	MAT *m_save(), *m_load();
extern	VEC *v_save();
extern	double d_save();
#endif

/* complex variant */
#ifdef COMPLEX
#include "zmatrix.h"

#ifdef ANSI_C
extern ZMAT	*zm_save(FILE *fp,ZMAT *A,char *name);
extern ZVEC	*zv_save(FILE *fp,ZVEC *x,char *name);
extern complex	z_save(FILE *fp,complex z,char *name);
extern ZMAT	*zm_load(FILE *fp,char **name);

#else

extern ZMAT	*zm_save();
extern ZVEC	*zv_save();
extern complex	z_save();
extern ZMAT	*zm_load();

#endif

#endif

#endif
