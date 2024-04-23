/* machine.h.  Generated automatically by configure.  */
/* Any machine specific stuff goes here */
/* Add details necessary for your own installation here! */

/* RCS id: $Id: machine.h.in,v 1.3 1995/03/27 15:36:21 des Exp $ */

/* This is for use with "configure" -- if you are not using configure
	then use machine.van for the "vanilla" version of machine.h */

/* Note special macros: ANSI_C (ANSI C syntax)
			SEGMENTED (segmented memory machine e.g. MS-DOS)
			MALLOCDECL (declared if malloc() etc have
					been declared) */

#ifndef _MACHINE_H
#define _MACHINE_H 1

/* #undef const */

/* #undef MALLOCDECL */
#define NOT_SEGMENTED 1
#define HAVE_MEMORY_H 1
/* #undef HAVE_COMPLEX_H */
#define HAVE_MALLOC_H 1
#define STDC_HEADERS 1
/* #undef HAVE_BCOPY */
/* #undef HAVE_BZERO */
#define CHAR0ISDBL0 1
#undef WORDS_BIGENDIAN 
/* #undef U_INT_DEF */
#define VARARGS 1
#define HAVE_PROTOTYPES 1
/* #undef HAVE_PROTOTYPES_IN_STRUCT */

/* for inclusion into C++ files */
#ifdef __cplusplus
#define ANSI_C 1
#ifndef HAVE_PROTOTYPES 
#define HAVE_PROTOTYPES 1
#endif
#ifndef HAVE_PROTOTYPES_IN_STRUCT
#define HAVE_PROTOTYPES_IN_STRUCT 1
#endif
#endif /* __cplusplus */

/* example usage: VEC *PROTO(v_get,(int dim)); */
#ifdef HAVE_PROTOTYPES
#define	PROTO(name,args)	name args
#else
#define PROTO(name,args)	name()
#endif /* HAVE_PROTOTYPES */
#ifdef HAVE_PROTOTYPES_IN_STRUCT
/* PROTO_() is to be used instead of PROTO() in struct's and typedef's */
#define	PROTO_(name,args)	name args
#else
#define PROTO_(name,args)	name()
#endif /* HAVE_PROTOTYPES_IN_STRUCT */

/* for basic or larger versions */
#define COMPLEX 1
#define SPARSE 1

/* for loop unrolling */
/* #undef VUNROLL */
/* #undef MUNROLL */

/* for segmented memory */
#ifndef NOT_SEGMENTED
#define	SEGMENTED
#endif

/* if the system has malloc.h */
#ifdef HAVE_MALLOC_H
#define	MALLOCDECL	1
#include	<malloc.h>
#endif

/* any compiler should have this header */
/* if not, change it */
#include        <stdio.h>


/* Check for ANSI C memmove and memset */
#ifdef STDC_HEADERS

/* standard copy & zero functions */
#define	MEM_COPY(from,to,size)	memmove((to),(from),(size))
#define	MEM_ZERO(where,size)	memset((where),'\0',(size))

#ifndef ANSI_C
#define ANSI_C 1
#endif

#endif

/* standard headers */
#ifdef ANSI_C
#include	<stdlib.h>
#include	<stddef.h>
#include	<string.h>
#include	<float.h>
#endif


/* if have bcopy & bzero and no alternatives yet known, use them */
#ifdef HAVE_BCOPY
#ifndef MEM_COPY
/* nonstandard copy function */
#define	MEM_COPY(from,to,size)	bcopy((char *)(from),(char *)(to),(int)(size))
#endif
#endif

#ifdef HAVE_BZERO
#ifndef MEM_ZERO
/* nonstandard zero function */
#define	MEM_ZERO(where,size)	bzero((char *)(where),(int)(size))
#endif
#endif

/* if the system has complex.h */
#ifdef HAVE_COMPLEX_H
#include	<complex.h>
#endif

/* If prototypes are available & ANSI_C not yet defined, then define it,
	but don't include any header files as the proper ANSI C headers
        aren't here */
#ifdef HAVE_PROTOTYPES
#ifndef ANSI_C
#define ANSI_C  1
#endif
#endif

/* floating point precision */

/* you can choose single, double or long double (if available) precision */

#define FLOAT 		1
#define DOUBLE 		2
#define LONG_DOUBLE 	3

/* #undef REAL_FLT */
/* #undef REAL_DBL */

/* if nothing is defined, choose double precision */
#ifndef REAL_DBL
#ifndef REAL_FLT
#define REAL_DBL 1
#endif
#endif

/* single precision */
#ifdef REAL_FLT
#define  Real float
#define  LongReal float
#define REAL FLOAT
#define LONGREAL FLOAT
#endif

/* double precision */
#ifdef REAL_DBL
#define Real double
#define LongReal double
#define REAL DOUBLE
#define LONGREAL DOUBLE
#endif


/* machine epsilon or unit roundoff error */
/* This is correct on most IEEE Real precision systems */
#ifdef DBL_EPSILON
#if REAL == DOUBLE
#define	MACHEPS	DBL_EPSILON
#elif REAL == FLOAT
#define	MACHEPS	FLT_EPSILON
#elif REAL == LONGDOUBLE
#define MACHEPS LDBL_EPSILON
#endif
#endif

#define F_MACHEPS 
#define D_MACHEPS 

#ifndef MACHEPS
#if REAL == DOUBLE
#define	MACHEPS	D_MACHEPS
#elif REAL == FLOAT  
#define MACHEPS F_MACHEPS
#elif REAL == LONGDOUBLE
#define MACHEPS D_MACHEPS
#endif
#endif

/* #undef M_MACHEPS */

/********************
#ifdef DBL_EPSILON
#define	MACHEPS	DBL_EPSILON
#endif
#ifdef M_MACHEPS
#ifndef MACHEPS
#define MACHEPS	M_MACHEPS
#endif
#endif
********************/

#define	M_MAX_INT 
#ifdef	M_MAX_INT
#ifndef MAX_RAND
#ifdef RAND_MAX
#define	MAX_RAND RAND_MAX
#else
#define	MAX_RAND ((double)(M_MAX_INT))
#endif /* MAX_RAND */
#endif
#endif

/* for non-ANSI systems */
#ifndef HUGE_VAL
#define HUGE_VAL HUGE
#else
#ifndef HUGE
#define HUGE HUGE_VAL
#endif
#endif
#ifdef HUGE_VAL
        /* HUGE_VAL later defined in math.h */
        #undef HUGE_VAL
#endif


#ifdef ANSI_C
extern	int	isatty(int);
#endif

#endif
