From boyer@jumpjibe.stanford.edu Fri Jun  9 14:17:09 1995
Received: from jumpjibe.stanford.edu (jumpjibe.Stanford.EDU [36.4.0.23]) by gluttony.isc.tamu.edu (8.6.11/8.6.11) with ESMTP id OAA24615 for <des@isc.tamu.edu>; Fri, 9 Jun 1995 14:17:07 -0500
Received: (from boyer@localhost) by jumpjibe.stanford.edu (8.6.10/8.6.10) id MAA15164 for des@isc.tamu.edu; Fri, 9 Jun 1995 12:17:24 -0700
Message-Id: <199506091917.MAA15164@jumpjibe.stanford.edu>
From: boyer@jumpjibe.stanford.edu (Brent Boyer)
Date: Fri, 9 Jun 1995 12:17:24 PDT
In-Reply-To: David Stewart <des@isc.tamu.edu>
       "Re: Meschach setup question" (Jun  8, 19:07)
X-Mailer: Mail User's Shell (7.2.0 10/31/90)
To: David Stewart <des@isc.tamu.edu>
Subject: Re: Meschach setup question
Content-Length: 9498
X-Lines: 369
Status: RO

david,

	did this file get thru to you last nite?  (someone else told
me that i sent it them instead; maybe i accidentally cced it to them).

	-brent

below is the new machine.h file for Macs:





/* ================================================================================ */


/* machine.h.  Generated automatically by configure.  */
/* Any machine specific stuff goes here */
/* Add details necessary for your own installation here! */

/* RCS id: $Id: machine.h.in,v 1.2 1994/03/13 23:07:30 des Exp $ */

/* This is for use with "configure" -- if you are not using configure
	then use machine.van for the "vanilla" version of machine.h */

/* Note special macros: ANSI_C (ANSI C syntax)
			SEGMENTED (segmented memory machine e.g. MS-DOS)
			MALLOCDECL (declared if malloc() etc have
					been declared) */


/* ================================================================================ */


/* #undef const */						/* leave this commented out -- THINK C has no keyword named "const" */

/* #undef MALLOCDECL */					/* leave this commented out -- THINK C doesn't supply <malloc.h>  */

#define NOT_SEGMENTED 1					/* this must #defined -- Mac's don't have segmented memory */

#undef HAVE_MEMORY_H					/* make sure this is #undefined -- THINK C doesn't supply <memory.h> */

#undef HAVE_COMPLEX_H					/* make sure this is #undefined -- THINK C doesn't supply <complex.h> */

#undef HAVE_MALLOC_H					/* make sure this is #undefined -- THINK C doesn't supply <malloc.h> */

#define STDC_HEADERS 1					/* this must be #defined -- it will cause precisely two effects below:
											1) the macros MEM_COPY(...) & MEM_ZERO(...)	will be correctly
												defined using memmove(...) & memset(...)
											2) the macro ANSI_C will be #defined */

#undef HAVE_BCOPY						/* make sure this is #undefined -- bcopy is for a BSD system? */

#undef HAVE_BZERO						/* make sure this is #undefined -- bzero is for a BSD system? */

/* #undef CHAR0ISDBL0 1	*/				/* for safety, this should be commented out (Dave Stewart's advice) */

#define WORDS_BIGENDIAN 1				/* 68K Macs use big endian microprocessors */

#undef U_INT_DEF						/* make sure this is #undefined (Dave Stewart's advice) */

#define VARARGS 1						/* this must be #defined (Dave Stewart's advice) */


/* ================================================================================ */


/* for prototypes */

#define HAVE_PROTOTYPES 1				/* this must be #defined (Dave Stewart's advice) */

#define HAVE_PROTOTYPES_IN_STRUCT 1		/* this must be #defined (Dave Stewart's advice) */

	/* for inclusion into C++ files */
#ifdef __cplusplus						/* (Note: THINK C must #define this somewhere, since it is used in "ctype.h") */
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



/* ================================================================================ */


/* for basic or larger versions */

#define COMPLEX 1						/* this must be #defined (I want all the complex routines) */
#define SPARSE 1						/* this must be #defined (I want all the sparse routines) */


/* ================================================================================ */


/* for loop unrolling */

/* #undef VUNROLL */
/* #undef MUNROLL */


/* ================================================================================ */


/* for segmented memory */

#ifndef NOT_SEGMENTED
#define	SEGMENTED
#endif


/* ================================================================================ */


/* if the system has malloc.h */

#ifdef HAVE_MALLOC_H
#define	MALLOCDECL	1
#include	<malloc.h>
#endif


/* ================================================================================ */


/* any compiler should have this header */
/* 	if not, change it */

#include	<stdio.h>


/* ================================================================================ */


/* Check for ANSI C memmove and memset */

#ifdef STDC_HEADERS

	/* standard copy & zero functions */
#define	MEM_COPY(from,to,size)	memmove((to),(from),(size))
#define	MEM_ZERO(where,size)	memset((where),'\0',(size))

#ifndef ANSI_C
#define ANSI_C 1
#endif

#endif


/* ================================================================================ */


/* standard headers */

#ifdef ANSI_C
#include	<stdlib.h>
#include	<stddef.h>
#include	<string.h>
#include	<float.h>
#include	<math.h>					/* #include <math.h> so that the macro HUGE_VAL will be available to us */
#endif


/* ================================================================================ */


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


/* ================================================================================ */


/* if the system has complex.h */

#ifdef HAVE_COMPLEX_H
#include	<complex.h>
#endif


/* ================================================================================ */


/* If prototypes are available & ANSI_C not yet defined, then define it,
	but don't include any header files as the proper ANSI C headers
        aren't here */

#ifdef HAVE_PROTOTYPES
#ifndef ANSI_C
#define ANSI_C  1
#endif
#endif


/* ================================================================================ */


/* floating point precision */

/* you can choose single, double or long double (if available) precision */

#define FLOAT 		1
#define DOUBLE 		2
#define LONG_DOUBLE 	3

/* #undef REAL_FLT */
/* #undef REAL_DBL */					/* leave these both commented out, so that the dafault of double is used */

/* choose double precision by default */
#ifndef REAL_DBL
#ifndef REAL_FLT
#define REAL_DBL 1						/* this is what we want: all reals to be of type double */
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


/* Note: under THINK C, the type "double" gets mapped to the type "long double" as long as you DO NOT turn on
			the "8-byte doubles" option.
			
	Recall: this project was compiled with the "8-byte doubles" option turned OFF (so double == long double)
	
	Also Recall: this project was compiled with the "Generate 68881 instructions" and "Native floating-point format"
					options turned ON; this means that double will be a 96 bit MC68881 floating point extended
					precision type; these options give the best speed.
					
	(See the THINK C 6.0 User's Guide, pp. 313-317)
					
	--Brent Boyer 6/7/95 */


/* ================================================================================ */


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

#define F_MACHEPS 1.19209e-07
#define D_MACHEPS 2.22045e-16


/* Note: the extended precision floating point type we are using actually has DBL_EPSILON = 1.08420E-19
			(THINK C 6.0 User's Guide, p. 317); out of caution, I will let the above value for D_MACHEPS
			stay the same.
			
	--Brent Boyer 6/7/95 */


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


/* ================================================================================ */


#define	M_MAX_INT 2147483647			/* this value only works if ints are 32 bits */
	

/* Recall: this project was compiled with the "4-byte ints" option turned ON (so int == long int <==> 32 bits);
			if you do not turn this option on, then ints will be 16 bits so that you will need to do
			#define M_MAX_INT 32767 instead
			
	--Brent Boyer 6/7/95 */


#ifdef	M_MAX_INT
#ifndef MAX_RAND
#define	MAX_RAND ((double)(M_MAX_INT))
#endif
#endif


/* ================================================================================ */


/* for non-ANSI systems */

	/* we #included <math.h> above precisely so that HUGE_VAL will be #defined here */
#ifndef HUGE_VAL
#define HUGE_VAL HUGE
#else
#ifndef HUGE
#define HUGE HUGE_VAL		/* actually, since HUGE is used in several Meschach routines, you need this
									line to be executed even on ANSI systems */
#endif
#endif


/* ================================================================================ */


#ifdef ANSI_C
extern	int	isatty(int);
#endif


