
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
	2nd header file for Meschach's complex routines.
	This file contains declarations for complex factorisation/solve
	routines.

*/


#ifndef ZMATRIX2H
#define ZMATRIX2H

#include "zmatrix.h"

#ifdef ANSI_C
extern ZVEC	*zUsolve(ZMAT *matrix, ZVEC *b, ZVEC *out, double diag);
extern ZVEC	*zLsolve(ZMAT *matrix, ZVEC *b, ZVEC *out, double diag);
extern ZVEC	*zUAsolve(ZMAT *U, ZVEC *b, ZVEC *out, double diag);
extern ZVEC	*zDsolve(ZMAT *A, ZVEC *b, ZVEC *x);
extern ZVEC	*zLAsolve(ZMAT *L, ZVEC *b, ZVEC *out, double diag);

extern ZVEC	*zhhvec(ZVEC *,int,Real *,ZVEC *,complex *);
extern ZVEC	*zhhtrvec(ZVEC *,double,int,ZVEC *,ZVEC *);
extern ZMAT	*zhhtrrows(ZMAT *,int,int,ZVEC *,double);
extern ZMAT	*zhhtrcols(ZMAT *,int,int,ZVEC *,double);
extern ZMAT	*_zhhtrcols(ZMAT *,int,int,ZVEC *,double,ZVEC *);
extern ZMAT     *zHfactor(ZMAT *,ZVEC *);
extern ZMAT     *zHQunpack(ZMAT *,ZVEC *,ZMAT *,ZMAT *);

extern ZMAT	*zQRfactor(ZMAT *A, ZVEC *diag);
extern ZMAT	*zQRCPfactor(ZMAT *A, ZVEC *diag, PERM *px);
extern ZVEC	*_zQsolve(ZMAT *QR, ZVEC *diag, ZVEC *b, ZVEC *x, ZVEC *tmp);
extern ZMAT	*zmakeQ(ZMAT *QR, ZVEC *diag, ZMAT *Qout);
extern ZMAT	*zmakeR(ZMAT *QR, ZMAT *Rout);
extern ZVEC	*zQRsolve(ZMAT *QR, ZVEC *diag, ZVEC *b, ZVEC *x);
extern ZVEC	*zQRAsolve(ZMAT *QR, ZVEC *diag, ZVEC *b, ZVEC *x);
extern ZVEC	*zQRCPsolve(ZMAT *QR,ZVEC *diag,PERM *pivot,ZVEC *b,ZVEC *x);
extern ZVEC	*zUmlt(ZMAT *U, ZVEC *x, ZVEC *out);
extern ZVEC	*zUAmlt(ZMAT *U, ZVEC *x, ZVEC *out);
extern double	zQRcondest(ZMAT *QR);

extern ZVEC	*zLsolve(ZMAT *, ZVEC *, ZVEC *, double);
extern ZMAT	*zset_col(ZMAT *, int, ZVEC *);

extern ZMAT	*zLUfactor(ZMAT *A, PERM *pivot);
extern ZVEC	*zLUsolve(ZMAT *A, PERM *pivot, ZVEC *b, ZVEC *x);
extern ZVEC	*zLUAsolve(ZMAT *LU, PERM *pivot, ZVEC *b, ZVEC *x);
extern ZMAT	*zm_inverse(ZMAT *A, ZMAT *out);
extern double	zLUcondest(ZMAT *LU, PERM *pivot);

extern void	zgivens(complex, complex, Real *, complex *);
extern ZMAT	*zrot_rows(ZMAT *A, int i, int k, double c, complex s,
			   ZMAT *out);
extern ZMAT	*zrot_cols(ZMAT *A, int i, int k, double c, complex s,
			   ZMAT *out);
extern ZVEC	*rot_zvec(ZVEC *x, int i, int k, double c, complex s,
			  ZVEC *out);
extern ZMAT	*zschur(ZMAT *A,ZMAT *Q);
/* extern ZMAT	*schur_vecs(ZMAT *T,ZMAT *Q,X_re,X_im) */
#else
extern ZVEC	*zUsolve(), *zLsolve(), *zUAsolve(), *zDsolve(), *zLAsolve();

extern ZVEC	*zhhvec();
extern ZVEC	*zhhtrvec();
extern ZMAT	*zhhtrrows();
extern ZMAT     *zhhtrcols();
extern ZMAT     *_zhhtrcols();
extern ZMAT     *zHfactor();
extern ZMAT     *zHQunpack();


extern ZMAT	*zQRfactor(), *zQRCPfactor();
extern ZVEC	*_zQsolve();
extern ZMAT	*zmakeQ(), *zmakeR();
extern ZVEC	*zQRsolve(), *zQRAsolve(), *zQRCPsolve();
extern ZVEC	*zUmlt(), *zUAmlt();
extern double	zQRcondest();

extern ZVEC	*zLsolve();
extern ZMAT	*zset_col();

extern ZMAT	*zLUfactor();
extern ZVEC	*zLUsolve(), *zLUAsolve();
extern ZMAT	*zm_inverse();
extern double	zLUcondest();

extern void	zgivens();
extern ZMAT	*zrot_rows(), *zrot_cols();
extern ZVEC	*rot_zvec();
extern ZMAT	*zschur();
/* extern ZMAT	*schur_vecs(); */
#endif /* ANSI_C */

#endif /* ZMATRIX2H */

