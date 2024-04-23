
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


/* Sparse matrix factorise/solve header */
/* RCS id: $Id: sparse2.h,v 1.4 1994/01/13 05:33:46 des Exp $ */



#ifndef SPARSE2H

#define SPARSE2H

#include "sparse.h"


#ifdef ANSI_C
SPMAT	*spCHfactor(SPMAT *A), *spICHfactor(SPMAT *A), *spCHsymb(SPMAT *A);
VEC	*spCHsolve(SPMAT *CH, const VEC *b, VEC *x);

SPMAT	*spLUfactor(SPMAT *A,PERM *pivot,double threshold);
SPMAT	*spILUfactor(SPMAT *A,double theshold);
VEC	*spLUsolve(const SPMAT *LU,PERM *pivot, const VEC *b,VEC *x),
	*spLUTsolve(SPMAT *LU,PERM *pivot, const VEC *b,VEC *x);

SPMAT	*spBKPfactor(SPMAT *, PERM *, PERM *, double);
VEC	*spBKPsolve(SPMAT *, PERM *, PERM *, const VEC *, VEC *);

VEC	*pccg(VEC *(*A)(),void *A_par,VEC *(*M_inv)(),void *M_par,VEC *b,
						double tol,VEC *x);
VEC	*sp_pccg(SPMAT *,SPMAT *,VEC *,double,VEC *);
VEC	*cgs(VEC *(*A)(),void *A_par,VEC *b,VEC *r0,double tol,VEC *x);
VEC	*sp_cgs(SPMAT *,VEC *,VEC *,double,VEC *);
VEC	*lsqr(VEC *(*A)(),VEC *(*AT)(),void *A_par,VEC *b,double tol,VEC *x);
VEC	*sp_lsqr(SPMAT *,VEC *,double,VEC *);
int	cg_set_maxiter(int);

void	lanczos(VEC *(*A)(),void *A_par,int m,VEC *x0,VEC *a,VEC *b,
						Real *beta_m1,MAT *Q);
void	sp_lanczos(SPMAT *,int,VEC *,VEC *,VEC *,Real *,MAT *);
VEC	*lanczos2(VEC *(*A)(),void *A_par,int m,VEC *x0,VEC *evals,
						VEC *err_est);
VEC	*sp_lanczos2(SPMAT *,int,VEC *,VEC *,VEC *);
extern  void    scan_to(SPMAT *,IVEC *,IVEC *,IVEC *,int);
extern  row_elt  *chase_col(const SPMAT *,int,int *,int *,int);
extern  row_elt  *chase_past(const SPMAT *,int,int *,int *,int);
extern  row_elt  *bump_col(const SPMAT *,int,int *,int *);

#else
extern SPMAT	*spCHfactor(), *spICHfactor(), *spCHsymb();
extern VEC	*spCHsolve();

extern SPMAT	*spLUfactor();
extern SPMAT	*spILUfactor();
extern VEC	*spLUsolve(), *spLUTsolve();

extern SPMAT	*spBKPfactor();
extern VEC	*spBKPsolve();

extern VEC	*pccg(), *sp_pccg(), *cgs(), *sp_cgs(), *lsqr(), *sp_lsqr();
extern int	cg_set_maxiter();

void	lanczos(), sp_lanczos();
VEC	*lanczos2(), *sp_lanczos2();
extern  void    scan_to();
extern  row_elt  *chase_col();
extern  row_elt  *chase_past();
extern  row_elt  *bump_col();

#endif


#endif
