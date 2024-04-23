
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


/* Main include file for zmeschach library -- complex vectors and matrices */

#ifndef ZMATRIXH
#define ZMATRIXH

#include "matrix.h"


          /*  Type definitions for complex vectors and matrices  */


/* complex definition */
typedef struct  {
                Real re,im;
        } complex;

/* complex vector definition */
typedef struct  {
                unsigned int   dim, max_dim;
                complex  *ve;
                } ZVEC;

/* complex matrix definition */
typedef struct  {
                unsigned int   m, n;
                unsigned int   max_m, max_n, max_size;
                complex *base;          /* base is base of alloc'd mem */
                complex **me;
                } ZMAT;

#define ZVNULL  ((ZVEC *)NULL)
#define ZMNULL  ((ZMAT *)NULL)

#define	Z_CONJ		1
#define	Z_NOCONJ	0


#define	zm_entry(A,i,j)		zm_get_val(A,i,j)
#define	zv_entry(x,i)		zv_get_val(x,i)
#ifdef DEBUG
#define	zm_set_val(A,i,j,val)	( m_chk_idx(A,i,j) ? \
	(A)->me[(i)][(j)] = (val) : (error(E_BOUNDS,"zm_set_val"), zmake(0.0,0.0)))
#define	zm_add_val(A,i,j,val)	( m_chk_idx(A,i,j) ? \
	(A)->me[(i)][(j)] = zadd((A)->me[(i)][(j)],(val)) : \
	(error(E_BOUNDS,"zm_add_val"), zmake(0.0,0.0)))
#define	zm_sub_val(A,i,j,val)	( m_chk_idx(A,i,j) ? \
	(A)->me[(i)][(j)] = zsub((A)->me[(i)][(j)],(val)) : \
	(error(E_BOUNDS,"zm_sub_val"), zmake(0.0,0.0)))
#define	zm_get_val(A,i,j)	( m_chk_idx(A,i,j) ? \
	(A)->me[(i)][(j)] : (error(E_BOUNDS,"zm_get_val"), zmake(0.0,0.0)))
#define	zv_set_val(x,i,val)	( v_chk_idx(x,i) ? (x)->ve[(i)] = (val) : \
	(error(E_BOUNDS,"zv_set_val"), zmake(0.0,0.0)))
#define	zv_add_val(x,i,val)	( v_chk_idx(x,i) ? \
	(x)->ve[(i)] = zadd((x)->ve[(i)],(val)) : \
	(error(E_BOUNDS,"zv_set_val"), zmake(0.0,0.0)))
#define	zv_sub_val(x,i,val)	( v_chk_idx(x,i) ? \
	(x)->ve[(i)] = zsub((x)->ve[(i)],(val)) : \
	(error(E_BOUNDS,"zv_set_val"), zmake(0.0,0.0)))
#define	zv_get_val(x,i)	( v_chk_idx(x,i) ? (x)->ve[(i)] : \
	(error(E_BOUNDS,"zv_get_val"), zmake(0.0,0.0)))
#else /* no DEBUG */
#define	zm_set_val(A,i,j,val)	((A)->me[(i)][(j)] = (val))
#define	zm_add_val(A,i,j,val)	((A)->me[(i)][(j)] = zadd((A)->me[(i)][(j)],(val)))
#define	zm_sub_val(A,i,j,val)	((A)->me[(i)][(j)] = zsub((A)->me[(i)][(j)],(val)))
#define	zm_get_val(A,i,j)	((A)->me[(i)][(j)])
#define	zv_set_val(x,i,val)	((x)->ve[(i)] = (val))
#define	zv_add_val(x,i,val)	((x)->ve[(i)] = zadd((x)->ve[(i)],(val)))
#define	zv_sub_val(x,i,val)	((x)->ve[(i)] = zsub((x)->ve[(i)],(val)))
#define	zv_get_val(x,i)		((x)->ve[(i)])
#endif /* DEBUG */

/* memory functions */

#ifdef ANSI_C
int zv_get_vars(int dim,...);
int zm_get_vars(int m,int n,...);
int zv_resize_vars(int new_dim,...);
int zm_resize_vars(int m,int n,...);
int zv_free_vars(ZVEC **,...);
int zm_free_vars(ZMAT **,...);

#elif VARARGS
int zv_get_vars();
int zm_get_vars();
int zv_resize_vars();
int zm_resize_vars();
int zv_free_vars();
int zm_free_vars();

#endif




#ifdef ANSI_C
extern ZMAT	*_zm_copy(const ZMAT *in,ZMAT *out, int i0, int j0);
extern ZMAT	* zm_move(const ZMAT *, int, int, int, int, ZMAT *, int, int);
extern ZMAT	*zvm_move(const ZVEC *, int, ZMAT *, int, int, int, int);
extern ZVEC	*_zv_copy(const ZVEC *in,ZVEC *out,int i0);
extern ZVEC	* zv_move(const ZVEC *, int, int, ZVEC *, int);
extern ZVEC	*zmv_move(const ZMAT *, int, int, int, int, ZVEC *, int);
extern complex	z_finput(FILE *fp);
extern ZMAT	*zm_finput(FILE *fp,ZMAT *a);
extern ZVEC     *zv_finput(FILE *fp,ZVEC *x);
extern ZMAT	*zm_add(ZMAT *mat1,ZMAT *mat2,ZMAT *out);
extern ZMAT	*zm_sub(ZMAT *mat1,ZMAT *mat2,ZMAT *out);
extern ZMAT	*zm_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT);
extern ZMAT	*zmma_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT);
extern ZMAT	*zmam_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT);
extern ZVEC	*zmv_mlt(ZMAT *A,ZVEC *b,ZVEC *out);
extern ZMAT	*zsm_mlt(complex scalar,ZMAT *matrix,ZMAT *out);
extern ZVEC	*zvm_mlt(ZMAT *A,ZVEC *b,ZVEC *out);
extern ZMAT	*zm_adjoint(ZMAT *in,ZMAT *out);
extern ZMAT	*zswap_rows(ZMAT *A,int i,int j,int lo,int hi);
extern ZMAT	*zswap_cols(ZMAT *A,int i,int j,int lo,int hi);
extern ZMAT	*mz_mltadd(ZMAT *A1,ZMAT *A2,complex s,ZMAT *out);
extern ZVEC	*zmv_mltadd(ZVEC *v1,ZVEC *v2,ZMAT *A,complex alpha,ZVEC *out);
extern ZVEC	*zvm_mltadd(ZVEC *v1,ZVEC *v2,ZMAT *A,complex alpha,ZVEC *out);
extern ZVEC	*zv_zero(ZVEC *x);
extern ZMAT	*zm_zero(ZMAT *A);
extern ZMAT	*zm_get(int m,int n);
extern ZVEC	*zv_get(int dim);
extern ZMAT	*zm_resize(ZMAT *A,int new_m,int new_n);
extern complex	_zin_prod(const ZVEC *x, const ZVEC *y,unsigned int i0,unsigned int flag);
extern ZVEC	*zv_resize(ZVEC *x,int new_dim);
extern ZVEC	*zv_mlt(complex scalar,const ZVEC *vector,ZVEC *out);
extern ZVEC	*zv_add(const ZVEC *vec1,const ZVEC *vec2,ZVEC *out);
extern ZVEC	*zv_mltadd(const ZVEC *v1,const ZVEC *v2,complex scale,ZVEC *out);
extern ZVEC	*zv_sub(const ZVEC *vec1,const ZVEC *vec2,ZVEC *out);
#ifdef PROTOTYPES_IN_STRUCT
extern ZVEC	*zv_map(complex (*f)(),const ZVEC *x,ZVEC *out);
extern ZVEC	*_zv_map(complex (*f)(),void *params,const ZVEC *x,ZVEC *out);
#else
extern ZVEC	*zv_map(complex (*f)(complex),const ZVEC *x,ZVEC *out);
extern ZVEC	*_zv_map(complex (*f)(void *,complex),void *params,const ZVEC *x,ZVEC *out);
#endif
extern ZVEC	*zv_lincomb(int n,const ZVEC *v[],const complex a[],ZVEC *out);
extern ZVEC	*zv_linlist(ZVEC *out,ZVEC *v1,complex a1,...);
extern ZVEC	*zv_star(const ZVEC *x1, const ZVEC *x2, ZVEC *out);
extern ZVEC	*zv_slash(const ZVEC *x1, const ZVEC *x2, ZVEC *out);
extern complex	zv_sum(const ZVEC *x);
extern int	zm_free(ZMAT *mat);
extern int	zv_free(ZVEC *vec);

extern ZVEC	*zv_rand(ZVEC *x);
extern ZMAT	*zm_rand(ZMAT *A);

extern ZVEC	*zget_row(ZMAT *A, int i, ZVEC *out);
extern ZVEC	*zget_col(ZMAT *A, int j, ZVEC *out);
extern ZMAT	*zset_row(ZMAT *A, int i, ZVEC *in);
extern ZMAT	*zset_col(ZMAT *A, int j, ZVEC *in);

extern ZVEC	*px_zvec(PERM *pi, ZVEC *in, ZVEC *out);
extern ZVEC	*pxinv_zvec(PERM *pi, ZVEC *in, ZVEC *out);

extern void	__zconj__(complex zp[], int len);
extern complex	__zip__(const complex zp1[], const complex zp2[],
			int len,int flag);
extern void	__zmltadd__(complex zp1[], const complex zp2[],
			    complex s,int len,int flag);
extern void	__zmlt__(const complex zp[],complex s,complex out[],int len);
extern void	__zadd__(const complex zp1[],const complex zp2[],
			 complex out[],int len);
extern void	__zsub__(const complex zp1[],const complex zp2[],
			 complex out[],int len);
extern void	__zzero__(complex zp[],int len);
extern void	z_foutput(FILE *fp,complex z);
extern void     zm_foutput(FILE *fp,ZMAT *a);
extern void     zv_foutput(FILE *fp,ZVEC *x);
extern void     zm_dump(FILE *fp,ZMAT *a);
extern void     zv_dump(FILE *fp,ZVEC *x);

extern double	_zv_norm1(ZVEC *x, VEC *scale);
extern double	_zv_norm2(ZVEC *x, VEC *scale);
extern double	_zv_norm_inf(ZVEC *x, VEC *scale);
extern double	zm_norm1(ZMAT *A);
extern double	zm_norm_inf(ZMAT *A);
extern double	zm_norm_frob(ZMAT *A);

complex	zmake(double real, double imag);
double	zabs(complex z);
complex zadd(complex z1,complex z2);
complex zsub(complex z1,complex z2);
complex	zmlt(complex z1,complex z2);
complex	zinv(complex z);
complex	zdiv(complex z1,complex z2);
complex	zsqrt(complex z);
complex	zexp(complex z);
complex	zlog(complex z);
complex	zconj(complex z);
complex	zneg(complex z);
#else
extern ZMAT	*_zm_copy();
extern ZVEC	*_zv_copy();
extern ZMAT	*zm_finput();
extern ZVEC     *zv_finput();
extern ZMAT	*zm_add();
extern ZMAT	*zm_sub();
extern ZMAT	*zm_mlt();
extern ZMAT	*zmma_mlt();
extern ZMAT	*zmam_mlt();
extern ZVEC	*zmv_mlt();
extern ZMAT	*zsm_mlt();
extern ZVEC	*zvm_mlt();
extern ZMAT	*zm_adjoint();
extern ZMAT	*zswap_rows();
extern ZMAT	*zswap_cols();
extern ZMAT	*mz_mltadd();
extern ZVEC	*zmv_mltadd();
extern ZVEC	*zvm_mltadd();
extern ZVEC	*zv_zero();
extern ZMAT	*zm_zero();
extern ZMAT	*zm_get();
extern ZVEC	*zv_get();
extern ZMAT	*zm_resize();
extern ZVEC	*zv_resize();
extern complex	_zin_prod();
extern ZVEC	*zv_mlt();
extern ZVEC	*zv_add();
extern ZVEC	*zv_mltadd();
extern ZVEC	*zv_sub();
extern ZVEC	*zv_map();
extern ZVEC	*_zv_map();
extern ZVEC	*zv_lincomb();
extern ZVEC	*zv_linlist();
extern ZVEC	*zv_star();
extern ZVEC	*zv_slash();

extern ZVEC	*px_zvec();
extern ZVEC	*pxinv_zvec();

extern ZVEC	*zv_rand();
extern ZMAT	*zm_rand();

extern ZVEC	*zget_row();
extern ZVEC	*zget_col();
extern ZMAT	*zset_row();
extern ZMAT	*zset_col();

extern int	zm_free();
extern int	zv_free();
extern void	__zconj__();
extern complex	__zip__();
extern void	__zmltadd__();
extern void	__zmlt__();
extern void	__zadd__();
extern void	__zsub__();
extern void	__zzero__();
extern void    zm_foutput();
extern void    zv_foutput();
extern void    zm_dump();
extern void    zv_dump();

extern double	_zv_norm1();
extern double	_zv_norm2();
extern double	_zv_norm_inf();
extern double	zm_norm1();
extern double	zm_norm_inf();
extern double	zm_norm_frob();

complex	zmake();
double	zabs();
complex zadd();
complex zsub();
complex	zmlt();
complex	zinv();
complex	zdiv();
complex	zsqrt();
complex	zexp();
complex	zlog();
complex	zconj();
complex	zneg();
#endif

#define	zv_copy(x,y)	_zv_copy(x,y,0)
#define	zm_copy(A,B)	_zm_copy(A,B,0,0)

#define	z_input()	z_finput(stdin)
#define	zv_input(x)	zv_finput(stdin,x)
#define	zm_input(A)	zm_finput(stdin,A)
#define	z_output(z)	z_foutput(stdout,z)
#define	zv_output(x)	zv_foutput(stdout,x)
#define	zm_output(A)	zm_foutput(stdout,A)

#define	ZV_FREE(x)	( zv_free(x), (x) = ZVNULL )
#define	ZM_FREE(A)	( zm_free(A), (A) = ZMNULL )

#define	zin_prod(x,y)	_zin_prod(x,y,0,Z_CONJ)

#define	zv_norm1(x)	_zv_norm1(x,VNULL)
#define	zv_norm2(x)	_zv_norm2(x,VNULL)
#define	zv_norm_inf(x)	_zv_norm_inf(x,VNULL)


#endif
