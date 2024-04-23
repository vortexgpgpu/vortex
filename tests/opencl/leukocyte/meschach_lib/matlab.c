
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
	This file contains routines for import/exporting data to/from
		MATLAB. The main routines are:
			MAT *m_save(FILE *fp,MAT *A,char *name)
			VEC *v_save(FILE *fp,VEC *x,char *name)
			MAT *m_load(FILE *fp,char **name)
*/

#include        <stdio.h>
#include        "matrix.h"
#include	"matlab.h"

static char rcsid[] = "$Id: matlab.c,v 1.8 1995/02/14 20:12:36 des Exp $";

/* m_save -- save matrix in ".mat" file for MATLAB
	-- returns matrix to be saved */
#ifndef ANSI_C
MAT     *m_save(fp,A,name)
FILE    *fp;
MAT     *A;
char    *name;
#else
MAT     *m_save(FILE *fp, MAT *A, const char *name)
#endif
{
	int     i, j;
	matlab  mat;

	if ( ! A )
		error(E_NULL,"m_save");

	mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
	mat.m = A->m;
	mat.n = A->n;
	mat.imag = FALSE;
	mat.namlen = (name == (char *)NULL) ? 1 : strlen(name)+1;

	/* write header */
	fwrite(&mat,sizeof(matlab),1,fp);
	/* write name */
	if ( name == (char *)NULL )
		fwrite("",sizeof(char),1,fp);
	else
		fwrite(name,sizeof(char),(int)(mat.namlen),fp);
	/* write actual data */
#if ORDER == ROW_ORDER
	for ( i = 0; i < A->m; i++ )
		fwrite(A->me[i],sizeof(Real),(int)(A->n),fp);
#else /* column major order: ORDER == COL_ORDER */
	for ( j = 0; j < A->n; j++ )
	  for ( i = 0; i < A->m; i++ )
	    fwrite(&(A->me[i][j]),sizeof(Real),1,fp);
#endif

	return A;
}


/* v_save -- save vector in ".mat" file for MATLAB
	-- saves it as a row vector
	-- returns vector to be saved */
#ifndef ANSI_C
VEC     *v_save(fp,x,name)
FILE    *fp;
VEC     *x;
char    *name;
#else
VEC     *v_save(FILE *fp, VEC *x, const char *name)
#endif
{
	matlab  mat;

	if ( ! x )
		error(E_NULL,"v_save");

	mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
	mat.m = x->dim;
	mat.n = 1;
	mat.imag = FALSE;
	mat.namlen = (name == (char *)NULL) ? 1 : strlen(name)+1;

	/* write header */
	fwrite(&mat,sizeof(matlab),1,fp);
	/* write name */
	if ( name == (char *)NULL )
		fwrite("",sizeof(char),1,fp);
	else
		fwrite(name,sizeof(char),(int)(mat.namlen),fp);
	/* write actual data */
	fwrite(x->ve,sizeof(Real),(int)(x->dim),fp);

	return x;
}

/* d_save -- save double in ".mat" file for MATLAB
	-- saves it as a row vector
	-- returns vector to be saved */
#ifndef ANSI_C
double	d_save(fp,x,name)
FILE    *fp;
double	x;
char    *name;
#else
double	d_save(FILE *fp, double x, const char *name)
#endif
{
	matlab  mat;
	Real x1 = x;

	mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
	mat.m = 1;
	mat.n = 1;
	mat.imag = FALSE;
	mat.namlen = (name == (char *)NULL) ? 1 : strlen(name)+1;

	/* write header */
	fwrite(&mat,sizeof(matlab),1,fp);
	/* write name */
	if ( name == (char *)NULL )
		fwrite("",sizeof(char),1,fp);
	else
		fwrite(name,sizeof(char),(int)(mat.namlen),fp);
	/* write actual data */
	fwrite(&x1,sizeof(Real),1,fp);

	return x;
}

/* m_load -- loads in a ".mat" file variable as produced by MATLAB
	-- matrix returned; imaginary parts ignored */
#ifndef ANSI_C
MAT     *m_load(fp,name)
FILE    *fp;
char    **name;
#else
MAT     *m_load(FILE *fp, char **name)
#endif
{
	MAT     *A;
	int     i;
	int     m_flag, o_flag, p_flag, t_flag;
	float   f_temp;
	Real    d_temp;
	matlab  mat;

	if ( fread(&mat,sizeof(matlab),1,fp) != 1 )
	    error(E_FORMAT,"m_load");
	if ( mat.type >= 10000 )	/* don't load a sparse matrix! */
	    error(E_FORMAT,"m_load");
	m_flag = (mat.type/1000) % 10;
	o_flag = (mat.type/100) % 10;
	p_flag = (mat.type/10) % 10;
	t_flag = (mat.type) % 10;
	if ( m_flag != MACH_ID )
		error(E_FORMAT,"m_load");
	if ( t_flag != 0 )
		error(E_FORMAT,"m_load");
	if ( p_flag != DOUBLE_PREC && p_flag != SINGLE_PREC )
		error(E_FORMAT,"m_load");
	*name = (char *)malloc((unsigned)(mat.namlen)+1);
	if ( fread(*name,sizeof(char),(unsigned)(mat.namlen),fp) == 0 )
		error(E_FORMAT,"m_load");
	A = m_get((unsigned)(mat.m),(unsigned)(mat.n));
	for ( i = 0; i < A->m*A->n; i++ )
	{
		if ( p_flag == DOUBLE_PREC )
		    fread(&d_temp,sizeof(double),1,fp);
		else
		{
		    fread(&f_temp,sizeof(float),1,fp);
		    d_temp = f_temp;
		}
		if ( o_flag == ROW_ORDER )
		    A->me[i / A->n][i % A->n] = d_temp;
		else if ( o_flag == COL_ORDER )
		    A->me[i % A->m][i / A->m] = d_temp;
		else
		    error(E_FORMAT,"m_load");
	}

	if ( mat.imag )         /* skip imaginary part */
	for ( i = 0; i < A->m*A->n; i++ )
	{
		if ( p_flag == DOUBLE_PREC )
		    fread(&d_temp,sizeof(double),1,fp);
		else
		    fread(&f_temp,sizeof(float),1,fp);
	}

	return A;
}

