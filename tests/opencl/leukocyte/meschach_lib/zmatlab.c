
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
	This file contains routines for import/exporting complex data
	to/from MATLAB. The main routines are:
			ZMAT *zm_save(FILE *fp,ZMAT *A,char *name)
			ZVEC *zv_save(FILE *fp,ZVEC *x,char *name)
			complex z_save(FILE *fp,complex z,char *name)
			ZMAT *zm_load(FILE *fp,char **name)
*/

#include        <stdio.h>
#include        "zmatrix.h"
#include	"matlab.h"

static char rcsid[] = "$Id: zmatlab.c,v 1.2 1995/02/14 20:13:27 des Exp $";

/* zm_save -- save matrix in ".mat" file for MATLAB
   -- returns matrix to be saved */
ZMAT    *zm_save(fp,A,name)
FILE    *fp;
ZMAT    *A;
char    *name;
{
    int     i, j;
    matlab  mat;
    
    if ( ! A )
	error(E_NULL,"zm_save");
    
    mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
    mat.m = A->m;
    mat.n = A->n;
    mat.imag = TRUE;
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
	for ( j = 0; j < A->n; j++ )
	    fwrite(&(A->me[i][j].re),sizeof(Real),1,fp);
    for ( i = 0; i < A->m; i++ )
	for ( j = 0; j < A->n; j++ )
	    fwrite(&(A->me[i][j].im),sizeof(Real),1,fp);
#else /* column major order: ORDER == COL_ORDER */
    for ( j = 0; j < A->n; j++ )
	for ( i = 0; i < A->m; i++ )
	    fwrite(&(A->me[i][j].re),sizeof(Real),1,fp);
    for ( j = 0; j < A->n; j++ )
	for ( i = 0; i < A->m; i++ )
	    fwrite(&(A->me[i][j].im),sizeof(Real),1,fp);
#endif
    
    return A;
}


/* zv_save -- save vector in ".mat" file for MATLAB
   -- saves it as a row vector
   -- returns vector to be saved */
ZVEC    *zv_save(fp,x,name)
FILE    *fp;
ZVEC    *x;
char    *name;
{
    int	i, j;
    matlab  mat;
    
    if ( ! x )
	error(E_NULL,"zv_save");
    
    mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
    mat.m = x->dim;
    mat.n = 1;
    mat.imag = TRUE;
    mat.namlen = (name == (char *)NULL) ? 1 : strlen(name)+1;
    
    /* write header */
    fwrite(&mat,sizeof(matlab),1,fp);
    /* write name */
    if ( name == (char *)NULL )
	fwrite("",sizeof(char),1,fp);
    else
	fwrite(name,sizeof(char),(int)(mat.namlen),fp);
    /* write actual data */
    for ( i = 0; i < x->dim; i++ )
	fwrite(&(x->ve[i].re),sizeof(Real),1,fp);
    for ( i = 0; i < x->dim; i++ )
	fwrite(&(x->ve[i].im),sizeof(Real),1,fp);
    
    return x;
}

/* z_save -- saves complex number in ".mat" file for MATLAB
	-- returns complex number to be saved */
complex	z_save(fp,z,name)
FILE	*fp;
complex	z;
char	*name;
{
    matlab  mat;
    
    mat.type = 1000*MACH_ID + 100*ORDER + 10*PRECISION + 0;
    mat.m = 1;
    mat.n = 1;
    mat.imag = TRUE;
    mat.namlen = (name == (char *)NULL) ? 1 : strlen(name)+1;
    
    /* write header */
    fwrite(&mat,sizeof(matlab),1,fp);
    /* write name */
    if ( name == (char *)NULL )
	fwrite("",sizeof(char),1,fp);
    else
	fwrite(name,sizeof(char),(int)(mat.namlen),fp);
    /* write actual data */
    fwrite(&z,sizeof(complex),1,fp);
    
    return z;
}



/* zm_load -- loads in a ".mat" file variable as produced by MATLAB
   -- matrix returned; imaginary parts ignored */
ZMAT    *zm_load(fp,name)
FILE    *fp;
char    **name;
{
    ZMAT     *A;
    int     i;
    int     m_flag, o_flag, p_flag, t_flag;
    float   f_temp;
    double  d_temp;
    matlab  mat;
    
    if ( fread(&mat,sizeof(matlab),1,fp) != 1 )
	error(E_FORMAT,"zm_load");
    if ( mat.type >= 10000 )	/* don't load a sparse matrix! */
	error(E_FORMAT,"zm_load");
    m_flag = (mat.type/1000) % 10;
    o_flag = (mat.type/100) % 10;
    p_flag = (mat.type/10) % 10;
    t_flag = (mat.type) % 10;
    if ( m_flag != MACH_ID )
	error(E_FORMAT,"zm_load");
    if ( t_flag != 0 )
	error(E_FORMAT,"zm_load");
    if ( p_flag != DOUBLE_PREC && p_flag != SINGLE_PREC )
	error(E_FORMAT,"zm_load");
    *name = (char *)malloc((unsigned)(mat.namlen)+1);
    if ( fread(*name,sizeof(char),(unsigned)(mat.namlen),fp) == 0 )
	error(E_FORMAT,"zm_load");
    A = zm_get((unsigned)(mat.m),(unsigned)(mat.n));
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
	    A->me[i / A->n][i % A->n].re = d_temp;
	else if ( o_flag == COL_ORDER )
	    A->me[i % A->m][i / A->m].re = d_temp;
	else
	    error(E_FORMAT,"zm_load");
    }
    
    if ( mat.imag )         /* skip imaginary part */
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
		A->me[i / A->n][i % A->n].im = d_temp;
	    else if ( o_flag == COL_ORDER )
		A->me[i % A->m][i / A->m].im = d_temp;
	    else
		error(E_FORMAT,"zm_load");
	}
    
    return A;
}

