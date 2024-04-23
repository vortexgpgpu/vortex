
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


/* 1.6 matrixio.c 11/25/87 */


#include        <stdio.h>
#include        <ctype.h>
#include        "matrix.h"

static char rcsid[] = "$Id: matrixio.c,v 1.4 1994/01/13 05:31:10 des Exp $";


/* local variables */
static char line[MAXLINE];


/**************************************************************************
  Input routines
  **************************************************************************/
/* skipjunk -- skips white spaces and strings of the form #....\n
   Here .... is a comment string */
#ifndef ANSI_C
int     skipjunk(fp)
FILE    *fp;
#else
int	skipjunk(FILE *fp)
#endif
{
     int        c;
     
     for ( ; ; )        /* forever do... */
     {
	  /* skip blanks */
	  do
	       c = getc(fp);
	  while ( isspace(c) );
	  
	  /* skip comments (if any) */
	  if ( c == '#' )
	       /* yes it is a comment (line) */
	       while ( (c=getc(fp)) != '\n' )
		    ;
	  else
	  {
	       ungetc(c,fp);
	       break;
	  }
     }
     return 0;
}

/* m_finput -- input matrix
	-- input from a terminal is handled interactively
	-- batch/file input has the same format as produced by m_foutput
	except that whitespace and comments ("#..\n") are skipped
	-- returns a, which is created if a == NULL on entry */
#ifndef ANSI_C
MAT     *m_finput(fp,a)
FILE    *fp;
MAT     *a;
#else
MAT	*m_finput(FILE *fp, MAT *a)
#endif
{
     MAT        *im_finput(),*bm_finput();
     
     if ( isatty(fileno(fp)) )
	  return im_finput(fp,a);
     else
	  return bm_finput(fp,a);
}

/* im_finput -- interactive input of matrix */
#ifndef ANSI_C
MAT     *im_finput(fp,mat)
FILE    *fp;
MAT     *mat;
#else
MAT     *im_finput(FILE *fp,MAT *mat)
#endif
{
     char       c;
     unsigned int      i, j, m, n, dynamic;
     /* dynamic set to TRUE if memory allocated here */
     
     /* get matrix size */
     if ( mat != (MAT *)NULL && mat->m<MAXDIM && mat->n<MAXDIM )
     {  m = mat->m;     n = mat->n;     dynamic = FALSE;        }
     else
     {
	  dynamic = TRUE;
	  do
	  {
	       fprintf(stderr,"Matrix: rows cols:");
	       if ( fgets(line,MAXLINE,fp)==NULL )
		    error(E_INPUT,"im_finput");
	  } while ( sscanf(line,"%u%u",&m,&n)<2 || m>MAXDIM || n>MAXDIM );
	  mat = m_get(m,n);
     }
     
     /* input elements */
     for ( i=0; i<m; i++ )
     {
     redo:
	  fprintf(stderr,"row %u:\n",i);
	  for ( j=0; j<n; j++ )
	       do
	       {
	       redo2:
		    fprintf(stderr,"entry (%u,%u): ",i,j);
		    if ( !dynamic )
			 fprintf(stderr,"old %14.9g new: ",
				 mat->me[i][j]);
		    if ( fgets(line,MAXLINE,fp)==NULL )
			 error(E_INPUT,"im_finput");
		    if ( (*line == 'b' || *line == 'B') && j > 0 )
		    {   j--;    dynamic = FALSE;        goto redo2;     }
		    if ( (*line == 'f' || *line == 'F') && j < n-1 )
		    {   j++;    dynamic = FALSE;        goto redo2;     }
#if REAL == DOUBLE
	       } while ( *line=='\0' || sscanf(line,"%lf",&mat->me[i][j])<1 );
#elif REAL == FLOAT
	       } while ( *line=='\0' || sscanf(line,"%f",&mat->me[i][j])<1 );
#endif
	  fprintf(stderr,"Continue: ");
	  fscanf(fp,"%c",&c);
	  if ( c == 'n' || c == 'N' )
	  {    dynamic = FALSE;                 goto redo;      }
	  if ( (c == 'b' || c == 'B') /* && i > 0 */ )
	  {     if ( i > 0 )
		    i--;
		dynamic = FALSE;        goto redo;
	  }
     }
     
     return (mat);
}

/* bm_finput -- batch-file input of matrix */
#ifndef ANSI_C
MAT     *bm_finput(fp,mat)
FILE    *fp;
MAT     *mat;
#else
MAT     *bm_finput(FILE *fp,MAT *mat)
#endif
{
     unsigned int      i,j,m,n,dummy;
     int        io_code;
     
     /* get dimension */
     skipjunk(fp);
     if ((io_code=fscanf(fp," Matrix: %u by %u",&m,&n)) < 2 ||
	 m>MAXDIM || n>MAXDIM )
	  error(io_code==EOF ? E_EOF : E_FORMAT,"bm_finput");
     
     /* allocate memory if necessary */
     if ( mat==(MAT *)NULL )
	  mat = m_resize(mat,m,n);
     
     /* get entries */
     for ( i=0; i<m; i++ )
     {
	  skipjunk(fp);
	  if ( fscanf(fp," row %u:",&dummy) < 1 )
	       error(E_FORMAT,"bm_finput");
	  for ( j=0; j<n; j++ )
#if REAL == DOUBLE
	       if ((io_code=fscanf(fp,"%lf",&mat->me[i][j])) < 1 )
#elif REAL == FLOAT
	       if ((io_code=fscanf(fp,"%f",&mat->me[i][j])) < 1 )
#endif
		    error(io_code==EOF ? 7 : 6,"bm_finput");
     }
     
     return (mat);
}

/* px_finput -- inputs permutation from file/stream fp
	-- input from a terminal is handled interactively
	-- batch/file input has the same format as produced by px_foutput
	except that whitespace and comments ("#..\n") are skipped
	-- returns px, which is created if px == NULL on entry */
#ifndef ANSI_C
PERM    *px_finput(fp,px)
FILE    *fp;
PERM    *px;
#else
PERM    *px_finput(FILE *fp,PERM *px)
#endif
{
     PERM       *ipx_finput(),*bpx_finput();
     
     if ( isatty(fileno(fp)) )
	  return ipx_finput(fp,px);
     else
	  return bpx_finput(fp,px);
}


/* ipx_finput -- interactive input of permutation */
#ifndef ANSI_C
PERM    *ipx_finput(fp,px)
FILE    *fp;
PERM    *px;
#else
PERM    *ipx_finput(FILE *fp,PERM *px)
#endif
{
     unsigned int      i,j,size,dynamic; /* dynamic set if memory allocated here */
     unsigned int      entry,ok;
     
     /* get permutation size */
     if ( px!=(PERM *)NULL && px->size<MAXDIM )
     {  size = px->size;        dynamic = FALSE;        }
     else
     {
	  dynamic = TRUE;
	  do
	  {
	       fprintf(stderr,"Permutation: size: ");
	       if ( fgets(line,MAXLINE,fp)==NULL )
		    error(E_INPUT,"ipx_finput");
	  } while ( sscanf(line,"%u",&size)<1 || size>MAXDIM );
	  px = px_get(size);
     }
     
     /* get entries */
     i = 0;
     while ( i<size )
     {
	  /* input entry */
	  do
	  {
	  redo:
	       fprintf(stderr,"entry %u: ",i);
	       if ( !dynamic )
		    fprintf(stderr,"old: %u->%u new: ",
			    i,px->pe[i]);
	       if ( fgets(line,MAXLINE,fp)==NULL )
		    error(E_INPUT,"ipx_finput");
	       if ( (*line == 'b' || *line == 'B') && i > 0 )
	       {        i--;    dynamic = FALSE;        goto redo;      }
	  } while ( *line=='\0' || sscanf(line,"%u",&entry) < 1 );
	  /* check entry */
	  ok = (entry < size);
	  for ( j=0; j<i; j++ )
	       ok &= (entry != px->pe[j]);
	  if ( ok )
	  {
	       px->pe[i] = entry;
	       i++;
	  }
     }
     
     return (px);
}

/* bpx_finput -- batch-file input of permutation */
#ifndef ANSI_C
PERM    *bpx_finput(fp,px)
FILE    *fp;
PERM    *px;
#else
PERM    *bpx_finput(FILE *fp,PERM *px)
#endif
{
     unsigned int      i,j,size,entry,ok;
     int        io_code;
     
     /* get size of permutation */
     skipjunk(fp);
     if ((io_code=fscanf(fp," Permutation: size:%u",&size)) < 1 ||
	 size>MAXDIM )
	  error(io_code==EOF ? 7 : 6,"bpx_finput");
     
     /* allocate memory if necessary */
     if ( px==(PERM *)NULL || px->size<size )
	  px = px_resize(px,size);
     
     /* get entries */
     skipjunk(fp);
     i = 0;
     while ( i<size )
     {
	  /* input entry */
	  if ((io_code=fscanf(fp,"%*u -> %u",&entry)) < 1 )
	       error(io_code==EOF ? 7 : 6,"bpx_finput");
	  /* check entry */
	  ok = (entry < size);
	  for ( j=0; j<i; j++ )
	       ok &= (entry != px->pe[j]);
	  if ( ok )
	  {
	       px->pe[i] = entry;
	       i++;
	  }
	  else
	       error(E_BOUNDS,"bpx_finput");
     }
     
     return (px);
}

/* v_finput -- inputs vector from file/stream fp
	-- input from a terminal is handled interactively
	-- batch/file input has the same format as produced by px_foutput
	except that whitespace and comments ("#..\n") are skipped
	-- returns x, which is created if x == NULL on entry */
#ifndef ANSI_C
VEC     *v_finput(fp,x)
FILE    *fp;
VEC     *x;
#else
VEC     *v_finput(FILE *fp,VEC *x)
#endif
{
     VEC        *ifin_vec(),*bfin_vec();
     
     if ( isatty(fileno(fp)) )
	  return ifin_vec(fp,x);
     else
	  return bfin_vec(fp,x);
}

/* ifin_vec -- interactive input of vector */
#ifndef ANSI_C
VEC     *ifin_vec(fp,vec)
FILE    *fp;
VEC     *vec;
#else
VEC     *ifin_vec(FILE *fp,VEC *vec)
#endif
{
     unsigned int      i,dim,dynamic;  /* dynamic set if memory allocated here */
     
     /* get vector dimension */
     if ( vec != (VEC *)NULL && vec->dim<MAXDIM )
     {  dim = vec->dim; dynamic = FALSE;        }
     else
     {
	  dynamic = TRUE;
	  do
	  {
	       fprintf(stderr,"Vector: dim: ");
	       if ( fgets(line,MAXLINE,fp)==NULL )
		    error(E_INPUT,"ifin_vec");
	  } while ( sscanf(line,"%u",&dim)<1 || dim>MAXDIM );
	  vec = v_get(dim);
     }
     
     /* input elements */
     for ( i=0; i<dim; i++ )
	  do
	  {
	  redo:
	       fprintf(stderr,"entry %u: ",i);
	       if ( !dynamic )
		    fprintf(stderr,"old %14.9g new: ",vec->ve[i]);
	       if ( fgets(line,MAXLINE,fp)==NULL )
		    error(E_INPUT,"ifin_vec");
	       if ( (*line == 'b' || *line == 'B') && i > 0 )
	       {        i--;    dynamic = FALSE;        goto redo;         }
	       if ( (*line == 'f' || *line == 'F') && i < dim-1 )
	       {        i++;    dynamic = FALSE;        goto redo;         }
#if REAL == DOUBLE
	  } while ( *line=='\0' || sscanf(line,"%lf",&vec->ve[i]) < 1 );
#elif REAL == FLOAT
          } while ( *line=='\0' || sscanf(line,"%f",&vec->ve[i]) < 1 );
#endif
     
     return (vec);
}

/* bfin_vec -- batch-file input of vector */
#ifndef ANSI_C
VEC     *bfin_vec(fp,vec)
FILE    *fp;
VEC     *vec;
#else
VEC     *bfin_vec(FILE *fp,VEC *vec)
#endif
{
     unsigned int      i,dim;
     int        io_code;
     
     /* get dimension */
     skipjunk(fp);
     if ((io_code=fscanf(fp," Vector: dim:%u",&dim)) < 1 ||
	 dim>MAXDIM )
	  error(io_code==EOF ? 7 : 6,"bfin_vec");
     
     /* allocate memory if necessary */
     if ( vec==(VEC *)NULL )
	  vec = v_resize(vec,dim);
     
     /* get entries */
     skipjunk(fp);
     for ( i=0; i<dim; i++ )
#if REAL == DOUBLE
	  if ((io_code=fscanf(fp,"%lf",&vec->ve[i])) < 1 )
#elif REAL == FLOAT
	  if ((io_code=fscanf(fp,"%f",&vec->ve[i])) < 1 )
#endif
	       error(io_code==EOF ? 7 : 6,"bfin_vec");
     
     return (vec);
}

/**************************************************************************
  Output routines
  **************************************************************************/
static const char    *format = "%14.9g ";

/* setformat -- sets the printf format string for the Meschach I/O operations
	-- returns the previous format string */
#ifndef ANSI_C
char	*setformat(f_string)
char    *f_string;
#else
const char	*setformat(const char *f_string)
#endif
{
    const char	*old_f_string;
    old_f_string = format;
    if ( f_string != (char *)NULL && *f_string != '\0' )
	format = f_string;

    return old_f_string;
}

/* m_foutput -- prints a representation of the matrix a onto file/stream fp */
#ifndef ANSI_C
void    m_foutput(fp,a)
FILE    *fp;
MAT     *a;
#else
void    m_foutput(FILE *fp, const MAT *a)
#endif
{
     unsigned int      i, j, tmp;
     
     if ( a == (MAT *)NULL )
     {  fprintf(fp,"Matrix: NULL\n");   return;         }
     fprintf(fp,"Matrix: %d by %d\n",a->m,a->n);
     if ( a->me == (Real **)NULL )
     {  fprintf(fp,"NULL\n");           return;         }
     for ( i=0; i<a->m; i++ )   /* for each row... */
     {
	  fprintf(fp,"row %u: ",i);
	  for ( j=0, tmp=2; j<a->n; j++, tmp++ )
	  {             /* for each col in row... */
	       fprintf(fp,format,a->me[i][j]);
	       if ( ! (tmp % 5) )       putc('\n',fp);
	  }
	  if ( tmp % 5 != 1 )   putc('\n',fp);
     }
}

/* px_foutput -- prints a representation of px onto file/stream fp */
#ifndef ANSI_C
void    px_foutput(fp,px)
FILE    *fp;
PERM    *px;
#else
void	px_foutput(FILE *fp, const PERM *px)
#endif
{
     unsigned int      i;
     
     if ( px == (PERM *)NULL )
     {  fprintf(fp,"Permutation: NULL\n");      return;         }
     fprintf(fp,"Permutation: size: %u\n",px->size);
     if ( px->pe == (unsigned int *)NULL )
     {  fprintf(fp,"NULL\n");   return;         }
     for ( i=0; i<px->size; i++ )
	if ( ! (i % 8) && i != 0 )
	  fprintf(fp,"\n  %u->%u ",i,px->pe[i]);
	else
	  fprintf(fp,"%u->%u ",i,px->pe[i]);
     fprintf(fp,"\n");
}

/* v_foutput -- prints a representation of x onto file/stream fp */
#ifndef ANSI_C
void    v_foutput(fp,x)
FILE    *fp;
VEC     *x;
#else
void	v_foutput(FILE *fp, const VEC *x)
#endif
{
     unsigned int      i, tmp;
     
     if ( x == (VEC *)NULL )
     {  fprintf(fp,"Vector: NULL\n");   return;         }
     fprintf(fp,"Vector: dim: %d\n",x->dim);
     if ( x->ve == (Real *)NULL )
     {  fprintf(fp,"NULL\n");   return;         }
     for ( i=0, tmp=0; i<x->dim; i++, tmp++ )
     {
	  fprintf(fp,format,x->ve[i]);
	  if ( tmp % 5 == 4 )   putc('\n',fp);
     }
     if ( tmp % 5 != 0 )        putc('\n',fp);
}

/* m_dump -- prints a dump of all pointers and data in a onto fp
	-- suitable for low-level debugging */
#ifndef ANSI_C
void    m_dump(fp,a)
FILE    *fp;
MAT     *a;
#else
void	m_dump(FILE *fp, const MAT *a)
#endif
{
	unsigned int   i, j, tmp;
     
     if ( a == (MAT *)NULL )
     {  fprintf(fp,"Matrix: NULL\n");   return;         }
     fprintf(fp,"Matrix: %d by %d @ 0x%lx\n",a->m,a->n,(long)a);
     fprintf(fp,"\tmax_m = %d, max_n = %d, max_size = %d\n",
	     a->max_m, a->max_n, a->max_size);
     if ( a->me == (Real **)NULL )
     {  fprintf(fp,"NULL\n");           return;         }
     fprintf(fp,"a->me @ 0x%lx\n",(long)(a->me));
     fprintf(fp,"a->base @ 0x%lx\n",(long)(a->base));
     for ( i=0; i<a->m; i++ )   /* for each row... */
     {
	  fprintf(fp,"row %u: @ 0x%lx ",i,(long)(a->me[i]));
	  for ( j=0, tmp=2; j<a->n; j++, tmp++ )
	  {             /* for each col in row... */
	       fprintf(fp,format,a->me[i][j]);
	       if ( ! (tmp % 5) )       putc('\n',fp);
	  }
	  if ( tmp % 5 != 1 )   putc('\n',fp);
     }
}

/* px_dump --  prints a dump of all pointers and data in a onto fp
	-- suitable for low-level debugging */
#ifndef ANSI_C
void    px_dump(fp,px)
FILE    *fp;
PERM    *px;
#else
void	px_dump(FILE *fp, const PERM *px)
#endif
{
     unsigned int      i;
     
     if ( ! px )
     {  fprintf(fp,"Permutation: NULL\n");      return;         }
     fprintf(fp,"Permutation: size: %u @ 0x%lx\n",px->size,(long)(px));
     if ( ! px->pe )
     {  fprintf(fp,"NULL\n");   return;         }
     fprintf(fp,"px->pe @ 0x%lx\n",(long)(px->pe));
     for ( i=0; i<px->size; i++ )
	  fprintf(fp,"%u->%u ",i,px->pe[i]);
     fprintf(fp,"\n");
}


/* v_dump --  prints a dump of all pointers and data in a onto fp
	-- suitable for low-level debugging */
#ifndef ANSI_C
void    v_dump(fp,x)
FILE    *fp;
VEC     *x;
#else
void	v_dump(FILE *fp, const VEC *x)
#endif
{
     unsigned int      i, tmp;
     
     if ( ! x )
     {  fprintf(fp,"Vector: NULL\n");   return;         }
     fprintf(fp,"Vector: dim: %d @ 0x%lx\n",x->dim,(long)(x));
     if ( ! x->ve )
     {  fprintf(fp,"NULL\n");   return;         }
     fprintf(fp,"x->ve @ 0x%lx\n",(long)(x->ve));
     for ( i=0, tmp=0; i<x->dim; i++, tmp++ )
     {
	  fprintf(fp,format,x->ve[i]);
	  if ( tmp % 5 == 4 )   putc('\n',fp);
     }
     if ( tmp % 5 != 0 )        putc('\n',fp);
}


/* iv_foutput -- print a representation of iv on stream fp */
#ifndef ANSI_C
void	iv_foutput(fp,iv)
FILE	*fp;
IVEC	*iv;
#else
void	iv_foutput(FILE *fp, const IVEC *iv)
#endif
{
   int	i;
   
   fprintf(fp,"IntVector: ");
   if ( iv == IVNULL )
   {
      fprintf(fp,"**** NULL ****\n");
      return;
   }
   fprintf(fp,"dim: %d\n",iv->dim);
   for ( i = 0; i < iv->dim; i++ )
   {
      if ( (i+1) % 8 )
	fprintf(fp,"%8d ",iv->ive[i]);
      else
	fprintf(fp,"%8d\n",iv->ive[i]);
   }
   if ( i % 8 )
     fprintf(fp,"\n");
}


/* iv_finput -- input integer vector from stream fp
	-- input from a terminal is handled interactively
	-- batch/file input has the same format as produced by
	iv_foutput except that whitespace and comments ("#...\n") 
	are skipped */
#ifndef ANSI_C
IVEC	*iv_finput(fp,x)
FILE	*fp;
IVEC	*x;
#else
IVEC	*iv_finput(FILE *fp, IVEC *x)
#endif
{
   IVEC	*iiv_finput(),*biv_finput();
   
   if ( isatty(fileno(fp)) )
     return iiv_finput(fp,x);
   else
     return biv_finput(fp,x);
}

/* iiv_finput -- interactive input of IVEC iv */
#ifndef ANSI_C
IVEC	*iiv_finput(fp,iv)
FILE	*fp;
IVEC	*iv;
#else
IVEC	*iiv_finput(FILE *fp, IVEC *iv)
#endif
{
   unsigned int	i,dim,dynamic;	/* dynamic set if memory allocated here */
   
   /* get dimension */
   if ( iv != (IVEC *)NULL && iv->dim<MAXDIM )
   {	dim = iv->dim;	dynamic = FALSE;	}
   else
   {
      dynamic = TRUE;
      do
      {
	 fprintf(stderr,"IntVector: dim: ");
	 if ( fgets(line,MAXLINE,fp)==NULL )
	   error(E_INPUT,"iiv_finput");
      } while ( sscanf(line,"%u",&dim)<1 || dim>MAXDIM );
      iv = iv_get(dim);
   }
   
   /* input elements */
   for ( i=0; i<dim; i++ )
     do
     {
      redo:
	fprintf(stderr,"entry %u: ",i);
	if ( !dynamic )
	  fprintf(stderr,"old: %-9d  new: ",iv->ive[i]);
	if ( fgets(line,MAXLINE,fp)==NULL )
	  error(E_INPUT,"iiv_finput");
	if ( (*line == 'b' || *line == 'B') && i > 0 )
	{	i--;	dynamic = FALSE;	goto redo;	   }
	if ( (*line == 'f' || *line == 'F') && i < dim-1 )
	{	i++;	dynamic = FALSE;	goto redo;	   }
     } while ( *line=='\0' || sscanf(line,"%d",&iv->ive[i]) < 1 );
   
   return (iv);
}

/* biv_finput -- batch-file input of IVEC iv */
#ifndef ANSI_C
IVEC	*biv_finput(fp,iv)
FILE	*fp;
IVEC	*iv;
#else
IVEC	*biv_finput(FILE *fp, IVEC *iv)
#endif
{
   unsigned int	i,dim;
   int	io_code;
   
   /* get dimension */
   skipjunk(fp);
   if ((io_code=fscanf(fp," IntVector: dim:%u",&dim)) < 1 ||
       dim>MAXDIM )
     error(io_code==EOF ? 7 : 6,"biv_finput");
   
   /* allocate memory if necessary */
   if ( iv==(IVEC *)NULL || iv->dim<dim )
     iv = iv_resize(iv,dim);
   
   /* get entries */
   skipjunk(fp);
   for ( i=0; i<dim; i++ )
     if ((io_code=fscanf(fp,"%d",&iv->ive[i])) < 1 )
       error(io_code==EOF ? 7 : 6,"biv_finput");
   
   return (iv);
}

/* iv_dump -- dumps all the contents of IVEC iv onto stream fp */
#ifndef ANSI_C
void	iv_dump(fp,iv)
FILE*fp;
IVEC*iv;
#else
void	iv_dump(FILE *fp, const IVEC *iv)
#endif
{
   int		i;
   
   fprintf(fp,"IntVector: ");
   if ( ! iv )
   {
      fprintf(fp,"**** NULL ****\n");
      return;
   }
   fprintf(fp,"dim: %d, max_dim: %d\n",iv->dim,iv->max_dim);
   fprintf(fp,"ive @ 0x%lx\n",(long)(iv->ive));
   for ( i = 0; i < iv->max_dim; i++ )
   {
      if ( (i+1) % 8 )
	fprintf(fp,"%8d ",iv->ive[i]);
      else
	fprintf(fp,"%8d\n",iv->ive[i]);
   }
   if ( i % 8 )
     fprintf(fp,"\n");
}	

