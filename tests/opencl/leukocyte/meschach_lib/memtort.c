
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
  Tests for mem_info.c functions
  */

static char rcsid[] = "$Id: $";

#include        <stdio.h>
#include        <math.h>
#include        "matrix2.h"
#include 	"sparse2.h"
#include  	"zmatrix2.h"


#define errmesg(mesg)   printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)    printf("# Testing %s...\n",mesg)


/*  new types list */

extern MEM_CONNECT mem_connect[MEM_CONNECT_MAX_LISTS];

/* the number of a new list */
#define FOO_LIST 1

/* numbers of types */
#define TYPE_FOO_1    1
#define TYPE_FOO_2    2

typedef struct {
   int dim;
   int fix_dim;
   Real (*a)[10];
} FOO_1;

typedef struct {
  int dim;
  int fix_dim;
  Real (*a)[2];
} FOO_2;



FOO_1 *foo_1_get(dim)
int dim;
{
   FOO_1 *f;
   
   if ((f = (FOO_1 *)malloc(sizeof(FOO_1))) == NULL)
     error(E_MEM,"foo_1_get");
   else if (mem_info_is_on()) {
      mem_bytes_list(TYPE_FOO_1,0,sizeof(FOO_1),FOO_LIST);
      mem_numvar_list(TYPE_FOO_1,1,FOO_LIST);
   }
   
   f->dim = dim;
   f->fix_dim = 10;
   if ((f->a = (Real (*)[10])malloc(dim*sizeof(Real [10]))) == NULL)
      error(E_MEM,"foo_1_get");
   else if (mem_info_is_on())
     mem_bytes_list(TYPE_FOO_1,0,dim*sizeof(Real [10]),FOO_LIST); 

   return f;
}


FOO_2 *foo_2_get(dim)
int dim;
{
   FOO_2 *f;
   
   if ((f = (FOO_2 *)malloc(sizeof(FOO_2))) == NULL)
     error(E_MEM,"foo_2_get");
   else if (mem_info_is_on()) {
      mem_bytes_list(TYPE_FOO_2,0,sizeof(FOO_2),FOO_LIST);
      mem_numvar_list(TYPE_FOO_2,1,FOO_LIST);
   }

   f->dim = dim;
   f->fix_dim = 2;
   if ((f->a = (Real (*)[2])malloc(dim*sizeof(Real [2]))) == NULL)
      error(E_MEM,"foo_2_get");
   else if (mem_info_is_on())
     mem_bytes_list(TYPE_FOO_2,0,dim*sizeof(Real [2]),FOO_LIST); 

   return f;
}



int foo_1_free(f)
FOO_1 *f;
{
   if ( f != NULL) {
      if (mem_info_is_on()) {
	 mem_bytes_list(TYPE_FOO_1,sizeof(FOO_1)+
			f->dim*sizeof(Real [10]),0,FOO_LIST);
	 mem_numvar_list(TYPE_FOO_1,-1,FOO_LIST);
      }

      free(f->a);
      free(f);
   }
   return 0;
}

int foo_2_free(f)
FOO_2 *f;
{
   if ( f != NULL) {
      if (mem_info_is_on()) {
	 mem_bytes_list(TYPE_FOO_2,sizeof(FOO_2)+
			f->dim*sizeof(Real [2]),0,FOO_LIST);
	 mem_numvar_list(TYPE_FOO_2,-1,FOO_LIST);
      }

      free(f->a);
      free(f);
   }
   return 0;
}




char *foo_type_name[] = {
   "nothing",
   "FOO_1",
   "FOO_2"
};


#define FOO_NUM_TYPES  (sizeof(foo_type_name)/sizeof(*foo_type_name))


int (*foo_free_func[FOO_NUM_TYPES])() = {
   NULL, 
   foo_1_free, 
   foo_2_free
  };



static MEM_ARRAY foo_info_sum[FOO_NUM_TYPES];



  /* px_rand -- generates sort-of random permutation */
PERM    *px_rand(pi)
PERM    *pi;
{
   int         i, j, k;
   
   if ( ! pi )
     error(E_NULL,"px_rand");
   
   for ( i = 0; i < 3*pi->size; i++ )
   {
      j = (rand() >> 8) % pi->size;
      k = (rand() >> 8) % pi->size;
      px_transp(pi,j,k);
   }
   
   return pi;
}

#ifdef SPARSE
SPMAT  *gen_non_symm(m,n)
int     m, n;
{
    SPMAT      *A;
    static      PERM    *px = PNULL;
    int         i, j, k, k_max;
    Real        s1;

    A = sp_get(m,n,8);
    px = px_resize(px,n);
    MEM_STAT_REG(px,TYPE_PERM);
    for ( i = 0; i < A->m; i++ )
    {
        k_max = 1 + ((rand() >> 8) % 10);
        for ( k = 0; k < k_max; k++ )
        {
            j = (rand() >> 8) % A->n;
            s1 = rand()/((double)MAX_RAND);
            sp_set_val(A,i,j,s1);
        }
    }
    /* to make it likely that A is nonsingular, use pivot... */
    for ( i = 0; i < 2*A->n; i++ )
    {
        j = (rand() >> 8) % A->n;
        k = (rand() >> 8) % A->n;
        px_transp(px,j,k);
    }
    for ( i = 0; i < A->n; i++ )
        sp_set_val(A,i,px->pe[i],1.0);

    
    return A;
}
#endif

void stat_test1(par)
int par;
{
   static MAT *AT = MNULL;
   static VEC *xt1 = VNULL, *yt1 = VNULL;
   static VEC *xt2 = VNULL, *yt2 = VNULL;
   static VEC *xt3 = VNULL, *yt3 = VNULL;
   static VEC *xt4 = VNULL, *yt4 = VNULL;

   AT = m_resize(AT,10,10);
   xt1 = v_resize(xt1,10);
   yt1 = v_resize(yt1,10);
   xt2 = v_resize(xt2,10);
   yt2 = v_resize(yt2,10);
   xt3 = v_resize(xt3,10);
   yt3 = v_resize(yt3,10);
   xt4 = v_resize(xt4,10);
   yt4 = v_resize(yt4,10);

   MEM_STAT_REG(AT,TYPE_MAT);

#ifdef ANSI_C
   mem_stat_reg_vars(0,TYPE_VEC,__FILE__,__LINE__,&xt1,&xt2,&xt3,&xt4,&yt1,
		     &yt2,&yt3,&yt4,NULL);
#else
#ifdef VARARGS
   mem_stat_reg_vars(0,TYPE_VEC,__FILE__,__LINE__,&xt1,&xt2,&xt3,&xt4,&yt1,
		     &yt2,&yt3,&yt4,NULL);
#else
   MEM_STAT_REG(xt1,TYPE_VEC);
   MEM_STAT_REG(yt1,TYPE_VEC);
   MEM_STAT_REG(xt2,TYPE_VEC);
   MEM_STAT_REG(yt2,TYPE_VEC);
   MEM_STAT_REG(xt3,TYPE_VEC);
   MEM_STAT_REG(yt3,TYPE_VEC);
   MEM_STAT_REG(xt4,TYPE_VEC);
   MEM_STAT_REG(yt4,TYPE_VEC);
#endif
#endif

   v_rand(xt1);
   m_rand(AT);
   mv_mlt(AT,xt1,yt1);
   
}


void stat_test2(par)
int par;
{
   static PERM *px = PNULL;
   static IVEC *ixt = IVNULL, *iyt = IVNULL;
   
   px = px_resize(px,10);
   ixt = iv_resize(ixt,10);
   iyt = iv_resize(iyt,10);

   MEM_STAT_REG(px,TYPE_PERM);
   MEM_STAT_REG(ixt,TYPE_IVEC);
   MEM_STAT_REG(iyt,TYPE_IVEC);

   px_rand(px);
   px_inv(px,px);
}

#ifdef SPARSE
void stat_test3(par)
int par;
{
   static SPMAT *AT = (SPMAT *)NULL;
   static VEC *xt = VNULL, *yt = VNULL;
   static SPROW *r = (SPROW *) NULL;
   
   if (AT == (SPMAT *)NULL)
     AT = gen_non_symm(100,100);
   else
     AT = sp_resize(AT,100,100);
   xt = v_resize(xt,100);
   yt = v_resize(yt,100);
   if (r == NULL) r = sprow_get(100);

   MEM_STAT_REG(AT,TYPE_SPMAT);
   MEM_STAT_REG(xt,TYPE_VEC);
   MEM_STAT_REG(yt,TYPE_VEC);
   MEM_STAT_REG(r,TYPE_SPROW);

   v_rand(xt);
   sp_mv_mlt(AT,xt,yt);
   
}
#endif

#ifdef COMPLEX
void stat_test4(par)
int par;
{
   static ZMAT *AT = ZMNULL;
   static ZVEC *xt = ZVNULL, *yt = ZVNULL;
   
   AT = zm_resize(AT,10,10);
   xt = zv_resize(xt,10);
   yt = zv_resize(yt,10);

   MEM_STAT_REG(AT,TYPE_ZMAT);
   MEM_STAT_REG(xt,TYPE_ZVEC);
   MEM_STAT_REG(yt,TYPE_ZVEC);

   zv_rand(xt);
   zm_rand(AT);
   zmv_mlt(AT,xt,yt);
   
}
#endif


void main(argc, argv)
int     argc;
char    *argv[];
{
   VEC  *x = VNULL, *y = VNULL, *z = VNULL;
   PERM  *pi1 = PNULL, *pi2 = PNULL, *pi3 = PNULL;
   MAT   *A = MNULL, *B = MNULL, *C = MNULL;
#ifdef SPARSE
   SPMAT *sA, *sB;
   SPROW *r;
#endif
   IVEC *ix = IVNULL, *iy = IVNULL, *iz = IVNULL;
   int m,n,i,j,deg,k;
   Real s1,s2;
#ifdef COMPLEX
   ZVEC        *zx = ZVNULL, *zy = ZVNULL, *zz = ZVNULL;
   ZMAT        *zA = ZMNULL, *zB = ZMNULL, *zC = ZMNULL;
   complex     ONE;
#endif
   /* variables for testing attaching new lists of types  */
   FOO_1 *foo_1;
   FOO_2 *foo_2;


   mem_info_on(TRUE);

#if defined(ANSI_C) || defined(VARARGS)

   notice("vector initialize, copy & resize");
   
   n = v_get_vars(15,&x,&y,&z,(VEC **)NULL);
   if (n != 3) {
      errmesg("v_get_vars");
      printf(" n = %d (should be 3)\n",n);
   }

   v_rand(x);
   v_rand(y);
   z = v_copy(x,z);
   if ( v_norm2(v_sub(x,z,z)) >= MACHEPS )
     errmesg("v_get_vars");
   v_copy(x,y);
   n = v_resize_vars(10,&x,&y,&z,NULL);
   if ( n != 3 || v_norm2(v_sub(x,y,z)) >= MACHEPS )
     errmesg("VEC copy/resize");

   n = v_resize_vars(20,&x,&y,&z,NULL);
   if ( n != 3 || v_norm2(v_sub(x,y,z)) >= MACHEPS )
     errmesg("VEC resize"); 

   n = v_free_vars(&x,&y,&z,NULL);
   if (n != 3)
     errmesg("v_free_vars");
   
   /* IVEC */
   notice("int vector initialise, copy & resize");
   n = iv_get_vars(15,&ix,&iy,&iz,NULL);

   if (n != 3) {
      errmesg("iv_get_vars");
      printf(" n = %d (should be 3)\n",n);
   }
   for (i=0; i < ix->dim; i++) {
      ix->ive[i] = 2*i-1;
      iy->ive[i] = 3*i+2;
   }
   iz = iv_add(ix,iy,iz);
   for (i=0; i < ix->dim; i++) 
     if ( iz->ive[i] != 5*i+1)
       errmesg("iv_get_vars");
   
   n = iv_resize_vars(10,&ix,&iy,&iz,NULL);
   if ( n != 3) errmesg("IVEC copy/resize");
   
   iv_add(ix,iy,iz);
   for (i=0; i < ix->dim; i++)
     if (iz->ive[i] != 5*i+1)
       errmesg("IVEC copy/resize");
   
   n = iv_resize_vars(20,&ix,&iy,&iz,NULL);
   if ( n != 3 ) errmesg("IVEC resize");
   
   iv_add(ix,iy,iz);
   for (i=0; i < 10; i++)
     if (iz->ive[i] != 5*i+1)
       errmesg("IVEC copy/resize");
   
   n = iv_free_vars(&ix,&iy,&iz,NULL);
   if (n != 3) 
     errmesg("iv_free_vars");
   
   /* MAT */
   notice("matrix initialise, copy & resize");
   n = m_get_vars(10,10,&A,&B,&C,NULL);
   if (n != 3) {
      errmesg("m_get_vars");
      printf(" n = %d (should be 3)\n",n);
   }
   
   m_rand(A);
   m_rand(B);
   C = m_copy(A,C);
   if ( m_norm_inf(m_sub(A,C,C)) >= MACHEPS )
     errmesg("MAT copy");
   m_copy(A,B);
   n = m_resize_vars(5,5,&A,&B,&C,NULL);
   if ( n != 3 || m_norm_inf(m_sub(A,B,C)) >= MACHEPS )
     errmesg("MAT copy/resize");
   
   n = m_resize_vars(20,20,&A,&B,NULL);
   if ( m_norm_inf(m_sub(A,B,C)) >= MACHEPS )
     errmesg("MAT resize"); 
   
   k = m_free_vars(&A,&B,&C,NULL);
   if ( k != 3 )
     errmesg("MAT free");
   
   /* PERM */
   notice("permutation initialise, inverting & permuting vectors");
   n = px_get_vars(15,&pi1,&pi2,&pi3,NULL);
   if (n != 3) {
      errmesg("px_get_vars");
      printf(" n = %d (should be 3)\n",n);
   }

   v_get_vars(15,&x,&y,&z,NULL);
   
   px_rand(pi1);
   v_rand(x);
   px_vec(pi1,x,z);
   y = v_resize(y,x->dim);
   pxinv_vec(pi1,z,y);
   if ( v_norm2(v_sub(x,y,z)) >= MACHEPS )
     errmesg("PERMute vector");
   pi2 = px_inv(pi1,pi2);
   pi3 = px_mlt(pi1,pi2,pi3);
   for ( i = 0; i < pi3->size; i++ )
     if ( pi3->pe[i] != i )
       errmesg("PERM inverse/multiply");
   
   px_resize_vars(20,&pi1,&pi2,&pi3,NULL);
   v_resize_vars(20,&x,&y,&z,NULL);
   
   px_rand(pi1);
   v_rand(x);
   px_vec(pi1,x,z);
   pxinv_vec(pi1,z,y);
   if ( v_norm2(v_sub(x,y,z)) >= MACHEPS )
     errmesg("PERMute vector");
   pi2 = px_inv(pi1,pi2);
   pi3 = px_mlt(pi1,pi2,pi3);
   for ( i = 0; i < pi3->size; i++ )
     if ( pi3->pe[i] != i )
       errmesg("PERM inverse/multiply");
   
   n = px_free_vars(&pi1,&pi2,&pi3,NULL);
   if ( n != 3 )
     errmesg("PERM px_free_vars"); 

#ifdef SPARSE   
   /* set up two random sparse matrices */
   m = 120;
   n = 100;
   deg = 5;
   notice("allocating sparse matrices");
   k = sp_get_vars(m,n,deg,&sA,&sB,NULL);
   if (k != 2) {
      errmesg("sp_get_vars");
      printf(" n = %d (should be 2)\n",k);
   }
   
   notice("setting and getting matrix entries");
   for ( k = 0; k < m*deg; k++ )
   {
      i = (rand() >> 8) % m;
      j = (rand() >> 8) % n;
      sp_set_val(sA,i,j,rand()/((Real)MAX_RAND));
      i = (rand() >> 8) % m;
      j = (rand() >> 8) % n;
      sp_set_val(sB,i,j,rand()/((Real)MAX_RAND));
   }
   for ( k = 0; k < 10; k++ )
   {
      s1 = rand()/((Real)MAX_RAND);
      i = (rand() >> 8) % m;
      j = (rand() >> 8) % n;
      sp_set_val(sA,i,j,s1);
      s2 = sp_get_val(sA,i,j);
      if ( fabs(s1 - s2) >= MACHEPS ) {
	 printf(" s1 = %g, s2 = %g, |s1 - s2| = %g\n", 
		s1,s2,fabs(s1-s2));
	 break;
      }
   }
   if ( k < 10 )
     errmesg("sp_set_val()/sp_get_val()");
   
   /* check column access paths */
   notice("resizing and access paths");
   k = sp_resize_vars(sA->m+10,sA->n+10,&sA,&sB,NULL);
   if (k != 2) {
      errmesg("sp_get_vars");
      printf(" n = %d (should be 2)\n",k);
   }
   
   for ( k = 0 ; k < 20; k++ )
   {
      i = sA->m - 1 - ((rand() >> 8) % 10);
      j = sA->n - 1 - ((rand() >> 8) % 10);
      s1 = rand()/((Real)MAX_RAND);
      sp_set_val(sA,i,j,s1);
      if ( fabs(s1 - sp_get_val(sA,i,j)) >= MACHEPS )
	break;
   }
   if ( k < 20 )
     errmesg("sp_resize()");
   sp_col_access(sA);
   if ( ! chk_col_access(sA) )
   {
      errmesg("sp_col_access()");
   }
   sp_diag_access(sA);
   for ( i = 0; i < sA->m; i++ )
   {
      r = &(sA->row[i]);
      if ( r->diag != sprow_idx(r,i) )
	break;
   }
   if ( i < sA->m )
   {
      errmesg("sp_diag_access()");
   }
   
   k = sp_free_vars(&sA,&sB,NULL);
   if (k != 2)
     errmesg("sp_free_vars");
#endif  /* SPARSE */   


#ifdef COMPLEX
   /* complex stuff */
   
   ONE = zmake(1.0,0.0);
   printf("# ONE = "); z_output(ONE);
   printf("# Check: MACHEPS = %g\n",MACHEPS);
   /* allocate, initialise, copy and resize operations */
   /* ZVEC */
   notice("vector initialise, copy & resize");
   zv_get_vars(12,&zx,&zy,&zz,NULL);
   
   zv_rand(zx);
   zv_rand(zy);
   zz = zv_copy(zx,zz);
   if ( zv_norm2(zv_sub(zx,zz,zz)) >= MACHEPS )
     errmesg("ZVEC copy");
   zv_copy(zx,zy);
   
   zv_resize_vars(10,&zx,&zy,NULL);
   if ( zv_norm2(zv_sub(zx,zy,zz)) >= MACHEPS )
     errmesg("ZVEC copy/resize");
   
   zv_resize_vars(20,&zx,&zy,NULL);
   if ( zv_norm2(zv_sub(zx,zy,zz)) >= MACHEPS )
     errmesg("VZEC resize");
   zv_free_vars(&zx,&zy,&zz,NULL);

   
   /* ZMAT */
   notice("matrix initialise, copy & resize");
   zm_get_vars(8,5,&zA,&zB,&zC,NULL);
   
   zm_rand(zA);
   zm_rand(zB);
   zC = zm_copy(zA,zC);
   if ( zm_norm_inf(zm_sub(zA,zC,zC)) >= MACHEPS )
     errmesg("ZMAT copy");
   
   zm_copy(zA,zB);
   zm_resize_vars(3,5,&zA,&zB,&zC,NULL);
   
   if ( zm_norm_inf(zm_sub(zA,zB,zC)) >= MACHEPS )
     errmesg("ZMAT copy/resize");
   zm_resize_vars(20,20,&zA,&zB,&zC,NULL);
   
   if ( zm_norm_inf(zm_sub(zA,zB,zC)) >= MACHEPS )
     errmesg("ZMAT resize");
   
   zm_free_vars(&zA,&zB,&zC,NULL);
#endif /* COMPLEX */

#endif  /* if defined(ANSI_C) || defined(VARARGS) */

   printf("# test of mem_info_bytes and mem_info_numvar\n");
   printf("  TYPE VEC: %ld bytes allocated, %d variables allocated\n",
	  mem_info_bytes(TYPE_VEC,0),mem_info_numvar(TYPE_VEC,0));

   notice("static memory test");
   mem_info_on(TRUE);
   mem_stat_mark(1);
   for (i=0; i < 100; i++)
     stat_test1(i);
   mem_stat_free(1);

   mem_stat_mark(1);
   for (i=0; i < 100; i++) {
     stat_test1(i);
#ifdef COMPLEX
     stat_test4(i);
#endif
  }

   mem_stat_mark(2);
   for (i=0; i < 100; i++)
     stat_test2(i);

   mem_stat_mark(3);
#ifdef SPARSE
   for (i=0; i < 100; i++)
     stat_test3(i);
#endif

   mem_info();
   mem_dump_list(stdout,0);

   mem_stat_free(1);
   mem_stat_free(3);
   mem_stat_mark(4);

   for (i=0; i < 100; i++) {
      stat_test1(i);
#ifdef COMPLEX
      stat_test4(i);
#endif
   } 

   mem_stat_dump(stdout,0);
   if (mem_stat_show_mark() != 4) {
      errmesg("not 4 in mem_stat_show_mark()");
   }
   
   mem_stat_free(2);
   mem_stat_free(4);

   if (mem_stat_show_mark() != 0) {
      errmesg("not 0 in mem_stat_show_mark()");
   }

   /* add new list of types */

   mem_attach_list(FOO_LIST,FOO_NUM_TYPES,foo_type_name,
		   foo_free_func,foo_info_sum);
   if (!mem_is_list_attached(FOO_LIST))
     errmesg("list FOO_LIST is not attached");

   mem_dump_list(stdout,FOO_LIST);
   foo_1 = foo_1_get(6);
   foo_2 = foo_2_get(3);
   for (i=0; i < foo_1->dim; i++)
     for (j=0; j < foo_1->fix_dim; j++)
       foo_1->a[i][j] = i+j;
   for (i=0; i < foo_2->dim; i++)
     for (j=0; j < foo_2->fix_dim; j++)
       foo_2->a[i][j] = i+j;
   printf(" foo_1->a[%d][%d] = %g\n",5,9,foo_1->a[5][9]);
   printf(" foo_2->a[%d][%d] = %g\n",2,1,foo_2->a[2][1]);
   
   mem_stat_mark(5);
   mem_stat_reg_list((void **)&foo_1,TYPE_FOO_1,FOO_LIST,__FILE__,__LINE__);
   mem_stat_reg_list((void **)&foo_2,TYPE_FOO_2,FOO_LIST,__FILE__,__LINE__);
   mem_stat_dump(stdout,FOO_LIST);
   mem_info_file(stdout,FOO_LIST);
   mem_stat_free_list(5,FOO_LIST);
   mem_stat_dump(stdout,FOO_LIST);
   if ( foo_1 != NULL )
     errmesg(" foo_1 is not released");
   if ( foo_2 != NULL )
     errmesg(" foo_2 is not released");
   mem_dump_list(stdout,FOO_LIST);
   mem_info_file(stdout,FOO_LIST);

   mem_free_vars(FOO_LIST);
   if ( mem_is_list_attached(FOO_LIST) )
     errmesg("list FOO_LIST is not detached");

   mem_info();
   
#if REAL == FLOAT
   printf("# SINGLE PRECISION was used\n");
#elif REAL == DOUBLE
   printf("# DOUBLE PRECISION was used\n");
#endif

#define ANSI_OR_VAR

#ifndef ANSI_C
#ifndef VARARGS
#undef ANSI_OR_VAR
#endif
#endif

#ifdef ANSI_OR_VAR

   printf("# you should get: \n");
#if (REAL == FLOAT)
     printf("#   type VEC: 276 bytes allocated, 3 variables allocated\n");
#elif (REAL == DOUBLE)
     printf("#   type VEC: 516 bytes allocated, 3 variables allocated\n");
#endif
   printf("#   and other types are zeros\n");

#endif /*#if defined(ANSI_C) || defined(VARAGS) */

   printf("# Finished memory torture test\n");

   dmalloc_shutdown();
   return;
}
