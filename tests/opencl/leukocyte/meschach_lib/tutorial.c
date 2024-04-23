
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

/* tutorial.c 10/12/1993 */

/* routines from Chapter 1 of Meschach */

static char rcsid[] = "$Id: tutorial.c,v 1.3 1994/01/16 22:53:09 des Exp $";

#include <math.h>
#include "matrix.h"

/* rk4 -- 4th order Runge--Kutta method */
double rk4(f,t,x,h)
double t, h;
VEC    *(*f)(), *x;
{
   static VEC *v1=VNULL, *v2=VNULL, *v3=VNULL, *v4=VNULL;
   static VEC *temp=VNULL;
   
   /* do not work with NULL initial vector */
   if ( x == VNULL )
     error(E_NULL,"rk4");

   /* ensure that v1, ..., v4, temp are of the correct size */
   v1   = v_resize(v1,x->dim);
   v2   = v_resize(v2,x->dim);
   v3   = v_resize(v3,x->dim);
   v4   = v_resize(v4,x->dim);
   temp = v_resize(temp,x->dim);

   /* register workspace variables */
   MEM_STAT_REG(v1,TYPE_VEC);
   MEM_STAT_REG(v2,TYPE_VEC);
   MEM_STAT_REG(v3,TYPE_VEC);
   MEM_STAT_REG(v4,TYPE_VEC);
   MEM_STAT_REG(temp,TYPE_VEC);
   /* end of memory allocation */

   (*f)(t,x,v1); /* most compilers allow: "f(t,x,v1);" */
   v_mltadd(x,v1,0.5*h,temp);    /* temp = x+.5*h*v1 */
   (*f)(t+0.5*h,temp,v2);
   v_mltadd(x,v2,0.5*h,temp);    /* temp = x+.5*h*v2 */
   (*f)(t+0.5*h,temp,v3);
   v_mltadd(x,v3,h,temp);        /* temp = x+h*v3 */
   (*f)(t+h,temp,v4);
   
   /* now add: v1+2*v2+2*v3+v4 */
   v_copy(v1,temp);              /* temp = v1 */
   v_mltadd(temp,v2,2.0,temp);   /* temp = v1+2*v2 */
   v_mltadd(temp,v3,2.0,temp);   /* temp = v1+2*v2+2*v3 */
   v_add(temp,v4,temp);          /* temp = v1+2*v2+2*v3+v4 */
   
   /* adjust x */
   v_mltadd(x,temp,h/6.0,x);     /* x = x+(h/6)*temp */
   
   return t+h;                   /* return the new time */
}



/* rk4 -- 4th order Runge-Kutta method */
/* another variant */
double rk4_var(f,t,x,h)
double t, h;
VEC    *(*f)(), *x;
{
   static VEC *v1, *v2, *v3, *v4, *temp;
   
   /* do not work with NULL initial vector */
   if ( x == VNULL )        error(E_NULL,"rk4");
   
   /* ensure that v1, ..., v4, temp are of the correct size */
   v_resize_vars(x->dim, &v1, &v2, &v3, &v4, &temp, NULL);

   /* register workspace variables */
   mem_stat_reg_vars(0, TYPE_VEC, __FILE__, __LINE__,
		     &v1, &v2, &v3, &v4, &temp, NULL);
   /* end of memory allocation */

   (*f)(t,x,v1);             v_mltadd(x,v1,0.5*h,temp);
   (*f)(t+0.5*h,temp,v2);    v_mltadd(x,v2,0.5*h,temp);
   (*f)(t+0.5*h,temp,v3);    v_mltadd(x,v3,h,temp);
   (*f)(t+h,temp,v4);
   
   /* now add: temp = v1+2*v2+2*v3+v4 */
   v_linlist(temp, v1, 1.0, v2, 2.0, v3, 2.0, v4, 1.0, VNULL);
   /* adjust x */
   v_mltadd(x,temp,h/6.0,x);     /* x = x+(h/6)*temp */
   
   return t+h;                   /* return the new time */
}


/* f -- right-hand side of ODE solver */
VEC	*f(t,x,out)
VEC	*x, *out;
double	t;
{
   if ( x == VNULL || out == VNULL )
     error(E_NULL,"f");
   if ( x->dim != 2 || out->dim != 2 )
     error(E_SIZES,"f");
   
   out->ve[0] = x->ve[1];
   out->ve[1] = - x->ve[0];
   
   return out;
}


void tutor_rk4()
{
   VEC        *x;
   VEC        *f();
   double     h, t, t_fin;
   double     rk4();
   
   input("Input initial time: ","%lf",&t);
   input("Input final time: ",  "%lf",&t_fin);
   x = v_get(2);    /* this is the size needed by f() */
   prompter("Input initial state:\n");	x = v_input(VNULL);
   input("Input step size: ",   "%lf",&h);
   
   printf("# At time %g, the state is\n",t);
   v_output(x);
   while (t < t_fin)
   {
      /* you can use t = rk4_var(f,t,x,min(h,t_fin-t)); */
      t = rk4(f,t,x,min(h,t_fin-t));   /* new t is returned */
      printf("# At time %g, the state is\n",t);
      v_output(x);
   }
}




#include "matrix2.h"

void tutor_ls()
{
   MAT *A, *QR;
   VEC *b, *x, *diag;
   
   /* read in A matrix */
   printf("Input A matrix:\n");
   
   A = m_input(MNULL);     /* A has whatever size is input */
   
   if ( A->m < A->n )
   {
      printf("Need m >= n to obtain least squares fit\n");
      exit(0);
   }
   printf("# A =\n");       m_output(A);
   diag = v_get(A->m);
   /* QR is to be the QR factorisation of A */
   QR = m_copy(A,MNULL);
   QRfactor(QR,diag);   
   /* read in b vector */
   printf("Input b vector:\n");
   b = v_get(A->m);
   b = v_input(b);
   printf("# b =\n");       v_output(b);
   
   /* solve for x */
   x = QRsolve(QR,diag,b,VNULL);
   printf("Vector of best fit parameters is\n");
   v_output(x);
   /* ... and work out norm of errors... */
   printf("||A*x-b|| = %g\n",
	  v_norm2(v_sub(mv_mlt(A,x,VNULL),b,VNULL)));
}


#include "iter.h"


#define N 50
#define VEC2MAT(v,m)  vm_move((v),0,(m),0,0,N,N);

#define PI 3.141592653589793116
#define index(i,j) (N*((i)-1)+(j)-1)

/* right hand side function (for generating b) */
double f1(x,y)
double x,y;
{
  /* return 2.0*PI*PI*sin(PI*x)*sin(PI*y); */
   return exp(x*y);
}

/* discrete laplacian */
SPMAT *laplacian(A)
SPMAT *A;
{
   Real h;
   int i,j;
   
   if (!A)
     A = sp_get(N*N,N*N,5);

   for ( i = 1; i <= N; i++ )
     for ( j = 1; j <= N; j++ )
     {
        if ( i < N )
	  sp_set_val(A,index(i,j),index(i+1,j),-1.0);
        if ( i > 1 )
	  sp_set_val(A,index(i,j),index(i-1,j),-1.0);
        if ( j < N )
	  sp_set_val(A,index(i,j),index(i,j+1),-1.0);
        if ( j > 1 )
	  sp_set_val(A,index(i,j),index(i,j-1),-1.0);
        sp_set_val(A,index(i,j),index(i,j),4.0);
     }
   return A;
}

/* generating right hand side */
VEC *rhs_lap(b)
VEC *b;
{
   Real h,h2,x,y;
   int i,j;
   
   if (!b)
     b = v_get(N*N);

   h = 1.0/(N+1);      /* for a unit square */
   h2 = h*h;
   x = 0.0;
   for ( i = 1; i <= N; i++ ) {
      x += h;
      y = 0.0;
     for ( j = 1; j <= N; j++ ) {
	y += h;
	b->ve[index(i,j)] = h2*f1(x,y);
     }
   }
   return b;
}
   
void tut_lap()
{
   SPMAT *A, *LLT;
   VEC *b, *out, *x;
   MAT *B;
   int num_steps;
   FILE *fp;

   A = sp_get(N*N,N*N,5);
   b = v_get(N*N);

   laplacian(A);
   LLT = sp_copy(A);
   spICHfactor(LLT);

   out = v_get(A->m);
   x = v_get(A->m);

   rhs_lap(b);   /* new rhs */
   iter_spcg(A,LLT,b,1e-6,out,1000,&num_steps);
   printf("Number of iterations = %d\n",num_steps);

   /* save b as a MATLAB matrix */

   fp = fopen("laplace.mat","w");  /* b will be saved in laplace.mat */
   if (fp == NULL) {
      printf("Cannot open %s\n","laplace.mat");
      exit(1);
   }
   
   /* b must be transformed to a matrix */
   
   B = m_get(N,N);
   VEC2MAT(out,B);
   m_save(fp,B,"sol");  /* sol is an internal name in MATLAB */

}


void main()
{
   int i;

   input("Choose the problem (1=Runge-Kutta, 2=least squares,3=laplace): ",
	 "%d",&i);
   switch (i) {
    case 1: tutor_rk4(); break;
    case 2: tutor_ls(); break;
    case 3: tut_lap(); break;
    default: 
      printf(" Wrong value of i (only 1, 2 or 3)\n\n");
      break;
   }

}

