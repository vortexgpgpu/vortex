
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

/* iotort.c  10/11/93 */
/* test of I/O functions */


static char rcsid[] = "$Id: $";

#include "sparse.h"
#include "zmatrix.h"


#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);


void main()
{
   VEC *x;
   MAT *A;
   PERM *pivot;
   IVEC *ix;
   SPMAT *spA;
   ZVEC *zx;
   ZMAT *ZA;
   char yes;
   int i;
   FILE *fp;

   mem_info_on(TRUE);

   if ((fp = fopen("iotort.dat","w")) == NULL) {
      printf(" !!! Cannot open file %s for writing\n\n","iotort.dat");
      exit(1);
   }
     
   x = v_get(10);
   A = m_get(3,3);
   zx = zv_get(10);
   ZA = zm_get(3,3);
   pivot = px_get(10);
   ix = iv_get(10);
   spA = sp_get(3,3,2);

   v_rand(x);
   m_rand(A);
   zv_rand(zx);
   zm_rand(ZA);
   px_ident(pivot);
   for (i=0; i < 10; i++)
     ix->ive[i] = i+1;
   for (i=0; i < spA->m; i++) {
      sp_set_val(spA,i,i,1.0);
      if (i > 0) sp_set_val(spA,i-1,i,-1.0);
   }

   notice(" VEC output");
   v_foutput(fp,x);
   notice(" MAT output");
   m_foutput(fp,A);
   notice(" ZVEC output");
   zv_foutput(fp,zx);
   notice(" ZMAT output");
   zm_foutput(fp,ZA);
   notice(" PERM output");
   px_foutput(fp,pivot);
   notice(" IVEC output");
   iv_foutput(fp,ix);
   notice(" SPMAT output");
   sp_foutput(fp,spA);
   fprintf(fp,"Y");
   fclose(fp);

   printf("\nENTER SOME VALUES:\n\n");

   if ((fp = fopen("iotort.dat","r")) == NULL) {
      printf(" !!! Cannot open file %s for reading\n\n","iotort.dat");
      exit(1);
   }

   notice(" VEC input/output");
   x = v_finput(fp,x);
   v_output(x);

   notice(" MAT input/output");
   A = m_finput(fp,A);
   m_output(A);

   notice(" ZVEC input/output");
   zx = zv_finput(fp,zx);
   zv_output(zx);

   notice(" ZMAT input/output");
   ZA = zm_finput(fp,ZA);
   zm_output(ZA);

   notice(" PERM input/output");
   pivot = px_finput(fp,pivot);
   px_output(pivot);

   notice(" IVEC input/output");
   ix = iv_finput(fp,ix);
   iv_output(ix);

   notice(" SPMAT input/output");
   SP_FREE(spA);
   spA = sp_finput(fp);
   sp_output(spA);

   notice(" general input");
   finput(fp," finish the test?  ","%c",&yes);
   if (yes == 'y' || yes == 'Y' )
     printf(" YES\n");
   else printf(" NO\n");
   fclose(fp);

   mem_info();
}
