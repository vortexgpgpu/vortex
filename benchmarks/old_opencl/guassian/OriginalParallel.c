/*-----------------------------------------------------------
** ge_p.c -- The program is to solve a linear system Ax = b
**   by using Gaussian Elimination. The algorithm on page 101
**   ("Foundations of Parallel Programming") is used.  
**   The sequential version is ge_s.c.  This parallel 
**   implementation converts three independent for() loops 
**   into three Fans.  Use the data file ge_3.dat to verify 
**   the correction of the output. 
**
** Written by Andreas Kura, 02/15/95
** Modified by Chong-wei Xu, /04/20/95
**-----------------------------------------------------------
*/
#include <us.h>
#include <stdio.h>

int Size, t;
float **a, *b;
BEGIN_SHARED_DECL
  float **m;
END_SHARED_DECL;
FILE *fp;

void InitProblemOnce();
void InitPerRun();
void ForwardSub();
void Fan1();
void Fan2();
void Fan3();
void InitMat();
void InitAry();
void PrintMat();
void PrintAry();

main ()
{
  InitializeUs();
  MakeSharedVariables;  /* to make SHARED m */

  InitProblemOnce();
  InitPerRun();
  ForwardSub();

  printf("The result of matrix m is: \n");
  PrintMat(SHARED m, Size, Size);
  printf("The result of matrix a is: \n");
  PrintMat(a, Size, Size);
  printf("The result of array b is: \n");
  PrintAry(b, Size);
}

/*------------------------------------------------------
** InitProblemOnce -- Initialize all of matrices and
** vectors by opening a data file specified by the user.
**
** We used dynamic array **a, *b, and **m to allocate
** the memory storages.
**------------------------------------------------------
*/
void InitProblemOnce()
{
  char filename[30];
 
  printf("Enter the data file name: ");
  scanf("%s", filename);
  printf("The file name is: %s\n", filename);
 
  fp = fopen(filename, "r");
 
  fscanf(fp, "%d", &Size);
  a = (float **) UsAllocScatterMatrix(Size, Size, sizeof(float));
  /* 
  a = (float **) malloc(Size * sizeof(float *));
  for (i=0; i<Size; i++) {
    a[i] = (float *) malloc(Size * sizeof(float));
  }
  */
  InitMat(a, Size, Size);
  printf("The input matrix a is:\n");
  PrintMat(a, Size, Size);
 
  b = (float *) UsAlloc(Size * sizeof(float));
  /* 
  b = (float *) malloc(Size * sizeof(float));
  */
  InitAry(b, Size);
  printf("The input array b is:\n");
  PrintAry(b, Size);
 
  SHARED m = (float **) UsAllocScatterMatrix(Size, Size, sizeof(float));
  /*
  m = (float **) malloc(Size * sizeof(float *));
  for (i=0; i<Size; i++) {
    m[i] = (float *) malloc(Size * sizeof(float));
  }
  */

  Share(&Size);
  Share(&a);
  Share(&b);
}

/*------------------------------------------------------
** InitPerRun() -- Initialize the contents of the
** multipier matrix **m
**------------------------------------------------------
*/
void InitPerRun() 
{
  int i, j;

  for (i=0; i<Size; i++)
    for (j=0; j<Size; j++) 
      SHARED m[i][j] = 0.0;
}

/*------------------------------------------------------
** ForwardSub() -- Forward substitution of Gaussian
** elimination.
**------------------------------------------------------
*/
void ForwardSub()
{
  for (t=0; t<(Size-1); t++) {
    Share(&t);
    GenOnI(Fan1, Size-1-t);  /* t=0 to (Size-2), the range is
                             ** Size-2-t+1 = Size-1-t
                             */   
    GenOnA(Fan2, Size-1-t, Size-t);
    GenOnI(Fan3, Size-1-t);
  }
}

/*-------------------------------------------------------
** Fan1() -- Calculate multiplier matrix
** Pay attention to the index.  Index i give the range
** which starts from 0 to range-1.  The real values of
** the index should be adjust and related with the value
** of t which is defined on the ForwardSub().
**-------------------------------------------------------
*/
void Fan1(dummy, i)
int dummy, i;
{   
  /* Use these printf() to display the nodes and index */
  printf("from node #%d\n", PhysProcToUsProc(Proc_Node));
  SHARED m[i+t+1][t] = a[i+t+1][t] / a[t][t];
  printf("i=%d, a[%d][%d]=%.2f, a[%d][%d]=%.2f, m[%d][%d]=%.2f\n",
     (i+t+1),t,t,a[t][t],(i+t+1),t,a[i+t+1][t],(i+t+1),t,
     SHARED m[i+t+1][t]);
}

/*-------------------------------------------------------
** Fan2() -- Modify the matrix A into LUD
**-------------------------------------------------------
*/ 
void Fan2(dummy, i, j)
int dummy, i, j;
{
  a[i+1+t][j+t] -= SHARED m[i+1+t][t] * a[t][j+t];
  Share (&a);
}

/*-------------------------------------------------------
** Fan3() -- Modify the array b
**-------------------------------------------------------
*/
void Fan3(dummy, i)
int dummy, i;
{
  b[i+1+t] -= SHARED m[i+1+t][t] * b[t];
}

/*------------------------------------------------------
** InitMat() -- Initialize the matrix by reading data
** from the data file
**------------------------------------------------------
*/
void InitMat(ary, nrow, ncol)
float **ary;
int nrow, ncol;
{
  int i, j;

  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      fscanf(fp, "%f",  &ary[i][j]);
    }
  }  
}

/*------------------------------------------------------
** PrintMat() -- Print the contents of the matrix
**------------------------------------------------------
*/
void PrintMat(ary, nrow, ncol)
float **ary;
int nrow, ncol;
{
  int i, j;
 
  for (i=0; i<nrow; i++) {
    for (j=0; j<ncol; j++) {
      printf("%8.2f ", ary[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

/*------------------------------------------------------
** InitAry() -- Initialize the array (vector) by reading
** data from the data file
**------------------------------------------------------
*/
void InitAry(ary, ary_size)
float *ary;
int ary_size;
{
  int i;
 
  for (i=0; i<ary_size; i++) {
    fscanf(fp, "%f",  &ary[i]);
  }
}  
 
/*------------------------------------------------------
** PrintAry() -- Print the contents of the array (vector)
**------------------------------------------------------
*/
void PrintAry(ary, ary_size)
float *ary;
int ary_size;
{
  int i;
 
  for (i=0; i<ary_size; i++) {
    printf("%.2f ", ary[i]);
  }
  printf("\n");
}
