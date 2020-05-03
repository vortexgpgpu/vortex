#include <stdio.h>
#include <stdlib.h>
#include "../../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec_sgemm_nn.h"

//---------------------------------------------------------------
/* # void sgemm_nn(size_t n, size_t m, size_t k, 
#          int *a,   // m * k matri size_t lda, 
#          int *b,   // k * n matrix  size_t ldb, 
#          int *c,   // m * n matrix  size_t ldc)
#  c += a*b (alpha=1, no transpose on input matrices)
#  matrices stored in C row-major order */
//---------------------------------------------------------------

int main()
{
    vx_tmc(1);

    int m = 64;
    int k = 64;
    int n = 64;

    int* a1 = (int*)malloc(sizeof(int) * m * k);
    int* b1 = (int*)malloc(sizeof(int) * k * n);
    int* c1 = (int*)malloc(sizeof(int) * m * n);
    int* d1 = (int*)malloc(sizeof(int) * m * n); //verfication

    for (int i = 0; i < (m * k); ++i) a1[i] = i;
    for (int i = 0; i < (k * n); ++i) b1[i] = 1;
    for (int i = 0; i < (m * n); ++i) c1[i] = 0;
    for (int i = 0; i < (m * n); ++i) d1[i] = 0;


#if 0
    printf("sgemm_nn\na[%d]:", m*k);
    for (int i = 0; i < m*k; ++i) {
        if(!(i % k)) printf("\n");
        printf("%d ", a1[i]);
    }
    printf("\n\nb[%d]:", k*n);
    for (int i = 0; i < k*n; ++i) {
        if (!(i % n)) printf("\n");
        printf("%d ", b1[i]);
    }
#endif

    int lda = 4;
    int ldb = 4;
    int ldc = 64; //64;
    int vsize = 32;


  int startCycles = vx_num_cycles();
  int startInst = vx_num_instrs();
  for (int r = 0; r < m; r++) {
       for (int c = 0; c < n; c++) {
           for (int i = 0; i < k;) {
//               d1[r*k+i] += a1[r*k+c]*b1[i*n+c];
                 vx_vec_sgemm_nn(i, r, c, a1, b1, c1, ldc, vsize);
                 i = i + vsize;
           }
       }
    }
    int endCycles = vx_num_cycles();
    int endInst = vx_num_instrs();

    int totalInst = (endInst - startInst);
    int totalCycles = (endCycles - startCycles);
    printf("\nCycles = %d, Instructions = %d", totalCycles, totalInst);
//    vx_vec_sgemm_nn(n, a1, b1, c1);

#if 0
    printf("\n\nc[%d]:", m*n);
    for (int i = 0; i < m*n; ++i) {
        if (!(i % n)) printf("\n");
        printf("%d ", c1[i]);
    }
#endif

   for (int r = 0; r < m; r++) {
       for (int c = 0; c < n; c++) {
           for (int i = 0; i < k; i++) {
               d1[c*ldc+i] += a1[c*ldc+r]*b1[i + (r*ldc)];
               //printf("d[%d] += a[%d]*b[%d]\n", c*ldc+i, c*ldc+r , i + (r*ldc));
               //printf("%d %d %d\n", d1[c*ldc+i] , a1[c*ldc+r] , b1[i + (r*ldc)]);
           }
       }
    }

#if 0
   printf("\n\nc[%d]:\n", m*n);
   for(int i = 0; i < m; ++i) {
      for(int j = 0; j < n; ++j) {
          printf("%d ", d1[i*m+j]);
      }
      printf("\n");
    }
#endif


    for(int i = 0; i < m*n; ++i)
    {
        if(c1[i] != d1[i])
        {
           printf("\n<sgemm_nn> FAILED at <index: %d>! \n", i);
           return 1;
        }
    }

    printf("\nPASS.......................... <sgemm_nn> \n");


    free(a1); free(b1); free(c1);

    vx_tmc(0);

    return 0;

}
