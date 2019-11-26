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

    int w = 4;
    int h = 4;
    int d = 4;

    int* a1 = (int*)malloc(sizeof(int) * w * h);
    int* b1 = (int*)malloc(sizeof(int) * h * d);
    int* c1 = (int*)malloc(sizeof(int) * w * d);
    int* d1 = (int*)malloc(sizeof(int) * w * d); //verfication

    for (int i = 0; i < (w * h); ++i) a1[i] = i;
    for (int i = 0; i < (h * d); ++i) b1[i] = 1;
    for (int i = 0; i < (w * d); ++i) c1[i] = 0;
    for (int i = 0; i < (w * d); ++i) d1[i] = 0;


#if 0
    printf("sgemm_nn\na[%d]:", w*h);
    for (int i = 0; i < w*h; ++i) {
        if(!(i % h)) printf("\n");
        printf("%d ", a1[i]);
    }
    printf("\n\nb[%d]:", h*d);
    for (int i = 0; i < h*d; ++i) {
        if (!(i % d)) printf("\n");
        printf("%d ", b1[i]);
    }
#endif

    int lda = 4;
    int ldb = 4;
    int ldc = 4; //64;
    int vsize = 4;

  for (int n = 0; n < h; n++) {
       for (int i = 0; i < w; i=+4) {
           for (int m = 0; m < d; m++) {
                 vx_vec_sgemm_nn(i, m, n, a1, b1, c1, ldc, vsize);
                 i = i + vsize;
           }
       }
    }

#if 1 
    printf("\n\nc[%d]:", d*h);
    for (int i = 0; i < d*h; ++i) {
        if (!(i % h)) printf("\n");
        printf("%d ", c1[i]);
    }
#endif

   for (int r = 0; r < h; r++) {
       for (int c = 0; c < w; c++) {
           for (int i = 0; i < d; i++) {
               d1[r*h+i] += a1[r*h+c]*b1[i*d+c];
           }
       }
    }

#if 1
   printf("\n\nc[%d]:\n", w*d);
   for(int i = 0; i < w; ++i) {
      for(int j = 0; j < d; ++j) {
          printf("%d ", d1[i*w+j]);
      }
      printf("\n");
    }
#endif


    for(int i = 0; i < w*d; ++i)
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
