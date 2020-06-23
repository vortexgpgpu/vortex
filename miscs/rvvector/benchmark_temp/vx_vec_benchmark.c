#include <stdio.h>
#include <stdlib.h>
#include "../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec_benchmark.h"

int main()
{
    vx_tmc(1);

    int n = 5;
    int scalar = 10;

    int *a = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
    int *b = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
    int *c = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};

    for (int i = 0; i < n; ++i) { a[i] = 1; b[i] = 2; c[i] = 5; }

#if 0
//---------------------------------------------------------------
/* vvaddint32
 * # vector-vector add routine of 32-bit integers
 * # void vvaddint32(size_t n, const int*x, const int*y, int*z)
 * # { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } } */
    printf("vvaddint...\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", a[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", b[i]);
    printf("\nc[%d] = a[%d] + b[%d]: ", n, n, n);
    for(int i = 0; i < n; ++i) printf("%d ", c[i]);

    vx_vec_vvaddint32(n, a, b, c);

    for(int i = 0; i < n; ++i) 
    {
        if(c[i] != (a[i]+b[i])) 
        {
           printf("\n<vddint32> failed at <index: %d>! \n", i);
           return 1;   
        }
    }
    printf("\nPASSED.......................... <vddint32> \n");
#endif
#if 0
//---------------------------------------------------------------
/* #  vector-scalar add
   # for (i=0; i<N; i++) { C[i] = A[i] + B; } // 32-bit ints */
    for (int i = 0; i < n; ++i) { a[i] = 1; b[i] = 1;}
    printf("vsadd...scalar:%d\na[%d]: ", scalar, n);
    for(int i = 0; i < n; ++i) printf("%d \n", a[i]);
    printf("\nb: %d", scalar);
    
    vx_vec_vsadd(n, a, scalar);

    for(int i = 0; i < n; ++i) 
    {
        if(a[i] != (b[i] * scalar)) 
        {
           printf("\n<vsadd> failed at <index: %d>! \n", i);
           return 1;   
        }
    }
    printf("\nPASSED.......................... <vsadd> \n");

#endif
#if 0
//---------------------------------------------------------------
/*  # memory copy
    # void *memcpy(void* dest, const void* src, size_t n) */
    for (int i = 0; i < n; ++i) { a[i] = 1; b[i] = 2;}
    printf("memcpy\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", a[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", b[i]);

    vx_vec_memcpy(a, b, n);

    for(int i = 0; i < n; ++i) 
    {
        if(a[i] != b[i]) 
        {
           printf("\n<memcpy> failed at <index: %d>! \n", i);
<<<<<<< HEAD
           return;   
        }
    }
    printf("\nPASSED.......................... <memcpy> \n");
=======
           return 1;   
        }
    }
    printf("\nPASSED.......................... <memcpy> \n");
#endif
#if 1
//---------------------------------------------------------------
/* # void saxpy(size_t n, const float a, const float *x, float *y)
   # ==> convert to int!!
   # void saxpy(size_t n, const int a, const int *x, int *y)
   # {
   #   size_t i;
   #   for (i=0; i<n; i++) y[i] = a * x[i] + y[i];
   # } */
    for (int i = 0; i < n; ++i) { a[i] = 1; b[i] = 2; c[i] = 2;}

    printf("saxpy\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", a[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", b[i]);

    vx_vec_saxpy(n, scalar, a, b);

    printf("saxpy\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", a[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", b[i]);

    for(int i = 0; i < n; ++i) 
    {
        if(b[i] != ((a[i] * scalar) + c[i])) 
        {
           printf("\n<saxpy> failed at <index: %d>! \n", i);
           return;   
        }
    }
    printf("\nPASSED.......................... <saxpy> \n");

           return 1;   
        }
    }
    printf("\nPASSED.......................... <saxpy> \n");
#endif
#if 0
//---------------------------------------------------------------
/* # void sgemm_nn(size_t n, size_t m, size_t k, const float*a,   // m * k matrix
#          size_t lda, const float*b,   // k * n matrix 
#          size_t ldb, float*c,         // m * n matrix
#          size_t ldc)
#  c += a*b (alpha=1, no transpose on input matrices)
#  matrices stored in C row-major order */

    int m = 8;
    int k = 8;
    int n = 8
    int lda = 4;
    int ldb = 4;
    int ldc = 4;

    int* a1 = (int*)malloc(sizeof(m * k));
    int* b1 = (int*)malloc(sizeof(k * n));
    int* c1 = (int*)malloc(sizeof(m * n));

    for(int i = 0; i < (m * k); ++i) a1[i] = 1;
    for(int i = 0; i < (k * n); ++i) b1[i] = 1;
    for(int i = 0; i < (m * n); ++i) c1[i] = 1;    

    printf("sgemm_nn\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", a1[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d \n", b1[i]);

    vx_vec_sgemm_nn(n, m, k, a1, lda, b1, ldb, c1, ldc);

    //for(int i = 0; i < n; ++i) 
    //{
    //    if(b[i] != ((a[i] * scalar) + c[i])) 
    //    {
    //       printf("\n<sgemm_nn> failed at <index: %d>! \n", i);
    //       return;   
    //    }
    //}
    printf("\nNOT TESTED.......................... <sgemm_nn> \n");
//---------------------------------------------------------------
#endif
    
    vx_tmc(0);
    return 0;
}
