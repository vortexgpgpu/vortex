#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec_saxpy.h"

//---------------------------------------------------------------
/* # void saxpy(size_t n, const float a, const float *x, float *y)
   # ==> convert to int!!
   # void saxpy(size_t n, const int a, const int *x, int *y)
   # { size_t i;
   #   for (i=0; i<n; i++) y[i] = a * x[i] + y[i];  }           */
//---------------------------------------------------------------

int main()
{
    vx_tmc(1);

    int n = 64; //#define NUM_DATA 65536

    int *a = (int*)malloc(sizeof(int) * n); 
    int *b = (int*)malloc(sizeof(int) * n); 
    int *c = (int*)malloc(sizeof(int) * n); //verification

    //  float factor = ((float)rand()/(float)(RAND_MAX)) * 100.0;
    int factor = ((float)rand()/(RAND_MAX)) * 100.0;

    for (int i = 0; i < n; ++i) { 
        a[i] = ((float)rand()/(RAND_MAX)) * 100.0;
        b[i] = 0; 
        c[i] = 0;
    }
  //; c[i] = 2;}

#if 1
    printf("saxpy\nfactor: %d\na[%d]: ", factor, n);
    for(int i = 0; i < n; ++i) printf("%d ", a[i]);
//    printf("\nb[%d]: ", n);
//    for(int i = 0; i < n; ++i) printf("%d \n", b[i]);
#endif

    int startCycles = vx_num_cycles();
    int startInst = vx_num_instrs();
    vx_vec_saxpy(n, factor, a, b);
    int endCycles = vx_num_cycles();
    int endInst = vx_num_instrs();

    int totalInst = (endInst - startInst);
    int totalCycles = (endCycles - startCycles);

    printf("\nCycles = %d, Instructions = %d", totalCycles, totalInst);

#if 0
    printf("\nsaxpy\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", a[i]);
    printf("\n\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", b[i]);
#endif

    for(int i = 0; i < n; ++i) 
    {
        if(b[i] != ((a[i] * factor) + c[i])) 
        {
           printf("\n<saxpy> FAILED at <index: %d>! \n", i);
           return 1;   
        }
    }
    
    printf("\nPASSED.......................... <saxpy> \n");


    free(a); free(b); free(c);

    vx_tmc(0);

    return 0;

}
