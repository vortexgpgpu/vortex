#include <stdio.h>
#include <stdlib.h>
#include "../../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec_vecadd.h"

//---------------------------------------------------------------
/* vvaddint32
 * # vector-vector add routine of 32-bit integers
 * # void vvaddint32(size_t n, const int*x, const int*y, int*z)
 * # { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } } */
//---------------------------------------------------------------

int main()
{
    vx_tmc(1);

    int n = 4; //SIZE

    int *a = (int*)malloc(sizeof(int) * n); 
    int *b = (int*)malloc(sizeof(int) * n); 
    int *c = (int*)malloc(sizeof(int) * n); 

    // Initialize values for array members.  
    for (int i = 0; i < n; ++i) {
       a[i] = i * 2 + 0;
       b[i] = i * 2 + 1;
       c[i] = 0;
    }

#if 0
    printf("vvaddint...\na[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", a[i]);
    printf("\nb[%d]: ", n);
    for(int i = 0; i < n; ++i) printf("%d ", b[i]);
    printf("\nc[%d] = a[%d] + b[%d]: ", n, n, n);
    for(int i = 0; i < n; ++i) printf("%d ", c[i]);
#endif

    vx_vec_vvaddint32(n, a, b, c);

    for(int i = 0; i < n; ++i) 
    {
        if(c[i] != (a[i]+b[i])) 
        {
           printf("\n<vddint32> FAILED at <index: %d>! \n", i);
           return 1;   
        }
    }
    printf("\nPASSED.......................... <vddint32> \n");

    free(a); free(b); free(c);

    vx_tmc(0);

    return 0;

}
