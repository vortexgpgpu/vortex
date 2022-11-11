#include <stdio.h>
#include <stdlib.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

//---------------------------------------------------------------
/* vvaddint32
 * # vector-vector add routine of 32-bit integers
 * # void vvaddint32(size_t n, const int*x, const int*y, int*z)
 * # { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } } */
//---------------------------------------------------------------

int main()
{
    int n = 7; //SIZE

    int a[7] = {0,1,2,3,4,5,6};
    int b[7] = {0,1,2,3,4,5,6};
    int c[7] = {0,0,0,0,0,0,0};

    vx_vec_vvaddint32(n, (unsigned int)a, (unsigned int)b, (unsigned int)c);

    vx_printf("Start of program\n");

    for(int i = 0; i < n; ++i) 
    {
        if(c[i] != (a[i]+b[i])) 
        {
           vx_printf("\n<vddint32> FAILED at <index: %d>! \n", i);
           return 1;
        }
    }

    vx_printf("\nPASSED.......................... <vddint32> \n");

    return 0;

}
