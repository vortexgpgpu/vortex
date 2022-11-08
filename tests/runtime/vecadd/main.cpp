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
    int n = 4; //SIZE

    int a[4] = {0,1,2,3};
    int b[4] = {0,1,2,3};
    int c[4] = {0,0,0,0}; 

    // Initialize values for array members.  
    for (int i = 0; i < n; ++i)
    {
       a[i] = i * 2 + 0;
       b[i] = i * 2 + 1;
       c[i] = 0;
    }

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
