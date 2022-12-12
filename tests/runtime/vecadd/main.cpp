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
    const int n = 10; // SIZE

    // int a[7] = {0, 1, 2, 3, 4, 5, 200};
    // int b[7] = {0, 1, 2, 3, 4, 5, 200};
    // int c[7] = {0, 0, 0, 0, 0, 0, 0};

    int a[n] = {100, 150, 20, 0, 1, 150, 200, 150, 200, 4};
    int b[n] = {20, 0, 1, 152, 200, 3, 1, 152, 200, 3};
    int c[n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    vx_vec_vvaddint32(n, a, b, c);

    vx_printf("Start of program\n");

    for (int i = 0; i < n; ++i)
    {
        if (c[i] != (a[i] + b[i]))
        {
            vx_printf("\n<vddint32> FAILED at <index: %d> <value: %d>! \n", i, c[i]);
            return 1;
        }
    }

    vx_printf("\nPASSED.......................... <vddint32> \n");

    return 0;
}
