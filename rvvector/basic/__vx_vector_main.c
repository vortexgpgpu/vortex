#include <stdio.h>
#include <stdlib.h>
#include "../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec.h"

int main()
{
	vx_tmc(1);
#if 0
    # vector-vector add routine of 32-bit integers
    # void vvaddint32(size_t n, const int*x, const int*y, int*z)
    # { for (size_t i=0; i<n; i++) { z[i]=x[i]+y[i]; } }
    #
    # a0 = n, a1 = x, a2 = y, a3 = z
    # Non-vector instructions are indented
#endif   
#if 1      
        int n = 5;
        int *a = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *b = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *c = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};

        for(int i = 0; i < n; ++i)
        {
           a[i] = b[i] = c[i] = 1;
        }

        for(int i = 0; i < n; ++i) printf("%d, ", a[i]);
        printf("\n");
//        for(int i = 0; i < n; ++i) printf("%d, ", b[i]);
//        printf("\n");
//        for(int i = 0; i < n; ++i) printf("%d, ", c[i]);

        int *d;
        *d = 1;
	vx_vec_test(n, d, b, c);


        printf("(after: n = %d, %d)\n", n, *d);
        for(int i = 0; i < n; ++i) printf("%d, ", a[i]);
//        printf("\n");
//        for(int i = 0; i < n; ++i) printf("%d, ", b[i]);
//        printf("\n");
//        for(int i = 0; i < n; ++i) printf("%d, ", c[i]);

#endif
#if 0
	int * a = malloc(sizeof(int) * 10);
	for(int i = 0; i < 10; ++i) a[i] = 5;
   
       
	for(int i = 0; i < 10; ++i)
	    printf("%d, ", a[i]);

	vx_vec_test(a);
	//vx_vec_test(2, a, a, a);

	printf("after--------\n");
        for(int i = 0; i < 10; ++i) 
            printf("%d, ", a[i]);
#endif
#if 0
        int n = 5;
        int *a = (int*)malloc(sizeof(int) * 5); //{1, 1, 1, 1, 1};
        int *b = (int*)malloc(sizeof(int) * 5); //{1, 1, 1, 1, 1};
        int *c = (int*)malloc(sizeof(int) * 5); //{1, 1, 1, 1, 1}; 
        
        for(int i = 0; i < n; ++i)
        {
            a[i] = 1; 
            b[i] = 1;
            c[i] = 0;
        }

        printf("Value of a: %d, b: %d, c: %d, n: %d\n", a[0], b[0], c[0], n);
        vx_vec_test(n, a, b, c);
        printf("Value of a: %d, b: %d, c: %d, n: %d\n", a[0], b[0], c[0], n);
        
#endif

	// for (int i = 0; i < 4; i++)
	// {
	// 	if (c[i] != (a[i] + b[i]))
	// 	{
	// 		printf("Fail\n");
	// 		break;
	// 	}
	// }

	vx_tmc(0);
}
