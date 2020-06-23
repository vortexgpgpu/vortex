#include "../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec.h"

int main()
{
	vx_tmc(1);
        printf("----------------hello!!! \n");

        int n = 8;
        int *a = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *b = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *c = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
         
        printf("hello!!! \n");

        for(int i = 0; i < n; ++i)
        {
           a[i] = b[i] = c[i] = 1;
        }

	vx_vec_test(n, a, b, c);

        for(int i = 0; i < n; ++i)
           printf("%d ", c[i]);

	vx_tmc(0);
}
