#include "../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec.h"

int main()
{
        vx_tmc(1);

        printf("Hello\n");

        int n = 64;
        int *a = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *b = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};
        int *c = (int*)malloc(sizeof(int) * n); //{1, 1, 1, 1, 1};

        for(int i = 0; i < n; ++i)
        {
           a[i] = b[i] = c[i] = 1;
        }

        vx_vec_test(n, a, b, c);

        for (int i = 0; i < n; ++i)
        {
        	printf("a[%d]=%d, b[%d]=%d, c[%d]=%d\n", i, a[i], i, b[i], i, c[i]);
        }


        vx_tmc(0);
}