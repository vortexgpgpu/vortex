
#include "../../runtime/intrinsics/vx_intrinsics.h"
#include "vx_vec.h"

int main()
{
	vx_tmc(1);
	// int * a = malloc(4);
	// int * b = malloc(4);
	// int * c = malloc(4);


	int * a = malloc(4);
	*a = 5;
	printf("Value of a: %d\n", *a);

	vx_vec_test(a);

	printf("Value of a: %d\n", *a);


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