#include <stdio.h>

struct hello {
	int a;
	hello()
	{
		a = 55;
	}
};

hello nameing;

int main()
{
	nameing.a = 20;
	int b;
	printf("Passed!\n");

	return 0;
}