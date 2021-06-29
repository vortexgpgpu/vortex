#include <stdio.h>

const int Num = 9;
const int Ans = 34;

int fibonacci(int n) {
   if (n <= 1)
      return n;
   return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
	int errors = 0;

	int fib = fibonacci(Num);
	
	printf("fibonacci(%d) = %d\n", Num, fib);

	if (fib == Ans) {
		printf("Passed!\n");
	} else {
		printf("Failed! value=%d, expected=%d\n", fib, Ans);
		errors = 1;
	}

	return errors;
}





