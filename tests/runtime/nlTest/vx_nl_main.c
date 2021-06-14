#include <stdio.h>
#include <math.h>
#include <vx_print.h>

const int Num = 9;
const float fNum = 9.0f;

int fibonacci(int n) {
   if (n <= 1)
      return n;
   return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
	int fib = fibonacci(Num);
	float isq = 1.0f / sqrt(fNum);
	vx_printf("fibonacci(%d) = %d\n", Num, fib);
	vx_printf("invAqrt(%f) = %f\n", fNum, isq);
	vx_prints("Passed!\n");
	return 0;
}





