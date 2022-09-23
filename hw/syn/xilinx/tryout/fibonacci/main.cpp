#include <stdint.h>

const intptr_t src_addr = 0x7fff0; // source value address
const intptr_t dst_addr = 0x7fff4; // destination value address

int fibonacci(int n) {
   if (n <= 1)
      return n;
   return fibonacci(n-1) + fibonacci(n-2);
}

int main() {
	int* src_ptr  = (int*)(src_addr);
	int* dst_ptr = (int*)(src_addr);

	int src_data = *src_ptr;

	int result = fibonacci(src_data);

	*dst_ptr = result;

	return 0;
}