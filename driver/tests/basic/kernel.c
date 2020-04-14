#include <stdint.h>

void main() {
	int64_t* x = (int64_t*)0x10000000;
	int64_t* y = (int64_t*)0x20000000;
	for (int  i = 0; i < 8 * 4; ++i) {
		y[i] = x[i];
	}
}