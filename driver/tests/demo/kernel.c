#include <stdlib.h>
#include <stdio.h>
#include "intrinsics/vx_intrinsics.h"

void main() {
	unsigned *x = (unsigned*)0x10000000;
	unsigned *y = (unsigned*)0x20000000;
	unsigned *z = (unsigned*)0x30000000;

	unsigned wid = vx_warpID();

	unsigned tid = vx_threadID();

	unsigned i = (wid * MAX_THREADS) + tid;

	//if (i == 0) {
	//	printf("begin\n");
  //}

	z[i] = x[i] + y[i];

	//if (i == 0) {
	//	printf("end\n");
	//}
}