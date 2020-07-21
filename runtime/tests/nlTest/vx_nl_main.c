
#include <vx_intrinsics.h>

// #include <utlist.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>


// Newlib
#include <stdio.h>

int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);


	printf("printf: Newlib Main %d\n", 456);

	return 0;
}





