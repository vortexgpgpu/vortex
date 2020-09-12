
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


	vx_print_str("Newlib Main ");
	vx_print_hex(456);
	vx_print_str(" \n");
	vx_print_str("Passed!\n");

	return 0;
}





