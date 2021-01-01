#include <vx_intrinsics.h>
#include <vx_print.h>

int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);

	vx_prints("Newlib Main ");
	vx_printx(456);
	vx_prints(" \n");
	vx_prints("Passed!\n");

	return 0;
}





