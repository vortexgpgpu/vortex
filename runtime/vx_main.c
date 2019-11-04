
#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"
#include "tests/tests.h"

int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);

	/*
		NOTE: * when test_wspawn is called from instrinsic_tests, RA 80000458 is stored at address 6fffefbc,
		      but when read back again it reads zeros even though no other write request is made to that
		      address (when only test_wsapwn is called by itself).

		      * When test_wsapwn is called by itself from main new lines are not printed....

		      * when test_wspawn is called with other tests from main it works fine...
	*/
	// intrinsics_tests(); 


	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// Test wspawn
	vx_print_str("test_wspawn\n");
	test_wsapwn();

	return 0;
}