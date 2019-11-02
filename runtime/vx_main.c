
#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"
#include "tests/tests.h"

int main()
{
	vx_tmc(1);

	// TMC test
	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// // Test wspawn
	// vx_print_str("test_spawn\n");
	// test_wsapwn();


	return 0;
}