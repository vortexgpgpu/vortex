
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"

int main()
{
	// Main is called with all threads active of warp 0
	vx_tmc(1);

	vx_print_str("Simple Main1\n");

	// TMC test
	test_tmc();

	// Control Divergence Test
	vx_print_str("test_divergence\n");
	vx_tmc(4);
	test_divergence();
	vx_tmc(1);


	// Test wspawn
	vx_print_str("test_spawn\n");
	test_wsapwn();

	return 0;
}