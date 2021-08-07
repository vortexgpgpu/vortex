#include "tests.h"
#include <vx_print.h>

int main() {
	int errors = 0;

	errors += test_global_memory();

	errors += test_stack_memory();

	errors += test_shared_memory();

	errors += test_tmc();

	errors += test_divergence();

	errors += test_wsapwn();

	errors += test_spawn_tasks();

	errors += test_tmask();

	if (0 == errors) {	
		vx_printf("Passed!\n");
	} else {
		vx_printf("Failed!\n");
	}
	
	return errors;
}