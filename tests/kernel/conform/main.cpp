#include "tests.h"
#include <vx_print.h>

int main() {
	int errors = 0;

	errors += test_global_memory();

	errors += test_local_memory();

	errors += test_tmc();

	errors += test_pred();

	errors += test_divergence();

	errors += test_wsapwn();

	errors += test_spawn_tasks();

	errors += test_serial();

	errors += test_tmask();

	errors += test_barrier();

	errors += test_tls();

	if (0 == errors) {	
		PRINTF("Passed!\n");
	} else {
		PRINTF("Failed!\n");
	}
	
	return errors;
}