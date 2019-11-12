
#include "../../intrinsics/vx_intrinsics.h"
#include "../../io/vx_io.h"
#include "../../tests/tests.h"
#include "../../vx_api/vx_api.h"
#include "../../fileio/fileio.h"

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

	vx_close();


	return 0;
}





