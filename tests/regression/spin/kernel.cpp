#include <vx_spawn2.h>
#include "common.h"

// Livelocks on purpose: every thread spins forever. Used by the reset
// acceptance test to prove a hung kernel is recoverable by the next
// open's reset, without JTAG or a reboot.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	(void)arg;
	volatile uint32_t beat = 0;
	for (;;) {
		beat = beat + 1;
	}
}
