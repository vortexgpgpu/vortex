
#include "intrinsics/vx_intrinsics.h"
#include "io/vx_io.h"

int arr[4];

int main()
{
	// vx_print_str("Hello from runtime\n");

	vx_tmc(4); // Activate 4 threads

	unsigned tid = vx_threadID(); // Get TID


	arr[tid] = tid;

	// vx_tmc(1);

	// vx_print_hex(arr[0]);
	// vx_print_str("\n");
	// vx_print_hex(arr[1]);
	// vx_print_str("\n");
	// vx_print_hex(arr[2]);
	// vx_print_str("\n");
	// vx_print_hex(arr[3]);
	// vx_print_str("\n");


	return 0;
}