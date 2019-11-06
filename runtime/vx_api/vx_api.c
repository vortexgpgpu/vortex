
#include "../intrinsics/vx_intrinsics.h"
#include "vx_api.h"



func_t global_function_pointer;
// void (func_t)(void *)

void *   global_argument_struct;

unsigned global_num_threads;
void setup_call()
{
	vx_tmc(global_num_threads);
	global_function_pointer(global_argument_struct);

	unsigned wid = vx_warpID();
	if (wid != 0)
	{
		vx_tmc(0); // Halt Warp Execution
	}
	else
	{
		vx_tmc(1); // Only activate one thread
	}
}

void vx_spawnWarps(unsigned numWarps, unsigned numThreads, func_t func_ptr, void * args)
{
	global_function_pointer = func_ptr;
	global_argument_struct  = args;
	global_num_threads      = numThreads;
	vx_wspawn(numWarps, (unsigned) func_ptr);
	setup_call();

}
