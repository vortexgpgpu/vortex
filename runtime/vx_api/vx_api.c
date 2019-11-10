
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
	vx_wspawn(numWarps, (unsigned) setup_call);
	setup_call();

}


// void vx_cl_spawnWarps(char * args, char * pocl_context, long group_x, long group_y, long group_z)
// {
// 	if (group_z != 1)
// 	{
// 		vx_printf("ERROR: group_z should be set equal to 1");
// 		return;
// 	}

// 	vx_spawnWarps(group_y, group_x, )
// }
