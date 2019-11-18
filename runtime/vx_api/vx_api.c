
#include "../intrinsics/vx_intrinsics.h"
#include "vx_api.h"
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif


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


unsigned               pocl_threads;
uint8_t *              pocl_args;
uint8_t *              pocl_ctx;
vx_pocl_workgroup_func pocl_pfn;


void pocl_spawn_real()
{
	vx_tmc(pocl_threads);
	int x = vx_threadID();
	int y = vx_warpID();

	(pocl_pfn)( pocl_args, pocl_ctx, x, y, 0);

	if (y != 0)
	{
		vx_tmc(0);
	}
	vx_tmc(1);
}


void pocl_spawn(struct context_t * ctx, const void * pfn, void * arguments)
{

	if (ctx->num_groups[2] > 1)
	{
		printf("ERROR: pocl_spawn doesn't support Z dimension yet!\n");
		return;
	}

	pocl_threads = ctx->num_groups[0];
	pocl_pfn     = (vx_pocl_workgroup_func) pfn;
	pocl_ctx     = (uint8_t *) ctx;
	pocl_args    = (uint8_t *) arguments;

	if (ctx->num_groups[1] > 1)
	{
		vx_wspawn(ctx->num_groups[1], (unsigned) &pocl_spawn_real);
	}

	pocl_spawn_real();

 //   int z;
 //   int y;
 //   int x;
	// for (z = 0; z < ctx->num_groups[2]; ++z)
	// {
	// 	for (y = 0; y < ctx->num_groups[1]; ++y)
	// 	{
	// 		for (x = 0; x < ctx->num_groups[0]; ++x)
	// 		{
	// 			(use_pfn)((uint8_t *)arguments, (uint8_t *)ctx, x, y, z);
	// 		}
	// 	}
	// }
}

#ifdef __cplusplus
}
#endif
