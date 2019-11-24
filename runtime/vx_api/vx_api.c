
#include "../intrinsics/vx_intrinsics.h"
#include "vx_api.h"
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TOTAL_WARPS 2
#define TOTAL_THREADS 16

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

unsigned               global_z;
unsigned               global_y;
unsigned               global_x;


void pocl_spawn_real()
{
	vx_tmc(pocl_threads);
	int base_x = vx_threadID();
	int base_y = vx_warpID();

	int local_x;
	int local_y;

	for (int iter_z = 0; iter_z < global_z; iter_z++)
	{
		for (int iter_x = 0; iter_x < global_x; iter_x++)
		{
			for (int iter_y = 0; iter_y < global_y; iter_y++)
			{

				local_x = (iter_x * TOTAL_THREADS) + base_x;
				local_y = (iter_y * TOTAL_WARPS  ) + base_y;

				(pocl_pfn)( pocl_args, pocl_ctx, local_x, local_y, iter_z);

			}
		}
	}

	// (pocl_pfn)( pocl_args, pocl_ctx, x, y, 0);

	if (base_y != 0)
	{
		vx_tmc(0);
	}
	vx_tmc(1);
}


void pocl_spawn(struct context_t * ctx, const void * pfn, void * arguments)
{


	// printf("ctx->num_groups[0]: %d\n", ctx->num_groups[0]);
	// printf("ctx->num_groups[1]: %d\n", ctx->num_groups[1]);
	// printf("ctx->num_groups[2]: %d\n", ctx->num_groups[2]);

	// printf("\n\n");

	// printf("ctx->local_size[0]: %d\n", ctx->local_size[0]);
	// printf("ctx->local_size[1]: %d\n", ctx->local_size[1]);
	// printf("ctx->local_size[2]: %d\n", ctx->local_size[2]);
	if (ctx->num_groups[0] > TOTAL_THREADS)
	{
		pocl_threads = TOTAL_THREADS;
		global_x     = ctx->num_groups[0] / TOTAL_THREADS;
		printf("pocl_threads: %d\n", pocl_threads);
		// printf("global_x: %d\n", global_x);
	}
	else
	{
		pocl_threads = ctx->num_groups[0];
		global_x     = 1;
		// printf("pocl_threads: %d\n", pocl_threads);
		// printf("global_x: %d\n", global_x);
	}


	global_z     = ctx->num_groups[2];
	pocl_pfn     = (vx_pocl_workgroup_func) pfn;
	pocl_ctx     = (uint8_t *) ctx;
	pocl_args    = (uint8_t *) arguments;

	if (ctx->num_groups[1] > 1)
	{
		if (ctx->num_groups[1] > TOTAL_WARPS)
		{
			global_y = ctx->num_groups[1] / TOTAL_WARPS;
			vx_wspawn(TOTAL_WARPS, (unsigned) &pocl_spawn_real);
			// printf("global_y: %d\n", global_y);
			// printf("Warps: %d\n", TOTAL_WARPS);
		}
		else
		{
			global_y = 1;
			vx_wspawn(ctx->num_groups[1], (unsigned) &pocl_spawn_real);
			// printf("global_y: %d\n", global_y);
			// printf("Warps: %d\n", ctx->num_groups[1]);
		}
	}

	unsigned starting_cycles = vx_getCycles();
	unsigned starting_inst   = vx_getInst();

	pocl_spawn_real();

	unsigned end_cycles = vx_getCycles();
	unsigned end_inst   = vx_getInst();


	printf("pocl_spawn: Total Cycles: %d\n", (end_cycles - starting_cycles));
	printf("pocl_spawn: Total Inst  : %d\n", (end_inst   - starting_inst  ));

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
