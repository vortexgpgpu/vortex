#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

func_t global_function_pointer;
void * global_argument_struct;
int global_num_threads;

void spawn_warp_runonce() {
	// active all threads
	vx_tmc(global_num_threads);

	// call user routine
	global_function_pointer(global_argument_struct);

	// resume single-thread execution on exit
	int wid = vx_warp_id();
	unsigned tmask = (0 == wid) ? 0x1 : 0x0; 
	vx_tmc(tmask);
}

void vx_spawn_warps(int numWarps, int numThreads, func_t func_ptr, void * args) {
	global_function_pointer = func_ptr;
	global_argument_struct  = args;
	global_num_threads      = numThreads;
	if (numWarps > 1) {
		vx_wspawn(numWarps, (unsigned)spawn_warp_runonce);
	}
	spawn_warp_runonce();
}

int                    pocl_threads;
struct context_t *     pocl_ctx;
vx_pocl_workgroup_func pocl_pfn;
const void *           pocl_args;

void pocl_spawn_warp_runonce() {
	// active all threads
	vx_tmc(pocl_threads);

	int x = vx_thread_id();
	int y = vx_warp_gid();

	// call kernel routine
	(pocl_pfn)(pocl_args, pocl_ctx, x, y, 0);

	// resume single-thread execution on exit
	int wid = vx_warp_id();
	unsigned tmask = (0 == wid) ? 0x1 : 0x0;
	vx_tmc(tmask);
}

void pocl_spawn(struct context_t * ctx, vx_pocl_workgroup_func pfn, const void * args) {
	if (ctx->num_groups[2] > 1)	{
		printf("ERROR: pocl_spawn doesn't support Z dimension yet!\n");
		return;
	}

	pocl_threads = ctx->num_groups[0];
	pocl_ctx     = ctx;
	pocl_pfn     = pfn;
	pocl_args    = args;

	if (ctx->num_groups[1] > 1)	{
		vx_wspawn(ctx->num_groups[1], (unsigned)&pocl_spawn_warp_runonce);
	}

	pocl_spawn_warp_runonce();
}

#ifdef __cplusplus
}
#endif