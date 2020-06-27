#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	func_t function;
	void * arguments;
	int    nthreads;
} spawn_t;

spawn_t* g_spawn = NULL;

void spawn_warp_runonce() {
	// active all threads
	vx_tmc(g_spawn->nthreads);

	// call user routine
	g_spawn->function(g_spawn->arguments);

	// resume single-thread execution on exit
	int wid = vx_warp_id();
	unsigned tmask = (0 == wid) ? 0x1 : 0x0; 
	vx_tmc(tmask);
}

void vx_spawn_warps(int num_warps, int num_threads, func_t func_ptr , void * args) {
	spawn_t spawn = { func_ptr, args, num_threads };
	g_spawn = &spawn;

	if (num_warps > 1) {
		vx_wspawn(num_warps, (unsigned)spawn_warp_runonce);
	}
	spawn_warp_runonce();
}

#ifdef __cplusplus
}
#endif