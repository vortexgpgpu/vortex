#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_CORES_MAX 16

typedef struct {
	func_t function;
	void * arguments;
	int    nthreads;
} spawn_t;

spawn_t* g_spawn[NUM_CORES_MAX];

void spawn_warp_all() {
	// active all threads
	int num_threads = vx_num_threads();	
	vx_tmc(num_threads);

	int core_id = vx_core_id();
	spawn_t* p_spawn = g_spawn[core_id];

	// call user routine
	p_spawn->function(p_spawn->arguments);	

	// resume single-warp execution on exit
	int wid = vx_warp_id();
	unsigned tmask = (0 == wid) ? 0x1 : 0x0; 
	vx_tmc(tmask);
}

void spawn_warp_threads(int num_threads) {
	// active all threads	
	vx_tmc(num_threads);

	int core_id = vx_core_id();
	spawn_t* p_spawn = g_spawn[core_id];

	// call user routine
	p_spawn->function(p_spawn->arguments);	

	// resume single-warp execution on exit
	int wid = vx_warp_id();
	unsigned tmask = (0 == wid) ? 0x1 : 0x0; 
	vx_tmc(tmask);
}

void vx_spawn_warps(int num_warps, int num_threads, func_t func_ptr , void * args) {
	int core_id = vx_core_id();
	if (core_id >= NUM_CORES_MAX)
		return;
		
	spawn_t spawn = { func_ptr, args, num_threads };	
	g_spawn[core_id] = &spawn;

	if (num_warps > 1) {
		vx_wspawn(num_warps, (unsigned)spawn_warp_all);
	}
	spawn_warp_threads(num_threads);
}

#ifdef __cplusplus
}
#endif