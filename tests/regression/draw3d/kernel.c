#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>

typedef struct {
  	kernel_arg_t* state;
} tile_arg_t;

void kernel_body(int task_id, tile_arg_t* arg) {
	// TODO
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	// configure raster unit

	// configure rop unit

	tile_arg_t targ;
	targ.state = arg;
	
	vx_spawn_tasks(arg->num_tiles, (vx_spawn_tasks_cb)kernel_body, &targ);
	/*for (uint32_t t=0; t < arg->num_tiles; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}