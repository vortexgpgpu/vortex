#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include <algorithm>

typedef struct {
  	kernel_arg_t* state;
 	uint32_t tile_height;
} tile_arg_t;

void kernel_body(int task_id, tile_arg_t* arg) {
	auto state   = arg->state;	
	
	auto y_start = task_id * arg->tile_height;
	auto y_end   = std::min<uint32_t>(y_start + arg->tile_height, state->dst_height);

	auto x_start = 0;
	auto x_end = state->dst_width;

	for (uint32_t y = y_start; y < y_end; ++y) {
		for (uint32_t x = x_start; x < x_end; ++x) {
			vx_rop(x, y, state->backface, state->color, state->depth);	
		}
	}
}

int main() {
	auto arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;
	
	//vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
	for (uint32_t t = 0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}

	return 0;
}