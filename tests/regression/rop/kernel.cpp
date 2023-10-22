#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include <algorithm>
#include <math.h>

typedef struct {
  	uint32_t tile_height;
	float alpha;
} tile_info_t;

static tile_info_t g_tileinfo;

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {	
	auto y_start = task_id * g_tileinfo.tile_height;
	auto y_end   = std::min<uint32_t>(y_start + g_tileinfo.tile_height, arg->dst_height);

	auto x_start = 0;
	auto x_end = arg->dst_width;

	uint32_t alpha    = arg->blend_enable ? static_cast<uint32_t>(task_id * g_tileinfo.alpha) : 0xff;
	uint32_t color    = (alpha << 24) | (arg->color & 0x00ffffff);
	uint32_t backface = arg->backface;
	uint32_t depth    = arg->depth;

	for (uint32_t y = y_start; y < y_end; ++y) {
		for (uint32_t x = x_start; x < x_end; ++x) {
			vx_rop(x, y, backface, color, depth);	
		}
	}
}

int main() {
	auto arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	g_tileinfo.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;
	g_tileinfo.alpha = 255.0f / g_tileinfo.tile_height;
	
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
	/*for (uint32_t t = 0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}