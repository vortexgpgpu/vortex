#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include "texsw.h"

typedef struct {
  	kernel_arg_t* state;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
	float minification;
} tile_arg_t;

void kernel_body(int task_id, tile_arg_t* arg) {
	auto state = arg->state;

	auto y_start = task_id * arg->tile_height;
	auto y_end   = std::min<uint32_t>(y_start + arg->tile_height, state->dst_height);

	auto x_start = 0;
	auto x_end = state->dst_width;

	auto dst_ptr = (uint32_t*)(state->dst_addr + x_start * state->dst_stride + y_start * state->dst_pitch);

	TFixed<16> xj(arg->minification);

	/*vx_printf("task_id=%d, tile_width=%d, tile_height=%d, deltaX=%f, deltaY=%f, minification=%f\n", 
	 	task_id, arg->tile_width, arg->tile_height, arg->deltaX, arg->deltaY, arg->minification);*/

	auto fv = (y_start + 0.5f) * arg->deltaY;
	for (uint32_t y = y_start; y < y_end; ++y) {
		auto dst_row = dst_ptr;
		auto fu = (x_start + 0.5f) * arg->deltaX;
		for (uint32_t x = x_start; x < x_end; ++x) {
			TFixed<TEX_FXD_FRAC> xu(fu);
			TFixed<TEX_FXD_FRAC> xv(fv);
			auto color = tex_load(state, xu, xv, xj);
			//vx_printf("task_id=%d, x=%d, y=%d, fu=%f, fv=%f, xu=0x%x, xv=0x%x, color=0x%x\n", task_id, x, y, fu, fv, xu.data(), xv.data(), color);			
			dst_row[x] = color;
			fu += arg->deltaX;
		}
		dst_ptr += state->dst_pitch;
		fv += arg->deltaY;
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;    
	targ.deltaX      = 1.0f / arg->dst_width;
	targ.deltaY      = 1.0f / arg->dst_height;

	{
		uint32_t src_width  = (1 << arg->src_logwidth);
		uint32_t src_height = (1 << arg->src_logheight);
		float width_ratio   = float(src_width) / arg->dst_width;
		float height_ratio  = float(src_height) / arg->dst_height;
		targ.minification   = std::max<float>(width_ratio, height_ratio);
	}
	
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
	/*for (uint32_t t = 0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}