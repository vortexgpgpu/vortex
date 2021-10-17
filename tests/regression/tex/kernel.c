#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"
#include "texsw.h"

#define ENABLE_SW

typedef struct {
  	kernel_arg_t* state;	
  	uint32_t tile_width;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
} tile_arg_t;

void kernel_body(int task_id, tile_arg_t* arg) {
	kernel_arg_t* state = arg->state;
	
	uint32_t xoffset = 0;
	uint32_t yoffset = task_id * arg->tile_height;	
	uint8_t* dst_ptr = (uint8_t*)(state->dst_ptr + xoffset * state->dst_stride + yoffset * state->dst_pitch);

	float fv = yoffset * arg->deltaY;
	for (uint32_t y = 0; y < arg->tile_height; ++y) {
		uint32_t* dst_row = (uint32_t*)dst_ptr;
		float fu = xoffset * arg->deltaX;
		for (uint32_t x = 0; x < arg->tile_width; ++x) {
			int32_t u = (int32_t)(fu * (1<<20));
			int32_t v = (int32_t)(fv * (1<<20));
		#ifdef ENABLE_SW
			if (state->use_sw) {
				dst_row[x] = (state->filter == 2) ? tex3_sw(state, 0, u, v, state->lod) : tex_sw(state, 0, u, v, state->lod);
			} else {
		#endif
			dst_row[x] = (state->filter == 2) ? vx_tex3(0, u, v, state->lod) : vx_tex(0, u, v, state->lod);
		#ifdef ENABLE_SW
			}
		#endif
			fu += arg->deltaX;
		}
		dst_ptr += state->dst_pitch;
		fv += arg->deltaY;
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	// configure texture unit
	vx_csr_write(CSR_TEX_ADDR(0),   arg->src_ptr);
	vx_csr_write(CSR_TEX_MIPOFF(0), 0);	
	vx_csr_write(CSR_TEX_WIDTH(0),  arg->src_logWidth);
	vx_csr_write(CSR_TEX_HEIGHT(0), arg->src_logHeight);
	vx_csr_write(CSR_TEX_FORMAT(0), arg->format);
	vx_csr_write(CSR_TEX_WRAP(0),   (arg->wrap << 2) | arg->wrap);
	vx_csr_write(CSR_TEX_FILTER(0), (arg->filter ? 1 : 0));

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_width  = arg->dst_width;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;    
	targ.deltaX      = 1.0f / arg->dst_width;
	targ.deltaY      = 1.0f / arg->dst_height;
	
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
}