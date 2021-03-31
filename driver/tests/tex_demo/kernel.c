#include <stdint.h>
#include <vx_intrinsics.h>
#include "common.h"

uint32_t ilog2 (uint32_t value) {
  	return (uint32_t)(sizeof(uint32_t) * 8UL) - (uint32_t)__builtin_clzl((value << 1) - 1UL) - 1;
}
struct tile_arg_t {
  	struct kernel_arg_t karg;	
  	uint32_t tile_width;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
};

void kernel_body(int task_id, void* arg) {
	struct tile_arg_t* _arg = (struct tile_arg_t*)(arg);
	
	uint32_t xoffset = 0;
	uint32_t yoffset = task_id * _arg->tile_height;	
	uint8_t* dst_ptr = (uint8_t*)(_arg->karg.dst_ptr + xoffset * _arg->karg.dst_stride + yoffset * _arg->karg.dst_pitch);

	float fu = xoffset * _arg->deltaX;
	float fv = yoffset * _arg->deltaY;

	for (uint32_t y = 0; y < _arg->tile_height; ++y) {
		uint32_t* dst_row = (uint32_t*)dst_ptr;
		for (uint32_t x = 0; x < _arg->tile_width; ++x) {
			int32_t u = (int32_t)(fu * (1<<20));
			int32_t v = (int32_t)(fv * (1<<20));
			dst_row[x] = vx_tex(0, u, v, 0x0);
			fu += _arg->deltaX;
		}
		dst_ptr += _arg->karg.dst_pitch;
		fv += _arg->deltaY;
	}
}

int main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	// configure texture unit
	vx_csr_write(CSR_TEX_ADDR(0),   arg->src_ptr);
	vx_csr_write(CSR_TEX_MIPOFF(0), 0);	
	vx_csr_write(CSR_TEX_WIDTH(0),  ilog2(arg->src_width));
	vx_csr_write(CSR_TEX_HEIGHT(0), ilog2(arg->src_height));
	vx_csr_write(CSR_TEX_FORMAT(0), arg->format);
	vx_csr_write(CSR_TEX_WRAP(0),   (arg->wrap << 2) | arg->wrap);
	vx_csr_write(CSR_TEX_FILTER(0), arg->filter);

	struct tile_arg_t targ;
	targ.karg        = *arg;
	targ.tile_width  = arg->dst_width;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;    
	targ.deltaX      = 1.0f / arg->dst_width;
	targ.deltaY      = 1.0f / arg->dst_height;
	
	vx_spawn_tasks(arg->num_tasks, kernel_body, &targ);
}