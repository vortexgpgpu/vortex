#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include "texsw.h"

typedef struct {
  	kernel_arg_t* state;	
  	uint32_t tile_width;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
} tile_arg_t;

template <typename T, T Start, T End>
struct static_for_t {
    template <typename Fn>
    inline void operator()(const Fn& callback) const {
        callback(Start);
        static_for_t<T, Start+1, End>()(callback);
    }
};

template <typename T, T N>
struct static_for_t<T, N, N> {
    template <typename Fn>
    inline void operator()(const Fn& callback) const {}
};

void kernel_body(int task_id, tile_arg_t* arg) {
	kernel_arg_t* state = arg->state;
	
	uint32_t xoffset = 0;
	uint32_t yoffset = task_id * arg->tile_height;

	uint8_t* dst_ptr = (uint8_t*)(state->dst_addr + xoffset * state->dst_stride + yoffset * state->dst_pitch);

	Fixed<16> xlod(state->lod);

	/*vx_printf("task_id=%d, deltaX=%f, deltaY=%f, tile_width=%d, tile_height=%d\n", 
		task_id, arg->deltaX, arg->deltaY, arg->tile_width, arg->tile_height);*/

	float fv = (yoffset + 0.5f) * arg->deltaY;
	for (uint32_t y = 0; y < arg->tile_height; ++y) {
		uint32_t* dst_row = (uint32_t*)dst_ptr;
		float fu = (xoffset + 0.5f) * arg->deltaX;
		for (uint32_t x = 0; x < arg->tile_width; ++x) {
			Fixed<TEX_FXD_FRAC> xu(fu);
			Fixed<TEX_FXD_FRAC> xv(fv);
			uint32_t color;
		#ifdef ENABLE_SW
			if (state->use_sw)
				color = tex_load_sw(state, xu, xv, xlod);
			else
		#endif
			color = tex_load_hw(state, xu, xv, xlod);						
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

	// configure texture unit
	csr_write(CSR_TEX(0, TEX_STATE_WIDTH),  arg->src_logwidth);	
	csr_write(CSR_TEX(0, TEX_STATE_HEIGHT), arg->src_logheight);
	csr_write(CSR_TEX(0, TEX_STATE_FORMAT), arg->format);
	csr_write(CSR_TEX(0, TEX_STATE_WRAPU),  arg->wrapu);
	csr_write(CSR_TEX(0, TEX_STATE_WRAPV),  arg->wrapv);
	csr_write(CSR_TEX(0, TEX_STATE_FILTER), (arg->filter ? 1 : 0));
	csr_write(CSR_TEX(0, TEX_STATE_ADDR),   arg->src_addr);
	static_for_t<int, 0, TEX_LOD_MAX+1>()([&](int i) {
		csr_write(CSR_TEX(0, TEX_STATE_MIPOFF(i)), arg->mip_offs[i]);
	});	

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_width  = arg->dst_width;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;    
	targ.deltaX      = 1.0f / arg->dst_width;
	targ.deltaY      = 1.0f / arg->dst_height;
	
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
	/*for (uint32_t t=0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}