#include "common.h"
#include <vx_intrinsics.h>
#include <bitmanip.h>
#include <vx_spawn.h>
#include <vx_print.h>

using namespace graphics;

typedef struct {
  	kernel_arg_t* state;
 	uint32_t tile_height;
  	float deltaX;
  	float deltaY;
	float minification;
} tile_arg_t;

static void memory_cb(uint32_t* out,
                      const uint32_t* addr,    
                      uint32_t stride,
                      uint32_t size,
                      void* /*cb_arg*/) {
  switch (stride) {
  case 4:
	for (uint32_t i = 0; i < size; ++i) {
		out[i] = *reinterpret_cast<const uint32_t*>(addr[i]);
	}    
  	break;
  case 2:
	for (uint32_t i = 0; i < size; ++i) {
		out[i] = *reinterpret_cast<const uint16_t*>(addr[i]);
	}    
	break;
  case 1:
	for (uint32_t i = 0; i < size; ++i) {
		out[i] = *reinterpret_cast<const uint8_t*>(addr[i]);
	}    
	break;
  }
}

static TextureSampler g_sampler(memory_cb, nullptr);

void kernel_body(int task_id, tile_arg_t* __UNIFORM__ arg) {
	auto state = arg->state;

	auto y_start = task_id * arg->tile_height;
	auto y_end = std::min<uint32_t>(y_start + arg->tile_height, state->dst_height);

	auto x_start = 0;
	auto x_end = state->dst_width;

	cocogfx::TFixed<16> xj(arg->minification);
	
	auto dst_ptr = reinterpret_cast<uint8_t*>(state->dst_addr + x_start * state->dst_stride + y_start * state->dst_pitch);

	/*vx_printf("task_id=%d, tile_width=%d, tile_height=%d, deltaX=%f, deltaY=%f, minification=%f\n", 
	 	task_id, arg->tile_width, arg->tile_height, arg->deltaX, arg->deltaY, arg->minification);*/

	auto fv = (y_start + 0.5f) * arg->deltaY;
	for (uint32_t y = y_start; y < y_end; ++y) {
		auto dst_row = reinterpret_cast<uint32_t*>(dst_ptr);
		auto fu = (x_start + 0.5f) * arg->deltaX;
		for (uint32_t x = x_start; x < x_end; ++x) {
			uint32_t color;
			cocogfx::TFixed<TEX_FXD_FRAC> xu(fu);
			cocogfx::TFixed<TEX_FXD_FRAC> xv(fv);			
			uint32_t j = std::max<int32_t>(xj.data(), cocogfx::TFixed<16>::ONE);
			uint32_t lod = std::min<uint32_t>(log2floor(j) - 16, TEX_LOD_MAX);
			if (state->filter == 2) {        
				uint32_t lodn = std::min<uint32_t>(lod + 1, TEX_LOD_MAX);
				uint32_t frac = (j - (1 << (lod + 16))) >> (lod + 16 - 8);
				uint32_t texel0, texel1;
				if (state->use_sw) {
					texel0 = g_sampler.read(0, xu.data(), xv.data(), lod);  
					texel1 = g_sampler.read(0, xu.data(), xv.data(), lodn);
				} else {
					texel0 = vx_tex(0, xu.data(), xv.data(), lod);
					texel1 = vx_tex(0, xu.data(), xv.data(), lodn);
				}				
				{
					uint32_t c0l, c0h, c1l, c1h;
					Unpack8888(texel0, &c0l, &c0h);
					Unpack8888(texel1, &c1l, &c1h);
					uint32_t cl = Lerp8888(c0l, c1l, frac);
					uint32_t ch = Lerp8888(c0h, c1h, frac);
					color = Pack8888(cl, ch);
				}
			} else {
				if (state->use_sw) {
					color = g_sampler.read(0, xu.data(), xv.data(), lod);
				} else {
					color = vx_tex(0, xu.data(), xv.data(), lod);
				}
			}
			//vx_printf("task_id=%d, x=%d, y=%d, fu=%f, fv=%f, xu=0x%x, xv=0x%x, color=0x%x\n", task_id, x, y, fu, fv, xu.data(), xv.data(), color);			
			dst_row[x] = color;
			fu += arg->deltaX;
		}
		dst_ptr += state->dst_pitch;
		fv += arg->deltaY;
	}
}

int main() {
	auto __UNIFORM__ arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	auto base_addr = arg->dcrs.read(0, DCR_TEX_ADDR);

	tile_arg_t targ;
	targ.state       = arg;
	targ.tile_height = (arg->dst_height + arg->num_tasks - 1) / arg->num_tasks;    
	targ.deltaX      = 1.0f / arg->dst_width;
	targ.deltaY      = 1.0f / arg->dst_height;

	{
		auto tex_logdim     = arg->dcrs.read(0, DCR_TEX_LOGDIM);
		auto tex_logwidth   = tex_logdim & 0xffff;
  		auto tex_logheight  = tex_logdim >> 16;
		uint32_t src_width  = (1 << tex_logwidth);
		uint32_t src_height = (1 << tex_logheight);
		float width_ratio   = float(src_width) / arg->dst_width;
		float height_ratio  = float(src_height) / arg->dst_height;
		targ.minification   = std::max<float>(width_ratio, height_ratio);
	}

  	g_sampler.configure(arg->dcrs);

	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, &targ);
	/*for (uint32_t t = 0; t < arg->num_tasks; ++t) {		
		kernel_body(t, &targ);
	}*/

	return 0;
}