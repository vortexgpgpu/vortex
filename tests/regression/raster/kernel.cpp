#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>

#define DEFAULTS_i(i) \
	r[i] = fixed24_t(1.0f); \
	g[i] = fixed24_t(1.0f); \
	b[i] = fixed24_t(1.0f); \
	a[i] = fixed24_t(1.0f)

#define DEFAULTS \
	DEFAULTS_i(0); \
	DEFAULTS_i(1); \
	DEFAULTS_i(2); \
	DEFAULTS_i(3)  \

#define GRADIENTS_i(i) { \
	auto F0 = static_cast<float>(fixed16_t::make(csr_read(CSR_RASTER_BCOORD_X##i))); \
	auto F1 = static_cast<float>(fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Y##i))); \
	auto F2 = static_cast<float>(fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Z##i))); \
	auto r  = 1.0f / (F0 + F1 + F2); \
    dx[i] = fixed24_t(r * F0); 		 \
    dy[i] = fixed24_t(r * F1); 		 \
}

#define GRADIENTS \
	GRADIENTS_i(0) \
	GRADIENTS_i(1) \
	GRADIENTS_i(2) \
	GRADIENTS_i(3) \

#define INTERPOLATE_i(i, dst, src) { \
	auto tmp = int32_t(int64_t(src.x.data()) * int64_t(dx[i].data()) >> 24) + src.z.data(); \
        tmp  = int32_t(int64_t(src.y.data()) * int64_t(dy[i].data()) >> 24) + tmp; \
	dst[i] = fixed24_t::make(tmp); \
}

#define INTERPOLATE(dst, src) \
	INTERPOLATE_i(0, dst, src); \
	INTERPOLATE_i(1, dst, src); \
	INTERPOLATE_i(2, dst, src); \
	INTERPOLATE_i(3, dst, src)

#define TO_RGBA_i(i, dst, src_r, src_g, src_b, src_a) \
	dst[i].r = static_cast<uint8_t>((src_r[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].g = static_cast<uint8_t>((src_g[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].b = static_cast<uint8_t>((src_b[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].a = static_cast<uint8_t>((src_a[i].data() * 255) >> fixed24_t::FRAC)

#define TO_RGBA(dst, src_r, src_g, src_b, src_a) \
	TO_RGBA_i(0, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(1, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(2, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(3, dst, src_r, src_g, src_b, src_a)

#define OUTPUT_i(i, mask, x, y, color) \
	if (mask & (1 << i)) {				  \
		auto pos_x = (x << 1) + (i & 1);  \
		auto pos_y = (y << 1) + (i >> 1); \
		auto dst_ptr = reinterpret_cast<uint32_t*>(arg->cbuf_addr + pos_x * arg->cbuf_stride + pos_y * arg->cbuf_pitch); \
		*dst_ptr = color[i].value; \
	}

#define OUTPUT(color) \
	auto __DIVERGENT__ pos_mask = csr_read(CSR_RASTER_POS_MASK);  \
	auto mask = (pos_mask >> 0) & 0xf;                            \
	auto x    = (pos_mask >> 4) & ((1 << (RASTER_DIM_BITS-1))-1); \
	auto y    = (pos_mask >> (4 + (RASTER_DIM_BITS-1))) & ((1 << (RASTER_DIM_BITS-1))-1); \
	OUTPUT_i(0, mask, x, y, color) \
	OUTPUT_i(1, mask, x, y, color) \
	OUTPUT_i(2, mask, x, y, color) \
	OUTPUT_i(3, mask, x, y, color)

void shader_function(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto prim_ptr = (rast_prim_t*)arg->prim_addr;
	fixed24_t r[4], g[4], b[4], a[4];
	fixed24_t dx[4], dy[4];
	cocogfx::ColorARGB out_color[4];

	DEFAULTS;

	for (;;) {
		auto __DIVERGENT__ status = vx_rast();
		if (0 == (status & 0x1))
			return;

		auto pid      = status >> 1;
		auto& prim    = prim_ptr[pid];
		auto& attribs = prim.attribs;

		GRADIENTS
		
		INTERPOLATE(r, attribs.r);
		INTERPOLATE(g, attribs.g);
		INTERPOLATE(b, attribs.b);
		INTERPOLATE(a, attribs.a);
		
		TO_RGBA(out_color, r, g, b, a);
		
		OUTPUT(out_color)
	}
}

int main() {
	auto arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)shader_function, arg);
	//shader_function(0, arg);
	return 0;
}