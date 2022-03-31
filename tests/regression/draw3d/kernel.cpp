#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>

using fixeduv_t = cocogfx::TFixed<TEX_FXD_FRAC>;

#define DEFAULTS_i(i) \
	z[i] = fixed24_t(0.0f); \
	r[i] = fixed24_t(1.0f); \
	g[i] = fixed24_t(1.0f); \
	b[i] = fixed24_t(1.0f); \
	a[i] = fixed24_t(1.0f); \
	u[i] = fixed24_t(0.0f); \
	v[i] = fixed24_t(0.0f)

#define DEFAULTS \
	DEFAULTS_i(0); \
	DEFAULTS_i(1); \
	DEFAULTS_i(2); \
	DEFAULTS_i(3)  \

#define GRADIENTS_i(i) { \
	auto F0 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_X##i)); \
	auto F1 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Y##i)); \
	auto F2 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Z##i)); \
	auto r  = fixed24_t::make((1ll << (16+24)) / (int64_t(F0.data()) + int64_t(F1.data()) + int64_t(F2.data()))); \
    dx[i]   = cocogfx::Mul<fixed24_t>(r, F0); \
    dy[i]   = cocogfx::Mul<fixed24_t>(r, F1); \
}

#define GRADIENTS \
	GRADIENTS_i(0) \
	GRADIENTS_i(1) \
	GRADIENTS_i(2) \
	GRADIENTS_i(3) \

#define INTERPOLATE_i(i, dst, src) { \
	auto tmp = vx_imadd(src.x.data(), dx[i].data(), src.z.data(), 3); \
	     tmp = vx_imadd(src.y.data(), dy[i].data(), tmp, 3); \
	dst[i] = fixed24_t::make(tmp); \
}

#define INTERPOLATE(dst, src) \
	INTERPOLATE_i(0, dst, src); \
	INTERPOLATE_i(1, dst, src); \
	INTERPOLATE_i(2, dst, src); \
	INTERPOLATE_i(3, dst, src)

#define TEXTURING(dst, u, v) \
	dst[0] = vx_tex(fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data(), 0); \
	dst[1] = vx_tex(fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data(), 0); \
	dst[2] = vx_tex(fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data(), 0); \
	dst[3] = vx_tex(fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data(), 0)

#define MODULATE_i(i, dst, src1_r, src1_g, src1_b, src1_a, src2) \
	dst[i].r = (src1_r[i].data() * src2[i].r) >> fixed24_t::FRAC; \
	dst[i].g = (src1_g[i].data() * src2[i].g) >> fixed24_t::FRAC; \
	dst[i].b = (src1_b[i].data() * src2[i].b) >> fixed24_t::FRAC; \
	dst[i].a = (src1_a[i].data() * src2[i].a) >> fixed24_t::FRAC

#define MODULATE(dst, src1_r, src1_g, src1_b, src1_a, src2) \
	MODULATE_i(0, dst, src1_r, src1_g, src1_b, src1_a, src2); \
	MODULATE_i(1, dst, src1_r, src1_g, src1_b, src1_a, src2); \
	MODULATE_i(2, dst, src1_r, src1_g, src1_b, src1_a, src2); \
	MODULATE_i(3, dst, src1_r, src1_g, src1_b, src1_a, src2)

#define REPLACE(dst, src) \
	dst[0] = src[0]; \
	dst[1] = src[1]; \
	dst[2] = src[2]; \
	dst[3] = src[3]

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

#define OUTPUT_i(i, mask, x, y, face, color, depth) \
	if (mask & (1 << i)) {							\
		auto pos_x = (x << 1) + (i & 1);			\
		auto pos_y = (y << 1) + (i >> 1);			\
		vx_rop(pos_x, pos_y, face, color[i].value, depth[i].data()); \
	}

#define OUTPUT(face, color, depth) \
	auto __DIVERGENT__ pos_mask = csr_read(CSR_RASTER_POS_MASK); \
	auto mask = (pos_mask >> 0) & 0xf;			 \
	auto x    = (pos_mask >> 4) & ((1 << (RASTER_DIM_BITS-1))-1); \
	auto y    = (pos_mask >> (4 + (RASTER_DIM_BITS-1))) & ((1 << (RASTER_DIM_BITS-1))-1); \
	OUTPUT_i(0, mask, x, y, face, color, depth)  \
	OUTPUT_i(1, mask, x, y, face, color, depth)  \
	OUTPUT_i(2, mask, x, y, face, color, depth)  \
	OUTPUT_i(3, mask, x, y, face, color, depth)

void shader_function(int task_id, kernel_arg_t* kernel_arg) {
	auto prim_ptr = (rast_prim_t*)kernel_arg->prim_addr;
	fixed24_t z[4], r[4], g[4], b[4], a[4], u[4], v[4];
	fixed24_t dx[4], dy[4];
	cocogfx::ColorARGB tex_color[4], out_color[4];

	DEFAULTS;

	for (;;) {
		auto __DIVERGENT__ status = vx_rast();
		if (0 == status)
			return;

		auto pid      = status >> 1;
		auto& prim    = prim_ptr[pid];
		auto& attribs = prim.attribs;

		GRADIENTS

		if (kernel_arg->depth_enabled) {
			INTERPOLATE(z, attribs.z);
		}

		if (kernel_arg->color_enabled) {
			INTERPOLATE(r, attribs.r);
			INTERPOLATE(g, attribs.g);
			INTERPOLATE(b, attribs.b);
			INTERPOLATE(a, attribs.a);
		}
		
		if (kernel_arg->tex_enabled) {
			INTERPOLATE(u, attribs.u);
			INTERPOLATE(v, attribs.v);
			TEXTURING(tex_color, u, v);			
			if (kernel_arg->tex_modulate) {
				MODULATE(out_color, r, g, b, a, tex_color);
			} else {
				REPLACE(out_color, tex_color);
			}
		} else {
			TO_RGBA(out_color, r, g, b, a);
		}

		OUTPUT(0, out_color, z)
	}
}

int main() {
	auto arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
	auto num_warps = vx_num_warps();
	auto num_threads = vx_num_threads();
	auto total_threads = num_warps * num_threads;
	vx_spawn_tasks(total_threads, (vx_spawn_tasks_cb)shader_function, arg);
	//shader_function(0, arg);
	return 0;
}