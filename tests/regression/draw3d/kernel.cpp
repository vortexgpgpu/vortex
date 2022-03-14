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
	csr_write(CSR_RASTER_FRAG, i); \
	auto F0 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_X)); \
	auto F1 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Y)); \
	auto F2 = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Z)); \
	auto r  = cocogfx::Inverse<fixed24_t>(F0 + F1 + F2); \
    auto f0 = cocogfx::Mul<fixed24_t>(r, F0); \
    auto f1 = cocogfx::Mul<fixed24_t>(r, F1); \
	csr_write(CSR_RASTER_GRAD_X, f0.data()); \
	csr_write(CSR_RASTER_GRAD_Y, f1.data()); \
}

#define GRADIENTS \
	GRADIENTS_i(0) \
	GRADIENTS_i(1) \
	GRADIENTS_i(2) \
	GRADIENTS_i(3) \

#define INTERPOLATE_i(i, dst, src) \
	csr_write(CSR_RASTER_FRAG, i); \
	dst[i] = fixed24_t::make(vx_interp(src.x.data(), src.y.data(), src.z.data()))

#define INTERPOLATE(dst, src) \
	INTERPOLATE_i(0, dst, src); \
	INTERPOLATE_i(1, dst, src); \
	INTERPOLATE_i(2, dst, src); \
	INTERPOLATE_i(3, dst, src);

#define TEXTURING(dst, u, v) \
	dst[0] = vx_tex(0, fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data(), 0); \
	dst[1] = vx_tex(0, fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data(), 0); \
	dst[2] = vx_tex(0, fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data(), 0); \
	dst[3] = vx_tex(0, fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data(), 0)

#define MODULATE_i(i, dst_r, dst_g, dst_b, dst_a, src) \
	dst_r[i] = fixed24_t::make(cocogfx::Mul8(dst_r[i].data(), src[i].r)); \
	dst_g[i] = fixed24_t::make(cocogfx::Mul8(dst_g[i].data(), src[i].g)); \
	dst_b[i] = fixed24_t::make(cocogfx::Mul8(dst_b[i].data(), src[i].b)); \
	dst_a[i] = fixed24_t::make(cocogfx::Mul8(dst_a[i].data(), src[i].a)); \

#define MODULATE(dst_r, dst_g, dst_b, dst_a, src) \
	MODULATE_i(0, dst_r, dst_g, dst_b, dst_a, src); \
	MODULATE_i(1, dst_r, dst_g, dst_b, dst_a, src); \
	MODULATE_i(2, dst_r, dst_g, dst_b, dst_a, src); \
	MODULATE_i(3, dst_r, dst_g, dst_b, dst_a, src)

#define REPLACE(dst, src) \
	dst[0] = src[0]; \
	dst[1] = src[1]; \
	dst[2] = src[2]; \
	dst[3] = src[3]

#define TO_RGBA_i(i, dst, src_r, src_g, src_b, src_a) \
	dst[i].r = static_cast<uint8_t>((src_r[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].g = static_cast<uint8_t>((src_g[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].b = static_cast<uint8_t>((src_b[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].a = static_cast<uint8_t>((src_a[i].data() * 255) >> fixed24_t::FRAC); \

#define TO_RGBA(dst, src_r, src_g, src_b, src_a) \
	TO_RGBA_i(0, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(1, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(2, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(3, dst, src_r, src_g, src_b, src_a)

#define OUTPUT_i(i, color, z) \
	csr_write(CSR_RASTER_FRAG, i); \
	vx_rop(color[i].value, z[i].data()) \

#define OUTPUT(color, z) \
	OUTPUT_i(0, color, z); \
	OUTPUT_i(1, color, z); \
	OUTPUT_i(2, color, z); \
	OUTPUT_i(3, color, z)

void shader_function(int task_id, kernel_arg_t* kernel_arg) {
	auto prim_ptr = (rast_prim_t*)kernel_arg->prim_addr;
	fixed24_t z[4], r[4], g[4], b[4], a[4], u[4], v[4];
	cocogfx::ColorARGB tex_color[4], out_color[4];

	DEFAULTS;

	for (;;) {
		__DIVERGENT__ int status = vx_rast();
		if (0 == status)
			return;

		auto pid      = status >> 1;
		auto& prim    = prim_ptr[pid];
		auto& attribs = prim.attribs;

		GRADIENTS;

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
				MODULATE(r, g, b, a, tex_color);
			} else {
				REPLACE(out_color, tex_color);
			}
		} else {
			TO_RGBA(out_color, r, g, b, a);
		}	

		OUTPUT(out_color, z);
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	//int num_warps = vx_num_warps();
	//int num_threads = vx_num_threads();
	//int total_threads = num_warps * total_threads;
	//vx_spawn_tasks(total_threads, (vx_spawn_tasks_cb)shader_function, arg);
	shader_function(0, arg);
	return 0;
}