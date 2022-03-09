#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>

#define GRADIENTS_i(i) \
{ \
	csr_write(CSR_RASTER_FRAG, i); \
	auto cx = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_X)); \
	auto cy = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Y)); \
	auto cz = fixed16_t::make(csr_read(CSR_RASTER_BCOORD_Z)); \
	auto r  = cocogfx::Inverse<fixed24_t>(cx + cy + cz); \
    auto gx = cocogfx::Mul<fixed24_t>(cx, r); \
    auto gy = cocogfx::Mul<fixed24_t>(cy, r); \
	csr_write(CSR_RASTER_GRAD_X, gx.data()); \
	csr_write(CSR_RASTER_GRAD_Y, gy.data()); \
}

#define GRADIENTS  \
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

#define TEXTURING(dst, u, v)	\
	dst[0] = vx_tex(0, u[0].data(), v[0].data(), 0); \
	dst[1] = vx_tex(0, u[1].data(), v[1].data(), 0); \
	dst[2] = vx_tex(0, u[2].data(), v[2].data(), 0); \
	dst[3] = vx_tex(0, u[3].data(), v[3].data(), 0)

#define MODULATE_i(i, dst, in1, in2_r, in2_g, in2_b, in2_a) \
	dst[i].r = cocogfx::Mul8(in1[i].r, static_cast<uint8_t>(in2_r[i])); \
	dst[i].g = cocogfx::Mul8(in1[i].g, static_cast<uint8_t>(in2_g[i])); \
	dst[i].b = cocogfx::Mul8(in1[i].b, static_cast<uint8_t>(in2_b[i])); \
	dst[i].a = cocogfx::Mul8(in1[i].a, static_cast<uint8_t>(in2_a[i]))

#define MODULATE(dst, in1, in2_r, in2_g, in2_b, in2_a) \
	MODULATE_i(0, dst, in1, in2_r, in2_g, in2_b, in2_a); \
	MODULATE_i(1, dst, in1, in2_r, in2_g, in2_b, in2_a); \
	MODULATE_i(2, dst, in1, in2_r, in2_g, in2_b, in2_a); \
	MODULATE_i(3, dst, in1, in2_r, in2_g, in2_b, in2_a)

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

#define OUTPUT(color, z) \
	vx_rop(color[0].value, z[0].data()); \
	vx_rop(color[1].value, z[1].data()); \
	vx_rop(color[2].value, z[2].data()); \
	vx_rop(color[3].value, z[3].data())

void shader_function(int task_id, kernel_arg_t* kernel_arg) {
	auto prim_ptr = (rast_prim_t*)kernel_arg->prim_addr;
	fixed24_t z[4], r[4], g[4], b[4], a[4], u[4], v[4];
	cocogfx::ColorARGB tex_color[4], out_color[4];

	for (;;) {
		__DIVERGENT__ int status = vx_rast();
		if (0 == status)
			return;

		auto pid      = status >> 1;
		auto& prim    = prim_ptr[pid];
		auto& attribs = prim.attribs;

		GRADIENTS;

		INTERPOLATE(z, attribs.z);
		INTERPOLATE(r, attribs.r);
		INTERPOLATE(g, attribs.g);
		INTERPOLATE(b, attribs.b);
		INTERPOLATE(a, attribs.a);
		INTERPOLATE(u, attribs.u);
		INTERPOLATE(v, attribs.v);

		if (kernel_arg->tex_enabled) {
			TEXTURING(tex_color, u, v);
			MODULATE(out_color, tex_color, r, g, b, a);
		} else {
			TO_RGBA(out_color, r, g, b, a);
		}	

		OUTPUT(out_color, z);
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	int num_warps = vx_num_warps();
	int num_threads = vx_num_threads();
	int total_threads = num_warps * total_threads;
	//vx_spawn_tasks(total_threads, (vx_spawn_tasks_cb)shader_function, arg);
	shader_function(0, arg);
	return 0;
}