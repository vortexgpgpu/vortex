#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>

#ifdef SW_ENABLE
#include "gpu_sw.h"
static GpuSW g_gpu_sw;
#endif

using namespace graphics;

using fixeduv_t = cocogfx::TFixed<VX_TEX_FXD_FRAC>;

#define DEFAULTS_i(i) \
	z[i] = FloatA(0.0f); \
	r[i] = FloatA(1.0f); \
	g[i] = FloatA(1.0f); \
	b[i] = FloatA(1.0f); \
	a[i] = FloatA(1.0f); \
	u[i] = FloatA(0.0f); \
	v[i] = FloatA(0.0f)

#define GRADIENTS_HW_i(i) { \
    int32_t f0 = csr_read(VX_CSR_RASTER_BCOORD_X##i); \
	int32_t f1 = csr_read(VX_CSR_RASTER_BCOORD_Y##i); \
	int32_t f2 = csr_read(VX_CSR_RASTER_BCOORD_Z##i); \
	auto F0 = static_cast<float>(*reinterpret_cast<const FloatA*>(&f0)); \
	auto F1 = static_cast<float>(*reinterpret_cast<const FloatA*>(&f1)); \
	auto F2 = static_cast<float>(*reinterpret_cast<const FloatA*>(&f2)); \
	auto r  = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(r * F0); \
    dy[i] = FloatA(r * F1); \
}

#define GRADIENTS_SW_i(i) { \
	auto F0 = static_cast<float>(*reinterpret_cast<const FloatA*>(&bcoords[i].x)); \
	auto F1 = static_cast<float>(*reinterpret_cast<const FloatA*>(&bcoords[i].y)); \
	auto F2 = static_cast<float>(*reinterpret_cast<const FloatA*>(&bcoords[i].z)); \
	auto r  = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(r * F0); \
    dy[i] = FloatA(r * F1); \
}

#ifdef FIXEDPOINT_RASTERIZER

inline int32_t imadd(int32_t a, int32_t b, int32_t c, int32_t s) {
	int32_t p = ((int64_t)a * (int64_t)b) >> (s << 3);
	return p + c;
}

#define multadd_fx(a, b, c) \
	fixed24_t::make(imadd(a.data(), b.data(), c.data(), 3))

#define INTERPOLATE_i(i, dst, src) { \
	auto tmp = multadd_fx(src.x, dx[i], src.z);  \
	dst[i]   = multadd_fx(src.y, dy[i], tmp);    \
}

#define MODULATE_i(i, dst, src1_r, src1_g, src1_b, src1_a, src2) \
	dst[i].r = (src1_r[i].data() * src2[i].r) >> fixed24_t::FRAC; \
	dst[i].g = (src1_g[i].data() * src2[i].g) >> fixed24_t::FRAC; \
	dst[i].b = (src1_b[i].data() * src2[i].b) >> fixed24_t::FRAC; \
	dst[i].a = (src1_a[i].data() * src2[i].a) >> fixed24_t::FRAC

#define TO_RGBA_i(i, dst, src_r, src_g, src_b, src_a) \
	dst[i].r = static_cast<uint8_t>((src_r[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].g = static_cast<uint8_t>((src_g[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].b = static_cast<uint8_t>((src_b[i].data() * 255) >> fixed24_t::FRAC); \
	dst[i].a = static_cast<uint8_t>((src_a[i].data() * 255) >> fixed24_t::FRAC)

#define OUTPUT_i(i, mask, x, y, face, color, depth, func) \
	if (mask & (1 << i)) { \
		auto pos_x = (x << 1) + (i & 1); \
		auto pos_y = (y << 1) + (i >> 1); \
		auto pos_z = depth[i].data(); \
		func(pos_x, pos_y, face, color[i].value, pos_z); \
	}

#else

#define INTERPOLATE_i(i, dst, src) { \
	auto tmp = src.x * dx[i] + src.z;  \
	dst[i]   = src.y * dy[i] + tmp;    \
}

#define MODULATE_i(i, dst, src1_r, src1_g, src1_b, src1_a, src2) \
	dst[i].r = static_cast<uint8_t>(src1_r[i] * src2[i].r); \
	dst[i].g = static_cast<uint8_t>(src1_g[i] * src2[i].g); \
	dst[i].b = static_cast<uint8_t>(src1_b[i] * src2[i].b); \
	dst[i].a = static_cast<uint8_t>(src1_a[i] * src2[i].a)

#define TO_RGBA_i(i, dst, src_r, src_g, src_b, src_a) \
	dst[i].r = static_cast<uint8_t>(src_r[i] * 255); \
	dst[i].g = static_cast<uint8_t>(src_g[i] * 255); \
	dst[i].b = static_cast<uint8_t>(src_b[i] * 255); \
	dst[i].a = static_cast<uint8_t>(src_a[i] * 255)

#define OUTPUT_i(i, mask, x, y, face, color, depth, func) \
	if (mask & (1 << i)) { \
		auto pos_x = (x << 1) + (i & 1); \
		auto pos_y = (y << 1) + (i >> 1); \
		auto pos_z = static_cast<uint32_t>(depth[i] * 65336); \
		func(pos_x, pos_y, face, color[i].value, pos_z); \
	}

#endif

#define DEFAULTS \
	DEFAULTS_i(0); \
	DEFAULTS_i(1); \
	DEFAULTS_i(2); \
	DEFAULTS_i(3)  \

#define GRADIENTS_HW \
	GRADIENTS_HW_i(0) \
	GRADIENTS_HW_i(1) \
	GRADIENTS_HW_i(2) \
	GRADIENTS_HW_i(3) \

#define GRADIENTS_SW \
	GRADIENTS_SW_i(0) \
	GRADIENTS_SW_i(1) \
	GRADIENTS_SW_i(2) \
	GRADIENTS_SW_i(3) \

#define INTERPOLATE(dst, src) \
	INTERPOLATE_i(0, dst, src); \
	INTERPOLATE_i(1, dst, src); \
	INTERPOLATE_i(2, dst, src); \
	INTERPOLATE_i(3, dst, src)

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

#define TO_RGBA(dst, src_r, src_g, src_b, src_a) \
	TO_RGBA_i(0, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(1, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(2, dst, src_r, src_g, src_b, src_a); \
	TO_RGBA_i(3, dst, src_r, src_g, src_b, src_a)

#define TEXTURING(dst, u, v, func) \
	dst[0] = func(0, fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data(), 0); \
	dst[1] = func(0, fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data(), 0); \
	dst[2] = func(0, fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data(), 0); \
	dst[3] = func(0, fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data(), 0)

#define OUTPUT(pos_mask, face, color, depth, func) \
	auto mask = (pos_mask >> 0) & 0xf; \
	auto x    = (pos_mask >> 4) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
	auto y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS-1))) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
	OUTPUT_i(0, mask, x, y, face, color, depth, func) \
	OUTPUT_i(1, mask, x, y, face, color, depth, func) \
	OUTPUT_i(2, mask, x, y, face, color, depth, func) \
	OUTPUT_i(3, mask, x, y, face, color, depth, func)

void shader_function_hw(kernel_arg_t* __UNIFORM__  arg) {
	FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4];
	FloatA dx[4], dy[4];
	cocogfx::ColorARGB tex_color[4], out_color[4];

	DEFAULTS;

	auto prim_ptr = (rast_prim_t*)arg->prim_addr;

	for (;;) {
		auto status = vx_rast();
		auto stanp_valid = status & 0x1;
		if (!stanp_valid)
			return;

		auto pid = (status >> 1);
		auto& attribs = prim_ptr[pid].attribs;

		GRADIENTS_HW

		if (arg->depth_enabled) {
			INTERPOLATE(z, attribs.z);
		}
		if (arg->color_enabled) {
			INTERPOLATE(r, attribs.r);
			INTERPOLATE(g, attribs.g);
			INTERPOLATE(b, attribs.b);
			INTERPOLATE(a, attribs.a);
		}
		if (arg->tex_enabled) {
			INTERPOLATE(u, attribs.u);
			INTERPOLATE(v, attribs.v);
		}

		if (arg->tex_enabled) {
		#ifdef SW_ENABLE
			if (arg->sw_tex) {
				TEXTURING(tex_color, u, v, g_gpu_sw.tex);
			} else
		#endif
			{
				TEXTURING(tex_color, u, v, vx_tex);
			}
			if (arg->tex_modulate) {
				MODULATE(out_color, r, g, b, a, tex_color);
			} else {
				REPLACE(out_color, tex_color);
			}
		} else {
			TO_RGBA(out_color, r, g, b, a);
		}

		auto pos_mask = csr_read(VX_CSR_RASTER_POS_MASK);
	#ifdef SW_ENABLE
		if (arg->sw_om) {
			OUTPUT(pos_mask, 0, out_color, z, g_gpu_sw.om);
		} else
	#endif
		{
			OUTPUT(pos_mask, 0, out_color, z, vx_om);
		}
	}
}

#ifdef SW_ENABLE
void shader_function_sw_rast_cb(uint32_t pos_mask, graphics::vec3e_t bcoords[4], uint32_t pid, void* /*cb_arg*/) {
	FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4];
	FloatA dx[4], dy[4];
	cocogfx::ColorARGB tex_color[4], out_color[4];

	DEFAULTS;

	auto __UNIFORM__ arg = g_gpu_sw.kernel_arg();
	auto prim_ptr = (rast_prim_t*)arg->prim_addr;
	auto& attribs = prim_ptr[pid].attribs;

	GRADIENTS_SW

	if (arg->depth_enabled) {
		INTERPOLATE(z, attribs.z);
	}
	if (arg->color_enabled) {
		INTERPOLATE(r, attribs.r);
		INTERPOLATE(g, attribs.g);
		INTERPOLATE(b, attribs.b);
		INTERPOLATE(a, attribs.a);
	}
	if (arg->tex_enabled) {
		INTERPOLATE(u, attribs.u);
		INTERPOLATE(v, attribs.v);
	}

	if (arg->tex_enabled) {
		if (arg->sw_tex) {
			TEXTURING(tex_color, u, v, g_gpu_sw.tex);
		} else {
			TEXTURING(tex_color, u, v, vx_tex);
		}
		if (arg->tex_modulate) {
			MODULATE(out_color, r, g, b, a, tex_color);
		} else {
			REPLACE(out_color, tex_color);
		}
	} else {
		TO_RGBA(out_color, r, g, b, a);
	}

	if (arg->sw_om) {
		OUTPUT(pos_mask, 0, out_color, z, g_gpu_sw.om);
	} else {
		OUTPUT(pos_mask, 0, out_color, z, vx_om);
	}
}

void shader_function_sw(kernel_arg_t* /*arg*/) {
	g_gpu_sw.render(blockIdx.x);
}
#endif

int main() {
	auto __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

	auto callback = (vx_kernel_func_cb)shader_function_hw;
#ifdef SW_ENABLE
	g_gpu_sw.configure(arg);
	if (arg->sw_rast) {
		callback = (vx_kernel_func_cb)shader_function_sw;
	}
#endif

	uint32_t num_tasks = 1 << arg->log_num_tasks;
	return vx_spawn_threads(1, &num_tasks, nullptr, (vx_kernel_func_cb)callback, arg);
}
