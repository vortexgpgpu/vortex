#include "common.h"
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <cocogfx/include/color.hpp>

#define INTERPOLATE(dst, src) \
	dst[0] = fixed23_t::make(vx_interp(0, src.x.data(), src.y.data(), src.z.data())); \
	dst[1] = fixed23_t::make(vx_interp(1, src.x.data(), src.y.data(), src.z.data())); \
	dst[2] = fixed23_t::make(vx_interp(2, src.x.data(), src.y.data(), src.z.data())); \
	dst[3] = fixed23_t::make(vx_interp(3, src.x.data(), src.y.data(), src.z.data()))

#define TO_NATIVE_COLOR(dst, r, g, b, a) \
	dst[0] = {static_cast<uint8_t>(r[0]), static_cast<uint8_t>(g[0]), static_cast<uint8_t>(b[0]), static_cast<uint8_t>(a[0])}; \
	dst[1] = {static_cast<uint8_t>(r[1]), static_cast<uint8_t>(g[1]), static_cast<uint8_t>(b[1]), static_cast<uint8_t>(a[1])}; \
	dst[2] = {static_cast<uint8_t>(r[2]), static_cast<uint8_t>(g[2]), static_cast<uint8_t>(b[2]), static_cast<uint8_t>(a[2])}; \
	dst[3] = {static_cast<uint8_t>(r[3]), static_cast<uint8_t>(g[3]), static_cast<uint8_t>(b[3]), static_cast<uint8_t>(a[3])}

#define OUTPUT(color, z) \
	csr_write(CSR_ROP_Z+0,     z[0].data()); \
	csr_write(CSR_ROP_Z+1,     z[1].data()); \
	csr_write(CSR_ROP_Z+2,     z[2].data()); \
	csr_write(CSR_ROP_Z+3,     z[3].data()); \
	csr_write(CSR_ROP_COLOR+0, color[0].value); \
	csr_write(CSR_ROP_COLOR+1, color[1].value); \
	csr_write(CSR_ROP_COLOR+2, color[2].value); \
	csr_write(CSR_ROP_COLOR+3, color[3].value)

void kernel_body() {
	auto kernel_arg = (kernel_arg_t*)(KERNEL_ARG_DEV_MEM_ADDR);
	auto prim_ptr = (rast_prim_t*)kernel_arg->prim_addr;

	fixed23_t z[4], r[4], g[4], b[4], a[4], c[4], u[4], v[4];
	cocogfx::ColorARGB color[4];

	for (;;) {
		__DIVERGENT__ int status = csr_read(CSR_RASTER_FETCH);
		if (0 == status)
			return;

		auto pid = csr_read(CSR_RASTER_PID);		
		auto& attribs = prim_ptr[pid].attribs;

		INTERPOLATE(z, attribs.z);
		INTERPOLATE(r, attribs.r);
		INTERPOLATE(g, attribs.g);
		INTERPOLATE(b, attribs.b);
		INTERPOLATE(a, attribs.a);
		INTERPOLATE(u, attribs.u);
		INTERPOLATE(v, attribs.v);

		TO_NATIVE_COLOR(color, r, g, b, c);

		OUTPUT(color, z);
	}
}

int main() {
	int num_warps = vx_num_warps();
	vx_wspawn(num_warps, kernel_body);
	return 0;
}