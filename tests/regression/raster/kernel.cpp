#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include <graphics.h>

using namespace graphics;

#define OUTPUT_i(i, mask, x, y, color)    \
	if (mask & (1 << i)) {				  \
		auto pos_x = (x << 1) + (i & 1);  \
		auto pos_y = (y << 1) + (i >> 1); \
		auto dst_ptr = reinterpret_cast<uint32_t*>(arg->cbuf_addr + pos_x * arg->cbuf_stride + pos_y * arg->cbuf_pitch); \
		*dst_ptr = color[i].value; \
	}

#define OUTPUT(color) \
	auto pos_mask = csr_read(VX_CSR_RASTER_POS_MASK);  \
	auto mask = (pos_mask >> 0) & 0xf;                            \
	auto x    = (pos_mask >> 4) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
	auto y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS-1))) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
	OUTPUT_i(0, mask, x, y, color) \
	OUTPUT_i(1, mask, x, y, color) \
	OUTPUT_i(2, mask, x, y, color) \
	OUTPUT_i(3, mask, x, y, color)

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	const cocogfx::ColorARGB out_color[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

	for (;;) {
		auto status = vx_rast();
		if (0 == (status & 0x1))
			return;
		OUTPUT(out_color)
	}
}

int main() {
	auto __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}