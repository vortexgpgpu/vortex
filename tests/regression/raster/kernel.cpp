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

void shader_function(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	const cocogfx::ColorARGB out_color[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};

	for (;;) {
		auto status = vx_rast();
		if (0 == (status & 0x1))
			return;
		OUTPUT(out_color)
	}
}

int main() {
	auto __UNIFORM__ arg = reinterpret_cast<kernel_arg_t*>(KERNEL_ARG_DEV_MEM_ADDR);
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)shader_function, arg);
	//shader_function(0, arg);
	return 0;
}