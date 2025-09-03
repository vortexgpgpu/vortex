
#include <vx_spawn.h>
#include <vx_print.h>
#include "render.h"

#define BLOCK_WIDTH  16
#define BLOCK_HEIGHT 4

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {

  //vx_printf("*** tile: %d %d\n", blockIdx.x, blockIdx.y);

  auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);

  for (uint32_t ty = 0; ty < BLOCK_HEIGHT; ++ty) {
    for (uint32_t tx = 0; tx < BLOCK_WIDTH; ++tx) {
      uint32_t x = blockIdx.x * BLOCK_WIDTH + tx;
      uint32_t y = blockIdx.y * BLOCK_HEIGHT + ty;
      if (x >= arg->dst_width || y >= arg->dst_height)
        continue;

      float3_t color = float3_t(0, 0, 0);
      for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
        auto ray = GenerateRay(x, y, arg);
        color += Trace(ray, arg);
      }

      uint32_t globalIdx = x + y * arg->dst_width;
      out_ptr[globalIdx] = RGB32FtoRGB8(color);
    }
  }
}

int main() {
  auto arg = reinterpret_cast<kernel_arg_t *>(csr_read(VX_CSR_MSCRATCH));
  const uint32_t grid_dim[2] = {
    (arg->dst_width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
    (arg->dst_height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT
  };
  return vx_spawn_threads(2, grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}