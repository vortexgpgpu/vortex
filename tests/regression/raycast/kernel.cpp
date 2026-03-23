
#include <vx_spawn2.h>
#include "render.h"

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t y = blockIdx.y;

  if (x >= arg->dst_width || y >= arg->dst_height)
    return;

  auto out_ptr = reinterpret_cast<uint32_t *>(arg->dst_addr);

  float3_t color = float3_t(0, 0, 0);
  for (uint32_t s = 0; s < arg->samples_per_pixel; ++s) {
    auto ray = GenerateRay(x, y, arg);
    color += Trace(ray, arg);
  }

  out_ptr[x + y * arg->dst_width] = RGB32FtoRGB8(color);
}
