#include <vx_spawn2.h>
#include "common.h"

// Read one word of every sector of each line, in line order.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const kernel_arg_t a = *arg;
  auto base = reinterpret_cast<volatile uint32_t*>(a.addr);
  uint32_t sum = 0;
  for (uint32_t i = 0; i < a.lines; ++i) {
    for (uint32_t s = 0; s < a.sectors; ++s) {
      sum += base[(i * a.line_stride + s * a.sector_bytes) / sizeof(uint32_t)];
    }
  }
  asm volatile ("" :: "r"(sum));
}
