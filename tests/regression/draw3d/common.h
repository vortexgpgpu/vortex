#ifndef _COMMON_H_
#define _COMMON_H_

#define SW_ENABLE

#include <stdint.h>
#include <VX_config.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

#include <graphics.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

class GpuSW;

typedef struct {
#ifdef SW_ENABLE
  graphics::RasterDCRS raster_dcrs;  
  graphics::TexDCRS    tex_dcrs;
  graphics::OMDCRS     om_dcrs;
#endif
  uint32_t log_num_tasks;
  uint64_t prim_addr;

  bool depth_enabled;
  bool color_enabled;
  bool tex_enabled; 
  bool tex_modulate;
  bool sw_rast;
  bool sw_tex;
  bool sw_om;
} kernel_arg_t;

#endif