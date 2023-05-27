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
  uint32_t log_num_tasks;   

  bool depth_enabled;
  bool color_enabled;
  bool tex_enabled; 
  bool tex_modulate;
  bool sw_tex;
  bool sw_rast;
  bool sw_rop;
  bool sw_interp;
  
  uint32_t dst_width;
  uint32_t dst_height;

  uint64_t cbuf_addr;  
  uint8_t  cbuf_stride;  
  uint32_t cbuf_pitch;    

  uint64_t zbuf_addr;  
  uint8_t  zbuf_stride;  
  uint32_t zbuf_pitch; 

  uint64_t prim_addr;

#ifdef SW_ENABLE
  graphics::RasterDCRS raster_dcrs;
  graphics::RopDCRS    rop_dcrs;
  graphics::TexDCRS    tex_dcrs;
#endif
} kernel_arg_t;

#endif