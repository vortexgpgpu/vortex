#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

using fixed16_t = cocogfx::TFixed<16>;
using fixed24_t = cocogfx::TFixed<24>;

typedef struct {
  fixed24_t x;
  fixed24_t y;
  fixed24_t z;
} rast_attrib_t;

typedef struct {
  rast_attrib_t z;
  rast_attrib_t r;
  rast_attrib_t g;
  rast_attrib_t b;
  rast_attrib_t a;
  rast_attrib_t u;
  rast_attrib_t v;
} rast_attribs_t;

typedef struct {  
  fixed16_t x;
  fixed16_t y;
  fixed16_t z;
} rast_edge_t;

typedef struct {  
  uint32_t left;
  uint32_t right;
  uint32_t top;
  uint32_t bottom;
} rast_bbox_t;

typedef struct {
  rast_edge_t    edges[3];
  rast_attribs_t attribs;
} rast_prim_t;

typedef struct {
  uint32_t tile_xy;
  uint32_t num_prims;
} rast_tile_header_t;

class RasterDCRS {
private:
  uint32_t states_[DCR_RASTER_STATE_COUNT];

public:
  RasterDCRS() {
    this->clear();
  }

  void clear() {
    for (auto& state : states_) {
      state = 0;
    }
  }

  uint32_t read(uint32_t addr) const {
    uint32_t state = DCR_RASTER_STATE(addr);
    return states_[state];
  }

  void write(uint32_t addr, uint32_t value) {
    uint32_t state = DCR_RASTER_STATE(addr);
    states_[state] = value;
  }
};  

class RopDCRS {
private:
  uint32_t states_[DCR_ROP_STATE_COUNT];

public:
  RopDCRS() {
    this->clear();
  }

  void clear() {
    for (auto& state : states_) {
      state = 0;
    }
  }

  uint32_t read(uint32_t addr) const {
    uint32_t state = DCR_ROP_STATE(addr);
    return states_[state];
  }

  void write(uint32_t addr, uint32_t value) {
    uint32_t state = DCR_ROP_STATE(addr);
    states_[state] = value;
  }
};

class GpuSW;

typedef struct {
  uint32_t log_num_tasks;
  
  uint32_t dst_width;
  uint32_t dst_height;

  uint32_t cbuf_addr;  
  uint8_t  cbuf_stride;  
  uint32_t cbuf_pitch;    

  uint32_t zbuf_addr;  
  uint8_t  zbuf_stride;  
  uint32_t zbuf_pitch; 

  uint32_t prim_addr;   

  bool depth_enabled;
  bool color_enabled;
  bool tex_enabled; 
  bool tex_modulate;
  bool sw_rast;
  bool sw_rop;
  bool sw_interp;

  RasterDCRS raster_dcrs;
  RopDCRS    rop_dcrs;
  GpuSW*     gpu_sw;
} kernel_arg_t;

#endif