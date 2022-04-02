#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <VX_types.h>
#include <cocogfx/include/fixed.hpp>
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

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;

  uint32_t cbuf_addr;  
  uint8_t  cbuf_stride;  
  uint32_t cbuf_pitch;  

  uint32_t prim_addr;
} kernel_arg_t;

#endif