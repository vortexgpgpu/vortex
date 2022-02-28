#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <VX_types.h>
#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  float     x;
  float     y;
  float     z;
  float     w;
  uint32_t  c;
  float     u;
  float     v;
} vertex_t;

typedef struct {  
  float x;
  float y;
  float z;
  float w;
} rast_vtx_t;

typedef struct {
  rast_vtx_t v0;
  rast_vtx_t v1;
  rast_vtx_t v2;
} rast_prim_t;

typedef struct {
  uint32_t tileXY;
  uint32_t num_prims;
} rast_tile_header_t;

typedef struct {
  uint32_t  vts_addr;
  uint32_t  dst_addr;
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint8_t   dst_stride;  
  uint32_t  dst_pitch;  
} kernel_arg_t;

#endif