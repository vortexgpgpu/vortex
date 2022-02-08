#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  float  x;
  float  y;
  float  z;
  uint32_t  c;
  float  u;
  float  v;
} vtx_t;

typedef struct {
  vtx_t  v0;
  vtx_t  v1;
  vtx_t  v2;
} prim_t;

typedef struct {
  uint32_t  top;
  uint32_t  left;
  uint32_t  width;
  uint32_t  height;
  uint32_t  num_prims;
  uint32_t* indices;
} tile_t;

typedef struct {
  uint32_t  num_tiles;
  uint32_t  tiles_addr;
  uint32_t  prims_addr;
  uint32_t  dst_addr;
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint8_t   dst_stride;  
  uint32_t  dst_pitch;  
} kernel_arg_t;

#endif