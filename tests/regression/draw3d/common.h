#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_config.h>
#include <VX_types.h>

// vortex2 KMU kernel-arg layout for the full TEX+RASTER+OM pipeline.
// The skybox v1 version embedded software-fallback state via
// graphics::{Raster,Tex,OM}DCRS; the v2 port assumes hardware-only
// execution since SimX graphics support is gated on simx_v3 Phase 5.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
} kernel_arg_t;

#endif
