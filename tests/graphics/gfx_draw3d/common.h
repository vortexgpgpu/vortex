#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_types.h>

// KMU kernel-arg layout for the full TEX+RASTER+OM pipeline.
// Assumes hardware-only execution; no software-fallback state.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
} kernel_arg_t;

#endif
