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
  // Fragment-export aperture encoding. The shader builds its own store address
  // from these; they MUST match the OM aperture DCRs the runtime programs, or the
  // ingress misreads every record. Both come from one call to set_aperture().
  uint32_t aperture_xbits;
  uint32_t aperture_ybits;
  uint32_t aperture_record_shift;
} kernel_arg_t;

// RASTER dispatch v2 (push): the raster work distributor launches the kernel once
// per covered-quad wave; it runs straight-line and exits (no worker loop).

#endif
