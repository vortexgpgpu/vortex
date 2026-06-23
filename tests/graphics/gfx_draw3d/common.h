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

#ifdef GFX_FWD
// RASTER dispatch v2 (FWD): the kernel is launched once per core as a "driver"
// warp that arms the Fragment Work Distributor; the FWD then injects fragment
// waves that re-enter the same kernel with role=FRAGMENT and read their per-lane
// payload from LMEM. One arg struct serves both roles.
#define GFX_ROLE_DRIVER   0
#define GFX_ROLE_FRAGMENT 1

typedef struct {
  uint32_t role;          // GFX_ROLE_DRIVER | GFX_ROLE_FRAGMENT
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
  uint64_t prim_addr;     // (fragment) primitive buffer base
  uint64_t frag_ctx;      // (driver) device address of the fragment arg struct
} fwd_arg_t;
#endif

#endif
