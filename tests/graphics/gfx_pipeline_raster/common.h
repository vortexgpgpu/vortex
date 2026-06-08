#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 on-device front end -> RASTER fixed-function unit, end to end.
//
// The fused setup+binning pipeline (shared pipe_abi.h / pipe_frontend.h) runs on
// the SIMT cores to produce RASTER's tilebuf + primbuf into pinned memory over a
// dense tile grid; those buffers bind to RASTER via its DCRs and a trivial
// fragment kernel writes the covered pixels. The rendered image is checked
// against the gfx-v1 reference — the device front end drives the FF unit with no
// host Binning() in the loop.

#include <stdint.h>
#include <pipe_abi.h>   // pipe_arg_t, PIPE_* constants/stages (-I gfx_setup_kernel)

// Fragment kernel args (writes covered pixels white).
typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t cbuf_addr;
  uint8_t  cbuf_stride;
  uint32_t cbuf_pitch;
  uint64_t prim_addr;
} frag_arg_t;

#endif
