#ifndef _SETUP_TYPES_H_
#define _SETUP_TYPES_H_

// gfx_v2 setup front end — test-side oracle dimensions.
//
// The shared host/device ABI (setup_vertex_t, setup_bbox_t, clip_tri_t,
// SETUP_MAX_SUB, SETUP_NEAR/FAR) now lives in <gfx_frontend_abi.h> (sw/common).
// This header keeps only the test-oracle render-target dimensions, which are
// NOT part of the ABI — real draws carry width/height in pipe_arg_t.

#include <stdint.h>
#include <gfx_frontend_abi.h>   // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

#define SETUP_W        512
#define SETUP_H        512
#define SETUP_BIN_LOG  7      // 128px coarse bin (Binning() tileLogSize; bbox-independent)

#endif
