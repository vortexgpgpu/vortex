#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 fused setup -> binning front end — shared host/device ABI.
//
// The CP sequences the WHOLE front end as one chain of device launches over
// resident buffers, host-untouched between stages: setup (clip + triangle
// setup) produces a dense per-prim screen bbox, which binning (bin-sort)
// consumes directly to produce sorted per-bin prim lists. Validated end-to-end
// against the host Binning() oracle's tilebuf.
//
// Nine CP-sequenced launches (the launch-drain is the device barrier):
//   SETUP  SCAN  EMIT        setup front end -> dense bbox[] + P  (meta[0])
//   BCOUNT BSCAN BEMIT       per-prim bin counts -> composite keys (meta[1]=keys)
//   BHIST  BBASE  BSCATTER   counting-sort keys -> headers[] + sorted pids[]
// The binning stages read the prim count from meta[0] (written by setup SCAN),
// so the CP never reads back to the host between stages.
//
// Non-indexed triangle list. Scene is non-crossing (all in front of the near
// plane) so the clip is a passthrough and Binning() stays a valid oracle for
// the whole chain; clip correctness is covered by the gfx_setup_kernel test.

#include <stdint.h>
#include <setup_types.h>   // setup_vertex_t, setup_bbox_t, clip_tri_t, SETUP_*

// Coarse-bin geometry over the setup render target (matches SETUP_BIN_LOG, and
// Binning()'s tileLogSize, so device bins == oracle tiles).
#define PIPE_BIN_LOG   SETUP_BIN_LOG
#define PIPE_BIN_COLS  ((SETUP_W + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_BIN_ROWS  ((SETUP_H + (1 << PIPE_BIN_LOG) - 1) >> PIPE_BIN_LOG)
#define PIPE_NUM_BINS  (PIPE_BIN_COLS * PIPE_BIN_ROWS)
#define PIPE_PRIM_BITS 20
#define PIPE_PRIM_MASK ((1u << PIPE_PRIM_BITS) - 1)

// Sparse bin header (sorted by bin_id == scan order).
typedef struct {
  uint16_t bin_x, bin_y;
  uint32_t pids_offset;   // start index into the sorted pid array
  uint32_t pids_count;    // prims overlapping this bin
} pipe_header_t;

#define PIPE_STAGE_SETUP    0   // multi-CTA: clip+setup -> per-tri slot bbox + keep
#define PIPE_STAGE_SCAN     1   // single-CTA: prefix-sum keep -> offset, P=meta[0]
#define PIPE_STAGE_EMIT     2   // multi-CTA: compact kept slots -> dense bbox[]
#define PIPE_STAGE_BCOUNT   3   // multi-CTA: bins covered per prim
#define PIPE_STAGE_BSCAN    4   // single-CTA: prefix-sum -> key offsets, keys=meta[1]
#define PIPE_STAGE_BEMIT    5   // multi-CTA: emit composite (bin,prim) keys
#define PIPE_STAGE_BHIST    6   // multi-CTA bin-stripe: per-thread histogram
#define PIPE_STAGE_BBASE    7   // single-CTA: bin-base scan + headers, nbins=meta[2]
#define PIPE_STAGE_BSCATTER 8   // multi-CTA bin-stripe: stable scatter -> sorted pids

typedef struct {
  uint32_t num_tris;      // input triangle count
  uint32_t stage;
  uint32_t width, height;
  uint32_t bin_stripe;    // bins per CTA (contiguous) for HIST/SCATTER
  uint32_t _pad;
  // setup front end
  uint64_t verts_addr;     // setup_vertex_t[3*num_tris]       (in)
  uint64_t slot_bbox_addr; // setup_bbox_t[num_tris*MAX_SUB]   (scratch)
  uint64_t keep_addr;      // uint32[num_tris]                 (scratch: 0..MAX_SUB)
  uint64_t offset_addr;    // uint32[num_tris + 1]             (scratch)
  uint64_t tsum_addr;      // uint32[T]                        (scratch: block scan)
  uint64_t bbox_addr;      // setup_bbox_t[num_tris*MAX_SUB]   (dense: setup out / binning in)
  // binning
  uint64_t bcount_addr;    // uint32[P]                        (scratch)
  uint64_t boffset_addr;   // uint32[P + 1]                    (scratch)
  uint64_t keys_addr;      // uint32[keys]                     (scratch: composite keys)
  uint64_t btsum_addr;     // uint32[T]                        (scratch: block scan)
  uint64_t thist_addr;     // uint32[T * NUM_BINS]             (scratch: hist/cursors)
  uint64_t bincount_addr;  // uint32[NUM_BINS]                 (scratch)
  uint64_t binbase_addr;   // uint32[NUM_BINS]                 (scratch)
  uint64_t headers_addr;   // pipe_header_t[NUM_BINS]          (out)
  uint64_t pids_addr;      // uint32[keys]                     (out: sorted pids)
  uint64_t meta_addr;      // uint32[3] = { P, keys, nbins }   (out)
} kernel_arg_t;

#endif
