#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 on-device bin-sort binning — shared host/device ABI.
// Tests the bin-sort pipeline (count -> prefix-sum -> emit -> sort ->
// header-scan) running on the SIMT cores, validated against the host
// bin-sort reference (which the gfx_binsort unit test already proved
// renders identically to gfx-v1).

#define BINSORT_W         512
#define BINSORT_H         512
#define BINSORT_BIN_LOG   7                                   // 128px coarse bin
#define BINSORT_BIN_COLS  ((BINSORT_W + (1 << BINSORT_BIN_LOG) - 1) >> BINSORT_BIN_LOG)
#define BINSORT_BIN_ROWS  ((BINSORT_H + (1 << BINSORT_BIN_LOG) - 1) >> BINSORT_BIN_LOG)
#define BINSORT_NUM_BINS  (BINSORT_BIN_COLS * BINSORT_BIN_ROWS)
#define BINSORT_PRIM_BITS 20
#define BINSORT_PRIM_MASK ((1u << BINSORT_PRIM_BITS) - 1)

// One primitive's screen-space bbox (pixels, clamped to the render target).
// Setup (edges/clip) is upstream; this test isolates the binning stage.
typedef struct {
  uint32_t bbL, bbR, bbT, bbB;
} binsort_prim_t;

// Sparse bin header (sorted by bin_id == scan order).
typedef struct {
  uint16_t bin_x, bin_y;
  uint32_t pids_offset;   // start index into the sorted pid array
  uint32_t pids_count;    // prims overlapping this bin
} binsort_header_t;

// Pipeline stage selected per launch (the host/CP sequences them).
// HIST/SCATTER are bin-striped multi-CTA (CTA owns bins bin_id % G == cta),
// so cross-core sharing collapses to the tiny BASE scan (B bin totals).
#define BINSORT_STAGE_COUNT   0   // multi-CTA: bins covered per prim
#define BINSORT_STAGE_SCAN    1   // single-CTA: prefix-sum -> offsets, P
#define BINSORT_STAGE_EMIT    2   // multi-CTA: emit composite keys
#define BINSORT_STAGE_HIST    3   // multi-CTA bin-stripe: per-thread histogram of owned bins
#define BINSORT_STAGE_BASE    4   // single-CTA: bin-base scan + headers
#define BINSORT_STAGE_SCATTER 5   // multi-CTA bin-stripe: stable scatter of owned bins

typedef struct {
  uint32_t num_prims;
  uint32_t stage;
  uint32_t bin_stripe;    // bins per CTA (contiguous): CTA c owns [c*bs, c*bs+bs)
  uint32_t _pad;
  uint64_t prims_addr;    // binsort_prim_t[num_prims]   (in)
  uint64_t count_addr;    // uint32[num_prims]           (scratch)
  uint64_t offset_addr;   // uint32[num_prims + 1]       (scratch)
  uint64_t keys_addr;     // uint32[P]                   (scratch: composite keys)
  uint64_t tsum_addr;     // uint32[T]                   (scratch: block-scan partials)
  uint64_t thist_addr;    // uint32[T * NUM_BINS]        (scratch: counting-sort hist/cursors)
  uint64_t bincount_addr; // uint32[NUM_BINS]            (scratch)
  uint64_t binbase_addr;  // uint32[NUM_BINS]            (scratch)
  uint64_t headers_addr;  // binsort_header_t[NUM_BINS]  (out)
  uint64_t pids_addr;     // uint32[P]                   (out: sorted pids)
  uint64_t meta_addr;     // uint32[2] = { P, num_bins } (out)
} kernel_arg_t;

#endif
