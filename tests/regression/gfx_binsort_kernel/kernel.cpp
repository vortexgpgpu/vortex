#include <vx_spawn2.h>
#include "common.h"

// gfx_v2 on-device bin-sort — first increment: single CTA, all six stages on
// device, intermediate arrays in global scratch, CTA barriers between stages.
// Parallel where embarrassingly parallel (setup+count, emit); the scan / sort /
// header-scan run on thread 0 here (correct, serial) and are parallelized in a
// follow-up increment (block scan + LSD radix).

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto prims   = reinterpret_cast<binsort_prim_t*>(arg->prims_addr);
  auto count   = reinterpret_cast<uint32_t*>(arg->count_addr);
  auto offset  = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto keys    = reinterpret_cast<uint32_t*>(arg->keys_addr);
  auto headers = reinterpret_cast<binsort_header_t*>(arg->headers_addr);
  auto pids    = reinterpret_cast<uint32_t*>(arg->pids_addr);
  auto meta    = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t n        = arg->num_prims;
  const uint32_t tid      = threadIdx.x;
  const uint32_t nthreads = blockDim.x;

  // Stage 1: setup+count (parallel, grid-stride) — bins covered per prim.
  for (uint32_t i = tid; i < n; i += nthreads) {
    binsort_prim_t p = prims[i];
    int bL = p.bbL >> BINSORT_BIN_LOG, bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
    int bT = p.bbT >> BINSORT_BIN_LOG, bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
    count[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
  }
  __syncthreads();

  // Stage 2: prefix-sum (thread 0; serial) -> offsets + total P.
  if (tid == 0) {
    uint32_t acc = 0;
    for (uint32_t i = 0; i < n; ++i) { offset[i] = acc; acc += count[i]; }
    offset[n] = acc;
    meta[0]   = acc;
  }
  __syncthreads();

  const uint32_t P = meta[0];

  // Stage 3: emit composite keys at private offsets (parallel, no atomics).
  for (uint32_t i = tid; i < n; i += nthreads) {
    binsort_prim_t p = prims[i];
    int bL = p.bbL >> BINSORT_BIN_LOG, bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
    int bT = p.bbT >> BINSORT_BIN_LOG, bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
    uint32_t w = offset[i];
    for (int by = bT; by <= bB; ++by)
      for (int bx = bL; bx <= bR; ++bx)
        keys[w++] = ((uint32_t)(by * BINSORT_BIN_COLS + bx) << BINSORT_PRIM_BITS) | i;
  }
  __syncthreads();

  // Stage 4: sort keys[0..P) ascending (thread 0; insertion sort for v1).
  // Composite-key ascending == bin bucket then draw order.
  if (tid == 0) {
    for (uint32_t i = 1; i < P; ++i) {
      uint32_t k = keys[i];
      int j = (int)i - 1;
      while (j >= 0 && keys[j] > k) { keys[j + 1] = keys[j]; --j; }
      keys[j + 1] = k;
    }
  }
  __syncthreads();

  // Stage 5: header-scan (thread 0) -> sparse headers + sorted pids.
  if (tid == 0) {
    uint32_t nb = 0, i = 0;
    while (i < P) {
      uint32_t bin = keys[i] >> BINSORT_PRIM_BITS;
      uint32_t start = i;
      while (i < P && (keys[i] >> BINSORT_PRIM_BITS) == bin) {
        pids[i] = keys[i] & BINSORT_PRIM_MASK;
        ++i;
      }
      headers[nb].bin_x       = (uint16_t)(bin % BINSORT_BIN_COLS);
      headers[nb].bin_y       = (uint16_t)(bin / BINSORT_BIN_COLS);
      headers[nb].pids_offset = start;
      headers[nb].pids_count  = i - start;
      ++nb;
    }
    meta[1] = nb;
  }
}
