#include <vx_spawn2.h>
#include "common.h"

// gfx_v2 on-device bin-sort — parallel single-CTA version.
// All six stages parallel within the CTA, intermediate arrays in global
// scratch, CTA barriers between stages:
//   1 count           (block-partitioned, per-prim)
//   2 prefix-sum       (two-level block scan)
//   3 emit             (per-prim, at private offsets)
//   4 sort + headers   (stable counting sort by bin == single-pass LSD radix
//                       for the bounded bin field; bin bases ARE the headers)
// The composite-key total order falls out of a stable sort on the bin field
// (the key's low bits are the prim id, ascending within a bin = draw order).

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto prims    = reinterpret_cast<binsort_prim_t*>(arg->prims_addr);
  auto count    = reinterpret_cast<uint32_t*>(arg->count_addr);
  auto offset   = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto keys     = reinterpret_cast<uint32_t*>(arg->keys_addr);
  auto tsum     = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  auto thist    = reinterpret_cast<uint32_t*>(arg->thist_addr);
  auto bincount = reinterpret_cast<uint32_t*>(arg->bincount_addr);
  auto binbase  = reinterpret_cast<uint32_t*>(arg->binbase_addr);
  auto headers  = reinterpret_cast<binsort_header_t*>(arg->headers_addr);
  auto pids     = reinterpret_cast<uint32_t*>(arg->pids_addr);
  auto meta     = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t n   = arg->num_prims;
  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;
  const uint32_t B   = BINSORT_NUM_BINS;

  // Block-partition the prims so the scan composes (each thread owns a
  // contiguous range; its count chunk is its scan chunk).
  const uint32_t pchunk = (n + T - 1) / T;
  const uint32_t plo = umin(tid * pchunk, n);
  const uint32_t phi = umin(plo + pchunk, n);

  // Stage 1: count bins covered per prim.
  for (uint32_t i = plo; i < phi; ++i) {
    binsort_prim_t p = prims[i];
    int bL = p.bbL >> BINSORT_BIN_LOG, bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
    int bT = p.bbT >> BINSORT_BIN_LOG, bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
    count[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
  }
  __syncthreads();

  // Stage 2: two-level block prefix-sum -> offset[], P.
  { uint32_t s = 0; for (uint32_t i = plo; i < phi; ++i) s += count[i]; tsum[tid] = s; }
  __syncthreads();
  if (tid == 0) {                                  // exclusive scan of per-thread sums
    uint32_t acc = 0;
    for (uint32_t t = 0; t < T; ++t) { uint32_t v = tsum[t]; tsum[t] = acc; acc += v; }
    meta[0]   = acc;                               // P
    offset[n] = acc;
  }
  __syncthreads();
  { uint32_t acc = tsum[tid]; for (uint32_t i = plo; i < phi; ++i) { offset[i] = acc; acc += count[i]; } }
  __syncthreads();

  const uint32_t P = meta[0];

  // Stage 3: emit composite keys at private offsets (no atomics).
  for (uint32_t i = plo; i < phi; ++i) {
    binsort_prim_t p = prims[i];
    int bL = p.bbL >> BINSORT_BIN_LOG, bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
    int bT = p.bbT >> BINSORT_BIN_LOG, bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
    uint32_t w = offset[i];
    for (int by = bT; by <= bB; ++by)
      for (int bx = bL; bx <= bR; ++bx)
        keys[w++] = ((uint32_t)(by * BINSORT_BIN_COLS + bx) << BINSORT_PRIM_BITS) | i;
  }
  __syncthreads();

  // Stage 4: stable counting sort by bin (parallel). Each thread owns a
  // contiguous key chunk so the global order is preserved (stable).
  const uint32_t kchunk = (P + T - 1) / T;
  const uint32_t klo = umin(tid * kchunk, P);
  const uint32_t khi = umin(klo + kchunk, P);

  for (uint32_t j = tid; j < T * B; j += T) thist[j] = 0;           // 4a: zero T×B hist
  __syncthreads();
  for (uint32_t k = klo; k < khi; ++k)                              // 4b: per-thread hist
    thist[tid * B + (keys[k] >> BINSORT_PRIM_BITS)]++;
  __syncthreads();
  for (uint32_t b = tid; b < B; b += T) {                           // 4c: bin totals (column sums)
    uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b];
    bincount[b] = s;
  }
  __syncthreads();
  if (tid == 0) {                                                   // 4d: bin bases (exclusive scan)
    uint32_t acc = 0; for (uint32_t b = 0; b < B; ++b) { binbase[b] = acc; acc += bincount[b]; }
  }
  __syncthreads();
  for (uint32_t b = tid; b < B; b += T) {                           // 4e: per-thread stable write cursors
    uint32_t run = binbase[b];
    for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; }
  }
  __syncthreads();
  for (uint32_t k = klo; k < khi; ++k) {                            // 4f: scatter (each thread owns its row)
    uint32_t key = keys[k], b = key >> BINSORT_PRIM_BITS;
    pids[thist[tid * B + b]++] = key & BINSORT_PRIM_MASK;
  }
  __syncthreads();

  // Stage 5: headers — compact non-empty bins (bin bases are the offsets).
  if (tid == 0) {
    uint32_t nb = 0;
    for (uint32_t b = 0; b < B; ++b) {
      if (bincount[b] == 0) continue;
      headers[nb].bin_x       = (uint16_t)(b % BINSORT_BIN_COLS);
      headers[nb].bin_y       = (uint16_t)(b / BINSORT_BIN_COLS);
      headers[nb].pids_offset = binbase[b];
      headers[nb].pids_count  = bincount[b];
      ++nb;
    }
    meta[1] = nb;
  }
}
