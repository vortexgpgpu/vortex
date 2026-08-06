#include <vx_spawn2.h>
#include "common.h"

// gfx_v2 on-device bin-sort — multi-CTA, CP-sequenced (multi-launch).
// One kernel, four stages selected by arg->stage; the host launches it four
// times with chained dependencies, so each launch's drain is the device-wide
// barrier between stages (the Command-Processor model). The per-prim stages
// (count, emit) run multi-CTA across the whole device (grid-stride, no
// barriers); the reductions (scan, sort) run as a single cooperating CTA.

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

static inline void bin_range(const binsort_prim_t& p, int& bL, int& bR, int& bT, int& bB) {
  bL = p.bbL >> BINSORT_BIN_LOG; bR = (p.bbR - 1) >> BINSORT_BIN_LOG;
  bT = p.bbT >> BINSORT_BIN_LOG; bB = (p.bbB - 1) >> BINSORT_BIN_LOG;
}
static inline int imin3(int a, int b, int c) { int m = a < b ? a : b; return m < c ? m : c; }
static inline int imax3(int a, int b, int c) { int m = a > b ? a : b; return m > c ? m : c; }
static inline int iclamp(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }


__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto verts    = reinterpret_cast<binsort_vertex_t*>(arg->verts_addr);
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

  switch (arg->stage) {

  case BINSORT_STAGE_COUNT: {                          // single-CTA: SETUP (verts -> bbox + cull) + COUNT
    uint32_t pchunk = (n + T - 1) / T;
    uint32_t lo = umin(tid * pchunk, n), hi = umin(lo + pchunk, n);
    for (uint32_t i = lo; i < hi; ++i) {
      binsort_vertex_t a = verts[3 * i], b = verts[3 * i + 1], c = verts[3 * i + 2];
      int det = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);   // 2x signed area
      int L = iclamp(imin3(a.x, b.x, c.x), BINSORT_W);
      int R = iclamp(imax3(a.x, b.x, c.x), BINSORT_W);
      int Tp = iclamp(imin3(a.y, b.y, c.y), BINSORT_H);
      int Bp = iclamp(imax3(a.y, b.y, c.y), BINSORT_H);
      int valid = (det != 0) && (R > L) && (Bp > Tp);   // branchless cull
      int bL = L >> BINSORT_BIN_LOG, bR = (R - 1) >> BINSORT_BIN_LOG;
      int bT = Tp >> BINSORT_BIN_LOG, bB = (Bp - 1) >> BINSORT_BIN_LOG;
      int bins = (bR - bL + 1) * (bB - bT + 1);
      binsort_prim_t p;
      p.bbL = valid ? (uint32_t)L : 0u; p.bbR = valid ? (uint32_t)R : 0u;
      p.bbT = valid ? (uint32_t)Tp : 0u; p.bbB = valid ? (uint32_t)Bp : 0u;
      prims[i] = p;
      count[i] = valid ? (uint32_t)bins : 0u;
    }
  } break;

  case BINSORT_STAGE_SCAN: {                           // single CTA: two-level block scan
    uint32_t pchunk = (n + T - 1) / T;
    uint32_t plo = umin(tid * pchunk, n), phi = umin(plo + pchunk, n);
    { uint32_t s = 0; for (uint32_t i = plo; i < phi; ++i) s += count[i]; tsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = tsum[t]; tsum[t] = acc; acc += v; }
      meta[0] = acc; offset[n] = acc;
    }
    __syncthreads();
    { uint32_t acc = tsum[tid]; for (uint32_t i = plo; i < phi; ++i) { offset[i] = acc; acc += count[i]; } }
  } break;

  case BINSORT_STAGE_EMIT: {                           // single-CTA, block-partition
    uint32_t pchunk = (n + T - 1) / T;
    uint32_t lo = umin(tid * pchunk, n), hi = umin(lo + pchunk, n);
    for (uint32_t i = lo; i < hi; ++i) {
      if (count[i] == 0) continue;                      // culled / empty bbox
      int bL, bR, bT, bB; bin_range(prims[i], bL, bR, bT, bB);
      uint32_t w = offset[i];
      for (int by = bT; by <= bB; ++by)
        for (int bx = bL; bx <= bR; ++bx)
          keys[w++] = ((uint32_t)(by * BINSORT_BIN_COLS + bx) << BINSORT_PRIM_BITS) | i;
    }
  } break;

  case BINSORT_STAGE_HIST: {                           // multi-CTA bin-stripe: per-thread hist of owned bins
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);  // contiguous stripe
    uint32_t P = meta[0];
    uint32_t kchunk = (P + T - 1) / T;
    uint32_t klo = umin(tid * kchunk, P), khi = umin(klo + kchunk, P);
    for (uint32_t b = lo; b < hi; ++b) thist[tid * B + b] = 0;   // zero owned columns
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) { uint32_t b = keys[k] >> BINSORT_PRIM_BITS; if (b >= lo && b < hi) thist[tid * B + b]++; }
    __syncthreads();
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b]; bincount[b] = s; }
  } break;

  case BINSORT_STAGE_BASE: {                           // single CTA: bin-base scan + headers
    if (tid == 0) {
      uint32_t acc = 0, nb = 0;
      for (uint32_t b = 0; b < B; ++b) {
        binbase[b] = acc; acc += bincount[b];
        if (bincount[b] == 0) continue;
        headers[nb].bin_x = (uint16_t)(b % BINSORT_BIN_COLS);
        headers[nb].bin_y = (uint16_t)(b / BINSORT_BIN_COLS);
        headers[nb].pids_offset = binbase[b];
        headers[nb].pids_count  = bincount[b];
        ++nb;
      }
      meta[1] = nb;
    }
  } break;

  case BINSORT_STAGE_SCATTER: {                        // multi-CTA bin-stripe: stable scatter of owned bins
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);  // contiguous stripe
    uint32_t P = meta[0];
    uint32_t kchunk = (P + T - 1) / T;
    uint32_t klo = umin(tid * kchunk, P), khi = umin(klo + kchunk, P);
    // owned hist columns -> stable write cursors (base + thread-exclusive prefix)
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t run = binbase[b]; for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; } }
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) {
      uint32_t key = keys[k], b = key >> BINSORT_PRIM_BITS;
      if (b >= lo && b < hi) pids[thist[tid * B + b]++] = key & BINSORT_PRIM_MASK;
    }
  } break;

  }
}
