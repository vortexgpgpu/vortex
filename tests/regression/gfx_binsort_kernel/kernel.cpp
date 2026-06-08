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

  switch (arg->stage) {

  case BINSORT_STAGE_COUNT: {                          // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t i = gid; i < n; i += gstride) {
      int bL, bR, bT, bB; bin_range(prims[i], bL, bR, bT, bB);
      count[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
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

  case BINSORT_STAGE_EMIT: {                           // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t i = gid; i < n; i += gstride) {
      int bL, bR, bT, bB; bin_range(prims[i], bL, bR, bT, bB);
      uint32_t w = offset[i];
      for (int by = bT; by <= bB; ++by)
        for (int bx = bL; bx <= bR; ++bx)
          keys[w++] = ((uint32_t)(by * BINSORT_BIN_COLS + bx) << BINSORT_PRIM_BITS) | i;
    }
  } break;

  case BINSORT_STAGE_SORT: {                           // single CTA: stable counting sort + headers
    uint32_t P = meta[0];
    uint32_t kchunk = (P + T - 1) / T;
    uint32_t klo = umin(tid * kchunk, P), khi = umin(klo + kchunk, P);
    for (uint32_t j = tid; j < T * B; j += T) thist[j] = 0;
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) thist[tid * B + (keys[k] >> BINSORT_PRIM_BITS)]++;
    __syncthreads();
    for (uint32_t b = tid; b < B; b += T) { uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b]; bincount[b] = s; }
    __syncthreads();
    if (tid == 0) { uint32_t acc = 0; for (uint32_t b = 0; b < B; ++b) { binbase[b] = acc; acc += bincount[b]; } }
    __syncthreads();
    for (uint32_t b = tid; b < B; b += T) { uint32_t run = binbase[b]; for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; } }
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) { uint32_t key = keys[k], b = key >> BINSORT_PRIM_BITS; pids[thist[tid * B + b]++] = key & BINSORT_PRIM_MASK; }
    __syncthreads();
    if (tid == 0) {
      uint32_t nb = 0;
      for (uint32_t b = 0; b < B; ++b) {
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

  }
}
