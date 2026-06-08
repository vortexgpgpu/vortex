#include <vx_spawn2.h>
#include "common.h"
#include <setup_math.h>   // gfx_setup::{clip_near, setup_triangle} (-I gfx_setup_kernel)

// gfx_v2 fused setup -> binning — nine CP-sequenced launches over resident
// buffers. Two kernel entries (the CP picks one per phase): setup_k runs the
// front end (stages 0-2) producing a dense per-prim bbox + the prim count P
// (meta[0]); binning_k runs the bin-sort back end (stages 3-8), consuming that
// bbox and reading P/keys from meta so the host never touches the intermediate
// data. (Splitting also keeps each entry's uniform-load count modest — the
// merged 9-stage function overran the Vortex uniform-lowering pass.)

using gfx_setup::rast_prim_t;
using gfx_setup::setup_triangle;
using gfx_setup::clip_near;

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

static inline void bin_range(const setup_bbox_t& p, int& bL, int& bR, int& bT, int& bB) {
  bL = p.bbL >> PIPE_BIN_LOG; bR = (p.bbR - 1) >> PIPE_BIN_LOG;
  bT = p.bbT >> PIPE_BIN_LOG; bB = (p.bbB - 1) >> PIPE_BIN_LOG;
}

// Clip + setup for one triangle, kept out-of-line so the FP math does not bloat
// the entry. Writes up to SETUP_MAX_SUB kept subtri bboxes, returns the count.
static uint32_t __attribute__((noinline))
clip_and_setup(const setup_vertex_t* v, int W, int H, setup_bbox_t* out) {
  clip_tri_t sub[SETUP_MAX_SUB];
  int ns = clip_near(v[0], v[1], v[2], sub);
  uint32_t kept = 0;
  for (int s = 0; s < ns; ++s) {
    rast_prim_t prim{};
    setup_bbox_t bb{};
    if (setup_triangle(sub[s].v[0], sub[s].v[1], sub[s].v[2], W, H,
                       SETUP_NEAR, SETUP_FAR, prim, bb))
      out[kept++] = bb;
  }
  return kept;
}

// ---- setup front end: clip + setup -> dense bbox[] + P (meta[0]) -----------
__kernel void setup_k(kernel_arg_t* __UNIFORM__ arg) {
  auto verts     = reinterpret_cast<setup_vertex_t*>(arg->verts_addr);
  auto slot_bbox = reinterpret_cast<setup_bbox_t*>(arg->slot_bbox_addr);
  auto keep      = reinterpret_cast<uint32_t*>(arg->keep_addr);
  auto offset    = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto tsum      = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t ntri = arg->num_tris;
  const int      W = (int)arg->width;
  const int      H = (int)arg->height;
  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;

  switch (arg->stage) {

  case PIPE_STAGE_SETUP: {                             // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      setup_bbox_t bb[SETUP_MAX_SUB];
      uint32_t kept = clip_and_setup(&verts[3 * t], W, H, bb);
      for (uint32_t s = 0; s < kept; ++s) slot_bbox[t * SETUP_MAX_SUB + s] = bb[s];
      keep[t] = kept;
    }
  } break;

  case PIPE_STAGE_SCAN: {                              // single CTA: block scan
    uint32_t chunk = (ntri + T - 1) / T;
    uint32_t lo = umin(tid * chunk, ntri), hi = umin(lo + chunk, ntri);
    { uint32_t s = 0; for (uint32_t i = lo; i < hi; ++i) s += keep[i]; tsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = tsum[t]; tsum[t] = acc; acc += v; }
      meta[0] = acc; offset[ntri] = acc;
    }
    __syncthreads();
    { uint32_t acc = tsum[tid]; for (uint32_t i = lo; i < hi; ++i) { offset[i] = acc; acc += keep[i]; } }
  } break;

  case PIPE_STAGE_EMIT: {                              // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      uint32_t w = offset[t];
      for (uint32_t s = 0; s < keep[t]; ++s)
        bbox[w++] = slot_bbox[t * SETUP_MAX_SUB + s];
    }
  } break;

  }
}

// ---- binning back end: bin-sort dense bbox[] (P=meta[0]) -> headers + pids --
__kernel void binning_k(kernel_arg_t* __UNIFORM__ arg) {
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto bcount    = reinterpret_cast<uint32_t*>(arg->bcount_addr);
  auto boffset   = reinterpret_cast<uint32_t*>(arg->boffset_addr);
  auto keys      = reinterpret_cast<uint32_t*>(arg->keys_addr);
  auto btsum     = reinterpret_cast<uint32_t*>(arg->btsum_addr);
  auto thist     = reinterpret_cast<uint32_t*>(arg->thist_addr);
  auto bincount  = reinterpret_cast<uint32_t*>(arg->bincount_addr);
  auto binbase   = reinterpret_cast<uint32_t*>(arg->binbase_addr);
  auto headers   = reinterpret_cast<pipe_header_t*>(arg->headers_addr);
  auto pids      = reinterpret_cast<uint32_t*>(arg->pids_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;
  const uint32_t B   = PIPE_NUM_BINS;

  switch (arg->stage) {

  case PIPE_STAGE_BCOUNT: {                            // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; bin_range(bbox[i], bL, bR, bT, bB);
      bcount[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
    }
  } break;

  case PIPE_STAGE_BSCAN: {                             // single CTA: block scan
    uint32_t P = meta[0];
    uint32_t chunk = (P + T - 1) / T;
    uint32_t lo = umin(tid * chunk, P), hi = umin(lo + chunk, P);
    { uint32_t s = 0; for (uint32_t i = lo; i < hi; ++i) s += bcount[i]; btsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = btsum[t]; btsum[t] = acc; acc += v; }
      meta[1] = acc; boffset[P] = acc;
    }
    __syncthreads();
    { uint32_t acc = btsum[tid]; for (uint32_t i = lo; i < hi; ++i) { boffset[i] = acc; acc += bcount[i]; } }
  } break;

  case PIPE_STAGE_BEMIT: {                             // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; bin_range(bbox[i], bL, bR, bT, bB);
      uint32_t w = boffset[i];
      for (int by = bT; by <= bB; ++by)
        for (int bx = bL; bx <= bR; ++bx)
          keys[w++] = ((uint32_t)(by * PIPE_BIN_COLS + bx) << PIPE_PRIM_BITS) | i;
    }
  } break;

  case PIPE_STAGE_BHIST: {                             // multi-CTA bin-stripe
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = umin(tid * kchunk, K), khi = umin(klo + kchunk, K);
    for (uint32_t b = lo; b < hi; ++b) thist[tid * B + b] = 0;
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) { uint32_t b = keys[k] >> PIPE_PRIM_BITS; if (b >= lo && b < hi) thist[tid * B + b]++; }
    __syncthreads();
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b]; bincount[b] = s; }
  } break;

  case PIPE_STAGE_BBASE: {                             // single CTA: base + headers
    if (tid == 0) {
      uint32_t acc = 0, nb = 0;
      for (uint32_t b = 0; b < B; ++b) {
        binbase[b] = acc; acc += bincount[b];
        if (bincount[b] == 0) continue;
        headers[nb].bin_x = (uint16_t)(b % PIPE_BIN_COLS);
        headers[nb].bin_y = (uint16_t)(b / PIPE_BIN_COLS);
        headers[nb].pids_offset = binbase[b];
        headers[nb].pids_count  = bincount[b];
        ++nb;
      }
      meta[2] = nb;
    }
  } break;

  case PIPE_STAGE_BSCATTER: {                          // multi-CTA bin-stripe
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = umin(tid * kchunk, K), khi = umin(klo + kchunk, K);
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t run = binbase[b]; for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; } }
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) {
      uint32_t key = keys[k], b = key >> PIPE_PRIM_BITS;
      if (b >= lo && b < hi) pids[thist[tid * B + b]++] = key & PIPE_PRIM_MASK;
    }
  } break;

  }
}
