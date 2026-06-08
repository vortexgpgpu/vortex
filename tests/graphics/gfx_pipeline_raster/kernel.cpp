#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"
#include <setup_math.h>   // gfx_setup::{clip_near, setup_triangle} (-I gfx_setup_kernel)

// gfx_v2 device front end + RASTER fragment kernel in one module:
//   setup_k / binning_k  -> produce RASTER's tilebuf + primbuf (pinned)
//   kernel_main ("main")  -> trivial fragment kernel: write covered pixels
// The setup/binning stages are the gfx_pipeline_kernel pipeline at the RASTER
// tile granularity; the fragment kernel is gfx_raster's.

using gfx_setup::rast_prim_t;
using gfx_setup::setup_triangle;
using gfx_setup::clip_near;
using vortex::graphics::rast_tile_header_t;

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

static inline void bin_range(const setup_bbox_t& p, int& bL, int& bR, int& bT, int& bB) {
  bL = p.bbL >> PIPE_BIN_LOG; bR = (p.bbR - 1) >> PIPE_BIN_LOG;
  bT = p.bbT >> PIPE_BIN_LOG; bB = (p.bbB - 1) >> PIPE_BIN_LOG;
}

static uint32_t __attribute__((noinline))
clip_and_setup(const setup_vertex_t* v, int W, int H,
               rast_prim_t* prim_out, setup_bbox_t* bbox_out) {
  clip_tri_t sub[SETUP_MAX_SUB];
  int ns = clip_near(v[0], v[1], v[2], sub);
  uint32_t kept = 0;
  for (int s = 0; s < ns; ++s) {
    rast_prim_t prim{};
    setup_bbox_t bb{};
    if (setup_triangle(sub[s].v[0], sub[s].v[1], sub[s].v[2], W, H,
                       SETUP_NEAR, SETUP_FAR, prim, bb)) {
      prim_out[kept] = prim;
      bbox_out[kept] = bb;
      ++kept;
    }
  }
  return kept;
}

// ---- setup front end --------------------------------------------------------
__kernel void setup_k(pipe_arg_t* __UNIFORM__ arg) {
  auto verts     = reinterpret_cast<setup_vertex_t*>(arg->verts_addr);
  auto slot_prim = reinterpret_cast<rast_prim_t*>(arg->slot_prim_addr);
  auto slot_bbox = reinterpret_cast<setup_bbox_t*>(arg->slot_bbox_addr);
  auto keep      = reinterpret_cast<uint32_t*>(arg->keep_addr);
  auto offset    = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto tsum      = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  auto prim      = reinterpret_cast<rast_prim_t*>(arg->prim_addr);
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t ntri = arg->num_tris;
  const int      W = (int)arg->width;
  const int      H = (int)arg->height;
  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;

  switch (arg->stage) {

  case PIPE_STAGE_SETUP: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      rast_prim_t pr[SETUP_MAX_SUB];
      setup_bbox_t bb[SETUP_MAX_SUB];
      uint32_t kept = clip_and_setup(&verts[3 * t], W, H, pr, bb);
      for (uint32_t s = 0; s < kept; ++s) {
        slot_prim[t * SETUP_MAX_SUB + s] = pr[s];
        slot_bbox[t * SETUP_MAX_SUB + s] = bb[s];
      }
      keep[t] = kept;
    }
  } break;

  case PIPE_STAGE_SCAN: {
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

  case PIPE_STAGE_EMIT: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      uint32_t w = offset[t];
      for (uint32_t s = 0; s < keep[t]; ++s) {
        uint32_t slot = t * SETUP_MAX_SUB + s;
        prim[w] = slot_prim[slot];
        bbox[w] = slot_bbox[slot];
        ++w;
      }
    }
  } break;

  }
}

// ---- binning back end -------------------------------------------------------
__kernel void binning_k(pipe_arg_t* __UNIFORM__ arg) {
  auto bbox      = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto bcount    = reinterpret_cast<uint32_t*>(arg->bcount_addr);
  auto boffset   = reinterpret_cast<uint32_t*>(arg->boffset_addr);
  auto keys      = reinterpret_cast<uint32_t*>(arg->keys_addr);
  auto btsum     = reinterpret_cast<uint32_t*>(arg->btsum_addr);
  auto thist     = reinterpret_cast<uint32_t*>(arg->thist_addr);
  auto bincount  = reinterpret_cast<uint32_t*>(arg->bincount_addr);
  auto binbase   = reinterpret_cast<uint32_t*>(arg->binbase_addr);
  auto tilebuf   = reinterpret_cast<uint8_t*>(arg->tilebuf_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;
  const uint32_t B   = PIPE_NUM_BINS;

  switch (arg->stage) {

  case PIPE_STAGE_BCOUNT: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; bin_range(bbox[i], bL, bR, bT, bB);
      bcount[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
    }
  } break;

  case PIPE_STAGE_BSCAN: {
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

  case PIPE_STAGE_BEMIT: {
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

  case PIPE_STAGE_BHIST: {
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

  case PIPE_STAGE_BBASE: {
    if (tid == 0) {
      uint32_t nb = 0;
      for (uint32_t b = 0; b < B; ++b) if (bincount[b]) ++nb;
      uint32_t acc = 0;
      for (uint32_t b = 0; b < B; ++b) { binbase[b] = acc; acc += bincount[b]; }
      auto hdr = reinterpret_cast<rast_tile_header_t*>(tilebuf);
      uint32_t i = 0;
      for (uint32_t b = 0; b < B; ++b) {
        if (bincount[b] == 0) continue;
        hdr[i].tile_x      = (uint16_t)(b % PIPE_BIN_COLS);
        hdr[i].tile_y      = (uint16_t)(b / PIPE_BIN_COLS);
        hdr[i].pids_offset = (uint16_t)(2 * (nb - 1 - i) + binbase[b]);
        hdr[i].pids_count  = (uint16_t)bincount[b];
        ++i;
      }
      meta[2] = nb;
    }
  } break;

  case PIPE_STAGE_BSCATTER: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1], nb = meta[2];
    auto pids = reinterpret_cast<uint32_t*>(tilebuf + nb * sizeof(rast_tile_header_t));
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

// ---- RASTER fragment kernel (gfx_raster): write covered pixels --------------
__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  const uint32_t out_color = 0xffffffff;
  vx_rast_begin();
  for (;;) {
    uint32_t pos_mask = vx_rast();
    if (pos_mask == 0) return;
    uint32_t mask = (pos_mask >> 0) & 0xf;
    uint32_t x    = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    uint32_t y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    for (uint32_t i = 0; i < 4; ++i) {
      if (mask & (1u << i)) {
        uint32_t px = (x << 1) + (i & 1);
        uint32_t py = (y << 1) + (i >> 1);
        auto dst_ptr = reinterpret_cast<uint32_t*>(
            arg->cbuf_addr + px * arg->cbuf_stride + py * arg->cbuf_pitch);
        *dst_ptr = out_color;
      }
    }
  }
}
