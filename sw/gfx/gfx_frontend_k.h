#pragma once

// gfx_v2 fused setup -> binning front end — device kernel stages.
//
// Defines the two kernel entries that produce RASTER's gfx-v1 buffers: setup_k
// (clip + triangle setup -> dense primbuf + bbox + P) and binning_k (bin-sort
// over the dense tile grid -> tilebuf). Include from a test's kernel.cpp; the
// test adds its own fragment/consumer kernel. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // pipe_arg_t, PIPE_STAGE_*, PIPE_PRIM_*, setup types
#include <gfx_setup.h>          // gfx_setup::{clip_near, setup_triangle}

// Coarse bin granularity the front end bins at (= VX_CFG_RASTER_BIN_LOG_SIZE,
// supplied as a -D in the kernel build; fallback for standalone include).
#ifndef VX_CFG_RASTER_BIN_LOG_SIZE
#define VX_CFG_RASTER_BIN_LOG_SIZE 7
#endif
#ifndef PIPE_BIN_LOG
#define PIPE_BIN_LOG   VX_CFG_RASTER_BIN_LOG_SIZE
#endif

namespace gfx_pipe {

using gfx_setup::rast_prim_t;
using gfx_setup::setup_triangle;
using gfx_setup::clip_near;
using vortex::graphics::rast_bin_header_t;

static inline uint32_t pipe_umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

static inline void pipe_bin_range(const setup_bbox_t& p, int& bL, int& bR, int& bT, int& bB) {
  bL = p.bbL >> PIPE_BIN_LOG; bR = (p.bbR - 1) >> PIPE_BIN_LOG;
  bT = p.bbT >> PIPE_BIN_LOG; bB = (p.bbB - 1) >> PIPE_BIN_LOG;
}

// Clip + setup for one triangle, kept out-of-line so the FP math does not bloat
// the entry (the merged 9-stage function otherwise overruns the uniform pass).
static uint32_t __attribute__((noinline))
pipe_clip_and_setup(const setup_vertex_t* v, int W, int H, uint32_t cull_mode,
                    const setup_viewport_t* vp,
                    rast_prim_t* prim_out, setup_bbox_t* bbox_out) {
  clip_tri_t sub[SETUP_MAX_SUB];
  int ns = clip_near(v[0], v[1], v[2], sub);
  uint32_t kept = 0;
  for (int s = 0; s < ns; ++s) {
    rast_prim_t prim{};
    setup_bbox_t bb{};
    if (setup_triangle(sub[s].v[0], sub[s].v[1], sub[s].v[2], W, H,
                       SETUP_NEAR, SETUP_FAR, prim, bb, cull_mode, vp)) {
      prim_out[kept] = prim;
      bbox_out[kept] = bb;
      ++kept;
    }
  }
  return kept;
}

} // namespace gfx_pipe

// ---- vertex assembly: resident VS records -> setup_vertex_t[] (no readback) --
// Routes each VS-output record into the front end's vertex form: slot 0 is the
// clip-space position, slots 1.. are generic varyings (16 bytes each), mapped
// by component count (2 -> texcoord, 3/4 -> colour) the gfx-v1 way. One
// thread/vertex; grid-strided so any launch geometry covers num_verts.
__kernel void expand_k(expand_arg_t* __UNIFORM__ arg) {
  auto recs = reinterpret_cast<const uint8_t*>(arg->vsrec_addr);
  auto out  = reinterpret_cast<setup_vertex_t*>(arg->verts_addr);
  const uint32_t n      = arg->num_verts;
  const uint32_t stride = arg->vstride;
  const uint32_t nv     = arg->num_varyings;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    const uint8_t* rec = recs + (size_t)i * stride;
    const float*   pos = reinterpret_cast<const float*>(rec);
    setup_vertex_t v;
    v.pos[0] = pos[0]; v.pos[1] = pos[1]; v.pos[2] = pos[2]; v.pos[3] = pos[3];
    v.color[0] = 1.0f; v.color[1] = 1.0f; v.color[2] = 1.0f; v.color[3] = 1.0f;
    v.texcoord[0] = 0.0f; v.texcoord[1] = 0.0f;
    for (uint32_t vi = 0; vi < nv; ++vi) {
      const float* a = reinterpret_cast<const float*>(rec + 16u * (1u + vi));
      uint32_t nc = arg->varying_comps[vi];
      if (nc == 2) {
        v.texcoord[0] = a[0]; v.texcoord[1] = a[1];
      } else if (nc >= 3) {
        v.color[0] = a[0]; v.color[1] = a[1];
        v.color[2] = a[2]; v.color[3] = nc >= 4 ? a[3] : 1.0f;
      }
    }
    out[i] = v;
  }
}

// ---- setup front end: clip + setup -> dense primbuf + bbox + P (meta[0]) ----
__kernel void setup_k(pipe_arg_t* __UNIFORM__ arg) {
  using namespace gfx_pipe;
  auto verts     = reinterpret_cast<setup_vertex_t*>(arg->verts_addr);
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

  // App viewport transform, or the default full-framebuffer y-down mapping when
  // unset (all four zero — what the standalone setup tests, which zero-init the
  // arg block, feed). Passing &vp / nullptr selects the branch inside setup.
  setup_viewport_t vp;
  const bool have_vp = (arg->vp_sx != 0.0f) || (arg->vp_tx != 0.0f)
                    || (arg->vp_sy != 0.0f) || (arg->vp_ty != 0.0f);
  vp.sx = arg->vp_sx; vp.tx = arg->vp_tx;
  vp.sy = arg->vp_sy; vp.ty = arg->vp_ty;
  const setup_viewport_t* vpp = have_vp ? &vp : nullptr;

  switch (arg->stage) {

  case PIPE_STAGE_SETUP: {
    // Count survivors only. The dense scatter (EMIT) recomputes clip+setup and
    // writes each rast_prim_t straight to its compacted slot, so the 120-byte
    // record is never staged to a scratch buffer and read back — clip+setup is
    // a pure function of the (unchanged) input vertices, so the count here and
    // the per-triangle output in EMIT match exactly. Halves primbuf write
    // traffic and removes the slot_prim/slot_bbox scratch entirely.
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      rast_prim_t pr[SETUP_MAX_SUB];
      setup_bbox_t bb[SETUP_MAX_SUB];
      keep[t] = pipe_clip_and_setup(&verts[3 * t], W, H, arg->cull_mode, vpp, pr, bb);
    }
  } break;

  case PIPE_STAGE_SCAN: {
    uint32_t chunk = (ntri + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, ntri), hi = pipe_umin(lo + chunk, ntri);
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
    // Recompute clip+setup and scatter directly into the dense primbuf at the
    // scanned offset — no round-trip through a per-triangle scratch slot. kept
    // here equals keep[t] from SETUP (same pure function, same inputs).
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < ntri; t += gstride) {
      rast_prim_t pr[SETUP_MAX_SUB];
      setup_bbox_t bb[SETUP_MAX_SUB];
      uint32_t kept = pipe_clip_and_setup(&verts[3 * t], W, H, arg->cull_mode, vpp, pr, bb);
      uint32_t w = offset[t];
      for (uint32_t s = 0; s < kept; ++s) {
        prim[w] = pr[s];
        bbox[w] = bb[s];
        ++w;
      }
    }
  } break;

  }
}

// ---- binning back end: bin-sort dense bbox[] (P=meta[0]) -> dense tilebuf ----
__kernel void binning_k(pipe_arg_t* __UNIFORM__ arg) {
  using namespace gfx_pipe;
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
  const uint32_t B   = arg->num_bins;     // dense tile grid sized to the render target

  switch (arg->stage) {

  case PIPE_STAGE_BCOUNT: {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    uint32_t P = meta[0];
    for (uint32_t i = gid; i < P; i += gstride) {
      int bL, bR, bT, bB; pipe_bin_range(bbox[i], bL, bR, bT, bB);
      bcount[i] = (uint32_t)((bR - bL + 1) * (bB - bT + 1));
    }
  } break;

  case PIPE_STAGE_BSCAN: {
    uint32_t P = meta[0];
    uint32_t chunk = (P + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, P), hi = pipe_umin(lo + chunk, P);
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
      int bL, bR, bT, bB; pipe_bin_range(bbox[i], bL, bR, bT, bB);
      uint32_t w = boffset[i];
      for (int by = bT; by <= bB; ++by)
        for (int bx = bL; bx <= bR; ++bx)
          keys[w++] = ((uint32_t)(by * arg->bin_cols + bx) << PIPE_PRIM_BITS) | i;
    }
  } break;

  case PIPE_STAGE_BHIST: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = pipe_umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = pipe_umin(tid * kchunk, K), khi = pipe_umin(klo + kchunk, K);
    for (uint32_t b = lo; b < hi; ++b) thist[tid * B + b] = 0;
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) { uint32_t b = keys[k] >> PIPE_PRIM_BITS; if (b >= lo && b < hi) thist[tid * B + b]++; }
    __syncthreads();
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t s = 0; for (uint32_t t = 0; t < T; ++t) s += thist[t * B + b]; bincount[b] = s; }
  } break;

  case PIPE_STAGE_BBASE: {
    // Dense bin grid: one rast_bin_header_t per bin (empty bins get
    // pids_count=0 and RASTER skips them), so the bin count is B =
    // bin_cols*bin_rows. pids_offset is the absolute index into the sorted_pids
    // array that follows the header block. Cooperative exclusive
    // prefix-scan of bincount[] across the CTA's T threads (mirrors BSCAN),
    // replacing the single-lane O(B) loop so a large bin grid parallelises.
    uint32_t chunk = (B + T - 1) / T;
    uint32_t lo = pipe_umin(tid * chunk, B), hi = pipe_umin(lo + chunk, B);
    { uint32_t s = 0; for (uint32_t b = lo; b < hi; ++b) s += bincount[b]; btsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = btsum[t]; btsum[t] = acc; acc += v; }
    }
    __syncthreads();
    auto hdr = reinterpret_cast<rast_bin_header_t*>(tilebuf);
    uint32_t nb_local = 0;
    { uint32_t acc = btsum[tid];
      for (uint32_t b = lo; b < hi; ++b) {
        hdr[b].bin_x       = (uint16_t)(b % arg->bin_cols);
        hdr[b].bin_y       = (uint16_t)(b / arg->bin_cols);
        hdr[b].pids_offset = acc;
        hdr[b].pids_count  = bincount[b];
        binbase[b]         = acc;
        if (bincount[b]) ++nb_local;
        acc += bincount[b];
      }
    }
    __syncthreads();
    btsum[tid] = nb_local;   // offsets consumed; reuse btsum for the nb reduction
    __syncthreads();
    if (tid == 0) {
      uint32_t nb = 0;
      for (uint32_t t = 0; t < T; ++t) nb += btsum[t];
      meta[2] = nb;  // non-empty bin count (informational)
    }
  } break;

  case PIPE_STAGE_BSCATTER: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = pipe_umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    auto pids = reinterpret_cast<uint32_t*>(tilebuf + B * sizeof(rast_bin_header_t));
    uint32_t kchunk = (K + T - 1) / T;
    uint32_t klo = pipe_umin(tid * kchunk, K), khi = pipe_umin(klo + kchunk, K);
    for (uint32_t b = lo + tid; b < hi; b += T) { uint32_t run = binbase[b]; for (uint32_t t = 0; t < T; ++t) { uint32_t c = thist[t * B + b]; thist[t * B + b] = run; run += c; } }
    __syncthreads();
    for (uint32_t k = klo; k < khi; ++k) {
      uint32_t key = keys[k], b = key >> PIPE_PRIM_BITS;
      if (b >= lo && b < hi) pids[thist[tid * B + b]++] = key & PIPE_PRIM_MASK;
    }
  } break;

  }
}
