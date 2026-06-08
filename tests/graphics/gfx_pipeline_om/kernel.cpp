#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <setup_math.h>   // gfx_setup::{clip_near, setup_triangle} (-I gfx_setup_kernel)

// gfx_v2 device front end + fragment-interpolation + OM in one module:
//   setup_k / binning_k  -> produce RASTER's tilebuf + primbuf (pinned)
//   kernel_main ("main")  -> interpolate prim attribs per RASTER quad, vx_om
// The setup/binning stages are the dense-tile-grid pipeline; the fragment
// kernel is gfx_draw3d's (RASTER bcoord CSRs -> interpolated colour -> OM).

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
  const uint32_t B   = arg->num_bins;     // dense tile grid sized to the render target

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
          keys[w++] = ((uint32_t)(by * arg->bin_cols + bx) << PIPE_PRIM_BITS) | i;
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
      // Dense tile grid: one header per tile (empty tiles get pids_count=0 and
      // RASTER skips them). So RASTER's tile count is just B = bin_cols*bin_rows,
      // a host-known function of the framebuffer size — no num_tiles readback.
      uint32_t acc = 0;
      for (uint32_t b = 0; b < B; ++b) { binbase[b] = acc; acc += bincount[b]; }
      auto hdr = reinterpret_cast<rast_tile_header_t*>(tilebuf);
      uint32_t nb = 0;
      for (uint32_t b = 0; b < B; ++b) {
        hdr[b].tile_x      = (uint16_t)(b % arg->bin_cols);
        hdr[b].tile_y      = (uint16_t)(b / arg->bin_cols);
        hdr[b].pids_offset = (uint16_t)(2 * (B - 1 - b) + binbase[b]);
        hdr[b].pids_count  = (uint16_t)bincount[b];
        if (bincount[b]) ++nb;
      }
      meta[2] = nb;  // non-empty tile count (informational)
    }
  } break;

  case PIPE_STAGE_BSCATTER: {
    uint32_t lo = blockIdx.x * arg->bin_stripe, hi = umin(lo + arg->bin_stripe, B);
    uint32_t K = meta[1];
    // dense grid: pid region starts after all B headers
    auto pids = reinterpret_cast<uint32_t*>(tilebuf + B * sizeof(rast_tile_header_t));
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

// ---- fragment kernel (gfx_draw3d): interpolate prim attribs -> vx_om --------
// Each thread pops a RASTER quad, reads its pid + bcoords (CSRs), interpolates
// the prim's colour from the device-produced primbuf, and writes via OM.

#define BCOORD_AS_FLOAT(csr) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(csr_read(csr))))

#define GRADIENTS_HW_i(i) { \
    auto F0 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_X##i); \
    auto F1 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Y##i); \
    auto F2 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Z##i); \
    auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); dy[i] = FloatA(recip * F1); }

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = src.x * dx[i] + src.z; dst[i] = src.y * dy[i] + tmp; }

#define TO_RGBA_i(i, dst, sr, sg, sb, sa) \
    dst[i].r = static_cast<uint8_t>(sr[i] * 255); \
    dst[i].g = static_cast<uint8_t>(sg[i] * 255); \
    dst[i].b = static_cast<uint8_t>(sb[i] * 255); \
    dst[i].a = static_cast<uint8_t>(sa[i] * 255)

#define OUTPUT_i(i, mask, x, y, face, color, depth) \
    if (mask & (1 << i)) { \
        auto pos_x = (x << 1) + (i & 1); auto pos_y = (y << 1) + (i >> 1); \
        auto pos_z = static_cast<uint32_t>(depth[i] * 65336); \
        vx_om(pos_x, pos_y, face, color[i].value, pos_z); }

#define GRADIENTS_HW   GRADIENTS_HW_i(0) GRADIENTS_HW_i(1) GRADIENTS_HW_i(2) GRADIENTS_HW_i(3)
#define INTERPOLATE(d, s) INTERPOLATE_i(0,d,s); INTERPOLATE_i(1,d,s); INTERPOLATE_i(2,d,s); INTERPOLATE_i(3,d,s)
#define TO_RGBA(d, r, g, b, a) TO_RGBA_i(0,d,r,g,b,a); TO_RGBA_i(1,d,r,g,b,a); TO_RGBA_i(2,d,r,g,b,a); TO_RGBA_i(3,d,r,g,b,a)
#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    auto mask = (pos_mask >> 0) & 0xf; \
    auto x = (pos_mask >> 4) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    auto y = (pos_mask >> (4 + (VX_RASTER_DIM_BITS-1))) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    OUTPUT_i(0,mask,x,y,face,color,depth) OUTPUT_i(1,mask,x,y,face,color,depth) \
    OUTPUT_i(2,mask,x,y,face,color,depth) OUTPUT_i(3,mask,x,y,face,color,depth)

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z[4], r[4], g[4], b[4], a[4], dx[4], dy[4];
  cocogfx::ColorARGB out_color[4];
  for (int i = 0; i < 4; ++i) {
    z[i] = FloatA(0.0f); r[i] = FloatA(1.0f); g[i] = FloatA(1.0f);
    b[i] = FloatA(1.0f); a[i] = FloatA(1.0f);
  }
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  vx_rast_begin();
  for (;;) {
    uint32_t pos_mask = vx_rast();
    if (pos_mask == 0) return;
    uint32_t pid = csr_read(VX_CSR_RASTER_PID);
    auto& attribs = prim_ptr[pid].attribs;
    GRADIENTS_HW
    if (arg->depth_enabled) { INTERPOLATE(z, attribs.z); }
    if (arg->color_enabled) {
      INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
      INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
    }
    TO_RGBA(out_color, r, g, b, a);
    OUTPUT_QUAD(pos_mask, 0, out_color, z);
  }
}
