#include <vx_spawn2.h>
#include "common.h"
#include "setup_math.h"

// gfx_v2 on-device triangle setup — three CP-sequenced launches.
// SETUP/EMIT run multi-CTA (grid-stride, no barriers); SCAN runs as a single
// cooperating CTA. SETUP does the float setup once per triangle into a per-tri
// slot; EMIT is a pure integer gather, so the FP math never runs twice and the
// dense output order is input order (= Binning()'s compacted primbuf order).

using gfx_setup::rast_prim_t;
using gfx_setup::setup_triangle;

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto verts     = reinterpret_cast<setup_vertex_t*>(arg->verts_addr);
  auto slot_prim = reinterpret_cast<rast_prim_t*>(arg->slot_prim_addr);
  auto slot_bbox = reinterpret_cast<setup_bbox_t*>(arg->slot_bbox_addr);
  auto keep      = reinterpret_cast<uint32_t*>(arg->keep_addr);
  auto offset    = reinterpret_cast<uint32_t*>(arg->offset_addr);
  auto tsum      = reinterpret_cast<uint32_t*>(arg->tsum_addr);
  auto out_prim  = reinterpret_cast<rast_prim_t*>(arg->prim_addr);
  auto out_bbox  = reinterpret_cast<setup_bbox_t*>(arg->bbox_addr);
  auto meta      = reinterpret_cast<uint32_t*>(arg->meta_addr);

  const uint32_t n = arg->num_prims;
  const int      W = (int)arg->width;
  const int      H = (int)arg->height;
  const uint32_t tid = threadIdx.x;
  const uint32_t T   = blockDim.x;

  switch (arg->stage) {

  case SETUP_STAGE_SETUP: {                            // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < n; t += gstride) {
      rast_prim_t prim{};
      setup_bbox_t bbox{};
      bool kept = setup_triangle(verts[3 * t + 0], verts[3 * t + 1],
                                 verts[3 * t + 2], W, H, SETUP_NEAR, SETUP_FAR,
                                 prim, bbox);
      slot_prim[t] = prim;
      slot_bbox[t] = bbox;
      keep[t] = kept ? 1u : 0u;
    }
  } break;

  case SETUP_STAGE_SCAN: {                             // single CTA: block scan
    uint32_t chunk = (n + T - 1) / T;
    uint32_t lo = umin(tid * chunk, n), hi = umin(lo + chunk, n);
    { uint32_t s = 0; for (uint32_t i = lo; i < hi; ++i) s += keep[i]; tsum[tid] = s; }
    __syncthreads();
    if (tid == 0) {
      uint32_t acc = 0;
      for (uint32_t t = 0; t < T; ++t) { uint32_t v = tsum[t]; tsum[t] = acc; acc += v; }
      meta[0] = acc; offset[n] = acc;
    }
    __syncthreads();
    { uint32_t acc = tsum[tid]; for (uint32_t i = lo; i < hi; ++i) { offset[i] = acc; acc += keep[i]; } }
  } break;

  case SETUP_STAGE_EMIT: {                             // multi-CTA, grid-stride
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gstride = gridDim.x * blockDim.x;
    for (uint32_t t = gid; t < n; t += gstride) {
      if (!keep[t]) continue;
      uint32_t w = offset[t];
      out_prim[w] = slot_prim[t];
      out_bbox[w] = slot_bbox[t];
    }
  } break;

  }
}
