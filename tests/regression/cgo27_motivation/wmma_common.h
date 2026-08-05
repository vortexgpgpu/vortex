#ifndef _CGO27_WMMA_COMMON_H_
#define _CGO27_WMMA_COMMON_H_

// Device-side helpers shared by the kernel entries. Included by kernel.cpp before
// the per-mode headers (k_core.h / k_tcu.h / k_dtcu.h).
//
// One tile geometry, derived from VX_CFG_NUM_THREADS:
//
//   ctx (NR=8)  tile 16x16x16, NRA=NRB=NRC=8 -> 24 f-registers single-buffered.
//
// All WMMA modes share it. A second, smaller geometry was tried so mode 5 could
// double-buffer its operands in registers; it does not work — mma_sync pins its
// operands to fixed physical registers (C/D f0-f7, A f10-f17, B f24-f31), and FragA
// and FragC/D are hard-wired to 8 registers regardless of NR, so the tile cannot be
// shrunk to buy room. See the mode 5 comment in k_tcu.h.
//
// The HOST mirrors this geometry with `cfg` in main.cpp to build the launch grid.
// Host and kernel MUST agree — a mismatch there is what broke NT=32 (RFC blocker 2).

#include "common.h"
#include "epilogue.h"
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx      = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE>;

// Software fp16 -> fp32 (IEEE half->single). Used by the SIMT path only: the scalar
// pipeline has no HW fp16 (march is rv*imaf, no Zfh), so mode 0 must widen in
// software while the TCU/DTCU paths consume fp16 natively.
static inline float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t out;
  if (e == 0) {
    if (m == 0) {
      out = s;                                   // +/- 0
    } else {                                     // subnormal
      e = 127u - 15u + 1u;
      while ((m & 0x400u) == 0) { m <<= 1; e--; }
      m &= 0x3ffu;
      out = s | (e << 23) | (m << 13);
    }
  } else if (e == 0x1fu) {
    out = s | 0x7f800000u | (m << 13);           // inf / nan
  } else {
    out = s | ((e + (127u - 15u)) << 23) | (m << 13);
  }
  union { uint32_t u; float f; } cvt;
  cvt.u = out;
  return cvt.f;
}

// ---- fragment helpers (templated on the geometry so a second one can be added) ----
//
// EVERY helper here is always_inline, and every fragment register index is a
// compile-time constant. Both are load-bearing (measured on mode 1/2 at NT=32):
//  * a non-inlined helper leaks `fragD`'s address, so the accumulator array must
//    live in memory instead of registers — 14.5k cycles became ~158k (10.8x);
//  * a runtime index (`for (r = 0; r < NR; ++r) frag.data[r]`) does the same thing,
//    which is also why mode 6 uses NAMED smem stages instead of `A_smem[cur]`.
// vx_tensor.h marks all of its own fragment-touching functions always_inline for
// exactly this reason.

template <typename CTX>
static __attribute__((always_inline)) void wmma_seed_C_of(typename CTX::fragment_acc& fragD,
                                                          typename CTX::output_t* pC,
                                                          uint32_t tile_row, uint32_t tile_col,
                                                          uint32_t N) {
  CTX::load_matrix_sync(fragD, pC + tile_row * N + tile_col, N);
}

template <typename CTX>
static __attribute__((always_inline)) void wmma_store_D_of(typename CTX::output_t* pD,
                                                           typename CTX::fragment_acc& fragD,
                                                           uint32_t tile_row, uint32_t tile_col,
                                                           uint32_t N) {
  CTX::store_matrix_sync(pD + tile_row * N + tile_col, fragD, N);
}

template <typename CTX, typename F>
static __attribute__((always_inline)) void wmma_map_frag_of(typename CTX::fragment_acc& fragD,
                                                            F&& fn) {
  vt::detail::unroll_for<CTX::fragment_acc::NR>([&](auto r) {
    fragD.data[r] = fn(fragD.data[r]);
  });
}

// FUSED epilogue for the in-core modes: apply the elementwise map while the
// accumulator is still in registers, so the tile is never round-tripped through
// memory. This is the in-core path's structural advantage over the DTCU, which has
// no epilogue HW and must run a separate pass (k_core.h::moti_epilogue).
// Coordinate-dependent epilogues (residual/scale) cannot use this path — see the
// notes in epilogue/residual.h.
template <typename CTX>
static __attribute__((always_inline)) void wmma_fuse_epilogue_of(typename CTX::fragment_acc& fragD,
                                                                 uint32_t app) {
  switch (app) {
  case 2: wmma_map_frag_of<CTX>(fragD, [](float v) { return epi_relu(v); }); break;
  case 3: wmma_map_frag_of<CTX>(fragD, [](float v) { return epi_gelu(v); }); break;
  default: break;   // app 1 baseline; apps 4-8 use separate passes (Phase B)
  }
}

// Convenience wrappers for the default (NR=8) geometry used by modes 1/2/6.
static __attribute__((always_inline)) void wmma_seed_C(ctx::fragment_acc& fragD, ctx::output_t* pC,
                                                       uint32_t tile_row, uint32_t tile_col,
                                                       uint32_t N) {
  wmma_seed_C_of<ctx>(fragD, pC, tile_row, tile_col, N);
}
static __attribute__((always_inline)) void wmma_store_D(ctx::output_t* pD, ctx::fragment_acc& fragD,
                                                        uint32_t tile_row, uint32_t tile_col,
                                                        uint32_t N) {
  wmma_store_D_of<ctx>(pD, fragD, tile_row, tile_col, N);
}
static __attribute__((always_inline)) void wmma_fuse_epilogue(ctx::fragment_acc& fragD, uint32_t app) {
  wmma_fuse_epilogue_of<ctx>(fragD, app);
}

#endif // _CGO27_WMMA_COMMON_H_
