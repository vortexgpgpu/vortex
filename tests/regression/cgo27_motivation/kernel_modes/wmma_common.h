#ifndef _CGO27_WMMA_COMMON_H_
#define _CGO27_WMMA_COMMON_H_

// Device-side helpers shared by the kernel entries. Included by kernel.cpp before
// the per-mode programs (kernel_modes/kernel_m<N>.cpp).
//
// One tile geometry, derived from VX_CFG_NUM_THREADS:
//
//   ctx (NR=8)  tile 16x16x16, NRA=NRB=NRC=8 -> 24 f-registers single-buffered.
//
// All WMMA modes share it. A second, smaller geometry was tried so mode 5 could
// double-buffer its operands in registers; it does not work — mma_sync pins its
// operands to fixed physical registers (C/D f0-f7, A f10-f17, B f24-f31), and FragA
// and FragC/D are hard-wired to 8 registers regardless of NR, so the tile cannot be
// shrunk to buy room. See the mode 5 comment in kernel_m5.cpp.
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
// no epilogue HW and must run a separate pass (k_epilogue.h::moti_epilogue).
// The helper below reconstructs the fragment's row/column mapping, so residual and
// per-channel scale fuse here too.

// Apply the epilogue to the accumulator, WITH each element's (row, col).
//
// wmma_map_frag_of() maps values and knows no positions, which is fine for ReLU and GELU
// and useless for a residual or a per-channel scale. This mirrors store_matrix_sync's own
// indexing (vx_tensor.h:477) so register r of lane `lane` is placed exactly where the
// store will put it -- the same technique the wgmma epilogue in k_wg_common.h uses, and
// correct for the same reason: the accumulator's layout is not a secret, the store spells
// it out.
//
// The app is decided by the preprocessor now (see common.h), so this is one expression at
// any given MOTI_APP and the switch that used to live here -- a SECOND place the app id
// was decoded, contradicting epilogue.h's claim to be the only one -- is gone.
// wmma_context keeps its wmma_config_t private, so reconstruct the same one here with the
// same template arguments it uses (vx_tensor.h:162). Only tileM/tileN/tileK are public on
// the context, and the micro-tile geometry is what the store's indexing is built from.
using moti_wcfg = vt::wmma_config_t<VX_CFG_NUM_THREADS, vt::fp32, vt::fp32>;

template <typename CTX>
static __attribute__((always_inline)) void wmma_fuse_epilogue_at_of(
    typename CTX::fragment_acc& fragD, const float* aux,
    uint32_t tile_row, uint32_t tile_col, uint32_t N) {
  const uint32_t lane      = vx_thread_id();
  const uint32_t block_row = lane / moti_wcfg::tcN;
  const uint32_t block_col = lane % moti_wcfg::tcN;
  vt::detail::unroll_for<CTX::fragment_acc::NR>([&](auto r) {
    const uint32_t row = tile_row + (r / moti_wcfg::n_steps) * moti_wcfg::tcM + block_row;
    const uint32_t col = tile_col + (r % moti_wcfg::n_steps) * moti_wcfg::tcN + block_col;
    fragD.data[r] = epi_apply_at(fragD.data[r], aux, row, col, N);
  });
}

static __attribute__((always_inline)) void wmma_fuse_epilogue_at(
    ctx::fragment_acc& fragD, const float* aux,
    uint32_t tile_row, uint32_t tile_col, uint32_t N) {
  wmma_fuse_epilogue_at_of<ctx>(fragD, aux, tile_row, tile_col, N);
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


#endif // _CGO27_WMMA_COMMON_H_
