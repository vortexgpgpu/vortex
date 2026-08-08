#ifndef _CGO27_K_EPILOGUE_H_
#define _CGO27_K_EPILOGUE_H_

// The standalone epilogue pass, used by the DTCU modes only. It is not a mode: the
// engines have no epilogue HW, so an elementwise app costs them a SECOND launch over D,
// and that extra M*N round-trip is exactly the cost asymmetry the app sweep measures.

#include "wmma_common.h"
#include <vx_spawn2.h>

// Standalone elementwise epilogue pass over D, used by the DTCU modes (3/4).
//
// The DTCU has GEMM HW only — no activation/bias/epilogue — so its descriptor
// produces a bare D and the epilogue must be applied afterwards as a SEPARATE
// full pass over the matrix (read D, apply, write D). That extra round-trip is
// precisely the asymmetry the app sweep is designed to expose: the in-core modes
// fuse the same math for free (see wmma_common.h::wmma_fuse_epilogue), while the
// DTCU pays memory traffic proportional to M*N for it.
//
// Launched with the same geometry as moti_simt: one thread per output element.
#if MOTI_APP_IS_ELEMENTWISE
__kernel void moti_epilogue(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, app = arg->app;
  auto pD = reinterpret_cast<float*>(arg->D_addr);

  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y;

  pD[row * N + col] = epi_apply(app, pD[row * N + col]);
}
#endif // MOTI_APP_IS_ELEMENTWISE

// Row-wise softmax over D (app 6), as a standalone pass.
//
// One block per row, one warp per block, so the three phases below need no barrier: a
// Vortex warp runs its threads in lockstep, and the reduction tree reads what the previous
// step wrote within the same instruction stream. Local Memory holds the blockDim-element
// reduction scratch.
//
// Launched for EVERY mode, not just the engines. That is the point of the app: a row
// spans more output tiles than any one warp owns, so the reduction cannot be folded into
// the accumulator the way ReLU and GELU are, and the in-core modes have to make the same
// extra pass the DTCU does.
#if MOTI_APP_NEEDS_ROW_PASS
__kernel void moti_softmax(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N;
  auto pD = reinterpret_cast<float*>(arg->D_addr);

  const uint32_t row = blockIdx.x;
  const uint32_t t   = threadIdx.x;
  const uint32_t nt  = blockDim.x;
  auto red = reinterpret_cast<float*>(__local_mem());
  float* r = pD + (size_t)row * N;

  // 1. row max
  float m = -3.4028235e38f;
  for (uint32_t j = t; j < N; j += nt) m = epi_softmax_max(m, r[j]);
  red[t] = m;
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] = epi_softmax_max(red[t], red[t + s]);
  }
  const float row_max = red[0];

  // 2. sum of exp(x - max)
  float acc = 0.0f;
  for (uint32_t j = t; j < N; j += nt) acc = epi_softmax_addexp(acc, r[j], row_max);
  red[t] = acc;
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
  }
  const float row_sum = red[0];

  // 3. normalise in place
  for (uint32_t j = t; j < N; j += nt) r[j] = epi_softmax_norm(r[j], row_max, row_sum);
}
#endif // MOTI_APP_NEEDS_ROW_PASS

#endif // _CGO27_K_EPILOGUE_H_
