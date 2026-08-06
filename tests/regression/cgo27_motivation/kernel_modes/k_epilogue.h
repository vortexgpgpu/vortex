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
__kernel void moti_epilogue(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, app = arg->app;
  auto pD = reinterpret_cast<float*>(arg->D_addr);

  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y;

  pD[row * N + col] = epi_apply(app, pD[row * N + col]);
}

#endif // _CGO27_K_EPILOGUE_H_
