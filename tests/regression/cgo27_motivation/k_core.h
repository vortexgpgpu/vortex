#ifndef _CGO27_K_CORE_H_
#define _CGO27_K_CORE_H_

// mode 0 — in-core SIMT: the no-tensor-unit baseline. One thread per output
// element, scalar MAC loop over K, fp16 widened in software (see h2f).
// Source lineage: tests/regression/sgemm/kernel.cpp.

#include "wmma_common.h"
#include <vx_spawn2.h>

__kernel void moti_simt(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;   // read up front, see k_tcu.h
  auto pA = reinterpret_cast<const uint16_t*>(arg->A_addr); // fp16 storage
  auto pB = reinterpret_cast<const uint16_t*>(arg->B_addr);
  auto pC = reinterpret_cast<const float*>(arg->C_addr);
  auto pD = reinterpret_cast<float*>(arg->D_addr);

  uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t row = blockIdx.y;

  float sum = pC[row * N + col];
  for (uint32_t k = 0; k < K; ++k) {
    sum += h2f(pA[row * K + k]) * h2f(pB[col * K + k]); // B col-major
  }
  // Fused epilogue: the value is still in a register, so no extra memory pass.
  pD[row * N + col] = epi_apply(app, sum);
}

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

#endif // _CGO27_K_CORE_H_
