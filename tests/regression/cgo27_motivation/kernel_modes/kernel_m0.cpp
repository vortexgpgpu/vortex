// GPU-side program for mode 0 -- in-core SIMT, scalar MAC loop.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m0.vxbin, which runs on
// the GPU. It has no main(); its entry point is the __kernel below. The host program that
// opens the device, uploads A/B/C, launches this and reads D back is main.cpp, built for
// x86 -- that is where main() lives.
//
// It contains this mode's kernel and NOTHING else, which is the point. In the old
// all-in-one kernel.vxbin every mode occupied address space even though only one ran, and
// address decides icache set: adding modes 3/4 moved mode 2 from 15,468 to 24,106 cycles
// with a byte-identical kernel body. Here every mode's code starts at 0x180000034
// whatever else is in the tree.

#include "wmma_common.h"
#include <vx_spawn2.h>
#ifdef MOTI_WITH_ROW_PASS
// App 6 (row-wise softmax) is a reduction across a whole row, so it cannot be fused and
// runs as a separate pass that EVERY mode needs in its own module. Carrying it
// unconditionally is not free: adding these two kernels to a binary moved mode 4 by
// +44.6 % (32,583 -> 47,118) and every other in-core mode by 0.1-1.2 %, because a mode's
// cycle count depends on where its code lands in the icache. So it is a build option, and
// an app-6 measurement takes its OWN app-1 baseline from the same binary.
#include "k_epilogue.h"
#endif

__kernel void moti_simt(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;   // read up front, see kernel_m1.cpp
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
