// GPU-side program for mode 1 -- in-core TCU, WMMA.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m1.vxbin, which runs on
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
// The standalone epilogue passes. k_epilogue.h compiles ONLY the one this build's
// MOTI_APP needs -- nothing at all at MOTI_APP=1 -- so no unused kernel lands in the
// binary and no mode's address moves. See common.h.
#include "k_epilogue.h"

__kernel void moti_tcu(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);                 // A row-major
    ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);  // B col-major
    ctx::mma_sync(fragD, fragA, fragB, fragD);
  }
  // aux (apps 4/5) sits behind C in the same buffer -- see common.h. The indexed form is
  // what lets a residual or a per-channel scale reach the right element.
  wmma_fuse_epilogue_at(fragD, MOTI_AUX_PTR(arg->C_addr, arg->M, N), tile_row, tile_col, N);
  (void)app;
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
}
