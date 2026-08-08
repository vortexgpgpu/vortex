#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Threads per warp. MUST equal the HW/kernel configuration: the host uses this
// for the launch geometry (block_dim) AND for the WMMA tile shapes
// (wmma_config_t<NUM_THREADS> in main.cpp), while the kernel uses
// VX_CFG_NUM_THREADS directly (wmma_context<VX_CFG_NUM_THREADS> in kernel.cpp).
// A hardcoded fallback here silently disagrees with the kernel whenever the
// Makefile sets a different -DVX_CFG_NUM_THREADS: the launch then spawns the
// wrong number of lanes (e.g. 4 of 32 -> uninitialized registers in the inactive
// lanes) and host/kernel compute different tile sizes, so the grid indexes past
// the matrices. That was the NT=32 failure (mode 1 looping ~33M iterations on a
// clobbered loop bound, plus the earlier out-of-range writes and the spurious
// dtensor_start). Derive it from the build config instead.
#ifndef NUM_THREADS
#ifdef VX_CFG_NUM_THREADS
#define NUM_THREADS VX_CFG_NUM_THREADS
#else
#define NUM_THREADS 32
#endif
#endif

#define NUM_WARPS 16
#define NUM_CORES 4
#define NUM_SOCKETS 1

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

// GEMM computed by every mode: D = C + A * B  (dtcu_compare convention).
//   A : row-major  [M x K]   (A[i*K + k])
//   B : col-major  [K x N]   (B[j*K + k])
//   C : row-major  [M x N]   (C[i*N + j])  -- accumulator preload
//   D : row-major  [M x N]   (D[i*N + j])
//
// mode selects the execution path; the GEMM and input are identical across all
// five so the ONLY difference is which compute/memory unit runs it:
//   0 = in-core SIMT      (plain scalar MAC loop, no tensor unit)
//   1 = in-core TCU       (WMMA fragments, register path)
//   2 = in-core TCU + DXA (WMMA fed by DXA-staged smem tiles)
//   3 = DTCU (no TMA)     (descriptor engine, DTENSOR_FLAG_NO_TMA: blocking)
//   4 = DTCU + DTCU_TMA   (descriptor engine with TMA overlap; see main.cpp)
typedef struct {
  uint32_t mode;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t D_addr;
  uint64_t desc_addr;   // DTCU descriptor (modes 3,4)
  // NOTE: append new fields HERE, at the end. Inserting a field in the middle
  // shifts every following offset and measurably perturbs codegen/layout — see
  // the mode-2 investigation in 260718_moti_RFC.md.
  uint32_t app;         // 1..8, selects the prologue/epilogue (see epilogue.h)
} kernel_arg_t;

// DXA descriptor slots programmed host-side (mode 2).
#define DESC_A 0
#define DESC_B 1

// ---------------------------------------------------------------------------------
// Tiling knobs the HOST and the KERNEL must agree on, defined ONCE here.
//
// Both of these used to carry an #ifndef default in two files at the same time -- the
// host's host_types.h and the kernel that reads them. They agreed only because nobody had
// edited one of the two, and a disagreement is silent: the kernel tiles one way, the host
// sizes the grid, the Local Memory and the DXA descriptors the other way, and D comes out
// wrong with no error. That is the same failure NUM_THREADS is defined here to prevent,
// and the reason this file exists.
//
// -D on the command line still overrides, and now overrides both sides at once.

// K-steps held in ONE staged tile (modes 3/4/5). Reuse along K: at S=1 a staged tile
// feeds one MMA, at S=4 it feeds four. Both the kernel's sub-tile indexing and the host's
// lmem/descriptor sizing scale with it. Measured to lose at S=2 and S=4; the reason is
// still unexplained, but it is NOT occupancy -- see README.
#ifndef MOTI_WG_KSTEPS
#define MOTI_WG_KSTEPS 1
#endif

// Column tiles one CTA sweeps against a resident A (mode 5). Reuse along N, and the one
// that pays: it divides global A traffic by this factor. The host derives the grid width
// and the lmem size from it, the kernel loops over it.
#ifndef MOTI_WG_NCOLS
#define MOTI_WG_NCOLS 4
#endif

// WHICH EPILOGUE THIS BINARY CONTAINS. Compile-time, deliberately.
//
// The apps used to be selected at RUNTIME from kernel_arg_t::app, which meant every app's
// code had to be present in every binary. That is not free here: adding two standalone
// epilogue kernels that a mode never calls moved mode 4 by +44.6 % (32,583 -> 47,118),
// because a mode's cycle count depends on where its code lands in a 16 KB icache and an
// unused kernel is enough to relocate it.
//
// So the selection is a preprocessor one. Exactly ONE epilogue is compiled -- epi_apply()
// collapses to a single expression and the standalone passes exist only for the app that
// needs them -- and comparing app N against app 1 means comparing two builds that each
// contain one epilogue rather than one build carrying all of them.
//
// The host reads this too, so `-a` is checked against it instead of selecting anything.
//   1 baseline (no epilogue)   2 ReLU   3 GELU   6 row-wise softmax
//   4, 5, 7, 8: need operands the kernel has no pointer for -- see MOTI_AUX_ELEM_OFFSET.
#ifndef MOTI_APP
#define MOTI_APP 1
#endif

// True when the app is a pure float->float map that fuses into the accumulator.
#define MOTI_APP_IS_ELEMENTWISE (MOTI_APP == 2 || MOTI_APP == 3)
// True when the app needs a row-wise reduction, which nothing can fuse: a tile holds only
// part of a row, so the row max and row sum are not known until every tile is written.
#define MOTI_APP_NEEDS_ROW_PASS (MOTI_APP == 6)

// Where the auxiliary epilogue operands live, for the apps that need one.
//
// Apps 4, 5, 7 and 8 need an operand the GEMM does not have -- a residual matrix, a
// per-channel scale, a bias. NONE of them gets a kernel_arg_t field: growing that struct
// reshuffles codegen in paths nobody touched (64 -> 80 B moved mode 2 by +15.8 % and mode
// 5 by -32.9 %), which is why the note above says to leave it alone.
//
// Instead the host allocates the auxiliary array immediately after C in the SAME buffer
// and still passes that buffer's base as C_addr. The kernel derives the address from
// C_addr, M and N, all of which it already has:
//
//     aux = (otype*)C_addr + MOTI_AUX_ELEM_OFFSET(M, N)
//
//   app 4  R    [M x N]   residual, added elementwise
//   app 5  s    [N]       per-channel scale, broadcast down the column
//   app 7  bias [N]       added before the activation
//
// App 6 (row-wise softmax) needs no operand at all, only a reduction across D's rows, and
// apps 7/8's int8 inputs are a build variant of ITYPE above rather than a runtime app id.
#define MOTI_AUX_ELEM_OFFSET(M, N) ((M) * (N))

#endif
