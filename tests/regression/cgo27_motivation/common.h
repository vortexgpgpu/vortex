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

#endif
