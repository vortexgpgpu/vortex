#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

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
} kernel_arg_t;

// DXA descriptor slots programmed host-side (mode 2).
#define DESC_A 0
#define DESC_B 1

#endif
