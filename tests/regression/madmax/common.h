#pragma once

#include <stdint.h>


//==============================================================================
// DFV (Design-for-Verification) CSR Definitions
//==============================================================================
#define VX_CSR_DFV_CTRL           0x7C0
#define VX_CSR_DFV_ICACHE_STALL   0x7C1
#define VX_CSR_DFV_RANDOM_SEED    0x7C2
#define VX_CSR_DFV_SET_THRESHOLD   0x7C3
#define VX_CSR_DFV_DCACHE_STALL   0x7C4
#define VX_CSR_DFV_WRITEBACK_STALL 0x7C5 // Writeback stall enable (bit 0)
#define VX_CSR_DFV_FILL_STALL      0x7C6 // Cache fill stall enable (bit 0)
#define VX_CSR_DFV_RELEASE_THRESHOLD 0x7C7 // Release probability (0-255)
#define VX_CSR_DFV_RELEASE_SEED    0x7C8 // LFSR2 seed for release timing
#define VX_CSR_DFV_RELEASE_DELAY   0x7C9 // Per-point release delay [3:0]=ic [7:4]=dc [11:8]=wb [15:12]=fill
#define VX_CSR_DFV_RELEASE_FOREVER 0x7CA // When 1: once released, stalls stay off permanently
#define VX_CSR_DFV_THROTTLE_THRESHOLD 0x7CB // Throttle counter threshold (16-bit)

typedef struct {
  uint32_t grid_dim[2];
  uint32_t size;
  uint64_t dst_addr;
  uint32_t enable_dfv_test;
} kernel_arg_t;

inline float madmax_compute(uint32_t row, uint32_t col, uint32_t size) {
  // Initialize 16 independent accumulators using thread indices
  float a0 = (row * size + col) * 0.5f;
  float a1 = (col * size + row) * 0.5f;
  float a2 = a0 + a1;
  float a3 = a0 - a1;
  float a4 = a2 * 0.5f;
  float a5 = a3 * 0.5f;
  float a6 = a4 + a5;
  float a7 = a4 - a5;
  float a8 = a6 * 0.5f;
  float a9 = a7 * 0.5f;
  float a10 = a8 + a9;
  float a11 = a8 - a9;
  float a12 = a10 * 0.5f;
  float a13 = a11 * 0.5f;
  float a14 = a12 + a13;
  float a15 = a12 - a13;

  // Perform massive independent FMADD chains (1024 iterations)
  for (int i = 0; i < 256; ++i) {
    a0 = a0 * a1 + a2;
    a1 = a1 * a2 + a3;
    a2 = a2 * a3 + a4;
    a3 = a3 * a4 + a5;
    a4 = a4 * a5 + a6;
    a5 = a5 * a6 + a7;
    a6 = a6 * a7 + a8;
    a7 = a7 * a8 + a9;
    a8 = a8 * a9 + a10;
    a9 = a9 * a10 + a11;
    a10 = a10 * a11 + a12;
    a11 = a11 * a12 + a13;
    a12 = a12 * a13 + a14;
    a13 = a13 * a14 + a15;
    a14 = a14 * a15 + a0;
    a15 = a15 * a0 + a1;
  }

  // Combine results to force dependency and write output
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15;
}
