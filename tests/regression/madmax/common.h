#pragma once

#include <stdint.h>

typedef struct {
  uint32_t grid_dim[2];
  uint32_t size;
  uint64_t dst_addr;
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
