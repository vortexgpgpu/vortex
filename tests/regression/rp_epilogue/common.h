#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <string.h>

// Accumulator row length (one tile row read out of tensor memory).
#ifndef ROW
#define ROW 128
#endif

#ifndef CHUNK
#define CHUNK 16
#endif

typedef struct {
  uint32_t num_rows;
  float    act_scale;
  uint64_t acc_addr;    // [num_rows][ROW] fp32 accumulators
  uint64_t bias_addr;   // [ROW] fp32 bias
  uint64_t q_addr;      // [num_rows][ROW / 4] packed int8 outputs
  uint64_t scale_addr;  // [num_rows] dequantization scales
} kernel_arg_t;

// exp(x) as 2^(x*log2e) with a degree-5 polynomial; shared with the host.
static inline float rp_exp(float x) {
  float t = x * 1.44269504f;
  t = __builtin_fmaxf(t, -126.0f);
  t = __builtin_fminf(t, 126.0f);
  int i = (int)t;
  i -= (int)(t < (float)i);
  float f = t - (float)i;
  float p = 1.0f + f * (0.69314718f + f * (0.24022650f + f * (0.05550411f +
            f * (0.00961813f + f * 0.00133336f))));
  int bits = (i + 127) << 23;
  float e;
  memcpy(&e, &bits, sizeof(e));
  return p * e;
}

// GELU in its sigmoid form, x * sigmoid(1.702 x).
static inline float rp_gelu(float x) {
  return x / (1.0f + rp_exp(-1.702f * x));
}

// Round to nearest and clamp to int8, branch-free.
static inline int32_t rp_quant(float y, float s) {
  float v = __builtin_fminf(__builtin_fmaxf(y * s, -127.0f), 127.0f);
  return (int32_t)(v + __builtin_copysignf(0.5f, v));
}

#endif
