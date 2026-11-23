#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <string.h>

#ifndef CHUNK
#define CHUNK 16
#endif

// Output row length (head dimension) and the column group its rescale loads.
#ifndef HEAD
#define HEAD 64
#endif
#define OGROUP 16

typedef struct {
  uint32_t num_rows;
  uint32_t num_cols;  // multiple of CHUNK
  uint32_t padding;
  float    scale;
  uint64_t s_addr;    // [num_rows][num_cols] scores
  uint64_t p_addr;    // [num_rows][num_cols] unnormalized probabilities
  uint64_t o_addr;    // [num_rows][head_dim] output row, rescaled in place
  uint64_t ml_addr;   // [num_rows][2] final max and sum
} kernel_arg_t;

// exp(x) as 2^(x*log2e): integer part through the exponent bits, fraction by
// a degree-5 polynomial. Host and device share it so results match.
static inline float rp_exp(float x) {
  float t = x * 1.44269504f;
  t = __builtin_fmaxf(t, -126.0f);
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

// fmax, not a ternary: a float select becomes a divergent branch.
static inline float rp_max(float a, float b) { return __builtin_fmaxf(a, b); }

#endif
