#ifndef _CGO27_EPI_SOFTMAX_H_
#define _CGO27_EPI_SOFTMAX_H_

// Apps 6 and 8 epilogue: row-wise softmax over D.
//
// This is the hardest app in the sweep and the reason it is in it: softmax needs a
// reduction across the WHOLE row, so it CANNOT be fused into a single output tile
// the way ReLU/GELU can. A tile only holds tileN of the row's N columns, so every
// mode — in-core included — needs either a cross-tile reduction or a second pass.
// That removes the in-core path's fusion advantage and is where the DTCU's
// GEMM-only design should look least bad relative to the others.
//
// Numerically-stable three phase form, per row i:
//   1. m = max_j  x[i][j]
//   2. s = sum_j  exp(x[i][j] - m)
//   3. D[i][j] = exp(x[i][j] - m) / s
//
// exp uses the shared approximation below rather than libm's expf, for the same
// reason gelu.h avoids tanhf: this header is included by BOTH the host CPU
// reference and the kernel, so identical arithmetic makes verification exact.
// Accuracy is adequate for a cycle-measuring harness, not for real softmax.
//
// WIRING STATUS: math is here; the cross-tile reduction (an extra pass, plus
// row-max/row-sum scratch buffers) is Phase B — see 260718_moti_RFC.md §8.

// exp(x) via 2^(x*log2(e)) using exact ldexp-style scaling for the integer part and
// a degree-3 minimax-ish polynomial on the [0,1) fraction. Uses only basic float
// ops + a float->int truncation so host and device agree.
static inline float epi_exp_approx(float x) {
  if (x < -87.0f) return 0.0f;        // underflows fp32
  if (x >  88.0f) return 3.4028235e38f; // saturate instead of inf
  const float log2e = 1.4426950409f;
  float t  = x * log2e;               // want 2^t
  int   n  = (int)(t + (t >= 0.0f ? 0.5f : -0.5f)); // round to nearest
  float f  = t - (float)n;            // f in [-0.5, 0.5]
  // 2^f on [-0.5,0.5] via polynomial in (f * ln2)
  float u  = f * 0.6931471805f;       // f*ln2
  float p  = 1.0f + u * (1.0f + u * (0.5f + u * (0.1666666667f + u * 0.0416666667f)));
  // scale by 2^n by direct exponent arithmetic (n is small: |t| <= ~127)
  float scale = 1.0f;
  if (n > 0) { for (int i = 0; i < n; ++i) scale *= 2.0f; }
  else       { for (int i = 0; i < -n; ++i) scale *= 0.5f; }
  return p * scale;
}

// Phase 1/2 helpers: callers accumulate across the row (possibly across tiles).
static inline float epi_softmax_max(float acc, float x) {
  return x > acc ? x : acc;
}
static inline float epi_softmax_addexp(float acc, float x, float row_max) {
  return acc + epi_exp_approx(x - row_max);
}
// Phase 3: finalize one element given the row's max and exp-sum.
static inline float epi_softmax_norm(float x, float row_max, float row_sum) {
  return epi_exp_approx(x - row_max) / row_sum;
}

#endif // _CGO27_EPI_SOFTMAX_H_
