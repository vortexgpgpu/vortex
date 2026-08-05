#ifndef _CGO27_EPI_GELU_H_
#define _CGO27_EPI_GELU_H_

// App 3 epilogue: GELU (tanh formulation).
//
//   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3) ))
//
// tanh is computed with a small rational approximation rather than libm's tanhf.
// Reason: this header is shared by the host CPU reference and the kernel, and the
// harness verifies device output against that reference with a tight ULP bound.
// libm on the host and the Vortex libm need not agree bit-for-bit, which would
// produce spurious mismatches. Using the same basic-arithmetic sequence on both
// sides makes verification exact. The harness measures CYCLES, not numerical
// fidelity, so an approximation with ~1e-3 error is fine here — but do NOT reuse
// this header anywhere that needs true GELU accuracy.
//
// Coordinate-independent, so in-core modes fuse it on the accumulator fragment;
// DTCU modes run it as a separate SIMT pass.

// tanh via the (27 + z^2)/(27 + 9 z^2) Pade-style form, saturated outside |z|>3
// where fp32 tanh is already within ~1e-4 of +/-1.
static inline __attribute__((always_inline)) float epi_tanh_approx(float z) {
  if (z >  3.0f) return  1.0f;
  if (z < -3.0f) return -1.0f;
  float z2 = z * z;
  return z * (27.0f + z2) / (27.0f + 9.0f * z2);
}

static inline __attribute__((always_inline)) float epi_gelu(float x) {
  const float k0 = 0.7978845608f;   // sqrt(2/pi)
  const float k1 = 0.044715f;
  float inner = k0 * (x + k1 * x * x * x);
  return 0.5f * x * (1.0f + epi_tanh_approx(inner));
}

#endif // _CGO27_EPI_GELU_H_
