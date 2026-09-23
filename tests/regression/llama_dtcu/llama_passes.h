#ifndef _LLAMA_DTCU_PASSES_H_
#define _LLAMA_DTCU_PASSES_H_

// Host/device ABI for the non-GEMM passes of the llama forward (RMSNorm, RoPE + head
// packing, causal softmax, head gather, SiLU gate) and of the batched decode step (RoPE +
// KV-cache append, per-(sequence, head) attention). Every pass is a plain SIMT kernel
// that every mode runs identically; only the GEMMs change with the mode.
//
// fp16 conversion is done here in software on both sides (the SIMT core has no Zfh), with
// one implementation shared by the device kernels and the host reference so that the
// reference reproduces the device bit for bit wherever the arithmetic is the same.

#include <stdint.h>

typedef struct {
  uint32_t T;         // rows (tokens in this batch)
  uint32_t dim;       // model dim (288)
  uint32_t hidden;    // FFN hidden dim (768)
  uint32_t n_heads;   // attention heads (6)
  uint32_t head;      // head size (48)
  uint32_t head_pad;  // padded head size (64) used by the attention GEMMs
  uint32_t pos0;      // position of row 0 (prefill: 0)
  uint32_t kv_stride; // decode: rows per sequence in the KV cache (positions), 0 otherwise
  float    eps;       // RMSNorm epsilon (1e-5)
  float    scale;     // attention score scale, 1/sqrt(head)
  uint64_t src;       // input
  uint64_t dst;       // output
  uint64_t aux0;      // pass-specific
  uint64_t aux1;
  uint64_t aux2;
} llama_pass_arg_t;

// IEEE fp32 -> fp16, round to nearest even. Same result as softfloat f32_to_f16 (RNE) for
// every finite input and for infinities; NaN payloads are not preserved.
static inline uint16_t llama_f2h(float f) {
  union { float f; uint32_t u; } cvt; cvt.f = f;
  const uint32_t x = cvt.u;
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t e32  = (x >> 23) & 0xffu;
  uint32_t mant = x & 0x7fffffu;
  if (e32 == 0xffu)                       // inf / nan
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  const int32_t e16 = (int32_t)e32 - 127 + 15;
  if (e16 >= 0x1f)                        // overflow -> inf
    return (uint16_t)(sign | 0x7c00u);
  if (e16 <= 0) {                         // subnormal or zero
    if (e16 < -10) return (uint16_t)sign; // rounds to zero
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - e16);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) ++half;
    return (uint16_t)(sign | half);
  }
  uint32_t half = sign | ((uint32_t)e16 << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;   // carry into exp is correct
  return (uint16_t)half;
}

// IEEE fp16 -> fp32 (exact).
static inline float llama_h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t out;
  if (e == 0) {
    if (m == 0) {
      out = s;
    } else {
      e = 127u - 15u + 1u;
      while ((m & 0x400u) == 0) { m <<= 1; e--; }
      m &= 0x3ffu;
      out = s | (e << 23) | (m << 13);
    }
  } else if (e == 0x1fu) {
    out = s | 0x7f800000u | (m << 13);
  } else {
    out = s | ((e + (127u - 15u)) << 23) | (m << 13);
  }
  union { uint32_t u; float f; } cvt; cvt.u = out;
  return cvt.f;
}

// exp(x) without libm, identical arithmetic on host and device: 2^n * e^u with n the nearest
// integer of x*log2(e) and u = (x*log2(e) - n)*ln2 in [-0.35, 0.35], degree-5 Taylor
// (relative error ~1e-7, far below fp16 output precision). Saturates outside fp32 range.
static inline float llama_fast_exp(float x) {
  if (x < -87.0f) return 0.0f;
  if (x >  88.0f) return 3.4028235e38f;
  const float t = x * 1.4426950409f;
  const int   n = (int)(t + (t >= 0.0f ? 0.5f : -0.5f));
  const float u = (t - (float)n) * 0.6931471805f;
  const float p = 1.0f + u * (1.0f + u * (0.5f + u * (0.1666666667f + u * (0.0416666667f + u * 0.0083333333f))));
  union { uint32_t u; float f; } s; s.u = (uint32_t)(n + 127) << 23;
  return p * s.f;
}

#endif // _LLAMA_DTCU_PASSES_H_
