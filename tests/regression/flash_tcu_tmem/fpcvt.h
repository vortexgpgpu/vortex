#ifndef _FPCVT_H_
#define _FPCVT_H_

// fp32 -> narrow-format packing for TCU input operands.
//
// TCU's input formats are packed, right-aligned bit layouts (e.g. tf32 is 19
// bits: sign at bit 18, 8 exponent bits, 10 mantissa bits). FlashAttention
// produces P on-device from fp32 softmax, and there is no device-side float
// conversion in Vortex kernel surface so the pack has to exist here.
//
// Uses mask select for data-dependent decision, never branch. P differs per
// lane so a branch on it needs a vx_split, but the Vortex backend stops
// inserting vx_split for functions above -vortex-divergence-max-bbs blocks.
//
// Host references do not share one rounding semantics, so two paths:
//   f32_to_custom -- mirrors cvt_f32_to_custom() in sw/common/softfloat_ext.cpp
//                    (rv_ftotf32_s). Used for tf32.
//   f32_to_ieee   -- mirrors Berkeley softfloat (rv_ftoh_s / rv_ftob_s). Used
//                    for fp16/bf16.

#include <stdint.h>

namespace fpcvt {

static inline uint32_t sel(uint32_t c, uint32_t a, uint32_t b) {
  return (a & (0u - c)) | (b & (c - 1u));
}

// RNE pack into <1 + EXP_BITS + SIG_BITS> bits
// Layout: (sign << (EXP_BITS+SIG_BITS)) | (exp << SIG_BITS) | mant
template <uint32_t EXP_BITS, uint32_t SIG_BITS>
static inline uint32_t f32_to_custom(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = bits >> 31;
  const uint32_t e    = (bits >> 23) & 0xFFu;
  const uint32_t sig  = bits & 0x7FFFFFu;
  const uint32_t exp_max   = (1u << EXP_BITS) - 1u;
  const uint32_t mant_mask = (1u << SIG_BITS) - 1u;
  const uint32_t sbit      = sign << (EXP_BITS + SIG_BITS);
  const int32_t  bias_out  = (int32_t)((1u << (EXP_BITS - 1u)) - 1u);
  const int32_t  emax = bias_out, emin = 1 - bias_out;

  const uint32_t is_nan_inf = (e == 0xFFu);
  const uint32_t is_zero    = (e == 0u) & (sig == 0u);
  const uint32_t is_denorm  = (e == 0u) & (sig != 0u);

  // Normalize to 24 bits (hidden 1 at bit 23). Both candidates computed and one
  // selected; sh is in [1, 23] for denormals and every shift stays < 32.
  const uint32_t sh = (uint32_t)__builtin_clz(sig | 1u) - 8u;
  const uint32_t significand = sel(is_denorm, sig << (sh & 31u), (1u << 23) | sig);
  int32_t exponent = (int32_t)sel(is_denorm, (uint32_t)(-126 - (int32_t)sh), e - 127u);

  // Round to SIG_BITS+1 bits. Carry out of the top bumps the exponent.
  const uint32_t s1 = 23u - SIG_BITS;
  uint32_t kept = significand >> s1;
  const uint32_t g1 = (significand >> (s1 - 1u)) & 1u;
  const uint32_t l1 = (significand & ((1u << (s1 - 1u)) - 1u)) != 0u;
  kept += g1 & (l1 | (kept & 1u));
  const uint32_t carry = (kept == (1u << (SIG_BITS + 1u)));
  kept >>= carry;
  exponent += (int32_t)carry;

  const uint32_t ovf       = (exponent > emax);
  const uint32_t is_normal = (exponent >= emin);
  const uint32_t norm_out  = ((uint32_t)(exponent + bias_out) << SIG_BITS) | (kept & mant_mask);

  // Subnormal: shift down by t = emin - exponent, in [1, 23], and round.
  const uint32_t t  = sel(is_normal, 1u, (uint32_t)(emin - exponent)) & 31u;
  const uint32_t kp = kept >> t;
  const uint32_t g2 = (kept >> ((t - 1u) & 31u)) & 1u;
  const uint32_t l2 = (kept & ((1u << ((t - 1u) & 31u)) - 1u)) != 0u;
  const uint32_t sub_out = kp + (g2 & (l2 | (kp & 1u)));

  const uint32_t inf_out = exp_max << SIG_BITS;
  const uint32_t nan_out = inf_out | sel(sig != 0u, 1u << (SIG_BITS - 1u), 0u);
  uint32_t out = sel(is_normal, sel(ovf, inf_out, norm_out), sub_out);
  out = sel(is_zero, 0u, out);
  out = sel(is_nan_inf, nan_out, out);
  return sbit | out;
}

// RNE pack into IEEE-style narrow format, Berkeley softfloat semantics
template <uint32_t EXP_BITS, uint32_t SIG_BITS>
static inline uint32_t f32_to_ieee(float value) {
  uint32_t bits;
  __builtin_memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = bits >> 31;
  const uint32_t e    = (bits >> 23) & 0xFFu;
  const uint32_t sig  = bits & 0x7FFFFFu;
  const uint32_t exp_max   = (1u << EXP_BITS) - 1u;
  const uint32_t mant_mask = (1u << SIG_BITS) - 1u;
  const uint32_t sbit      = sign << (EXP_BITS + SIG_BITS);
  const int32_t  bias_out  = (int32_t)((1u << (EXP_BITS - 1u)) - 1u);
  const int32_t  emax = bias_out, emin = 1 - bias_out;

  const uint32_t is_nan_inf = (e == 0xFFu);
  const uint32_t is_zero    = (e == 0u) & (sig == 0u);
  const uint32_t is_denorm  = (e == 0u) & (sig != 0u);

  const uint32_t sh = (uint32_t)__builtin_clz(sig | 1u) - 8u;
  const uint32_t significand = sel(is_denorm, sig << (sh & 31u), (1u << 23) | sig);
  const int32_t exponent = (int32_t)sel(is_denorm, (uint32_t)(-126 - (int32_t)sh), e - 127u);

  // One combined RNE shift. Subnormals shift further by emin - exponent, any
  // shift >= 25 rounds 24-bit input to 0.
  const uint32_t s0 = 23u - SIG_BITS;
  const uint32_t is_sub = (exponent < emin);
  uint32_t shtot = sel(is_sub, s0 + (uint32_t)(emin - exponent), s0);
  shtot = sel(shtot > 25u, 25u, shtot);
  uint32_t m = significand >> shtot;
  const uint32_t g  = (significand >> (shtot - 1u)) & 1u;
  const uint32_t st = (significand & ((1u << (shtot - 1u)) - 1u)) != 0u;
  m += g & (st | (m & 1u));

  const uint32_t sub_out  = sbit | m;
  const uint32_t carry    = (m == (1u << (SIG_BITS + 1u)));
  const uint32_t mn       = m >> carry;
  const int32_t  en       = exponent + (int32_t)carry;
  const uint32_t ovf      = (en > emax);
  const uint32_t norm_out = sbit | ((uint32_t)(en + bias_out) << SIG_BITS) | (mn & mant_mask);
  const uint32_t inf_out  = sbit | (exp_max << SIG_BITS);
  const uint32_t nan_out  = sel(sig != 0u, (exp_max << SIG_BITS) | (1u << (SIG_BITS - 1u)), inf_out);

  uint32_t out = sel(is_sub, sub_out, sel(ovf, inf_out, norm_out));
  out = sel(is_zero, sbit, out);
  out = sel(is_nan_inf, nan_out, out);
  return out;
}

// TCU input formats
static inline uint32_t f32_to_tf32(float v) { return f32_to_custom<8, 10>(v); }
static inline uint32_t f32_to_fp16(float v) { return f32_to_ieee<5, 10>(v); }
static inline uint32_t f32_to_bf16(float v) { return f32_to_ieee<8, 7>(v); }

} // namespace fpcvt

#endif
