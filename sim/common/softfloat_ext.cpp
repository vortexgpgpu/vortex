/*============================================================================

This C source file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3e, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016 The Regents of the University of
California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================*/

#include "softfloat_ext.h"
#include <../RISCV/specialize.h>
#include <assert.h>
#include <internals.h>
#include <softfloat.h>
#include <stdbool.h>
#include <cstring>
#include <util.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

uint_fast16_t f16_classify(float16_t a) {
  union ui16_f16 uA;
  uint_fast16_t uiA;

  uA.f = a;
  uiA = uA.ui;

  uint_fast16_t infOrNaN = expF16UI(uiA) == 0x1F;
  uint_fast16_t subnormalOrZero = expF16UI(uiA) == 0;
  bool sign = signF16UI(uiA);
  bool fracZero = fracF16UI(uiA) == 0;
  bool isNaN = isNaNF16UI(uiA);
  bool isSNaN = softfloat_isSigNaNF16UI(uiA);

  return (sign && infOrNaN && fracZero) << 0 |
         (sign && !infOrNaN && !subnormalOrZero) << 1 |
         (sign && subnormalOrZero && !fracZero) << 2 |
         (sign && subnormalOrZero && fracZero) << 3 |
         (!sign && infOrNaN && fracZero) << 7 |
         (!sign && !infOrNaN && !subnormalOrZero) << 6 |
         (!sign && subnormalOrZero && !fracZero) << 5 |
         (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 |
         (isNaN && !isSNaN) << 9;
}

uint_fast16_t f32_classify(float32_t a) {
  union ui32_f32 uA;
  uint_fast32_t uiA;

  uA.f = a;
  uiA = uA.ui;

  uint_fast16_t infOrNaN = expF32UI(uiA) == 0xFF;
  uint_fast16_t subnormalOrZero = expF32UI(uiA) == 0;
  bool sign = signF32UI(uiA);
  bool fracZero = fracF32UI(uiA) == 0;
  bool isNaN = isNaNF32UI(uiA);
  bool isSNaN = softfloat_isSigNaNF32UI(uiA);

  return (sign && infOrNaN && fracZero) << 0 |
         (sign && !infOrNaN && !subnormalOrZero) << 1 |
         (sign && subnormalOrZero && !fracZero) << 2 |
         (sign && subnormalOrZero && fracZero) << 3 |
         (!sign && infOrNaN && fracZero) << 7 |
         (!sign && !infOrNaN && !subnormalOrZero) << 6 |
         (!sign && subnormalOrZero && !fracZero) << 5 |
         (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 |
         (isNaN && !isSNaN) << 9;
}

uint_fast16_t f64_classify(float64_t a) {
  union ui64_f64 uA;
  uint_fast64_t uiA;

  uA.f = a;
  uiA = uA.ui;

  uint_fast16_t infOrNaN = expF64UI(uiA) == 0x7FF;
  uint_fast16_t subnormalOrZero = expF64UI(uiA) == 0;
  bool sign = signF64UI(uiA);
  bool fracZero = fracF64UI(uiA) == 0;
  bool isNaN = isNaNF64UI(uiA);
  bool isSNaN = softfloat_isSigNaNF64UI(uiA);

  return (sign && infOrNaN && fracZero) << 0 |
         (sign && !infOrNaN && !subnormalOrZero) << 1 |
         (sign && subnormalOrZero && !fracZero) << 2 |
         (sign && subnormalOrZero && fracZero) << 3 |
         (!sign && infOrNaN && fracZero) << 7 |
         (!sign && !infOrNaN && !subnormalOrZero) << 6 |
         (!sign && subnormalOrZero && !fracZero) << 5 |
         (!sign && subnormalOrZero && fracZero) << 4 | (isNaN && isSNaN) << 8 |
         (isNaN && !isSNaN) << 9;
}

static inline uint64_t extract64(uint64_t val, int pos, int len) {
  assert(pos >= 0 && len > 0 && len <= 64 - pos);
  return (val >> pos) & (~UINT64_C(0) >> (64 - len));
}

static inline uint64_t make_mask64(int pos, int len) {
  assert(pos >= 0 && len > 0 && pos < 64 && len <= 64);
  return (UINT64_MAX >> (64 - len)) << pos;
}

// user needs to truncate output to required length
static inline uint64_t rsqrte7(uint64_t val, int e, int s, bool sub) {
  uint64_t exp = extract64(val, s, e);
  uint64_t sig = extract64(val, 0, s);
  uint64_t sign = extract64(val, s + e, 1);
  const int p = 7;

  static const uint8_t table[] = {
      52,  51,  50,  48,  47,  46,  44,  43,  42,  41,  40,  39,  38,  36,  35,
      34,  33,  32,  31,  30,  30,  29,  28,  27,  26,  25,  24,  23,  23,  22,
      21,  20,  19,  19,  18,  17,  16,  16,  15,  14,  14,  13,  12,  12,  11,
      10,  10,  9,   9,   8,   7,   7,   6,   6,   5,   4,   4,   3,   3,   2,
      2,   1,   1,   0,   127, 125, 123, 121, 119, 118, 116, 114, 113, 111, 109,
      108, 106, 105, 103, 102, 100, 99,  97,  96,  95,  93,  92,  91,  90,  88,
      87,  86,  85,  84,  83,  82,  80,  79,  78,  77,  76,  75,  74,  73,  72,
      71,  70,  70,  69,  68,  67,  66,  65,  64,  63,  63,  62,  61,  60,  59,
      59,  58,  57,  56,  56,  55,  54,  53};

  if (sub) {
    while (extract64(sig, s - 1, 1) == 0)
      exp--, sig <<= 1;

    sig = (sig << 1) & make_mask64(0, s);
  }

  int idx = ((exp & 1) << (p - 1)) | (sig >> (s - p + 1));
  uint64_t out_sig = (uint64_t)(table[idx]) << (s - p);
  uint64_t out_exp = (3 * make_mask64(0, e - 1) + ~exp) / 2;

  return (sign << (s + e)) | (out_exp << s) | out_sig;
}

float16_t f16_rsqrte7(float16_t in) {
  union ui16_f16 uA;

  uA.f = in;
  unsigned int ret = f16_classify(in);
  bool sub = false;
  switch (ret) {
  case 0x001: // -inf
  case 0x002: // -normal
  case 0x004: // -subnormal
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF16UI;
    break;
  case 0x008: // -0
    uA.ui = 0xfc00;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7c00;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +num
    uA.ui = rsqrte7(uA.ui, 5, 10, sub);
    break;
  }

  return uA.f;
}

float32_t f32_rsqrte7(float32_t in) {
  union ui32_f32 uA;

  uA.f = in;
  unsigned int ret = f32_classify(in);
  bool sub = false;
  switch (ret) {
  case 0x001: // -inf
  case 0x002: // -normal
  case 0x004: // -subnormal
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF32UI;
    break;
  case 0x008: // -0
    uA.ui = 0xff800000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7f800000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +num
    uA.ui = rsqrte7(uA.ui, 8, 23, sub);
    break;
  }

  return uA.f;
}

float64_t f64_rsqrte7(float64_t in) {
  union ui64_f64 uA;

  uA.f = in;
  unsigned int ret = f64_classify(in);
  bool sub = false;
  switch (ret) {
  case 0x001: // -inf
  case 0x002: // -normal
  case 0x004: // -subnormal
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF64UI;
    break;
  case 0x008: // -0
    uA.ui = 0xfff0000000000000ul;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7ff0000000000000ul;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +num
    uA.ui = rsqrte7(uA.ui, 11, 52, sub);
    break;
  }

  return uA.f;
}

// user needs to truncate output to required length
static inline uint64_t recip7(uint64_t val, int e, int s, int rm, bool sub,
                              bool *round_abnormal) {
  uint64_t exp = extract64(val, s, e);
  uint64_t sig = extract64(val, 0, s);
  uint64_t sign = extract64(val, s + e, 1);
  const int p = 7;

  static const uint8_t table[] = {
      127, 125, 123, 121, 119, 117, 116, 114, 112, 110, 109, 107, 105, 104, 102,
      100, 99,  97,  96,  94,  93,  91,  90,  88,  87,  85,  84,  83,  81,  80,
      79,  77,  76,  75,  74,  72,  71,  70,  69,  68,  66,  65,  64,  63,  62,
      61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,
      46,  45,  44,  43,  42,  41,  40,  40,  39,  38,  37,  36,  35,  35,  34,
      33,  32,  31,  31,  30,  29,  28,  28,  27,  26,  25,  25,  24,  23,  23,
      22,  21,  21,  20,  19,  19,  18,  17,  17,  16,  15,  15,  14,  14,  13,
      12,  12,  11,  11,  10,  9,   9,   8,   8,   7,   7,   6,   5,   5,   4,
      4,   3,   3,   2,   2,   1,   1,   0};

  if (sub) {
    while (extract64(sig, s - 1, 1) == 0)
      exp--, sig <<= 1;

    sig = (sig << 1) & make_mask64(0, s);

    if (exp != 0 && exp != UINT64_MAX) {
      *round_abnormal = true;
      if (rm == 1 || (rm == 2 && !sign) || (rm == 3 && sign))
        return ((sign << (s + e)) | make_mask64(s, e)) - 1;
      else
        return (sign << (s + e)) | make_mask64(s, e);
    }
  }

  int idx = sig >> (s - p);
  uint64_t out_sig = (uint64_t)(table[idx]) << (s - p);
  uint64_t out_exp = 2 * make_mask64(0, e - 1) + ~exp;
  if (out_exp == 0 || out_exp == UINT64_MAX) {
    out_sig = (out_sig >> 1) | make_mask64(s - 1, 1);
    if (out_exp == UINT64_MAX) {
      out_sig >>= 1;
      out_exp = 0;
    }
  }

  return (sign << (s + e)) | (out_exp << s) | out_sig;
}

float16_t f16_recip7(float16_t in) {
  union ui16_f16 uA;

  uA.f = in;
  unsigned int ret = f16_classify(in);
  bool sub = false;
  bool round_abnormal = false;
  switch (ret) {
  case 0x001: // -inf
    uA.ui = 0x8000;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x008: // -0
    uA.ui = 0xfc00;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7c00;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF16UI;
    break;
  case 0x004: // -subnormal
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +- normal
    uA.ui = recip7(uA.ui, 5, 10, softfloat_roundingMode, sub, &round_abnormal);
    if (round_abnormal) {
      softfloat_exceptionFlags |= softfloat_flag_inexact | softfloat_flag_overflow;
    }
    break;
  }

  return uA.f;
}

float32_t f32_recip7(float32_t in) {
  union ui32_f32 uA;

  uA.f = in;
  unsigned int ret = f32_classify(in);
  bool sub = false;
  bool round_abnormal = false;
  switch (ret) {
  case 0x001: // -inf
    uA.ui = 0x80000000;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x008: // -0
    uA.ui = 0xff800000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7f800000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF32UI;
    break;
  case 0x004: // -subnormal
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +- normal
    uA.ui = recip7(uA.ui, 8, 23, softfloat_roundingMode, sub, &round_abnormal);
    if (round_abnormal) {
      softfloat_exceptionFlags |= softfloat_flag_inexact | softfloat_flag_overflow;
    }
    break;
  }

  return uA.f;
}

float64_t f64_recip7(float64_t in) {
  union ui64_f64 uA;

  uA.f = in;
  unsigned int ret = f64_classify(in);
  bool sub = false;
  bool round_abnormal = false;
  switch (ret) {
  case 0x001: // -inf
    uA.ui = 0x8000000000000000;
    break;
  case 0x080: //+inf
    uA.ui = 0x0;
    break;
  case 0x008: // -0
    uA.ui = 0xfff0000000000000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x010: // +0
    uA.ui = 0x7ff0000000000000;
    softfloat_exceptionFlags |= softfloat_flag_infinite;
    break;
  case 0x100: // sNaN
    softfloat_exceptionFlags |= softfloat_flag_invalid;
    [[fallthrough]];
  case 0x200: // qNaN
    uA.ui = defaultNaNF64UI;
    break;
  case 0x004: // -subnormal
  case 0x020: //+ sub
    sub = true;
    [[fallthrough]];
  default: // +- normal
    uA.ui = recip7(uA.ui, 11, 52, softfloat_roundingMode, sub, &round_abnormal);
    if (round_abnormal) {
      softfloat_exceptionFlags |= softfloat_flag_inexact | softfloat_flag_overflow;
    }
    break;
  }

  return uA.f;
}

// Convert a float to a custom floating-point format
uint32_t cvt_f32_to_custom(float value, uint32_t exp_bits, uint32_t sig_bits,
                           uint32_t frm, uint32_t *fflags) {
  // RISC-V rounding modes
  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };

  // RISC-V exception flags
  const uint32_t FLAG_NX = 1u << 0; // inexact
  const uint32_t FLAG_UF = 1u << 1; // underflow
  const uint32_t FLAG_OF = 1u << 2; // overflow
  //const uint32_t FLAG_DZ = 1u << 3; // div-by-zero
  const uint32_t FLAG_NV = 1u << 4; // invalid

  uint32_t flags = 0;
  if (fflags) {
    *fflags = 0;
  }

  if (exp_bits == 0 || (1u + exp_bits + sig_bits) > 32u) {
    if (fflags) {
      *fflags |= FLAG_NV;
    }
    return 0;
  }

  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = bits >> 31;
  const uint32_t exp_ieee = (bits >> 23) & 0xFFu;
  const uint32_t sig_ieee = bits & 0x7FFFFFu;

  const uint32_t exp_max = (1u<<exp_bits) - 1u;
  const uint32_t exp_max_finite = exp_bits ? exp_max - 1u : 0u;
  const uint32_t mant_mask = sig_bits ? ((1u<<sig_bits) - 1u) : 0u;
  const uint32_t sign_shift = exp_bits + sig_bits;
  const int32_t  bias_out = (int32_t)((1u<<(exp_bits-1u)) - 1u);
  const int32_t  emax = bias_out;
  const int32_t  emin = 1 - bias_out;

  auto pack_custom = [&](uint32_t exp, uint32_t mant){
    return (sign << sign_shift) | (exp << sig_bits) | (mant & mant_mask);
  };

  // NaN / Inf / Zero
  if (exp_ieee == 0xFFu) {
    if (sig_ieee) {
      if ((sig_ieee & 0x400000u) == 0) {
        flags |= FLAG_NV; // sNaN
      }
      if (fflags) {
        *fflags |= flags;
      }
      return pack_custom(exp_max, sig_bits ? (1u<<(sig_bits-1u)) : 0u); // qNaN
    }
    if (fflags) {
      *fflags |= flags;
    }
    return pack_custom(exp_max, 0); // Inf
  }
  if (exp_ieee == 0 && sig_ieee == 0) {
    if (fflags) {
      *fflags |= flags;
    }
    return pack_custom(0, 0); // ±0
  }

  // Normalize source to 24-bit (1.hidden + 23)
  uint32_t significand;
  int32_t  exponent;
  if (exp_ieee) {
    significand = (1u<<23) | sig_ieee;
    exponent = (int32_t)exp_ieee - 127;
  } else {
    int lz = __builtin_clz(sig_ieee); // sig_ieee != 0 here
    int sh = lz - 8;                  // bring leading 1 to bit 23
    significand = sig_ieee << sh;
    exponent = -126 - sh;
  }

  // Map precision: create (sig_bits+1) bits with hidden 1 at bit sig_bits
  int32_t shift_amount = 23 - (int32_t)sig_bits;
  uint64_t main = significand; // (up to) 24 bits
  // IMPORTANT: do NOT add shift_amount to exponent (bug fix)
  if (shift_amount > 0) {
    uint32_t sh = (uint32_t)shift_amount;
    uint64_t remainder = ((uint64_t)main) & ((((uint64_t)1)<<sh) - 1u);
    uint64_t kept      = ((uint64_t)main) >> sh;

    if (remainder) {
      flags |= FLAG_NX;
    }

    uint32_t lsb    = (uint32_t)(kept & 1u);
    uint32_t guard  = (sh>=1) ? (uint32_t)((main >> (sh-1u)) & 1u) : 0u;
    uint32_t roundb = (sh>=2) ? (uint32_t)((main >> (sh-2u)) & 1u) : 0u;
    uint32_t sticky = (sh>=2) ? ((remainder & ((((uint64_t)1)<<(sh-1u)) - 1u)) != 0u) : 0u;

    int inc = 0;
    switch (frm) {
      default:
      case RNE: inc = guard && ((roundb | sticky) || lsb); break;
      case RTZ: inc = 0; break;
      case RDN: inc = sign && (remainder != 0); break;
      case RUP: inc = !sign && (remainder != 0); break;
      case RMM: inc = guard ? 1 : 0; break;
    }
    if (inc) {
      kept += 1u;
      if (kept == (1ull<<(sig_bits+1u))) { kept >>= 1u; exponent += 1; }
    }
    main = kept;
  } else if (shift_amount < 0) {
    main <<= (uint32_t)(-shift_amount); // exact; no NX
    // exponent unchanged
  }

  // Overflow?
  if (exponent > emax) {
    flags |= (FLAG_OF | FLAG_NX);
    int to_inf =
      (frm == RNE || frm == RMM) ? 1 :
      (frm == RTZ) ? 0 :
      (frm == RUP) ? (sign == 0) :
      (frm == RDN) ? (sign == 1) : 1;
    if (fflags) {
      *fflags |= flags;
    }
    return to_inf ? pack_custom(exp_max, 0) : pack_custom(exp_max_finite, mant_mask);
  }

  // Normal?
  if (exponent >= emin) {
    uint32_t expf = (uint32_t)(exponent + bias_out);
    uint32_t mant = sig_bits ? ((uint32_t)main & mant_mask) : 0u;
    if (fflags) {
      *fflags |= flags;
    }
    return pack_custom(expf, mant);
  }

  // Subnormal (E < emin): round with GRS; may promote to min-normal
  uint32_t t = (uint32_t)(emin - exponent); // >= 1
  uint64_t keep, disc;
  if (t < 64u) {
    keep = main >> t;
    disc = main & ((((uint64_t)1) << t) - 1u);
  } else {
    keep = 0;
    disc = main ? 1u : 0u;
  }

  uint32_t lsb    = sig_bits ? (uint32_t)(keep & 1u) : 0u;
  uint32_t guard  = (t>0) ? (uint32_t)((main >> (t-1u)) & 1u) : 0u;
  uint32_t roundb = (t>1) ? (uint32_t)((main >> (t-2u)) & 1u) : 0u;
  uint32_t sticky = (t>1) ? ((disc & ((((uint64_t)1)<<(t-1u)) - 1u)) != 0u) : 0u;

  int inc = 0;
  switch (frm) {
    default:
    case RNE: inc = guard && ((roundb | sticky) || lsb); break;
    case RTZ: inc = 0; break;
    case RDN: inc = sign && (disc != 0); break;
    case RUP: inc = !sign && (disc != 0); break;
    case RMM: inc = guard ? 1 : 0; break;
  }

  uint64_t mant_sub = keep + (inc ? 1u : 0u);
  if (disc != 0) {
    flags |= FLAG_NX;
  }

  // Promotion to min-normal?
  if (sig_bits && mant_sub == (1ull<<sig_bits)) {
    if (fflags) {
      *fflags |= flags; // NX already set if inexact
    }
    return pack_custom(1u, 0u);   // exp=1, mant=0
  }

  // Remain subnormal. UF only when tiny AFTER rounding and inexact.
  if (disc != 0 && mant_sub != 0) {
    flags |= FLAG_UF;
  }

  if (fflags) {
    *fflags |= flags;
  }
  return pack_custom(0u, (uint32_t)mant_sub);
}

// Convert a custom floating-point format to float
float cvt_custom_to_f32(uint32_t value, uint32_t exp_bits, uint32_t sig_bits,
                        uint32_t frm, uint32_t *fflags) {
  // RISC-V rounding modes
  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };

  // RISC-V exception flags
  const uint32_t FLAG_NX = 1u << 0; // inexact
  const uint32_t FLAG_UF = 1u << 1; // underflow
  const uint32_t FLAG_OF = 1u << 2; // overflow
  //const uint32_t FLAG_DZ = 1u << 3; // div-by-zero
  const uint32_t FLAG_NV = 1u << 4; // invalid

  uint32_t flags = 0;
  if (fflags) {
    *fflags = 0;
  }

  // Validate format parameters
  if (exp_bits == 0 || (1 + exp_bits + sig_bits) > 32) {
    if (fflags) {
      *fflags |= FLAG_NV;
    }
    return 0.0f;
  }

  // Extract components from custom format
  const uint32_t sign = value >> (exp_bits + sig_bits);
  const uint32_t exp_field = (value >> sig_bits) & ((1u << exp_bits) - 1);
  const uint32_t sig_field = value & ((1u << sig_bits) - 1);

  // Precompute format-specific constants
  const uint32_t exp_max = (1u << exp_bits) - 1;
  const int32_t bias_custom = (1 << (exp_bits - 1)) - 1;
  const int32_t bias_ieee = 127;

  // Helper to pack IEEE float
  auto pack_float = [](uint32_t bits) -> float {
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
  };

  // Handle special cases (NaN, Infinity, Zero)
  if (exp_field == exp_max) {
    if (sig_field != 0) {
      // sNaN if top mantissa bit is 0 (when present)
      if (sig_bits > 0 && ((sig_field >> (sig_bits - 1)) & 1u) == 0) {
        flags |= FLAG_NV;
      }
      if (fflags) {
        *fflags |= flags;
      }
      return pack_float((sign << 31) | 0x7FC00000u); // quiet NaN
    } else {
      if (fflags) {
        *fflags |= flags;
      }
      return pack_float((sign << 31) | 0x7F800000u); // infinity
    }
  }

  if (exp_field == 0 && sig_field == 0) {
    // Zero
    if (fflags) {
      *fflags |= flags;
    }
    return pack_float(sign << 31);
  }

  // Calculate unbiased exponent and 1+sig_bits significand (with hidden 1)
  int32_t exponent;
  uint64_t significand;

  if (exp_field == 0) {
    // Subnormal in custom: normalize
    exponent = 1 - bias_custom;
    significand = sig_field;
    if (sig_field != 0) {
      int shift = __builtin_clz(sig_field) - (32 - sig_bits);
      significand <<= shift;
      exponent -= shift;
    }
  } else {
    // Normal in custom
    exponent = (int32_t)exp_field - bias_custom;
    significand = (1ull << sig_bits) | sig_field; // include hidden 1
  }

  // Convert to a 24-bit main (hidden1 + 23) aligned to float32
  int32_t shift_amount = (int32_t)sig_bits - 23;
  int32_t ieee_exponent = exponent + bias_ieee; // unbiased->biased (NO precision-offset)
  uint64_t main24 = significand;                // will hold hidden1+23 bits
  uint64_t dropped = 0;

  if (shift_amount > 0) {
    // Right shift (need rounding) to 24 bits
    uint32_t sh = (uint32_t)shift_amount;
    dropped = main24 & ((((uint64_t)1) << sh) - 1u);
    main24 >>= sh;

    // GRS rounding
    uint32_t lsb    = (uint32_t)(main24 & 1u);
    uint32_t guard  = (sh >= 1) ? (uint32_t)((significand >> (sh - 1u)) & 1u) : 0u;
    uint32_t roundb = (sh >= 2) ? (uint32_t)((significand >> (sh - 2u)) & 1u) : 0u;
    uint32_t sticky = (sh >= 2) ? ((dropped & ((((uint64_t)1) << (sh - 1u)) - 1u)) != 0u) : 0u;

    int inc = 0;
    switch (frm) {
      default:
      case RNE: inc = guard && ((roundb | sticky) || lsb); break;
      case RTZ: inc = 0; break;
      case RDN: inc = sign && (dropped != 0); break;
      case RUP: inc = !sign && (dropped != 0); break;
      case RMM: inc = guard ? 1 : 0; break;
    }

    if (inc) {
      main24 += 1u;
      if (main24 == (1ull << 24)) { // carry into next bit
        main24 >>= 1u;
        ieee_exponent += 1;         // FIX: only bump exponent on carry-out
      }
    }
    if (dropped) {
      flags |= FLAG_NX;
    }
  } else if (shift_amount < 0) {
    // Left shift (exact)
    main24 <<= (uint32_t)(-shift_amount);
    // FIX: do NOT adjust exponent here
  }
  // else: already 24 bits

  // Check overflow in float32
  if (ieee_exponent >= 0xFF) {
    flags |= FLAG_OF | FLAG_NX;
    int produce_inf =
      (frm == RNE || frm == RMM) ? 1 :
      (frm == RTZ) ? 0 :
      (frm == RUP) ? (sign == 0) :
      (frm == RDN) ? (sign == 1) : 1;
    uint32_t out = (sign << 31) | (0xFFu << 23) | (produce_inf ? 0u : 0x7FFFFFu);
    if (fflags) {
      *fflags |= flags;
    }
    return pack_float(out);
  }

  // Normal?
  if (ieee_exponent > 0) {
    // Only now drop the hidden 1 for packing
    uint32_t mant = (uint32_t)(main24 & 0x7FFFFFu); // FIX: mask here, not earlier
    uint32_t out  = (sign << 31) | ((uint32_t)ieee_exponent << 23) | mant;
    if (fflags) {
      *fflags |= flags;
    }
    return pack_float(out);
  }

  // Subnormal (ieee_exponent <= 0): round from full 24-bit main
  {
    uint32_t t = (uint32_t)(1 - ieee_exponent); // shift to exp=0
    uint64_t keep, disc;
    if (t < 64u) {
      keep = main24 >> t;
      disc = main24 & ((((uint64_t)1) << t) - 1u);
    } else {
      keep = 0;
      disc = main24 ? 1u : 0u;
    }

    // GRS on subnormal
    uint32_t lsb    = (uint32_t)(keep & 1u);
    uint32_t guard  = (t > 0) ? (uint32_t)((main24 >> (t - 1u)) & 1u) : 0u;
    uint32_t roundb = (t > 1) ? (uint32_t)((main24 >> (t - 2u)) & 1u) : 0u;
    uint32_t sticky = (t > 1) ? ((disc & ((((uint64_t)1) << (t - 1u)) - 1u)) != 0u) : 0u;

    int inc = 0;
    switch (frm) {
      default:
      case RNE: inc = guard && ((roundb | sticky) || lsb); break;
      case RTZ: inc = 0; break;
      case RDN: inc = sign && (disc != 0); break;
      case RUP: inc = !sign && (disc != 0); break;
      case RMM: inc = guard ? 1 : 0; break;
    }

    uint64_t mant_sub = keep + (inc ? 1u : 0u);

    // Promote to min-normal?
    if (mant_sub == (1ull << 23)) {
      uint32_t out = (sign << 31) | (1u << 23); // exp=1, mant=0
      if (dropped || disc) {
        flags |= FLAG_NX;
      }
      if (fflags) {
        *fflags |= flags;
      }
      return pack_float(out);
    }

    // Stay subnormal
    uint32_t mant = (uint32_t)mant_sub; // already no hidden 1 when exp=0
    uint32_t out  = (sign << 31) | mant;
    if (dropped || disc) {
      flags |= FLAG_NX;
      if (mant != 0) {
        flags |= FLAG_UF; // tiny AFTER rounding and inexact
      }
    }
    if (fflags) {
      *fflags |= flags;
    }
    return pack_float(out);
  }
}

#ifdef __cplusplus
}
#endif
