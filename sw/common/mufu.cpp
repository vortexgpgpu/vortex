// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mufu.h"
#include "mufu_tables.h"

namespace vortex::mufu {

namespace {

constexpr uint32_t FRAC    = 30;
constexpr uint32_t XL_BITS = 18;

constexpr uint32_t POS_INF = 0x7f800000;
constexpr uint32_t NEG_INF = 0xff800000;
constexpr uint32_t ONE     = 0x3f800000;

// Output of the range-reduction stage; mirrors the RTL stage-1 register.
struct reduced_t {
  bool     special;  // result is `value`, bypassing interpolation
  uint32_t value;
  uint32_t addr;     // coefficient ROM address
  uint32_t xl;       // XL_BITS-wide position inside the interval
  int32_t  exp;      // result scale: value = P * 2^-FRAC * 2^exp
  bool     neg;      // result sign
  bool     zsign;    // sign of an exactly-zero result
  bool     is_lg2;   // add `k` as the integer part before normalizing
  int32_t  k;
};

reduced_t reduce(uint32_t fn, uint32_t x) {
  reduced_t r{};
  uint32_t s = x >> 31;
  uint32_t e = (x >> 23) & 0xff;
  uint32_t m = x & 0x7fffff;
  bool is_zero = (e == 0);
  bool is_inf  = (e == 0xff) && (m == 0);
  bool is_nan  = (e == 0xff) && (m != 0);
  uint32_t sig = m | (1u << 23);

  auto special = [&](uint32_t v) { r.special = true; r.value = v; return r; };

  if (is_nan)
    return special(QNAN_F32);

  switch (fn) {
  case FN_RCP:
    if (is_inf)  return special(s << 31);
    if (is_zero) return special(s ? NEG_INF : POS_INF);
    r.addr = TBL_RCP_BASE + (m >> 16);
    r.xl   = (m & 0xffff) << 2;
    r.exp  = 127 - int32_t(e);
    r.neg  = s;
    return r;

  case FN_RSQRT: {
    if (is_zero) return special(s ? NEG_INF : POS_INF);
    if (s)       return special(QNAN_F32);
    if (is_inf)  return special(0);
    int32_t k = int32_t(e) - 127;
    r.addr = TBL_RSQRT_BASE + ((uint32_t(k) & 1) << 6) + (m >> 17);
    r.xl   = (m & 0x1ffff) << 1;
    r.exp  = -(k >> 1);
    return r;
  }

  case FN_EX2: {
    if (is_inf)  return special(s ? 0 : POS_INF);
    if (is_zero) return special(ONE);
    if (e >= 134) return special(s ? 0 : POS_INF);
    int32_t sh = int32_t(e) - 120;
    int64_t mag = (sh >= 0) ? (int64_t(sig) << sh) : ((sh > -24) ? int64_t(sig >> -sh) : 0);
    int64_t X = s ? -mag : mag;
    int64_t f = X & ((int64_t(1) << FRAC) - 1);
    r.addr = TBL_EX2_BASE + uint32_t(f >> 24);
    r.xl   = uint32_t(f >> 6) & ((1u << XL_BITS) - 1);
    r.exp  = int32_t(X >> FRAC);
    return r;
  }

  case FN_LG2:
    if (is_zero) return special(NEG_INF);
    if (s)       return special(QNAN_F32);
    if (is_inf)  return special(POS_INF);
    r.addr   = TBL_LG2_BASE + (m >> 17);
    r.xl     = (m & 0x1ffff) << 1;
    r.is_lg2 = true;
    r.k      = int32_t(e) - 127;
    return r;

  case FN_SINPI:
  case FN_COSPI: {
    bool is_sin = (fn == FN_SINPI);
    if (is_inf)  return special(QNAN_F32);
    if (is_zero) return special(is_sin ? (s << 31) : ONE);
    // u = 2|x| in quarter turns, 30 fraction bits, modulo 4
    int32_t sh = int32_t(e) - 119;
    uint32_t u;
    if (sh >= 32)
      u = 0;
    else if (sh >= 0)
      u = uint32_t(uint64_t(sig) << sh);
    else
      u = (sh > -24) ? (sig >> -sh) : 0;
    if (!is_sin)
      u += (1u << FRAC);
    uint32_t q = u >> FRAC;
    uint32_t t = u & ((1u << FRAC) - 1);
    r.neg   = ((q >> 1) & 1) ^ (is_sin ? s : 0);
    r.zsign = is_sin && s;
    // Odd quadrants need cos(pi/2 t) = sin(pi/2 (1 - t)); ~t stands in for
    // 1 - t, and t == 0 (exactly 1) is returned directly to stay exact.
    if (q & 1) {
      if (t == 0)
        return special(r.neg ? (ONE | 0x80000000) : ONE);
      t = ~t & ((1u << FRAC) - 1);
    }
    r.addr  = TBL_SIN_BASE + (t >> 24);
    r.xl    = (t >> 6) & ((1u << XL_BITS) - 1);
    return r;
  }

  case FN_TANH:
    if (is_inf)  return special(s ? (ONE | 0x80000000) : ONE);
    if (is_zero) return special(s << 31);
    // |x| >= 8 rounds to +-1; below 2^-5, tanh x = x within 2^-11.
    if (e >= 130) return special(s ? (ONE | 0x80000000) : ONE);
    if (e < 122)  return special(x);
    // Entries hold 2^-(j+4) * tanh(2^j (1+t)) for binade j = e - 127.
    r.addr = TBL_TANH_BASE + ((e - 122) << 3) + (m >> 20);
    r.xl   = (m >> 2) & ((1u << XL_BITS) - 1);
    r.exp  = int32_t(e) - 127 + 4;
    r.neg  = s;
    return r;

  default:
    return special(0);
  }
}

int64_t interpolate(uint32_t addr, uint32_t xl) {
  const coeff_t& c = kTable[addr];
  int64_t a0 = int64_t(c.a0) << S0;
  int64_t a1 = int64_t(c.a1) * (int64_t(1) << S1);
  int64_t a2 = int64_t(c.a2) * (int64_t(1) << S2);
  int64_t sq = (int64_t(xl) * xl) >> XL_BITS;
  int64_t t1 = (a1 * xl) >> XL_BITS;
  int64_t t2 = (a2 * sq) >> XL_BITS;
  int64_t p = a0 + t1 + t2;
  return (p < 0) ? 0 : p;
}

uint32_t normalize(uint64_t mag, int32_t exp, bool neg, bool zsign) {
  if (mag == 0)
    return uint32_t(zsign) << 31;
  int lead = 63 - __builtin_clzll(mag);
  uint64_t norm = (mag << (39 - lead)) & ((uint64_t(1) << 40) - 1);
  uint32_t sig = uint32_t(norm >> 16);
  uint32_t rnd = uint32_t(norm >> 15) & 1;
  int32_t bexp = lead - int32_t(FRAC) + exp + 127;
  sig += rnd;
  if (sig >> 24) {
    sig >>= 1;
    bexp += 1;
  }
  uint32_t sgn = uint32_t(neg) << 31;
  if (bexp >= 255)
    return sgn | POS_INF;
  if (bexp <= 0)
    return sgn;
  return sgn | (uint32_t(bexp) << 23) | (sig & 0x7fffff);
}

} // namespace

uint32_t eval(uint32_t fn, uint32_t x) {
  reduced_t r = reduce(fn, x);
  if (r.special)
    return r.value;
  int64_t p = interpolate(r.addr, r.xl);
  if (r.is_lg2) {
    int64_t l = (int64_t(r.k) << FRAC) + p;
    bool neg = (l < 0);
    return normalize(uint64_t(neg ? -l : l), 0, neg, false);
  }
  return normalize(uint64_t(p), r.exp, r.neg, r.zsign);
}

const char* fn_name(uint32_t fn) {
  static const char* names[FN_COUNT] = {"rcp", "rsqrt", "ex2", "lg2", "sinpi", "cospi", "tanh"};
  return (fn < FN_COUNT) ? names[fn] : "?";
}

} // namespace vortex::mufu
