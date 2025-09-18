// Copyright © 2019-2023
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

#pragma once

#include <algorithm>
#include <array>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

class Logger {
public:
  static void log(const char *fmt, ...) {
    if constexpr (FEDP_TRACE) {
      va_list args;
      va_start(args, fmt);
      std::vprintf(fmt, args);
      va_end(args);
    }
  }
};
#define LOG(...) Logger::log(__VA_ARGS__)

class FEDP {
public:
  // frm: RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4
  // lanes: product lanes per accumulation
  FEDP(int frm, uint32_t lanes)
    : frm_(frm)
    , lanes_(lanes) {
    assert(frm_ >= 0 && frm_ <= 4);
    assert(lanes_ >= 1 && lanes_ <= 8);
    LOG("[ctor] frm=%d, lanes=%u, super=TF32 e8m10, Wc=%u, Win=%u\n", frm_, lanes_, Wc_, Win_);
  }

  // Top-level: a_words/b_words contain n_words packed values with (exp_bits,sig_bits)
  float operator()(const std::vector<uint32_t> &a_words,
                   const std::vector<uint32_t> &b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits) {
    resetFlags();

    const uint32_t width = 1u + exp_bits + sig_bits;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t elems_per_word = packed ? (32u / width) : 1u;
    const uint32_t k = n_words * elems_per_word;

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%d, elems/word=%u, n_words=%u, k=%u\n",
      exp_bits, sig_bits, width, (packed ? 1 : 0), elems_per_word, n_words, k);

    // S1: decode packed inputs to dec_t pairs
    const auto ab_dec = decode_inputs(a_words, b_words, n_words, elems_per_word, exp_bits, sig_bits, packed);

    // S1: decode C
    const uint32_t c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);

    // Fast path: handle specials/zero if any; use sticky seen so far
    if (const uint32_t fast = decode_special_or_zero(ab_dec, c_dec)) {
      return f32FromBits(fast);
    }

    // S1: convert decoded C to common grid
    const auto c_term = decoded_to_common(c_dec, 8, 23);

    // S1: multiply and local n_groups-reduce into CSA pair per n_groups
    const auto [prods, mul_sticky] = multiply_to_common(ab_dec, exp_bits, sig_bits);

    // S2: align all CSA pairs to Emax, with sticky from dropped bits
    const auto [aligned, Emax, align_sticky] = alignment(prods, c_term);

    // S3: CSA-accumulate all aligned pairs and perform the CPA here
    const auto acc = accumulate(aligned);

    // Fast path: handle specials/zero if any; use sticky seen so far
    if (acc == 0) {
      if (mul_sticky || align_sticky) {
        fflags_ |= (FLAG_NX | FLAG_UF);
      }
      LOG("[final-fast] zero=1, fflags=0x%02x\n", fflags_);
      return f32FromBits(0);
    }

    // S4: normalize + round+pack
    const norm_t nrm = normalize(acc, Emax, mul_sticky || align_sticky);
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  // fflags accessor
  uint32_t fflags() const { return fflags_; }

private:
  // ------------------------------- Types ------------------------------------
  struct dec_t {
    uint32_t sign{0};   // sign bit
    uint32_t frac{0};   // fraction bits
    uint32_t exp{0};    // exponent bits
    bool is_zero{false};// is zero
    bool is_sub{false}; // is subnormal
    bool is_inf{false}; // is infinity
    bool is_nan{false}; // is NaN
  };

  struct term24_t {
    int32_t  S{0};
    uint32_t C{0};
    int32_t  E{0};
    bool is_zero{true};
    bool is_inf{false};
    bool is_nan{false};
  };

  struct norm_t {
    uint32_t sign{0};
    uint32_t kept24{0}; // 24-bit core (1+23) for FP32
    int32_t e_unb{0};
    uint32_t round_bit{0};
    bool sticky{false};
  };

  enum { RNE = 0, RTZ = 1, RDN = 2, RUP = 3, RMM = 4 };
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // ----------------------------- Decode Inputs ------------------------------
  std::vector<std::array<dec_t, 2>>
  decode_inputs(const std::vector<uint32_t> &a_words,
               const std::vector<uint32_t> &b_words,
               uint32_t n_words,
               uint32_t elems_per_word,
               int exp_bits,
               int sig_bits,
               bool packed) {
    const uint32_t width = 1u + exp_bits + sig_bits;
    assert(width < 32);
    const uint32_t enc_mask  = ((1u << width) - 1u);
    const uint32_t k = n_words * elems_per_word;
    std::vector<std::array<dec_t, 2>> terms(k);

    uint32_t out = 0;
    for (uint32_t i = 0; i < n_words; ++i) {
      uint32_t aw = a_words[i];
      uint32_t bw = b_words[i];
      for (uint32_t j = 0; j < elems_per_word; ++j, ++out) {
        const uint32_t aenc = packed ? (aw & enc_mask) : aw;
        const uint32_t benc = packed ? (bw & enc_mask) : bw;
        terms[out][0] = decode_input(aenc, exp_bits, sig_bits);
        terms[out][1] = decode_input(benc, exp_bits, sig_bits);
        if (packed) {
          aw >>= width;
          bw >>= width;
        }
        LOG("[decode] idx=%u, A(enc=0x%x, s=%u, e=%u, f=0x%x), B(enc=0x%x, s=%u, e=%u, f=0x%x)\n",
            out,
            aenc, terms[out][0].sign, terms[out][0].exp, terms[out][0].frac,
            benc, terms[out][1].sign, terms[out][1].exp, terms[out][1].frac);
      }
    }
    LOG("[decode_inputs] decoded=%u\n", out);
    return terms;
  }

  // ---------------------------- Decode C to CSA -----------------------------
  term24_t decoded_to_common(dec_t decoded, int exp_bits, int sig_bits) {
    const uint32_t frac_mask = ((1u << sig_bits) - 1u);
    const uint32_t exp_mask  = ((1u << exp_bits) - 1u);
    const uint32_t bias = (1u << (exp_bits - 1)) - 1u;

    const uint32_t s = decoded.sign;
    const uint32_t e = decoded.exp;
    const uint32_t f = decoded.frac;

    const int32_t Ec = e - bias + 127;
    const uint32_t M = ((e != 0) << sig_bits) | f;

    // Scale mantissa to Wc_-1
    const int dM = int(Wc_ - 1u) - sig_bits;
    assert(dM < 32);
    uint32_t m_c = M << dM;
    const int32_t addend = s ? -int32_t(m_c) : int32_t(m_c);
    LOG("[decodeC] s=%u, Ec=0x%x, m=0x%x -> add=0x%x\n", s, Ec, m_c, addend);

    term24_t p;
    p.S = addend;
    p.C = 0;
    p.E = Ec;
    p.is_zero = (m_c == 0);
    return p;
  }

  // -------------------- Special/zero fast finalize helper -------------------
  uint32_t decode_special_or_zero(const std::vector<std::array<dec_t, 2>> &ab_dec, const dec_t &c_dec) {
    bool has_nan = false;
    bool has_nv = false;
    bool has_neg_inf = false;
    bool has_pos_inf = false;
    bool has_any_special = false;

    const size_t N = ab_dec.size();
    for (size_t i = 0; i < N; ++i) {
      const dec_t &a = ab_dec[i][0];
      const dec_t &b = ab_dec[i][1];

      if (a.is_nan || b.is_nan)
        has_nan = true;
      if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero))
        has_nv = true;

      if (a.is_inf || b.is_inf) {
        const uint32_t s = a.sign ^ b.sign;
        if (s)
          has_neg_inf = true;
        else
          has_pos_inf = true;
        has_any_special = true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
        continue;
      }
    }
    if (has_nv
     || (has_pos_inf && has_neg_inf)
     || (c_dec.is_inf && ((has_pos_inf && c_dec.sign == 1u) || (has_neg_inf && c_dec.sign == 0u)))) {
      fflags_ |= FLAG_NV;
      LOG("[final-fast] invalid=1, out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_nan || c_dec.is_nan) {
      LOG("[final-fast] has_NaN=1, out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_pos_inf || has_neg_inf) {
      const uint32_t s = has_neg_inf ? 1u : 0u;
      return packInf32(s);
    }
    if (c_dec.is_inf) {
      return packInf32(c_dec.sign);
    }
    return 0;
  }

  // ------------------------- S1: Multiply & n_groups → CSA ------------------
  std::tuple<std::vector<term24_t>, bool>
  multiply_to_common(const std::vector<std::array<dec_t, 2>> &terms, int exp_bits, int sig_bits) {
    const uint32_t width = 1u + exp_bits + sig_bits;
    const int32_t  bias  = (1 << (exp_bits - 1)) - 1;
    const uint32_t Wm_in = uint32_t(sig_bits) + 1u;   // input mantissa width (incl. hidden 1)
    const uint32_t Wraw  = 2u * Wm_in;                // raw product width
    assert(Wraw < Wc_);                               // must fit Wc grid
    const uint32_t L_in  = Wc_ - Wraw;                // shift up to Wc grid

    struct Traw {
      uint32_t sign;
      uint32_t m_wc;
      int32_t  E;
    };

    const size_t N = terms.size();
    std::vector<Traw> v(N);

    // Phase A: raw multiply and shift to Wc grid
    for (size_t i = 0; i < N; ++i) {
      const dec_t &a = terms[i][0];
      const dec_t &b = terms[i][1];

      const uint32_t s = a.sign ^ b.sign;

      const int32_t Ea = a.exp - bias;
      const int32_t Eb = b.exp - bias;
      const int32_t E  = Ea + Eb + 1 + 127;

      const uint32_t Ma = ((a.exp != 0) << sig_bits) | a.frac;
      const uint32_t Mb = ((b.exp != 0) << sig_bits) | b.frac;
      const uint32_t P  = Ma * Mb; // Wraw bits

      // Shift up to Wc bits (no loss)
      const uint32_t m_wc = P << L_in;
      assert(L_in < 32);

      v[i] = Traw{s, m_wc, E};
      LOG("[mul-prod] i=%zu, s=%u, E=0x%x, P=0x%x, m_wc=0x%x, Wraw=%u\n", i, s, E, P, m_wc, Wraw);
    }

    // Phase B: reduce per group into CSA pair aligned to each group's Eg
    const uint32_t n_groups = 16 / width; // fp16:1, e5m2:2, e4m3:2, fp4:4
    const size_t G = (N + n_groups - 1) / n_groups;
    std::vector<term24_t> out(G);
    bool sticky = false;

    for (size_t base = 0; base < N; base += n_groups) {
      const size_t g = base / n_groups;
      const size_t end = std::min(N, base + n_groups);

      // Per-group max exponent Eg
      int32_t Eg = INT32_MIN;
      for (size_t i = base; i < end; ++i) {
        Eg = std::max(Eg, v[i].E);
      }

      term24_t acc{0, 0, Eg, false, false, false};

      for (size_t i = base; i < end; ++i) {
        const auto &t = v[i];

        // Align t to Eg in magnitude domain: right shift magnitude, gather sticky
        const uint32_t delta = uint32_t(Eg - t.E);
        uint32_t m_shifted = (t.m_wc >> delta);

        bool sticky_local = false;
        if (delta >= Wc_) {
          sticky_local = (t.m_wc != 0);
        } else if (delta > 0) {
          const uint32_t mask = (delta == 32) ? 0xFFFFFFFFu : ((1u << delta) - 1u);
          sticky_local = (t.m_wc & mask) != 0u;
        }

        // Signed addend into CSA rails at Wc weight
        const int32_t addend = t.sign ? -int32_t(m_shifted) : int32_t(m_shifted);
        auto [s1, c1] = csa32(acc.S, (acc.C << 1u), addend);
        acc.S = s1;
        acc.C = c1;

        sticky |= sticky_local;

        LOG("[s1-csa] g=%zu, i=%zu, s=%u, delta=0x%x, m_adj=0x%x, add=0x%x, S=0x%x, C=0x%x\n",
            g, i, t.sign, delta, m_shifted, addend, acc.S, acc.C);
      }

      acc.is_zero = (acc.S == 0 && acc.C == 0);
      out[g] = acc;
      LOG("[s1-csa] g=%zu, Eg=0x%x, S=0x%x, C=0x%x, sticky=%d, zero=%d\n",
          g, Eg, acc.S, acc.C, (sticky ? 1 : 0), (acc.is_zero ? 1 : 0));
    }

    LOG("[multiply] groups=%zu\n", out.size());
    return std::tuple{out, sticky};
  }

  // ------------------------- S2: Alignment to Emax (CSA-preserving, 32-bit) --
  std::tuple<std::vector<term24_t>, int32_t, bool>
  alignment(const std::vector<term24_t> &groups, const term24_t &cterm) {
    // Emax over all terms
    int32_t Emax = INT32_MIN;
    for (const auto &g : groups) Emax = std::max(Emax, g.E);
    Emax = std::max(Emax, cterm.E);

    std::vector<term24_t> out;
    out.reserve(groups.size() + 1);

    bool sticky = false;

    auto align_one = [&](const term24_t &t, const char *tag, size_t idx) {
      const uint32_t delta = uint32_t(Emax - t.E);

      // Shift rails independently
      const int32_t Sprime = asr32(t.S, delta);
      const int32_t Cprime = asr32(t.C, delta);

      // Sticky: any dropped bit from either rail counts
      const bool local_sticky = any_dropped32(t.S, delta) || any_dropped32(t.C, delta);
      sticky |= local_sticky;

      term24_t a{};
      a.S = Sprime;
      a.C = Cprime;
      a.E = Emax;
      a.is_zero = (a.S == 0 && a.C == 0);
      a.is_inf = false;
      a.is_nan = false;

      LOG("[align-%s] idx=%zu, delta=0x%x, S'=0x%x, C'=0x%x, sticky+=%d\n",
          tag, idx, delta, a.S, a.C, (local_sticky ? 1 : 0));

      out.push_back(a);
    };

    for (size_t i = 0; i < groups.size(); ++i) {
      align_one(groups[i], "p", i);
    }
    align_one(cterm, "c", 0);

    LOG("[alignment] Emax=0x%x, sticky_align=%d, terms=%zu\n",
        Emax, (sticky ? 1 : 0), out.size());

    return std::tuple{out, Emax, sticky};
  }

  // ------------------------- S3: CSA Accumulate + CPA -----------------------
  int32_t accumulate(const std::vector<term24_t> &aligned) {
    term24_t acc{0, 0, 0, true, false, false};
    acc.E = aligned[0].E;
    acc.is_zero = false;

    auto add_pair_into = [&](term24_t &dst, const term24_t &src) {
      // Fold S component
      auto [s1, c1] = csa32(dst.S, (dst.C << 1), src.S);
      dst.S = s1;
      dst.C = c1;
      // Fold C component (shifted by 1 in weight)
      auto [s2, c2] = csa32(dst.S, (dst.C << 1),(src.C << 1));
      dst.S = s2;
      dst.C = c2;
    };

    for (size_t i = 0; i < aligned.size(); ++i) {
      add_pair_into(acc, aligned[i]);
      LOG("[acc-csa] i=%zu, S=0x%x, C=0x%x\n", i, acc.S, acc.C);
    }

    // Single CPA here: V = S + (C<<1) with well-defined wrap via unsigned
    int32_t V = acc.S + (acc.C << 1);
    return V;
  }

  // ------------------------------ S4: Normalize -----------------------------
  norm_t normalize(int32_t acc, int32_t Emax, bool sticky_prev) {
    norm_t n{};
    n.sign = (acc < 0) ? 1u : 0u;
    uint32_t mag = (acc < 0) ? uint32_t(-acc) : uint32_t(acc);

    const uint32_t nbits = (mag == 0) ? 1u : (32u - uint32_t(clz32(mag))); // >=1
    // Our fixed-point point was at bit (Wc_-1) relative to exponent; now the leading 1 is at nbits-1
    n.e_unb = (Emax - int32_t(Wc_ - 1u)) + int32_t(nbits - 1u);

    // Normalize to FP32 24-bit core (1+23)
    const int FP_TOP = 23;
    const int sh = int(nbits - 1u) - FP_TOP;

    uint32_t kept24 = 0, round_bit = 0;
    bool sticky_norm = false;
    if (sh > 0) {
      const uint32_t mask = (sh >= 32) ? ~0ull : ((1ull << sh) - 1ull);
      const uint32_t rem = mag & mask;
      round_bit = (sh >= 1) ? ((rem >> (sh - 1)) & 1ull) : 0u;
      sticky_norm = (sh >= 2) ? ((rem & ((1ull << (sh - 1)) - 1ull)) != 0ull) : false;
      kept24 = (sh >= 32) ? 0u : uint32_t(mag >> sh);
    } else {
      const int lsh = -sh;
      kept24 = (lsh >= 32) ? 0u : uint32_t(mag << lsh);
    }
    kept24 &= ((1u << 24) - 1u);

    n.kept24 = kept24;
    n.round_bit = round_bit;
    n.sticky = (sticky_norm || sticky_prev);

    LOG("[normalize] sign=%u, kept24=0x%x, e_unb=%d, round=%u, stickyAny=%d\n",
        n.sign, n.kept24, n.e_unb, n.round_bit, n.sticky ? 1 : 0);
    return n;
  }

  // ----------------------------- Rounding/Pack ------------------------------
  uint32_t round_and_pack(const norm_t &nrm) {
    uint32_t kept24 = nrm.kept24;
    int32_t e_unb = nrm.e_unb;
    const uint32_t sign = nrm.sign;

    if (nrm.round_bit || nrm.sticky) {
      fflags_ |= FLAG_NX;
    }

    const uint32_t lsb = kept24 & 1u;
    if (roundInc(frm_, sign, lsb, nrm.round_bit, nrm.sticky)) {
      kept24 += 1u;
      if (kept24 >= (1u << 24)) {
        kept24 >>= 1;
        e_unb += 1;
      }
    }

    const int32_t e_bias = e_unb;

    if (e_bias >= 0xFF) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      LOG("[rounding] out=Inf\n");
      return packInf32(sign);
    }

    if (e_bias <= 0) {
      const int sh2 = 1 - e_bias;
      const uint32_t shifted = (sh2 >= 32) ? 0u : (kept24 >> sh2);
      const uint32_t mask2 = (sh2 >= 32) ? kept24 : ((1u << sh2) - 1u);
      const uint32_t rem2 = kept24 & mask2;
      const uint32_t rb2 = (sh2 >= 1) ? ((rem2 >> (sh2 - 1)) & 1u) : 0u;
      const bool st2 = (sh2 >= 2) ? ((rem2 & ((1u << (sh2 - 1)) - 1u)) != 0u) : false;

      uint32_t frac_keep = shifted & ((1u << 23) - 1u);
      const uint32_t lsb2 = frac_keep & 1u;

      if (rb2 || st2) {
        fflags_ |= FLAG_NX;
      }
      if (roundInc(frm_, sign, lsb2, rb2, st2)) {
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u << 23)) {
          if (fflags_ & FLAG_NX) {
            fflags_ |= FLAG_UF;
          }
          LOG("[rounding] subnormal_to_min_normal=1\n");
          return (sign << 31) | (1u << 23);
        }
        frac_keep = t;
      }
      if (fflags_ & FLAG_NX) {
        fflags_ |= FLAG_UF;
      }
      const uint32_t out_sub = (sign << 31) | frac_keep;
      LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out_sub, fflags_);
      return out_sub;
    }

    const uint32_t exp_out  = uint32_t(e_bias);
    const uint32_t frac_out = kept24 & ((1u << 23) - 1u);
    const uint32_t out_norm = (sign << 31) | (exp_out << 23) | frac_out;
    LOG("[rounding] normal_out=0x%x, fflags=0x%02x\n", out_norm, fflags_);
    return out_norm;
  }

  static inline dec_t decode_input(uint32_t enc, int exp_bits, int sig_bits) {
    const uint32_t frac_mask = ((1u << sig_bits) - 1u);
    const uint32_t exp_mask  = ((1u << exp_bits) - 1u);
    const uint32_t sign = (enc >> (exp_bits + sig_bits)) & 1u;
    const uint32_t exp  = (enc >> sig_bits) & exp_mask;
    const uint32_t frac = enc & frac_mask;

    dec_t d;
    d.sign    = sign;
    d.frac    = frac;
    d.exp     = exp;
    d.is_zero = (exp == 0 && frac == 0);
    d.is_sub  = (exp == 0 && frac != 0);
    d.is_inf  = (exp == exp_mask && frac == 0);
    d.is_nan  = (exp == exp_mask && frac != 0);
    return d;
  }

  static bool roundInc(int frm, uint32_t sign, uint32_t lsb, uint32_t round_bit, bool sticky) {
    switch (frm) {
    case RNE:
      if (!round_bit)
        return false;
      if (sticky)
        return true;
      return (lsb & 1u);
    case RTZ:
      return false;
    case RDN:
      return (round_bit || sticky) && (sign == 1);
    case RUP:
      return (round_bit || sticky) && (sign == 0);
    case RMM:
      return (round_bit || sticky);
    default:
      return false;
    }
  }

  // 3:2 compressor on equal-weight bit-vectors
  static inline std::pair<uint32_t, uint32_t> csa32(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t t = (a ^ b);
    uint32_t sum = t ^ c;
    uint32_t carry = ((a & b) | (b & c) | (a & c));
    return std::pair{sum, carry};
  }

  // portable arithmetic shift right
  static inline int32_t asr32(int32_t x, uint32_t k) {
    if (k == 0) return x;
    if (k >= 31) return (x < 0) ? -1 : 0; // sign saturation after huge shift
    return (x >= 0) ? (x >> k) : ~((~x) >> k);
  }

  // check if any of the lowest k bits are set
  static inline bool any_dropped32(int32_t x, uint32_t k) {
    if (k == 0) return false;
    if (k >= 31) return x != 0;
    const uint32_t mask = (1u << k) - 1u;
    return (uint32_t(x) & mask) != 0u;
  }

  // ------------------------------- Utilities ----------------------------------
  static inline uint32_t bitsFromF32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
  }
  static inline float f32FromBits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  static inline int clz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    if (!x)
      return 32;
    int n = 0;
    while (!(x & (1u << 31))) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }
  static inline uint32_t packInf32(uint32_t s) { return (s << 31) | (0xFFu << 23); }
  static inline uint32_t canonicalNaN32() { return (0xFFu << 23) | (1u << 22); }

  void resetFlags() {
    fflags_ = 0;
  }

  // ------------------------------ Members -----------------------------------
  // Config Superformat is TF32 (e8m10)
  const uint32_t Wc_{24};  // common product magnitude width
  const uint32_t Win_{25}; // signed addend width
  const int frm_;
  const uint32_t lanes_;

  // State / flags
  uint32_t fflags_{0};
};
