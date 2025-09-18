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

#include <algorithm>
#include <array>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
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
  // lanes: product lanes per accumulation "cycle" (1..8)
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    if (frm_ < 0 || frm_ > 4) frm_ = 0;
    if (lanes_ < 1) lanes_ = 1;
    if (lanes_ > 8) lanes_ = 8;

    Wc_   = 24u;                                // product magnitude grid
    Win_  = Wc_ + 1u;                           // signed addend width
    Wacc_ = Win_ + ceil_log2(lanes_ + 1) + 1u;  // accumulator width

    LOG("[ctor] frm=%d, lanes=%u, super=TF32, e8m10, Wc=%u, Win=%u, Wacc=%u\n",
        frm_, lanes_, Wc_, Win_, Wacc_);
  }

  float operator()(const std::vector<uint32_t> &a_words,
                   const std::vector<uint32_t> &b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits) {
    resetFlags();

    const uint32_t width = 1u + uint32_t(exp_bits) + uint32_t(sig_bits);
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t elems_per_word = packed ? (32u / width) : 1u;
    const uint32_t k = n_words * elems_per_word;

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%d, elems/word=%u, n_words=%u, k=%u\n",
        exp_bits, sig_bits, width, packed ? 1 : 0, elems_per_word, n_words, k);

    // 1) decode
    const auto terms = decodeInputs(a_words, b_words, n_words, elems_per_word,
                                    width, exp_bits, sig_bits, packed);

    // 2) multiply + CSA grouping (fp16:1, fp8:2, fp4:4)
    const auto groups = multiply_to_common(terms, sig_bits, width);

    // 3) decode C
    const term24_t cterm = decodeC_to_common(bitsFromF32(c));

    // 4) alignment (finish CSA + global align vs C)
    const AlignOut aout = alignment(groups, cterm);
    LOG("[PATH] CSA alignment with %zu groups\n", groups.size());

    // 5) accumulate (honor lanes)
    const int64_t acc = accumulate(aout);

    // 6) fast finalize specials/zero
    if (has_any_special_ || acc == 0) {
      if (const uint32_t out_fast = finalize_special_or_zero(acc, (sticky_mul_ || sticky_align_ || sticky_c32_))) {
        float ret = f32FromBits(out_fast);
        LOG("[RETURN] fedp_out_bits=0x%08x fedp_out=%g\n", out_fast, (double)ret);
        return ret;
      }
    }

    // 7) normalize
    const Norm nrm = normalize(acc, aout.max_exp);

    // 8) rounding + pack
    const uint32_t out = round_and_pack(nrm);
    float ret = f32FromBits(out);
    LOG("[RETURN] fedp_out_bits=0x%08x fedp_out=%g\n", out, (double)ret);
    return ret;
  }

  uint32_t fflags() const { return fflags_; }

private:
  // -------------------------------- Types ------------------------------------
  struct dec_t {
    uint32_t sign{}, frac{}, exp_field{};
    int32_t exp_unb{};
    bool is_zero{}, is_sub{}, is_inf{}, is_nan{};
  };

  // Common product term (value = m * 2^(E - (Wc_-1)))
  struct term24_t {
    uint32_t sign{0};
    uint32_t m{0}; // Wc_ bits magnitude
    int32_t E{0};
    bool is_zero{true}, is_inf{false}, is_nan{false};
  };

  struct AlignOut {
    std::vector<int32_t> addends;
    int32_t max_exp{0};
    bool sticky_align{false};
  };

  struct Norm {
    uint32_t sign{0};
    uint32_t kept24{0}; // 24-bit core (1+23)
    int32_t e_unb{0};
    uint32_t round_bit{0};
    bool sticky_any{false};
  };

  // Output of multiply_to_common(): per-group compressed products
  struct prod_t {
    uint32_t S{0};    // carry-save sum (Win_ bits)
    uint32_t C{0};    // carry-save carry (Win_ bits, not shifted)
    int32_t  E{0};    // group exponent (max of members)
    uint32_t negs{0}; // number of negative addends compressed
  };

  enum { RNE = 0, RTZ = 1, RDN = 2, RUP = 3, RMM = 4 };
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // -------------------------- decodeInputs --------------------------
  std::vector<std::array<dec_t,2>>
  decodeInputs(const std::vector<uint32_t> &a_words,
               const std::vector<uint32_t> &b_words,
               uint32_t n_words,
               uint32_t elems_per_word,
               uint32_t width,
               int exp_bits, int sig_bits,
               bool packed) const {
    const uint32_t enc_mask  = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
    const uint32_t frac_mask = (sig_bits >= 32) ? 0xFFFFFFFFu : ((1u << sig_bits) - 1u);
    const uint32_t exp_mask  = (exp_bits >= 32) ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u);
    const uint32_t bias      = (1u << (exp_bits - 1)) - 1u;

    const uint32_t k = n_words * elems_per_word;
    std::vector<std::array<dec_t,2>> terms(k);

    uint32_t out = 0;
    for (uint32_t i = 0; i < n_words; ++i) {
      uint32_t aw = a_words[i], bw = b_words[i];
      for (uint32_t j = 0; j < elems_per_word; ++j, ++out) {
        const uint32_t aenc = packed ? (aw & enc_mask) : aw;
        const uint32_t benc = packed ? (bw & enc_mask) : bw;

        terms[out][0] = decode_one(aenc, exp_bits, sig_bits, exp_mask, frac_mask, bias);
        terms[out][1] = decode_one(benc, exp_bits, sig_bits, exp_mask, frac_mask, bias);

        if (packed) { aw >>= width; bw >>= width; }

        LOG("[decode] idx=%u, A(enc=0x%x, s=%u,e=%u,f=0x%x), B(enc=0x%x, s=%u,e=%u,f=0x%x)\n",
            out,
            aenc, terms[out][0].sign, terms[out][0].exp_field, terms[out][0].frac,
            benc, terms[out][1].sign, terms[out][1].exp_field, terms[out][1].frac);
      }
    }
    LOG("[decodeInputs] decoded=%u\n", out);
    return terms;
  }

  static inline dec_t decode_one(uint32_t enc, int exp_bits, int sig_bits,
                                 uint32_t exp_mask, uint32_t frac_mask, uint32_t bias) {
    const uint32_t sign = (enc >> (exp_bits + sig_bits)) & 1u;
    const uint32_t exp  = (enc >> sig_bits) & exp_mask;
    const uint32_t frac = enc & frac_mask;

    dec_t d{};
    d.sign = sign; d.frac = frac; d.exp_field = exp;
    d.is_zero = (exp == 0 && frac == 0);
    d.is_sub  = (exp == 0 && frac != 0);
    d.is_inf  = (exp == exp_mask && frac == 0);
    d.is_nan  = (exp == exp_mask && frac != 0);
    d.exp_unb = (exp == 0) ? (1 - int32_t(bias))
                           : (exp == exp_mask ? 0 : int32_t(exp) - int32_t(bias));
    return d;
  }

  // ----------------------------- decodeC ---------------------------
  term24_t decodeC_to_common(uint32_t enc32) {
    const uint32_t s = (enc32 >> 31) & 1u;
    const uint32_t e = (enc32 >> 23) & 0xFFu;
    const uint32_t f = enc32 & 0x7FFFFFu;

    c_sign_ = s;
    c_is_inf_ = (e == 0xFFu && f == 0);
    c_is_nan_ = (e == 0xFFu && f != 0);

    if (e == 0 && f == 0) {
      LOG("[decodeC] zero=1\n");
      LOG("[decodeC] s=0, E=0, m=0x000000, specials=0\n");
      return term24_t{0, 0, 0, true, false, false};
    }
    if (c_is_inf_ || c_is_nan_) {
      LOG("[decodeC] special=%s\n", c_is_inf_ ? "Inf" : "NaN");
      LOG("[decodeC] special=1\n");
      return term24_t{0, 0, 0, true, c_is_inf_, c_is_nan_};
    }

    const bool is_sub = (e == 0 && f != 0);
    const int32_t Ec = is_sub ? (1 - 127) : (int32_t(e) - 127);
    const uint32_t M = is_sub ? f : ((1u << 23) | f); // 24b

    // Scale FP32 mantissa to Wc_-1 top (23 for TF32)
    const int dM = int(Wc_ - 1u) - 23;
    uint32_t m_c = 0;
    if (dM >= 0) {
      m_c = (dM >= 32) ? 0u : (M << dM);
    } else {
      const int r = -dM;
      if (r >= 32)
        m_c = 0u;
      else {
        const uint32_t dropped = M & ((1u << r) - 1u);
        if (dropped)
          sticky_c32_ = true;
        m_c = M >> r;
      }
    }
    LOG("[decodeC] s=%u, Ec=%d, m=0x%x, sticky_c32=%d\n", s, Ec, m_c, sticky_c32_ ? 1 : 0);
    return term24_t{s, m_c, Ec, false, false, false};
  }

  // ---------------------------- CSA helpers --------------------------
  // 3:2 compressor for Win_-bit words. Returns {sum, carry}.
  static inline std::pair<uint32_t,uint32_t>
  carry_save_add(uint32_t a, uint32_t b, uint32_t c, uint32_t W) {
    const uint32_t maskW = (W >= 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);
    const uint32_t sum   = (a ^ b ^ c) & maskW;
    const uint32_t carry = ((a & b) | (a & c) | (b & c)) & maskW; // not shifted
    return {sum, carry};
  }

  // Reduce a vector of Win_-bit addends into (sum,carry) with iterative CSA.
  static inline std::pair<uint32_t,uint32_t>
  carry_save_add(const std::vector<uint32_t> &as, uint32_t W) {
    uint32_t S = 0, C = 0;
    const uint32_t maskW = (W >= 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);
    for (uint32_t a : as) {
      const uint32_t Cshift = (C << 1) & maskW;
      auto [nS, nC] = carry_save_add(S, a, Cshift, W);
      S = nS; C = nC;
    }
    return {S, C};
  }

  // CPA: propagate carry and return (sumW, cout) where cout is the bit leaving MSB.
  static inline std::pair<uint32_t,uint32_t>
  carry_propagate_add(uint32_t sum, uint32_t carry, uint32_t W) {
    const uint64_t wide = uint64_t(sum) + (uint64_t(carry) << 1);
    const uint32_t sumW = (W >= 64) ? uint32_t(wide) : uint32_t(wide & ((1ull << W) - 1ull));
    const uint32_t cout = (W >= 64) ? 0u : uint32_t((wide >> W) & 1u);
    return {sumW, cout};
  }

  // ----------------------------- multiply ---------------------------
  // Raw multiply -> shift to 24b grid -> CSA grouping (G=1/2/4)
  std::vector<prod_t>
  multiply_to_common(const std::vector<std::array<dec_t,2>> &terms,
                     int sig_bits_in, uint32_t width_in) {
    has_any_special_ = false;
    sticky_mul_ = false;

    struct Raw { uint32_t sign, P; int32_t E; bool is_zero, is_inf, is_nan; };

    const uint32_t Wm_in   = uint32_t(sig_bits_in) + 1u;
    const uint32_t Wraw_in = 2u * Wm_in;
    const uint32_t L_in    = Wc_ - Wraw_in;

    // Phase A: raw multiply
    const size_t n = terms.size();
    std::vector<Raw> raw(n);
    for (size_t i = 0; i < n; ++i) {
      const dec_t &a = terms[i][0];
      const dec_t &b = terms[i][1];

      if (a.is_nan || b.is_nan) has_nan_ = true;
      if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero)) has_nv_ = true;

      if (a.is_inf || b.is_inf) {
        const uint32_t s = a.sign ^ b.sign;
        if (s) has_neg_inf_ = true; else has_pos_inf_ = true;
        raw[i] = Raw{0,0,0,true,true,false};
        has_any_special_ = true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
        continue;
      }
      if (a.is_zero || b.is_zero) {
        raw[i] = Raw{0,0,0,true,false,false};
        LOG("[mul-prod] i=%zu, zero=1\n", i);
        continue;
      }

      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sig_bits_in) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sig_bits_in) | b.frac);
      const uint32_t P  = Ma * Mb;         // Wraw_in bits
      const int32_t  E  = (a.exp_unb + b.exp_unb) + 1;
      const uint32_t s  = a.sign ^ b.sign;

      raw[i] = Raw{s, P, E, false, false, false};
      LOG("[mul-prod] i=%zu, s=%u, E=%d, P=0x%x, Wraw_in=%u\n", i, s, E, P, Wraw_in);
    }

    // Phase B: shift to 24-bit common grid (exact)
    struct WC { uint32_t sign, m; int32_t E; bool is_zero, is_inf, is_nan; };
    std::vector<WC> wc(n);
    LOG("[mul-sup] L_in=%u, Wc=%u, Wraw_in=%u\n", L_in, Wc_, Wraw_in);

    for (size_t i = 0; i < n; ++i) {
      const auto &r = raw[i];
      if (r.is_zero || r.is_inf || r.is_nan) {
        wc[i] = WC{0,0,0,true,r.is_inf,r.is_nan};
        continue;
      }
      uint32_t m = (L_in >= 32) ? 0u : (r.P << L_in);
      wc[i] = WC{r.sign, m, r.E, false, false, false};
      LOG("[mul-sup] i=%zu, m_wc=0x%x, E=%d\n", i, m, r.E);
    }

    // Phase C: CSA grouping (fp16=1, fp8=2, fp4=4)
    uint32_t group = 1;
    if (width_in == 8u) group = 2;
    else if (width_in == 4u) group = 4;

    const size_t n_groups = (n + group - 1) / group;
    std::vector<prod_t> groups(n_groups);

    const uint32_t W     = Win_; // 25
    const uint32_t maskW = (W >= 32) ? 0xFFFFFFFFu : ((1u << W) - 1u);

    for (size_t g = 0; g < n_groups; ++g) {
      const size_t base = g * group;
      const size_t end  = std::min(n, base + group);

      // Find group exponent (max of valid terms)
      int32_t Egrp = INT32_MIN;
      for (size_t i = base; i < end; ++i) {
        const auto &t = wc[i];
        if (!t.is_zero && !t.is_inf && !t.is_nan && t.E > Egrp) Egrp = t.E;
      }
      if (Egrp == INT32_MIN) Egrp = 0; // still flow through

      // Build aligned two's-complement addends (size==group, 0-padded)
      std::vector<uint32_t> addends(group, 0u);
      uint32_t negs = 0;

      for (size_t i = base, slot = 0; i < base + group; ++i, ++slot) {
        uint32_t a_tc = 0;

        if (i < end) {
          const auto &t = wc[i];
          uint32_t m = 0;

          if (!(t.is_zero || t.is_inf || t.is_nan)) {
            const uint32_t delta = uint32_t(Egrp - t.E);
            m = t.m;
            if (delta >= Wc_) {
              if (m) sticky_mul_ = true;
              m = 0;
            } else if (delta) {
              const uint32_t mask = (1u << delta) - 1u;
              if (m & mask) sticky_mul_ = true;
              m >>= delta;
            }
            LOG("[mul-csa] elem delta=%u, m24=0x%x", (unsigned)delta, m);
          } else {
            LOG("[mul-csa] elem delta=%u, m24=0x%x", 0u, 0u);
          }

          // signed 25-bit addend (two's complement)
          int32_t a_signed = (t.sign ? -int32_t(m) : int32_t(m));
          if (a_signed < 0) ++negs;
          a_tc = uint32_t(a_signed) & maskW;
          LOG(", a_tc=0x%x\n", a_tc);
        } else {
          // padding beyond 'end' inside final (possibly partial) group
          LOG("[mul-csa] elem delta=%u, m24=0x0, a_tc=0x0\n", 0u);
        }

        addends[slot] = a_tc;
      }

      // Fold the whole group's addends with the vector CSA
      auto [S, C] = carry_save_add(addends, W);
      groups[g] = prod_t{S, C, Egrp, negs};

      LOG("[mul-csa] base=%zu, S=0x%x, C=0x%x, Egrp=%d\n", base, S, C, Egrp);
    }

    LOG("[multiply] groups_csa=%zu, sticky_mul=%d\n", groups.size(), sticky_mul_ ? 1 : 0);
    return groups;
  }

  // -------- alignment: finish CSA with signed CPA, then global align vs C ----
  AlignOut alignment(const std::vector<prod_t> &groups, const term24_t &cterm) {
    // 1) Finish each group to term24_t (signed-correct CPA in 64b, local normalize to Wc_)
    const size_t G = groups.size();
    std::vector<term24_t> reduced(G);
    bool sticky_local = false;
    const uint32_t maskWc = (Wc_ >= 32) ? 0xFFFFFFFFu : ((1u << Wc_) - 1u);

    for (size_t g = 0; g < G; ++g) {
      const auto &cg  = groups[g];
      int32_t  Egrp   = cg.E;

      // CPA of (S + (C<<1))
      auto [sumW, cout] = carry_propagate_add(cg.S, cg.C, Win_);

      // Signed correction for # of negative addends compressed into CSA
      const int64_t signed64 = int64_t(uint64_t(sumW) | (uint64_t(cout) << Win_))
                             - (int64_t(cg.negs) << Win_);

      // Normalize to Wc_ (track sticky on shifted-out bits)
      uint64_t mag = (signed64 < 0) ? (uint64_t)(-signed64) : (uint64_t)signed64;
      while (mag >> Wc_) {
        if (mag & 1ull) sticky_local = true;
        mag >>= 1;
        ++Egrp;
      }
      const uint32_t m_out = (uint32_t)(mag & maskWc);
      const bool is_zero = (m_out == 0);
      const uint32_t sgn = (signed64 < 0) ? 1u : 0u;

      LOG("[align-cpa] grp=%zu, S=0x%x, C=0x%x, negs=%u, sumW=0x%x, cout=%u, s=%u, E=%d, m=0x%x\n",
          g, cg.S, cg.C, cg.negs, sumW, cout, sgn, Egrp, m_out);

      reduced[g] = term24_t{sgn, m_out, Egrp, is_zero, false, false};
    }

    // 2) Global align the reduced terms + C into addends
    AlignOut out{};
    int32_t max_exp = INT32_MIN;

    auto note = [&](const term24_t &t) {
      if (!t.is_zero && !t.is_inf && !t.is_nan)
        if (t.E > max_exp) max_exp = t.E;
    };
    for (const auto &p : reduced) note(p);
    note(cterm);

    if (max_exp == INT32_MIN) {
      out.max_exp = 0;
      LOG("[alignment] all_zero=1\n");
      return out;
    }
    out.max_exp = max_exp;

    auto align_one = [&](const term24_t &t, const char *tag, size_t idx) -> int32_t {
      if (t.is_zero || t.is_inf || t.is_nan) {
        LOG("[align-%s] idx=%zu, zero_or_special=1\n", tag, idx);
        return 0;
      }
      uint32_t m = t.m;
      const uint32_t delta = uint32_t(max_exp - t.E);
      const uint8_t  sh8   = (delta > 255u) ? 255u : uint8_t(delta);
      if (delta >= Wc_) {
        if (m) { out.sticky_align = true; }
        m = 0;
      } else if (delta) {
        const uint32_t mask = (1u << delta) - 1u;
        if (m & mask) out.sticky_align = true;
        m >>= delta;
      }
      int32_t v = int32_t(m);
      if (t.sign) v = -v;
      LOG("[align-%s] idx=%zu, delta=%u, sh8=%u, m_adj=0x%x, signed=%d\n",
          tag, idx, (unsigned)delta, sh8, m, v);
      return v;
    };

    out.addends.resize(reduced.size() + 1);
    for (size_t i = 0; i < reduced.size(); ++i) out.addends[i] = align_one(reduced[i], "p", i);
    out.addends.back() = align_one(cterm, "c", 0);

    if (sticky_local) { out.sticky_align = true; sticky_align_ = true; }

    LOG("[alignment] max_exp=%d, sticky=%d, addends=%zu\n",
        out.max_exp, out.sticky_align ? 1 : 0, out.addends.size());
    return out;
  }

  // ---------------------------- accumulate --------------------------
  int64_t accumulate(const AlignOut &aout) {
    int64_t acc = 0;
    for (size_t i = 0; i < aout.addends.size(); ++i) {
       const auto &v = aout.addends[i];
       acc += int64_t(v);
       LOG("[acc-chunk] idx=%zu, addend=0x%x, acc=0x%lx\n", i, v, acc);
    }
    LOG("[accumulate] acc=0x%lx\n", acc);
    return acc;
  }

  // -------------------- Special/zero fast finalize helper --------------------
  uint32_t finalize_special_or_zero(int64_t acc, bool sticky_pre) {
    if (has_nv_ ||
        (has_pos_inf_ && has_neg_inf_) ||
        (c_is_inf_ && ((has_pos_inf_ && c_sign_ == 1u) || (has_neg_inf_ && c_sign_ == 0u)))) {
      fflags_ |= FLAG_NV;
      LOG("[final-fast] invalid=1, out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_nan_ || c_is_nan_) {
      LOG("[final-fast] has_NaN=1, out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_pos_inf_ || has_neg_inf_) {
      const uint32_t s = has_neg_inf_ ? 1u : 0u;
      return packInf32(s);
    }
    if (c_is_inf_) {
      return packInf32(c_sign_);
    }
    if (acc == 0) {
      if (sticky_pre) fflags_ |= (FLAG_NX | FLAG_UF);
      LOG("[final-fast] zero=1, fflags=0x%02x\n", fflags_);
      return packZero32(0);
    }
    return 0;
  }

  // ---------------------------- normalize ---------------------------
  struct Norm normalize(int64_t acc, int32_t max_exp) {
    Norm n{};
    n.sign = (acc < 0) ? 1u : 0u;
    uint64_t mag = (acc < 0) ? uint64_t(-acc) : uint64_t(acc);

    const uint32_t nbits = 64u - uint32_t(clz64(mag));
    n.e_unb = (max_exp - int32_t(Wc_ - 1u)) + int32_t(nbits - 1u);

    const int FP_TOP = 23;
    const int sh = int(nbits - 1u) - FP_TOP;

    uint32_t kept24 = 0, round_bit = 0;
    bool sticky_norm = false;
    if (sh > 0) {
      const uint64_t mask = (sh >= 64) ? ~0ull : ((1ull << sh) - 1ull);
      const uint64_t rem  = mag & mask;
      round_bit   = (sh >= 1) ? ((rem >> (sh - 1)) & 1ull) : 0u;
      sticky_norm = (sh >= 2) ? ((rem & ((1ull << (sh - 1)) - 1ull)) != 0ull) : false;
      kept24 = (sh >= 64) ? 0u : uint32_t(mag >> sh);
    } else {
      const int lsh = -sh;
      kept24 = (lsh >= 32) ? 0u : uint32_t(mag << lsh);
    }
    kept24 &= ((1u << 24) - 1u);

    n.kept24     = kept24;
    n.round_bit  = round_bit;
    n.sticky_any = sticky_norm || sticky_align_ || sticky_c32_ || sticky_mul_;

    LOG("[normalize] sign=%u, kept24=0x%x, e_unb=%d, round_bit=%u, sticky_any=%d\n",
        n.sign, n.kept24, n.e_unb, n.round_bit, n.sticky_any ? 1 : 0);
    return n;
  }

  // ----------------------------- rounding ---------------------------
  uint32_t round_and_pack(const Norm &nrm) {
    if (has_nv_ ||
        (has_pos_inf_ && has_neg_inf_) ||
        (c_is_inf_ && ((has_pos_inf_ && c_sign_ == 1u) || (has_neg_inf_ && c_sign_ == 0u)))) {
      fflags_ |= FLAG_NV;
      LOG("[rounding] out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_nan_ || c_is_nan_) {
      LOG("[rounding] has_NaN=1, out=qNaN\n");
      return canonicalNaN32();
    }
    if (has_pos_inf_ || has_neg_inf_) {
      const uint32_t s = has_neg_inf_ ? 1u : 0u;
      return packInf32(s);
    }
    if (c_is_inf_) {
      return packInf32(c_sign_);
    }

    uint32_t kept24 = nrm.kept24;
    int32_t  e_unb  = nrm.e_unb;
    const uint32_t sign = nrm.sign;

    if (nrm.round_bit || nrm.sticky_any) fflags_ |= FLAG_NX;

    const uint32_t lsb = kept24 & 1u;
    if (roundInc(sign, lsb, nrm.round_bit, nrm.sticky_any)) {
      kept24 += 1u;
      if (kept24 >= (1u << 24)) {
        kept24 >>= 1;
        e_unb += 1;
      }
    }

    const int32_t e_bias = e_unb + 127;

    if (e_bias >= 0xFF) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      LOG("[rounding] out=Inf\n");
      return packInf32(sign);
    }

    if (e_bias <= 0) {
      const int sh2 = 1 - e_bias;
      const uint32_t shifted = (sh2 >= 32) ? 0u : (kept24 >> sh2);
      const uint32_t mask2   = (sh2 >= 32) ? kept24 : ((1u << sh2) - 1u);
      const uint32_t rem2    = kept24 & mask2;
      const uint32_t rb2     = (sh2 >= 1) ? ((rem2 >> (sh2 - 1)) & 1u) : 0u;
      const bool st2         = (sh2 >= 2) ? ((rem2 & ((1u << (sh2 - 1)) - 1u)) != 0u) : false;

      uint32_t frac_keep = shifted & ((1u << 23) - 1u);
      const uint32_t lsb2 = frac_keep & 1u;

      if (rb2 || st2) fflags_ |= FLAG_NX;
      if (roundInc(sign, lsb2, rb2, st2)) {
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u << 23)) {
          if (fflags_ & FLAG_NX) fflags_ |= FLAG_UF;
          LOG("[rounding] subnormal_to_min_normal=1\n");
          return (sign << 31) | (1u << 23);
        }
        frac_keep = t;
      }
      if (fflags_ & FLAG_NX) fflags_ |= FLAG_UF;
      const uint32_t out_sub = (sign << 31) | frac_keep;
      LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out_sub, fflags_);
      return out_sub;
    }

    const uint32_t exp_out = uint32_t(e_bias);
    const uint32_t frac_out = kept24 & ((1u << 23) - 1u);
    const uint32_t out_norm = (sign << 31) | (exp_out << 23) | frac_out;
    LOG("[rounding] normal_out=0x%x, fflags=0x%02x\n", out_norm, fflags_);
    return out_norm;
  }

  bool roundInc(uint32_t sign, uint32_t lsb, uint32_t round_bit, bool sticky) const {
    switch (frm_) {
      case RNE: return round_bit ? (sticky || (lsb & 1u)) : false;
      case RTZ: return false;
      case RDN: return (round_bit || sticky) && (sign == 1);
      case RUP: return (round_bit || sticky) && (sign == 0);
      case RMM: return (round_bit || sticky);
      default:  return false;
    }
  }

  void resetFlags() {
    fflags_ = 0;
    has_nan_ = has_pos_inf_ = has_neg_inf_ = has_nv_ = false;
    c_is_nan_ = c_is_inf_ = false;
    c_sign_ = 0;
    sticky_c32_ = sticky_align_ = sticky_mul_ = false;
    has_any_special_ = false;
  }

  // ------------------------------- Utilities ---------------------------------
  static inline uint32_t bitsFromF32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
  static inline float f32FromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

  static inline int clz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#else
    if (!x) return 64;
    int n = 0; while (!(x & (1ull << 63))) { x <<= 1; ++n; } return n;
#endif
  }
  static inline uint32_t ceil_log2(uint32_t x) {
    if (x <= 1) return 0;
#if defined(__GNUC__) || defined(__clang__)
    return 32u - uint32_t(__builtin_clz(x - 1u));
#else
    uint32_t v = x - 1u, n = 0; while (v) { v >>= 1; ++n; } return n;
#endif
  }
  static inline uint32_t packZero32(uint32_t s) { return s << 31; }
  static inline uint32_t packInf32(uint32_t s)  { return (s << 31) | (0xFFu << 23); }
  static inline uint32_t canonicalNaN32()       { return (0xFFu << 23) | (1u << 22); }

  // ------------------------------ Members ------------------------------------
  // Config (TF32 superformat)
  uint32_t Wc_{24}, Win_{25}, Wacc_{29};
  int frm_{0};
  uint32_t lanes_{1};

  // State / flags
  uint32_t fflags_{0};
  bool has_nan_{false}, has_pos_inf_{false}, has_neg_inf_{false}, has_nv_{false};
  bool c_is_nan_{false}, c_is_inf_{false};
  uint32_t c_sign_{0};
  bool sticky_c32_{false}, sticky_align_{false}, sticky_mul_{false};
  bool has_any_special_{false};
};
