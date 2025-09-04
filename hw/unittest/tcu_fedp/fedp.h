// FEDP.hpp — RTL-faithful dot product (TF32 superformat) with tracing.
//
// Build with -DFEDP_TRACE=1 to enable logs.

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
  // lanes: product lanes per accumulation "cycle" (1..8). Fan-in per first chunk is (lanes + 1) due to C.
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    if (frm_ < 0 || frm_ > 4)
      frm_ = 0; // default RNE
    if (lanes_ < 1)
      lanes_ = 1;
    if (lanes_ > 8)
      lanes_ = 8;

    // Superformat is TF32 (e8m10): common product magnitude width
    Wc_ = 24u;       // 24-bit magnitude grid for products (s_prod)
    Win_ = Wc_ + 1u; // signed addend width (for alignment/accumulate)
    Wacc_ = Win_ + ceil_log2(lanes_ + 1) + 1u; // worst-case accumulator width

    LOG("[ctor] frm=%d, lanes=%u, super=TF32, e8m10, Wc=%u, Win=%u, Wacc=%u\n",
        frm_, lanes_, Wc_, Win_, Wacc_);
  }

  // Top-level: a_words/b_words contain n_words packed values with (exp_bits,sig_bits)
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

    // decode
    terms_.assign(k, {});
    decodeInputs(a_words, b_words, n_words, elems_per_word, width, exp_bits, sig_bits, packed);

    // multiply (and fp8/fp4 reduction)
    prods_.clear();
    multiply_to_common(sig_bits, width);

    // decode C
    const term24_t cterm = decodeC_to_common(bitsFromF32(c));

    // alignment (and max_exp)
    const AlignOut aout = alignment(prods_, cterm);

    // accumulate (honor lanes)
    const int64_t acc = accumulate(aout);

    // fast finalize specials/zero
    if (has_any_special_ || acc == 0) {
      if (const uint32_t out_fast = finalize_special_or_zero(acc, (sticky_mul_ || sticky_align_ || sticky_c32_))) {
        return f32FromBits(out_fast);
      }
    }

    // 6) normalize
    const Norm nrm = normalize(acc, aout.max_exp);

    // 7) rounding + pack
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  // fflags accessor (placed after operator() as requested)
  uint32_t fflags() const { return fflags_; }

private:
  // -------------------------------- Types ------------------------------------
  struct dec_t {
    uint32_t sign{}, frac{}, exp_field{};
    int32_t exp_unb{};
    bool is_zero{}, is_sub{}, is_inf{}, is_nan{};
  };

  // Common product term: value = m * 2^(E - (Wc_-1))
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
    uint32_t kept24{0}; // 24-bit core (1+23) for FP32
    int32_t e_unb{0};
    uint32_t round_bit{0};
    bool sticky_any{false};
  };

  enum { RNE = 0,
         RTZ = 1,
         RDN = 2,
         RUP = 3,
         RMM = 4 };
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // -------------------------- decodeInputs --------------------------
  void decodeInputs(const std::vector<uint32_t> &a_words,
                    const std::vector<uint32_t> &b_words,
                    uint32_t n_words,
                    uint32_t elems_per_word,
                    uint32_t width,
                    int exp_bits, int sig_bits,
                    bool packed) {
    const uint32_t enc_mask = (width >= 32) ? 0xFFFFFFFFu : ((1u << width) - 1u);
    const uint32_t frac_mask = (sig_bits >= 32) ? 0xFFFFFFFFu : ((1u << sig_bits) - 1u);
    const uint32_t exp_mask = (exp_bits >= 32) ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u);
    const uint32_t bias = (1u << (exp_bits - 1)) - 1u;

    uint32_t out = 0;
    for (uint32_t i = 0; i < n_words; ++i) {
      uint32_t aw = a_words[i], bw = b_words[i];
      for (uint32_t j = 0; j < elems_per_word; ++j, ++out) {
        const uint32_t aenc = packed ? (aw & enc_mask) : aw;
        const uint32_t benc = packed ? (bw & enc_mask) : bw;

        terms_[out][0] = decode_one(aenc, exp_bits, sig_bits, exp_mask, frac_mask, bias);
        terms_[out][1] = decode_one(benc, exp_bits, sig_bits, exp_mask, frac_mask, bias);

        if (packed) {
          aw >>= width;
          bw >>= width;
        }

        LOG("[decode] idx=%u, A(enc=0x%x, s=%u,e=%u,f=0x%x), B(enc=0x%x, s=%u,e=%u,f=0x%x)\n",
            out,
            aenc, terms_[out][0].sign, terms_[out][0].exp_field, terms_[out][0].frac,
            benc, terms_[out][1].sign, terms_[out][1].exp_field, terms_[out][1].frac);
      }
    }
    LOG("[decodeInputs] decoded=%u\n", out);
  }

  static inline dec_t decode_one(uint32_t enc, int exp_bits, int sig_bits,
                                 uint32_t exp_mask, uint32_t frac_mask, uint32_t bias) {
    const uint32_t sign = (enc >> (exp_bits + sig_bits)) & 1u;
    const uint32_t exp = (enc >> sig_bits) & exp_mask;
    const uint32_t frac = enc & frac_mask;

    dec_t d{};
    d.sign = sign;
    d.frac = frac;
    d.exp_field = exp;
    d.is_zero = (exp == 0 && frac == 0);
    d.is_sub = (exp == 0 && frac != 0);
    d.is_inf = (exp == exp_mask && frac == 0);
    d.is_nan = (exp == exp_mask && frac != 0);
    d.exp_unb = (exp == 0) ? (1 - int32_t(bias))
                           : (exp == exp_mask ? 0 : int32_t(exp) - int32_t(bias));
    return d;
  }

  // ----------------------------- multiply ---------------------------
  // Three phases: raw multiply -> shift each product to 24-bit common grid -> group reduce in 24b
  void multiply_to_common(int sig_bits_in, uint32_t width_in) {
    has_any_special_ = false;
    sticky_mul_ = false;

    struct Raw {
      uint32_t sign;
      uint32_t P;
      int32_t E;
      bool is_zero, is_inf, is_nan;
    };

    const uint32_t Wm_in = uint32_t(sig_bits_in) + 1u; // input mantissa width
    const uint32_t Wraw_in = 2u * Wm_in;               // raw product width
    const uint32_t L_in = Wc_ - Wraw_in;               // shift to 24-bit common grid

    // Phase A: raw multiply (no shift-up)
    std::vector<Raw> raw;
    raw.reserve(terms_.size());

    for (size_t i = 0; i < terms_.size(); ++i) {
      const dec_t &a = terms_[i][0];
      const dec_t &b = terms_[i][1];

      if (a.is_nan || b.is_nan)
        has_nan_ = true;
      if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero))
        has_nv_ = true;

      if (a.is_inf || b.is_inf) {
        const uint32_t s = a.sign ^ b.sign;
        if (s)
          has_neg_inf_ = true;
        else
          has_pos_inf_ = true;
        raw.push_back(Raw{0, 0, 0, true, true, false});
        has_any_special_ = true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
        continue;
      }
      if (a.is_zero || b.is_zero) {
        raw.push_back(Raw{0, 0, 0, true, false, false});
        LOG("[mul-prod] i=%zu, zero=1\n", i);
        continue;
      }

      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sig_bits_in) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sig_bits_in) | b.frac);
      const uint32_t P = Ma * Mb; // width Wraw_in
      const int32_t E = (a.exp_unb + b.exp_unb) + 1;
      const uint32_t s = a.sign ^ b.sign;

      raw.push_back(Raw{s, P, E, false, false, false});
      LOG("[mul-prod] i=%zu, s=%u, E=%d, P=0x%x, Wraw_in=%u\n", i, s, E, P, Wraw_in);
    }

    // Phase B: shift each product up to the 24-bit common grid (exact, no loss)
    struct WC {
      uint32_t sign;
      uint32_t m;
      int32_t E;
      bool is_zero, is_inf, is_nan;
    };
    std::vector<WC> wc;
    wc.reserve(raw.size());
    LOG("[mul-sup] L_in=%u, Wc=%u, Wraw_in=%u\n", L_in, Wc_, Wraw_in);

    for (size_t i = 0; i < raw.size(); ++i) {
      const auto &r = raw[i];
      if (r.is_zero || r.is_inf || r.is_nan) {
        wc.push_back(WC{0, 0, 0, true, r.is_inf, r.is_nan});
        continue;
      }
      uint32_t m = (L_in >= 32) ? 0u : (r.P << L_in); // now exactly Wc_ bits
      wc.push_back(WC{r.sign, m, r.E, false, false, false});
      LOG("[mul-sup] i=%zu, m_wc=0x%x, E=%d\n", i, m, r.E);
    }

    // Phase C: group reduce in 24-bit domain (fp8:add2, fp4:add4, else passthrough)
    prods_.clear();
    uint32_t group = 1;
    if (width_in == 8u)
      group = 2;
    else if (width_in == 4u)
      group = 4;

    for (size_t base = 0; base < wc.size(); base += group) {
      const size_t end = std::min(wc.size(), base + group);

      int32_t Egrp = INT32_MIN;
      for (size_t i = base; i < end; ++i) {
        const auto &t = wc[i];
        if (!t.is_zero && !t.is_inf && !t.is_nan && t.E > Egrp)
          Egrp = t.E;
      }
      if (Egrp == INT32_MIN) {
        prods_.push_back(term24_t{0, 0, 0, true, false, false});
        LOG("[mul-align] base=%zu, size=%zu, zero_group=1\n", base, end - base);
        continue;
      }

      int64_t S = 0;
      for (size_t i = base; i < end; ++i) {
        const auto &t = wc[i];
        if (t.is_zero || t.is_inf || t.is_nan)
          continue;

        const uint32_t delta = uint32_t(Egrp - t.E);
        uint32_t m = t.m; // 24-bit
        if (delta >= Wc_) {
          if (m)
            sticky_mul_ = true;
          m = 0;
        } else if (delta) {
          const uint32_t mask = (1u << delta) - 1u;
          if (m & mask)
            sticky_mul_ = true;
          m >>= delta;
        }
        int64_t v = int64_t(m);
        if (t.sign)
          v = -v;
        S += v;

        LOG("[mul-align] i=%zu, delta=%u, m_adj=0x%x, signed=0x%lx\n", i, (unsigned)delta, m, v);
      }

      // Normalize S to Wc_ bits (drop LSBs -> sticky), bump Egrp if needed
      uint32_t sgn = (S < 0) ? 1u : 0u;
      uint64_t mag = (S < 0) ? uint64_t(-S) : uint64_t(S);
      while (mag >> Wc_) {
        if (mag & 1ull)
          sticky_mul_ = true;
        mag >>= 1;
        ++Egrp;
      }
      const uint32_t m_out = uint32_t(mag & ((1ull << Wc_) - 1ull));
      const bool is_zero = (m_out == 0);

      prods_.push_back(term24_t{sgn, m_out, Egrp, is_zero, false, false});
      LOG("[mul-align] base=%zu, size=%zu, s=%u, E=%d, m=0x%x, zero=%d\n",
          base, end - base, sgn, Egrp, m_out, is_zero ? 1 : 0);
    }

    LOG("[multiply] groups=%zu, sticky_mul=%d\n", prods_.size(), sticky_mul_ ? 1 : 0);
  }

  // ------------------------------ decodeC ---------------------------
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

  // ----------------------------- alignment --------------------------
  AlignOut alignment(const std::vector<term24_t> &ps, const term24_t &cterm) {
    AlignOut out{};
    int32_t max_exp = INT32_MIN;

    auto note = [&](const term24_t &t) {
      if (!t.is_zero && !t.is_inf && !t.is_nan)
        if (t.E > max_exp)
          max_exp = t.E;
    };
    for (auto &p : ps)
      note(p);
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
      const uint8_t sh8 = (delta > 255u) ? 255u : uint8_t(delta); // trace
      if (delta >= Wc_) {
        if (m)
          out.sticky_align = true;
        m = 0;
      } else if (delta) {
        const uint32_t mask = (1u << delta) - 1u;
        if (m & mask)
          out.sticky_align = true;
        m >>= delta;
      }
      int32_t v = int32_t(m);
      if (t.sign)
        v = -v;
      LOG("[align-%s] idx=%zu, delta=%u, sh8=%u, m_adj=0x%x, signed=%d\n",
          tag, idx, (unsigned)delta, sh8, m, v);
      return v;
    };

    out.addends.reserve(ps.size()+1);
    for (size_t i = 0; i < ps.size(); ++i)
      out.addends.push_back(align_one(ps[i], "p", i));
    out.addends.push_back(align_one(cterm, "c", 0));

    LOG("[alignment] max_exp=%d, sticky=%d, addends=%zu\n",
        out.max_exp, out.sticky_align ? 1 : 0, out.addends.size());
    return out;
  }

  // ---------------------------- accumulate --------------------------
  int64_t accumulate(const AlignOut &aout) {
    int64_t acc = 0;
    for (size_t i = 0; i < aout.addends.size(); ++i) {
      auto &v = aout.addends[i];
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
      if (sticky_pre)
        fflags_ |= (FLAG_NX | FLAG_UF);
      LOG("[final-fast] zero=1, fflags=0x%02x\n", fflags_);
      return packZero32(0);
    }
    return 0;
  }

  // ---------------------------- normalize ---------------------------
  Norm normalize(int64_t acc, int32_t max_exp) {
    Norm n{};
    n.sign = (acc < 0) ? 1u : 0u;
    uint64_t mag = (acc < 0) ? uint64_t(-acc) : uint64_t(acc);

    const uint32_t nbits = 64u - uint32_t(clz64(mag)); // >=1
    n.e_unb = (max_exp - int32_t(Wc_ - 1u)) + int32_t(nbits - 1u);

    // Normalize to FP32 24-bit core (1+23)
    const int FP_TOP = 23;
    const int sh = int(nbits - 1u) - FP_TOP;

    uint32_t kept24 = 0, round_bit = 0;
    bool sticky_norm = false;
    if (sh > 0) {
      const uint64_t mask = (sh >= 64) ? ~0ull : ((1ull << sh) - 1ull);
      const uint64_t rem = mag & mask;
      round_bit = (sh >= 1) ? ((rem >> (sh - 1)) & 1ull) : 0u;
      sticky_norm = (sh >= 2) ? ((rem & ((1ull << (sh - 1)) - 1ull)) != 0ull) : false;
      kept24 = (sh >= 64) ? 0u : uint32_t(mag >> sh);
    } else {
      const int lsh = -sh;
      kept24 = (lsh >= 32) ? 0u : uint32_t(mag << lsh);
    }
    kept24 &= ((1u << 24) - 1u);

    n.kept24 = kept24;
    n.round_bit = round_bit;
    n.sticky_any = sticky_norm || sticky_align_ || sticky_c32_ || sticky_mul_;

    LOG("[normalize] sign=%u, kept24=0x%x, e_unb=%d, round_bit=%u, sticky_any=%d\n",
        n.sign, n.kept24, n.e_unb, n.round_bit, n.sticky_any ? 1 : 0);
    return n;
  }

  // ----------------------------- rounding ---------------------------
  uint32_t round_and_pack(const Norm &nrm) {
    // specials again (in case early exit didn't take)
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
    int32_t e_unb = nrm.e_unb;
    const uint32_t sign = nrm.sign;

    if (nrm.round_bit || nrm.sticky_any)
      fflags_ |= FLAG_NX;

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
      const uint32_t mask2 = (sh2 >= 32) ? kept24 : ((1u << sh2) - 1u);
      const uint32_t rem2 = kept24 & mask2;
      const uint32_t rb2 = (sh2 >= 1) ? ((rem2 >> (sh2 - 1)) & 1u) : 0u;
      const bool st2 = (sh2 >= 2) ? ((rem2 & ((1u << (sh2 - 1)) - 1u)) != 0u) : false;

      uint32_t frac_keep = shifted & ((1u << 23) - 1u);
      const uint32_t lsb2 = frac_keep & 1u;

      if (rb2 || st2)
        fflags_ |= FLAG_NX;
      if (roundInc(sign, lsb2, rb2, st2)) {
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u << 23)) {
          if (fflags_ & FLAG_NX)
            fflags_ |= FLAG_UF;
          LOG("[rounding] subnormal_to_min_normal=1\n");
          return (sign << 31) | (1u << 23);
        }
        frac_keep = t;
      }
      if (fflags_ & FLAG_NX)
        fflags_ |= FLAG_UF;
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

  void resetFlags() {
    fflags_ = 0;
    has_nan_ = has_pos_inf_ = has_neg_inf_ = has_nv_ = false;
    c_is_nan_ = c_is_inf_ = false;
    c_sign_ = 0;
    sticky_c32_ = sticky_align_ = sticky_mul_ = false;
    has_any_special_ = false;
  }

  // ------------------------------- Utilities ---------------------------------
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

  static inline int clz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#else
    if (!x)
      return 64;
    int n = 0;
    while (!(x & (1ull << 63))) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }
  static inline uint32_t ceil_log2(uint32_t x) {
    if (x <= 1)
      return 0;
#if defined(__GNUC__) || defined(__clang__)
    return 32u - uint32_t(__builtin_clz(x - 1u));
#else
    uint32_t v = x - 1u, n = 0;
    while (v) {
      v >>= 1;
      ++n;
    }
    return n;
#endif
  }
  static inline uint32_t packZero32(uint32_t s) { return s << 31; }
  static inline uint32_t packInf32(uint32_t s) { return (s << 31) | (0xFFu << 23); }
  static inline uint32_t canonicalNaN32() { return (0xFFu << 23) | (1u << 22); }

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

  // Decoded inputs and reduced products
  std::vector<std::array<dec_t, 2>> terms_; // terms_[i][0] = a_i, terms_[i][1] = b_i
  std::vector<term24_t> prods_;             // reduced product groups in 24-bit common grid
};
