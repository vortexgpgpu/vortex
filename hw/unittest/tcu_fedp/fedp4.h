// FEDP.hpp — RTL-faithful dot product (TF32 superformat)

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
      LOG("[final-fast] zero=1 fflags=0x%02x\n", fflags_);
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
    uint32_t sign{0}; // sign bit
    uint32_t frac{0}; // fraction bits
    uint32_t exp{0}; // exponent bits
    bool is_zero{false}; // is zero
    bool is_sub{false}; // is subnormal
    bool is_inf{false}; // is infinity
    bool is_nan{false}; // is NaN
  };

  struct term24_t {
    int64_t S{0};
    uint64_t C{0};
    int32_t E{0};
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
        LOG("[decode] idx=%u, A(enc=0x%x, s=%u,e=%u,f=0x%x), B(enc=0x%x, s=%u,e=%u,f=0x%x)\n",
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

    const int32_t Ec = decoded.is_sub ? (1 - bias) : (e - bias);
    const uint32_t M = decoded.is_sub ? f : ((1u << sig_bits) | f);

    // Scale mantissa to Wc_-1
    const int dM = int(Wc_ - 1u) - sig_bits;
    assert(dM < 32);
    uint32_t m_c = M << dM;
    const int64_t addend = s ? -int64_t(m_c) : int64_t(m_c);
    LOG("[decodeC] s=%u Ec=%d m=0x%x -> add=0x%llx\n", s, Ec, m_c, (long long)addend);

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
        LOG("[mul-prod] i=%zu special=Inf/NaN/0*Inf\n", i);
        continue;
      }
      if (a.is_zero || b.is_zero) {
        LOG("[mul-prod] i=%zu zero=1\n", i);
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
    const uint32_t bias  = (1u << (exp_bits - 1)) - 1u;
    const uint32_t Wm_in = uint32_t(sig_bits) + 1u;  // input mantissa width
    const uint32_t Wraw  = 2u * Wm_in;               // raw product width
    assert(Wraw < Wc_);                              // must fit into Wc grid
    const uint32_t L_in  = Wc_ - Wraw;               // shift up to Wc grid

    struct Traw {
      uint32_t sign;
      uint32_t m_wc;
      int32_t  E;
      bool is_zero;
      bool is_inf;
      bool is_nan;
    };

    const size_t N = terms.size();
    std::vector<Traw> v(N);

    // Phase A: raw multiply and shift to Wc grid
    for (size_t i = 0; i < N; ++i) {
      const dec_t &a = terms[i][0];
      const dec_t &b = terms[i][1];

      const uint32_t s = a.sign ^ b.sign;

      const uint32_t Ea = a.is_sub ? (1 - bias) : (a.exp - bias);
      const uint32_t Eb = b.is_sub ? (1 - bias) : (b.exp - bias);
      const int32_t  E  = Ea + Eb + 1;

      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sig_bits) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sig_bits) | b.frac);
      const uint32_t P  = Ma * Mb; // Wraw bits

      // Shift up to Wc bits (no loss)
      const uint32_t m_wc = P << L_in;
      assert(L_in < 32);

      v[i] = Traw{s, m_wc, E, false, false, false};
      LOG("[mul-prod] i=%zu s=%u E=%d P=0x%x Wraw=%u\n", i, s, E, P, Wraw);
    }

    // Phase B: n_groups size (fp16:1, fp8:2, fp4:4)
    const uint32_t n_groups = 16 / width; // max 16 values per word
    const size_t G = (N + n_groups - 1) / n_groups;
    std::vector<term24_t> out(G);
    bool sticky = false;

    for (size_t base = 0; base < N; base += n_groups) {
      const size_t g = base / n_groups;
      const size_t end = std::min(N, base + n_groups);

      // Find per-group max exponent Eg
      int32_t Eg = INT32_MIN;
      for (size_t i = base; i < end; ++i) {
        Eg = std::max(Eg, v[i].E);
      }

      term24_t acc{0, 0, Eg, false, false, false};

      for (size_t i = base; i < end; ++i) {
        const auto &t = v[i];

        // Align this term to Eg in magnitude domain (no CPA): right shift magnitude
        const uint32_t delta = uint32_t(Eg - t.E);
        bool sticky_local;
        uint32_t m_shifted;
        if (delta > Wc_) {
          sticky_local = (t.m_wc != 0);
          m_shifted = 0;
        } else {
          const uint32_t mask = (1u << delta) - 1u;
          sticky_local = (t.m_wc & mask) != 0;
          m_shifted = (t.m_wc >> delta);
        }

        int64_t addend = t.sign ? -int64_t(m_shifted) : int64_t(m_shifted);

        // per-group CSA accumulation (in Wc grid)
        auto [s1, c1] = csa32(acc.S, (acc.C << 1u), addend);
        acc.S = s1;
        acc.C = c1;

        sticky |= sticky_local;

        LOG("[s1-csa] g=%zu i=%zu s=%u delta=%u m_adj=0x%x add=0x%llx S=0x%llx C=0x%llx\n",
            g, i, t.sign, (unsigned)delta, m_shifted,
            (long long)addend, (long long)acc.S, (long long)acc.C);
      }

      // Mark zero if both rails are zero
      acc.is_zero = (acc.S == 0 && acc.C == 0);
      out[g] = acc;
      LOG("[s1-csa] g=%zu Eg=%d S=0x%llx C=0x%llx sticky=%d zero=%d\n",
          g, Eg, (long long)acc.S, (long long)acc.C, (sticky ? 1 : 0), acc.is_zero ? 1 : 0);
    }

    LOG("[multiply] groups=%zu\n", out.size());
    return std::tuple{out, sticky};
  }

  // ---------------------------- S2: Align CSA terms -------------------------
  std::tuple<std::vector<term24_t>, int32_t, bool>
  alignment(const std::vector<term24_t> &groups, const term24_t &cterm) {
    // Find Emax
    int32_t Emax = INT32_MIN;
    for (auto &g : groups) {
      Emax = std::max(Emax, g.E);
    }
    Emax = std::max(Emax, cterm.E);

    std::vector<term24_t> out;
    out.reserve(groups.size() + 1);

    bool sticky = false;

    auto align_one = [&](const term24_t &t, const char *tag, size_t idx) {
      const uint32_t delta = uint32_t(Emax - t.E);
      // align by dividing the full value V = S + (C<<1) by 2^delta
      // using truncate-toward-zero semantics; sticky comes from dropped bits.
      uint64_t Su = static_cast<uint64_t>(t.S);
      uint64_t Bu = (t.C << 1);
      int64_t V = Su + Bu;
      bool local_sticky = false;

      if (delta > 0) {
        if (delta >= 63) {
          if (V != 0) {
            local_sticky = true;
          }
          V = 0;
        } else {
          uint64_t absV = (V < 0) ? uint64_t(-V) : uint64_t(V);
          uint64_t mask = (uint64_t(1) << delta) - 1ull;
          if (absV & mask) {
            local_sticky = true;
          }
          // truncate toward zero
          V = (V >= 0) ? (V >> delta) : -(int64_t(((-V) >> delta)));
        }
      }

      sticky |= local_sticky;

      term24_t a;
      a.S = V;
      a.C = 0;
      a.E = Emax;
      a.is_zero = (V == 0);
      a.is_inf = false;
      a.is_nan = false;

      LOG("[align-%s] idx=%zu delta=%u S'=0x%llx C'=0x%llx sticky+=%d\n",
          tag, idx, (unsigned)delta, (long long)a.S, (long long)a.C, (local_sticky ? 1 : 0));
      out.push_back(a);
    };

    for (size_t i = 0; i < groups.size(); ++i) {
      align_one(groups[i], "p", i);
    }
    align_one(cterm, "c", 0);

    LOG("[alignment] Emax=%d sticky_align=%d terms=%zu\n", Emax, (sticky ? 1 : 0), out.size());
    return std::tuple{out, Emax, sticky};
  }

  // ------------------------- S3: CSA Accumulate + CPA -----------------------
  int64_t accumulate(const std::vector<term24_t> &aligned) {
    term24_t acc{0, 0, 0, true, false, false};
    acc.E = aligned[0].E;
    acc.is_zero = false;

    auto add_pair_into = [&](term24_t &dst, const term24_t &src) {
      // Fold S component
      {
        const uint64_t A = static_cast<uint64_t>(dst.S);
        const uint64_t B = (dst.C << 1);
        const uint64_t X = static_cast<uint64_t>(src.S);
        auto [s1, c1] = csa32(A, B, X);
        dst.S = static_cast<int64_t>(s1);
        dst.C = c1;
      }
      // Fold C component (shifted by 1 in weight)
      {
        const uint64_t A = static_cast<uint64_t>(dst.S);
        const uint64_t B = (dst.C << 1);
        const uint64_t X = (src.C << 1);
        auto [s1, c1] = csa32(A, B, X);
        dst.S = static_cast<int64_t>(s1);
        dst.C = c1;
      }
    };

    for (size_t i = 0; i < aligned.size(); ++i) {
      add_pair_into(acc, aligned[i]);
      LOG("[acc-csa] i=%zu S=0x%llx C=0x%llx\n", i, (long long)acc.S, (long long)acc.C);
    }

    // Single CPA here: V = S + (C<<1) with well-defined wrap via unsigned
    const uint64_t Su = static_cast<uint64_t>(acc.S);
    const uint64_t B = (acc.C << 1);
    int64_t V = Su + B;
    return V;
  }

  // ------------------------------ S4: Normalize -----------------------------
  norm_t normalize(int64_t acc, int32_t Emax, bool sticky_prev) {
    norm_t n{};
    n.sign = (acc < 0) ? 1u : 0u;
    uint64_t mag = (acc < 0) ? uint64_t(-acc) : uint64_t(acc);

    const uint32_t nbits = (mag == 0) ? 1u : (64u - uint32_t(clz64(mag))); // >=1
    // Our fixed-point point was at bit (Wc_-1) relative to exponent; now the leading 1 is at nbits-1
    n.e_unb = (Emax - int32_t(Wc_ - 1u)) + int32_t(nbits - 1u);

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
    n.sticky = (sticky_norm || sticky_prev);

    LOG("[normalize] sign=%u kept24=0x%x e_unb=%d round=%u stickyAny=%d\n",
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
      LOG("[rounding] subnormal_out=0x%x fflags=0x%02x\n", out_sub, fflags_);
      return out_sub;
    }

    const uint32_t exp_out  = uint32_t(e_bias);
    const uint32_t frac_out = kept24 & ((1u << 23) - 1u);
    const uint32_t out_norm = (sign << 31) | (exp_out << 23) | frac_out;
    LOG("[rounding] normal_out=0x%x fflags=0x%02x\n", out_norm, fflags_);
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
  static inline std::pair<uint64_t, uint64_t> csa32(uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t t = (a ^ b);
    uint64_t sum = t ^ c;
    uint64_t carry = ((a & b) | (b & c) | (a & c));
    return std::pair{sum, carry};
  }

  // Safe arithmetic right shift for signed 64-bit (handles large shifts)
  static inline int64_t asr64(int64_t v, uint32_t sh) {
    if (sh == 0)
      return v;
    if (sh >= 63)
      return (v < 0) ? int64_t(-1) : int64_t(0);
    return v >> sh;
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
