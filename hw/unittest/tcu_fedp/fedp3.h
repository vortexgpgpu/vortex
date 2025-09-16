// FEDP.hpp — CSA-based pipeline (SOTA) for fused dot product (TF32 superformat) with deep tracing.
// Pipeline:
//   S1: decode + multiply + group-reduce -> ***two separate CSAs: pos_cs and neg_cs***
//   S2: CPA(pos/neg) WIDE -> ***renorm each magnitude if it overflows 2^Wc (bump E)***
//       -> V = pos_mag - neg_mag (signed) -> arith-renorm of V if |V| ≥ 2^(Wc-1)
//       -> align (arith) to max_exp; emit Wacc-bit two’s-complement (sum=enc, carry=0)
//   S3: accumulate all aligned terms in Wacc-bit CSA
//   S4: single CPA (Wacc) -> sign-extend -> normalize -> round -> pack
//
// Build with: -DFEDP_TRACE=1   (enable verbose logs)

#include <algorithm>
#include <array>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
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
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    if (frm_ < 0 || frm_ > 4)
      frm_ = 0; // default RNE
    if (lanes_ < 1)
      lanes_ = 1;
    if (lanes_ > 8)
      lanes_ = 8;

    // TF32 (e8m10) common product grid
    Wc_ = 24u;       // CSA mantissa grid (bits)
    Win_ = Wc_ + 1u; // signed addend width (for +/-)
    // Headroom: allow up to ~64 inputs; add guard; cap to 63 (we use 64-bit).
    Wacc_ = Win_ + ceil_log2(64) + 4u; // ~35 in your runs
    if (Wacc_ > 63u)
      Wacc_ = 63u;

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

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%d, elems/word=%u, n_words=%u, k=%u\n",
        exp_bits, sig_bits, width, packed ? 1 : 0, elems_per_word, n_words, n_words * elems_per_word);

    // ---------------- S1: decode + multiply + group-reduce (pos/neg CSAs) -------------
    const auto terms = decodeInputs(a_words, b_words, n_words, elems_per_word, width,
                                    exp_bits, sig_bits, packed);
    const auto groups = s1_groups_to_csa_posneg(terms, sig_bits, width);

    // Scalar C as its own "group": pos/neg CSA selection by sign (still at Wc domain)
    const auto cgrp = decodeC_to_posneg_csa(bitsFromF32(c));

    // --------------------- S2: CPA(pos/neg) -> renorm(mags) -> V -> renorm(V) -> align ----------
    const auto aout = s2_cpa_posneg_then_align(groups, cgrp);
    sticky_align_ = aout.sticky_align; // visible to S4's sticky_any

    // --------------------- S3: accumulate in CSA (tree, Wacc) -----------------------
    const CS acc_csa = s3_accumulate_csa(aout); // Wacc-wide in 64-bit

    // --------------------- S4: finalize (CPA + NRZ + pack) --------------------------
    if (has_any_special_ || (acc_csa.sum == 0 && acc_csa.carry == 0)) {
      const bool sticky_pre = (sticky_mul_ || aout.sticky_align || sticky_c32_);
      if (const uint32_t out_fast = finalize_special_or_zero(0, sticky_pre)) {
        return f32FromBits(out_fast);
      }
    }

    const uint64_t maskWacc = lowbits_mask(Wacc_);
    const uint64_t cpa_sum = (acc_csa.sum + acc_csa.carry) & maskWacc;
    const int64_t acc_signed = sign_extend(cpa_sum, Wacc_);

    LOG("[CPA] Wacc=%u, sum=0x%llx, carry=0x%llx, cpa=0x%llx, signext=%lld\n",
        Wacc_,
        (unsigned long long)acc_csa.sum, (unsigned long long)acc_csa.carry,
        (unsigned long long)cpa_sum, (long long)acc_signed);

    const Norm nrm = normalize(acc_signed, aout.max_exp);
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  [[nodiscard]] uint32_t fflags() const { return fflags_; }

private:
  // -------------------------------- Types ---------------------------------
  struct dec_t {
    uint32_t sign{}, frac{}, exp_field{};
    int32_t exp_unb{};
    bool is_zero{}, is_sub{}, is_inf{}, is_nan{};
  };

  struct CS {
    uint64_t sum{0};
    uint64_t carry{0};
  };

  struct GroupPosNeg {
    CS pos_cs{}; // CSA of positive magnitudes aligned to E
    CS neg_cs{}; // CSA of negative magnitudes aligned to E
    int32_t E{0};
    bool is_zero{true}, is_inf{false}, is_nan{false};
    bool sticky_local{false};
  };

  struct AlignOut {
    std::vector<CS> aligned; // Wacc-bit two’s-complement patterns: sum=enc@Wacc, carry=0
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
  std::vector<std::array<dec_t, 2>>
  decodeInputs(const std::vector<uint32_t> &a_words,
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

    const uint32_t k = n_words * elems_per_word;
    std::vector<std::array<dec_t, 2>> terms(k);

    uint32_t out = 0;
    for (uint32_t i = 0; i < n_words; ++i) {
      uint32_t aw = a_words[i], bw = b_words[i];
      for (uint32_t j = 0; j < elems_per_word; ++j, ++out) {
        const uint32_t aenc = packed ? (aw & enc_mask) : aw;
        const uint32_t benc = packed ? (bw & enc_mask) : bw;

        terms[out][0] = decode_one(aenc, exp_bits, sig_bits, exp_mask, frac_mask, bias);
        terms[out][1] = decode_one(benc, exp_bits, sig_bits, exp_mask, frac_mask, bias);

        if (packed) {
          aw >>= width;
          bw >>= width;
        }

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

  // --------------------- S1: multiply + group reduce (pos/neg CSAs) -------------
  std::vector<GroupPosNeg>
  s1_groups_to_csa_posneg(const std::vector<std::array<dec_t, 2>> &terms,
                          int sig_bits_in, uint32_t width_in) {
    has_any_special_ = false;
    sticky_mul_ = false;

    struct Raw {
      uint32_t sign;
      uint32_t P;
      int32_t E;
      bool is_zero, is_inf, is_nan;
    };

    const uint32_t Wm_in = uint32_t(sig_bits_in) + 1u; // incl hidden 1 for normals
    const uint32_t Wraw_in = 2u * Wm_in;               // raw product width
    const uint32_t L_in = (Wc_ > Wraw_in) ? (Wc_ - Wraw_in) : 0u;

    const size_t N = terms.size();
    std::vector<Raw> raw(N);

    // Phase A: raw multiply (no alignment yet)
    for (size_t i = 0; i < N; ++i) {
      const dec_t &a = terms[i][0];
      const dec_t &b = terms[i][1];

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
        raw[i] = Raw{0, 0, 0, true, true, false};
        has_any_special_ = true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
        continue;
      }
      if (a.is_zero || b.is_zero) {
        raw[i] = Raw{0, 0, 0, true, false, false};
        LOG("[mul-prod] i=%zu, zero=1\n", i);
        continue;
      }

      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sig_bits_in) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sig_bits_in) | b.frac);
      const uint32_t P = Ma * Mb; // ≤ 2*Wm_in (fits 32b here)
      const int32_t E = (a.exp_unb + b.exp_unb) + 1;
      const uint32_t s = a.sign ^ b.sign;

      raw[i] = Raw{s, P, E, false, false, false};
      LOG("[mul-prod] i=%zu, s=%u, E=%d, P=0x%x, Wraw_in=%u\n", i, s, E, P, Wraw_in);
    }

    // Phase B: shift product to 24-bit grid (exact left shift; no loss)
    struct WC {
      uint32_t sign, m;
      int32_t E;
      bool is_zero, is_inf, is_nan;
    };
    std::vector<WC> wc(N);
    LOG("[mul-sup] L_in=%u, Wc=%u, Wraw_in=%u\n", L_in, Wc_, Wraw_in);

    for (size_t i = 0; i < N; ++i) {
      const auto &r = raw[i];
      if (r.is_zero || r.is_inf || r.is_nan) {
        wc[i] = WC{0, 0, 0, true, r.is_inf, r.is_nan};
        continue;
      }
      const uint32_t m = (L_in >= 32) ? 0u : (r.P << L_in); // Wc bits
      wc[i] = WC{r.sign, m, r.E, false, false, false};
      LOG("[mul-sup] i=%zu, m_wc=0x%x, E=%d\n", i, m, r.E);
    }

    // Phase C: group reduction by width (8→pairs, 4→quads, else passthrough)
    uint32_t group = 1;
    if (width_in == 8u)
      group = 2;
    else if (width_in == 4u)
      group = 4;

    const size_t G = (N + group - 1) / group;
    std::vector<GroupPosNeg> out(G);

    for (size_t base = 0, g = 0; base < N; base += group, ++g) {
      const size_t end = std::min(N, base + group);

      int32_t Egrp = INT32_MIN;
      for (size_t i = base; i < end; ++i) {
        const auto &t = wc[i];
        if (!t.is_zero && !t.is_inf && !t.is_nan && t.E > Egrp)
          Egrp = t.E;
      }
      if (Egrp == INT32_MIN) {
        out[g] = GroupPosNeg{CS{0, 0}, CS{0, 0}, 0, true, false, false, false};
        LOG("[mul-group] base=%zu size=%zu zero_group\n", base, end - base);
        continue;
      }

      bool sticky_local = false;
      std::vector<uint32_t> pos_mags;
      pos_mags.reserve(end - base);
      std::vector<uint32_t> neg_mags;
      neg_mags.reserve(end - base);

      for (size_t i = base; i < end; ++i) {
        const auto &t = wc[i];
        if (t.is_zero || t.is_inf || t.is_nan)
          continue;
        uint32_t m = t.m;
        const uint32_t delta = uint32_t(Egrp - t.E);
        if (delta >= Wc_) {
          if (m)
            sticky_local = true;
          m = 0;
        } else if (delta) {
          const uint32_t mask = (1u << delta) - 1u;
          if (m & mask)
            sticky_local = true;
          m >>= delta;
        }
        if (t.sign)
          neg_mags.push_back(m);
        else
          pos_mags.push_back(m);
      }

      CS pos{0, 0}, neg{0, 0};
      const uint64_t Wmask = lowbits_mask(Wc_);
      auto fold_many = [&](CS acc, const std::vector<uint32_t> &mags) {
        CS a = acc;
        size_t i = 0;
        while (i + 3 < mags.size()) {
          CS t = csa42(uint64_t(mags[i]) & Wmask,
                       uint64_t(mags[i + 1]) & Wmask,
                       uint64_t(mags[i + 2]) & Wmask,
                       uint64_t(mags[i + 3]) & Wmask);
          a = csa32_fold(a, t.sum, t.carry);
          i += 4;
        }
        for (; i + 1 < mags.size(); i += 2) {
          a = csa32_fold(a, uint64_t(mags[i]) & Wmask, uint64_t(mags[i + 1]) & Wmask);
        }
        if (i < mags.size())
          a = csa32_fold(a, uint64_t(mags[i]) & Wmask, 0ull);
        return a;
      };
      pos = fold_many(pos, pos_mags);
      neg = fold_many(neg, neg_mags);

      LOG("[s1-group-posneg] g=%zu, E=%d, pos(sum=0x%llx,car=0x%llx) neg(sum=0x%llx,car=0x%llx) sticky=%d\n",
          g, Egrp,
          (unsigned long long)pos.sum, (unsigned long long)pos.carry,
          (unsigned long long)neg.sum, (unsigned long long)neg.carry,
          sticky_local ? 1 : 0);

      sticky_mul_ |= sticky_local;
      out[g] = GroupPosNeg{pos, neg, Egrp, false, false, false, sticky_local};
    }

    LOG("[multiply] groups=%zu\n", out.size());
    return out;
  }

  // ------------------------------ decode C to pos/neg CSA -----------------------
  GroupPosNeg decodeC_to_posneg_csa(uint32_t enc32) {
    const uint32_t s = (enc32 >> 31) & 1u;
    const uint32_t e = (enc32 >> 23) & 0xFFu;
    a32_ = enc32; // for debugging if needed
    const uint32_t f = enc32 & 0x7FFFFFu;

    c_sign_ = s;
    c_is_inf_ = (e == 0xFFu && f == 0);
    c_is_nan_ = (e == 0xFFu && f != 0);

    if (e == 0 && f == 0) {
      LOG("[decodeC] zero=1\n");
      return GroupPosNeg{CS{0, 0}, CS{0, 0}, 0, true, false, false, false};
    }
    if (c_is_inf_ || c_is_nan_) {
      LOG("[decodeC] special=%s\n", c_is_inf_ ? "Inf" : "NaN");
      has_any_special_ = true;
      return GroupPosNeg{CS{0, 0}, CS{0, 0}, 0, true, c_is_inf_, c_is_nan_, false};
    }

    const bool is_sub = (e == 0 && f != 0);
    const int32_t Ec = is_sub ? (1 - 127) : (int32_t(e) - 127);
    const uint32_t M = is_sub ? f : ((1u << 23) | f); // 24b

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

    CS pos{0, 0}, neg{0, 0};
    const uint64_t Wmask = lowbits_mask(Wc_);
    if (s == 0) {
      pos = csa32_fold(pos, uint64_t(m_c) & Wmask, 0ull);
    } else {
      neg = csa32_fold(neg, uint64_t(m_c) & Wmask, 0ull);
    }
    LOG("[decodeC] s=%u, Ec=%d, m=0x%x, sticky_c32=%d, pos(sum=0x%llx,car=0x%llx) neg(sum=0x%llx,car=0x%llx)\n",
        s, Ec, m_c, sticky_c32_ ? 1 : 0,
        (unsigned long long)pos.sum, (unsigned long long)pos.carry,
        (unsigned long long)neg.sum, (unsigned long long)neg.carry);
    return GroupPosNeg{pos, neg, Ec, false, false, false, false};
  }

  // --------------- S2: CPA(pos/neg) WIDE -> renorm(mags) -> V=pos-neg -> renorm(V) -> align ------------
  AlignOut s2_cpa_posneg_then_align(const std::vector<GroupPosNeg> &groups,
                                    const GroupPosNeg &cgrp) {
    struct Tmp {
      bool special;
      int32_t E_in;
      int32_t E_after; // after renorm
      int64_t V;       // signed value after renorm
      bool sticky_renorm;
      char tag;
      size_t idx;
    };
    std::vector<Tmp> tmp;
    tmp.reserve(groups.size() + 1);

    auto cpa_posneg_renorm = [&](const GroupPosNeg &g, char tag, size_t idx) -> Tmp {
      if (g.is_zero || g.is_inf || g.is_nan) {
        LOG("[s2-pre] %c%zu special/zero: skip\n", tag, idx);
        return Tmp{true, g.E, g.E, 0, false, tag, idx};
      }
      const uint64_t twoWc = (Wc_ >= 63 ? ~0ull : (1ull << Wc_));
      const uint64_t maskWc = lowbits_mask(Wc_);

      // Wide CPA for pos/neg CSAs
      uint64_t pos_wide = g.pos_cs.sum + g.pos_cs.carry; // may exceed 2^Wc
      uint64_t neg_wide = g.neg_cs.sum + g.neg_cs.carry; // may exceed 2^Wc
      int32_t E = g.E;
      bool sticky_r = false;
      int ren_pos = 0, ren_neg = 0;

      // *** NEW: renormalize unsigned magnitudes BEFORE subtraction ***
      while (pos_wide >= twoWc) {
        sticky_r |= (pos_wide & 1ull) != 0ull;
        pos_wide >>= 1;
        ++E;
        ++ren_pos;
      }
      while (neg_wide >= twoWc) {
        sticky_r |= (neg_wide & 1ull) != 0ull;
        neg_wide >>= 1;
        ++E;
        ++ren_neg;
      }

      LOG("[s2-renorm-mags] %c%zu: pos_wide_pre=0x%llx neg_wide_pre=0x%llx -> pos_wide=0x%llx neg_wide=0x%llx E_bumps(+pos=%d,+neg=%d) E_now=%d\n",
          tag, idx,
          (unsigned long long)(g.pos_cs.sum + g.pos_cs.carry),
          (unsigned long long)(g.neg_cs.sum + g.neg_cs.carry),
          (unsigned long long)pos_wide, (unsigned long long)neg_wide,
          ren_pos, ren_neg, E);

      // Interpret as unsigned magnitudes in [0, 2^Wc-1] (guaranteed after renorm)
      const uint64_t pos_mag = pos_wide & maskWc;
      const uint64_t neg_mag = neg_wide & maskWc;

      // Now form signed difference at wide precision
      int64_t V = (int64_t)pos_mag - (int64_t)neg_mag;

      LOG("[s2-cpa-posneg] %c%zu: pos_mag=0x%llx neg_mag=0x%llx V_in=%lld\n",
          tag, idx, (unsigned long long)pos_mag, (unsigned long long)neg_mag, (long long)V);

      // Arithmetic renorm of V to keep |V| < 2^(Wc-1)
      const int64_t LIM = (int64_t(1) << (Wc_ - 1));
      int shifts = 0;
      while (V >= LIM || V < -LIM) {
        sticky_r |= (V & 1ll) != 0;
        V >>= 1;
        ++E;
        ++shifts;
      }
      LOG("[s2-renorm] %c%zu: E_in=%d, V_after=%lld, shifts=%d, sticky_r=%d\n",
          tag, idx, g.E, (long long)V, shifts, sticky_r ? 1 : 0);

      return Tmp{false, g.E, E, V, sticky_r, tag, idx};
    };

    for (size_t i = 0; i < groups.size(); ++i)
      tmp.push_back(cpa_posneg_renorm(groups[i], 'p', i));
    tmp.push_back(cpa_posneg_renorm(cgrp, 'c', 0));

    int32_t max_exp = INT32_MIN;
    for (const auto &t : tmp)
      if (!t.special && t.E_after > max_exp)
        max_exp = t.E_after;
    if (max_exp == INT32_MIN) {
      LOG("[alignment] all_zero_after_renorm=1\n");
      return AlignOut{};
    }
    LOG("[alignment] max_exp(after_renorm)=%d\n", max_exp);

    AlignOut out{};
    out.max_exp = max_exp;
    out.aligned.reserve(tmp.size());
    const uint64_t maskWacc = lowbits_mask(Wacc_);
    for (size_t i = 0; i < tmp.size(); ++i) {
      const auto &t = tmp[i];
      if (t.special) {
        out.aligned.push_back(CS{0, 0});
        continue;
      }
      int64_t V = t.V;
      const uint32_t delta = uint32_t(max_exp - t.E_after);
      if (delta > 0) {
        const uint64_t dropMask = (delta >= 64) ? ~0ull : ((1ull << delta) - 1ull);
        out.sticky_align |= ((uint64_t)V & dropMask) != 0ull;
        const int64_t before = V;
        V >>= std::min<uint32_t>(delta, 63u);
        LOG("[align-arith] %c%zu delta=%u dropMask=0x%llx dropped=0x%llx before=%lld after=%lld\n",
            t.tag, t.idx, (unsigned)delta,
            (unsigned long long)dropMask,
            (unsigned long long)((uint64_t)before & dropMask),
            (long long)before, (long long)V);
      } else {
        LOG("[align-arith] %c%zu delta=0 V=%lld\n", t.tag, t.idx, (long long)V);
      }
      const uint64_t enc = (uint64_t)V & maskWacc;
      out.aligned.push_back(CS{enc, 0});
    }
    return out;
  }

  // ------------------------------ S3: CSA accumulation (Wacc) -------------------
  CS s3_accumulate_csa(const AlignOut &aout) {
    std::vector<uint64_t> ops;
    ops.reserve(aout.aligned.size() * 2);
    for (const auto &x : aout.aligned) {
      ops.push_back(x.sum);
      ops.push_back(x.carry);
    }

    CS acc{0, 0};
    size_t i = 0;
    while (i + 3 < ops.size()) {
      CS t = csa42(ops[i], ops[i + 1], ops[i + 2], ops[i + 3]);
      acc = csa32_fold(acc, t.sum, t.carry);
      LOG("[acc-tree] i=%zu..%zu -> t(sum=0x%llx,car=0x%llx) acc(sum=0x%llx,car=0x%llx)\n",
          i, i + 3,
          (unsigned long long)t.sum, (unsigned long long)t.carry,
          (unsigned long long)acc.sum, (unsigned long long)acc.carry);
      i += 4;
    }
    for (; i + 1 < ops.size(); i += 2) {
      acc = csa32_fold(acc, ops[i], ops[i + 1]);
      LOG("[acc-fold2] i=%zu,%zu -> acc(sum=0x%llx,car=0x%llx)\n",
          i, i + 1, (unsigned long long)acc.sum, (unsigned long long)acc.carry);
    }
    if (i < ops.size()) {
      acc = csa32_fold(acc, ops[i], 0ull);
      LOG("[acc-fold1] i=%zu -> acc(sum=0x%llx,car=0x%llx)\n",
          i, (unsigned long long)acc.sum, (unsigned long long)acc.carry);
    }

    LOG("[accumulate-csa] sum=0x%llx carry=0x%llx (Wacc=%u)\n",
        (unsigned long long)acc.sum, (unsigned long long)acc.carry, Wacc_);
    return acc;
  }

  // -------------------- Special/zero fast finalize helper ----------------------
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

  // ---------------------------- S4: normalize ----------------------------

  Norm normalize(int64_t acc, int32_t max_exp) {
    Norm n{};
    n.sign = (acc < 0) ? 1u : 0u;
    uint64_t mag = (acc < 0) ? uint64_t(-acc) : uint64_t(acc);

    const uint32_t nbits = 64u - uint32_t(clz64(mag)); // >=1 if mag!=0
    n.e_unb = (max_exp - int32_t(Wc_ - 1u)) + int32_t(nbits - 1u);

    const int FP_TOP = 23;
    const int sh = int(nbits - 1u) - FP_TOP;

    uint32_t kept24 = 0, round_bit = 0;
    bool sticky_norm = false;
    if (sh > 0) {
      const uint64_t mask = (sh >= 64) ? ~0ull : ((1ull << sh) - 1ull);
      const uint64_t rem = mag & mask;
      round_bit = (sh >= 1) ? ((rem >> (sh - 1)) & 1ull) : 0u;
      sticky_norm = (sh >= 2) ? ((rem & ((1ull << (sh - 1)) - 1u)) != 0ull) : false;
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

  // ----------------------------- rounding + pack ------------------------------
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

  // ------------------------------- Primitives ---------------------------------

  static inline uint64_t lowbits_mask(uint32_t W) {
    return (W >= 64) ? ~0ull : ((1ull << W) - 1ull);
  }

  // 3:2 compressor on 64-bit lanes.
  static inline CS csa32(uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t s = (a ^ b) ^ c;
    const uint64_t g = (a & b) | (a & c) | (b & c);
    return CS{s, g << 1};
  }

  static inline CS csa32_fold(CS acc, uint64_t x, uint64_t y) {
    CS t1 = csa32(acc.sum, acc.carry, x);
    return csa32(t1.sum, t1.carry, y);
  }

  static inline CS csa42(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    CS t = csa32(a, b, c);
    return csa32(t.sum, t.carry, d);
  }

  static inline int64_t sign_extend(uint64_t v, uint32_t W) {
    if (W >= 64)
      return (int64_t)v;
    const uint64_t mask = (1ull << W) - 1ull;
    v &= mask;
    const uint64_t msb = 1ull << (W - 1);
    return (v & msb) ? (int64_t)(v | ~mask) : (int64_t)v;
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
  uint32_t Wc_{24}, Win_{25}, Wacc_{35};
  int frm_{0};
  uint32_t lanes_{1};

  uint32_t fflags_{0};
  bool has_nan_{false}, has_pos_inf_{false}, has_neg_inf_{false}, has_nv_{false};
  bool c_is_nan_{false}, c_is_inf_{false};
  uint32_t c_sign_{0};
  bool sticky_c32_{false}, sticky_align_{false}, sticky_mul_{false};
  bool has_any_special_{false};

  // debug scratch
  uint32_t a32_{0};
};
