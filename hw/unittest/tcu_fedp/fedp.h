#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

using namespace vortex;

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
  // exp_bits + sig_bits + 1 ∈ {4,8,16,19}; k = number of elements
  FEDP(int exp_bits, int sig_bits, int frm, uint32_t k)
      : ebits_(exp_bits)
      , sbits_(sig_bits)
      , frm_(frm)
      , k_(k) {
    width_ = 1u + ebits_ + sbits_;
    assert(width_ <= 19u);
    assert(width_ == 4u || width_ == 8u || width_ == 16u || width_ == 19u);
    assert(ebits_ >= 2 && ebits_ <= 8);
    assert(sbits_ >= 1 && sbits_ <= 10); // FP4..TF32

    bias_         = (1u << (ebits_ - 1)) - 1u;
    exp_all_ones_ = (1u << ebits_) - 1u;
    frac_mask_    = (1u << sbits_) - 1u;
    enc_mask_     = (1u << width_) - 1u;
    TOP_          = 2u * sbits_;

    // Packing configuration
    packed_ = (width_ <= 16);
    elems_per_word_ = packed_ ? (32u / width_) : 1u;
    assert(!packed_ || (32u % width_) == 0u);

    // Guard region for single-rounding dot product (no per-term rounding).
    const uint32_t Kdesign = elems_per_word_ * 4u; // up to 4 packed words
    uint32_t gb = sbits_ + log2ceil(std::max(Kdesign, 1u)) + 2u;  // +2 cushion
    guard_bits_ = std::clamp(gb, 1u, 8u);

    LOG("[cfg] ebits=%u, sbits=%u, frm=%u, k=%u, width=%u, packed=%d, elems/word=%u, bias=%u, G=%u\n",
        ebits_, sbits_, frm_, k_, width_, packed_, elems_per_word_, bias_, guard_bits_);
  }

  // Computes: sum_{i=0..k_-1} a[i]*b[i] + c
  float operator()(const std::vector<uint32_t> &a,
                   const std::vector<uint32_t> &b,
                   float c) {
    const uint32_t n_words = k_ / elems_per_word_;
    assert(n_words <= a.size() && n_words <= b.size());

    resetFlags();

    // decode exactly k_ elements
    auto [adec, bdec] = decodeInputs(a, b, n_words);

    // multiply → normalized product domain
    auto prods = multiply(adec, bdec);

    // decode c to product domain
    auto cenc = bitsFromF32(c);
    auto cprod = decodeC(cenc);

    // align to common exponent + guard grid
    auto plan = alignment(prods, cprod);

    // accumulate aligned terms
    auto acc32 = accumulate(plan.terms);

    // rounding and normalization
    auto out = finalize(acc32, plan.max_exp);

    // Cast result to float
    return f32FromBits(out);
  }

  uint32_t fflags() const { return fflags_; }

private:
  // ------------------------------- Types -------------------------------------
  struct fp_dec_t {
    uint32_t sign{}, frac{}, exp_field{};
    int32_t exp_unb{};
    bool is_zero{}, is_sub{}, is_inf{}, is_nan{};
  };
  struct fp_prod_t {
    uint32_t sign{};
    int32_t exp{};    // value = mnorm * 2^(exp - TOP_)
    uint32_t mnorm{}; // MSB near TOP_
    bool is_zero{};
    bool sticky{};        // right-shift inexact
    uint32_t lost_bits{}; // dropped bits
    uint32_t sh_norm{};   // right shift amount
  };
  struct AlignPlan {
    std::vector<int32_t> terms; // aligned terms on guard grid
    int32_t max_exp = 0;        // already (max_exp - guard_bits_)
    uint32_t K_eff = 0;
    uint32_t M_max = 0;
    uint32_t W_sum = 0;
  };

  // ------------------------------ Utilities ----------------------------------
  static inline int clz32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    if (!x)
      return 32;
    int n = 0;
    while ((x & (1u << 31)) == 0u) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }
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
  static inline uint32_t packZero32(uint32_t s) { return s << 31; }
  static inline uint32_t packInf32(uint32_t s) { return (s << 31) | (0xFFu << 23); }
  static inline uint32_t canonicalNaN32() { return (0xFFu << 23) | (1u << 22); }

  std::pair<std::vector<fp_dec_t>, std::vector<fp_dec_t>>
  decodeInputs(const std::vector<uint32_t> &a,
               const std::vector<uint32_t> &b,
               uint32_t n_words) const {
    LOG("[decode] words=%u, width=%u, packed=%d, elems/word=%u, total=%u\n",
        n_words, width_, packed_ ? 1 : 0, elems_per_word_, k_);

    std::vector<fp_dec_t> A(k_), B(k_);
    uint32_t out = 0;

    for (uint32_t i = 0; i < n_words && out < k_; ++i) {
      uint32_t aw = a[i], bw = b[i];
      for (uint32_t j = 0; j < elems_per_word_ && out < k_; ++j) {
        const uint32_t aenc = packed_ ? (aw & enc_mask_) : (a[i] & enc_mask_);
        const uint32_t benc = packed_ ? (bw & enc_mask_) : (b[i] & enc_mask_);
        A[out] = decodeCustom(aenc);
        B[out] = decodeCustom(benc);
        LOG("  [%u,%u] a(e=%u f=0x%x s=%u) b(e=%u f=0x%x s=%u)\n",
            i, j, A[out].exp_field, A[out].frac, A[out].sign,
            B[out].exp_field, B[out].frac, B[out].sign);
        if (packed_) {
          aw >>= width_;
          bw >>= width_;
        }
        ++out;
      }
    }
    return {std::move(A), std::move(B)};
  }

  fp_dec_t decodeCustom(uint32_t enc) const {
    enc &= enc_mask_;
    const uint32_t sign = (enc >> (ebits_ + sbits_)) & 1u;
    const uint32_t exp = (enc >> sbits_) & ((1u << ebits_) - 1u);
    const uint32_t frac = enc & frac_mask_;
    fp_dec_t d{};
    d.sign = sign;
    d.frac = frac;
    d.exp_field = exp;
    d.is_zero = (exp == 0 && frac == 0);
    d.is_sub = (exp == 0 && frac != 0);
    d.is_inf = (exp == exp_all_ones_ && frac == 0);
    d.is_nan = (exp == exp_all_ones_ && frac != 0);
    d.exp_unb = (exp == 0) ? (1 - int32_t(bias_))
                           : (exp == exp_all_ones_ ? 0 : int32_t(exp) - int32_t(bias_));
    return d;
  }

  std::vector<fp_prod_t>
  multiply(const std::vector<fp_dec_t> &A, const std::vector<fp_dec_t> &B) {
    assert(A.size() == B.size());
    resetSpecialFlags();

    std::vector<fp_prod_t> out;
    out.reserve(A.size());

    for (size_t i = 0; i < A.size(); ++i) {
      const auto &a = A[i], &b = B[i];

      if (a.is_nan || b.is_nan)
        has_nan_ = true;
      if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero))
        has_nv_ = true;

      if (a.is_inf || b.is_inf) {
        const uint32_t s = a.sign ^ b.sign;
        (s ? has_neg_inf_ : has_pos_inf_) = true;
        out.push_back(fp_prod_t{0, 0, 0, true, false, 0, 0});
        continue;
      }
      if (a.is_zero || b.is_zero) {
        out.push_back(fp_prod_t{0, 0, 0, true, false, 0, 0});
        continue;
      }

      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sbits_) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sbits_) | b.frac);
      uint32_t m = Ma * Mb; // ≤ 22 bits (sbits ≤ 10)
      int32_t E = a.exp_unb + b.exp_unb;

      const int pos = 31 - clz32(m); // nbits-1
      const int sh = pos - int(TOP_);

      bool sticky1 = false;
      uint32_t lost = 0, sh_norm = 0;
      if (sh > 0) {
        const uint32_t mask = (1u << sh) - 1u;
        lost = (m & mask);
        sticky1 = (lost != 0);
        m >>= sh;
        E += sh;
        sh_norm = uint32_t(sh);
      } else if (sh < 0) {
        m <<= -sh;
        E -= -sh;
      }

      out.push_back(fp_prod_t{a.sign ^ b.sign, E, m, false, sticky1, lost, sh_norm});
      LOG("[multiply] i=%zu s=%u E=%d M=0x%08x sh=%d sticky1=%u\n",
          i, out.back().sign, out.back().exp, out.back().mnorm, sh, sticky1 ? 1u : 0u);
    }
    return out;
  }

  // FP32 c → product domain; track c specials; numeric term is zero if non-finite
  fp_prod_t decodeC(uint32_t enc) {
    const uint32_t sign = (enc >> 31) & 1u;
    const uint32_t exp  = (enc >> 23) & 0xFFu;
    const uint32_t frac = enc & 0x7FFFFFu;

    c_sign_ = sign;
    c_is_inf_ = (exp == 0xFFu && frac == 0);
    c_is_nan_ = (exp == 0xFFu && frac != 0);

    const bool is_zero = (exp == 0 && frac == 0);
    const bool is_sub = (exp == 0 && frac != 0);

    if (is_zero || c_is_inf_ || c_is_nan_) {
      LOG("[cprod] non-finite-or-zero c -> numeric 0 (s=%u)\n", sign);
      return fp_prod_t{0, 0, 0, true, false, 0, 0};
    }

    const int32_t exp_unb = int32_t(exp) - 127; // exp != 0,0xFF
    const uint32_t M = is_sub ? frac : ((1u << 23) | frac);

    uint32_t m = M;
    int32_t E = exp_unb - 23;

    const int pos = 31 - clz32(m);
    const int sh = pos - int(TOP_);

    bool sticky1 = false;
    uint32_t lost = 0, sh_norm = 0;
    if (sh > 0) {
      const uint32_t mask = (1u << sh) - 1u;
      lost = (m & mask);
      sticky1 = (lost != 0);
      m >>= sh;
      E += sh;
      sh_norm = uint32_t(sh);
    } else if (sh < 0) {
      m <<= -sh;
      E -= -sh;
    }

    E += int(TOP_);
    LOG("[cprod] s=%u, E=%d, M=0x%08x\n", sign, E, m);
    return fp_prod_t{sign, E, m, false, sticky1, lost, sh_norm};
  }

  AlignPlan alignment(const std::vector<fp_prod_t> &ps, const fp_prod_t &cprod) {
    LOG("[alignment]\n");
    AlignPlan plan;

    int32_t max_exp = std::numeric_limits<int32_t>::min();
    auto note = [&](const fp_prod_t &p) {
      if (!p.is_zero) {
        max_exp = std::max(max_exp, p.exp);
      }
    };
    for (const auto &p : ps)
      note(p);
    note(cprod);

    if (max_exp == std::numeric_limits<int32_t>::min()) {
      plan.max_exp = 0;
      plan.K_eff = 0;
      plan.M_max = 0;
      plan.W_sum = 0;
      plan.terms.assign(ps.size() + 1, 0);
      LOG("  all-zero terms\n");
      return plan;
    }

    const uint32_t G = guard_bits_;
    plan.max_exp = max_exp - int32_t(G);
    plan.terms.reserve(ps.size() + 1);

    auto align_one = [&](const fp_prod_t &p, const char *tag, int idx) -> int32_t {
      if (p.is_zero) {
        LOG("  %s%02d: zero\n", tag, idx);
        return 0;
      }

      const int32_t delta = max_exp - p.exp; // ≥ 0 wrt original grid
      uint32_t m = p.mnorm;

      // Align to guard grid (keep G guards; anything below → sticky)
      if (delta >= int32_t(G)) {
        const uint32_t sh = uint32_t(delta - int32_t(G));
        if (sh >= 32u) {
          if (m)
            any_sticky_ = true;
          m = 0;
        } else {
          if (sh > 0) {
            const uint32_t rem_mask = (1u << sh) - 1u;
            if (m & rem_mask)
              any_sticky_ = true;
          }
          m >>= sh;
        }
      } else {
        const uint32_t lsh = uint32_t(int32_t(G) - delta);
        m <<= lsh;
      }

      // Re-inject lost bits: integer part to grid; fractional → sticky
      if (p.lost_bits) {
        const int32_t sd_signed = (max_exp - p.exp) + int32_t(p.sh_norm) - int32_t(G);
        if (sd_signed <= 0) {
          const uint32_t lsh = uint32_t(-sd_signed);
          if (lsh < 32u)
            m += (p.lost_bits << lsh);
        } else {
          const uint32_t sd = uint32_t(sd_signed);
          const uint32_t extra = (sd < 32u) ? (p.lost_bits >> sd) : 0u;
          if (extra)
            m += extra;
          if (sd < 32u) {
            const uint32_t rem_mask = (1u << sd) - 1u;
            if (p.lost_bits & rem_mask)
              any_sticky_ = true;
          } else if (p.lost_bits) {
            any_sticky_ = true;
          }
        }
      }

      if (p.sticky)
        any_sticky_ = true;

      const uint32_t mbits = m ? uint32_t(32 - clz32(m)) : 0u;
      plan.M_max = std::max(plan.M_max, mbits);
      if (m)
        plan.K_eff++;

      int32_t val = int32_t(m);
      if (p.sign)
        val = -val;

      LOG("  %s%02d: δ=%d mbits=%u sign=%u (G=%u)\n", tag, idx, delta, mbits, p.sign, G);
      return val;
    };

    for (size_t i = 0; i < ps.size(); ++i)
      plan.terms.push_back(align_one(ps[i], "p", int(i)));
    plan.terms.push_back(align_one(cprod, "c", 0));

    // Width proof (no cancellation). Must fit signed 32-bit magnitude.
    plan.W_sum = (plan.K_eff == 0) ? 0u : (plan.M_max + log2ceil(plan.K_eff));
    assert(plan.W_sum <= 31u && "Accumulator magnitude would exceed 31 bits");

    LOG("  summary: K=%u, M_max=%u, W_sum=%u (<=31 OK), sticky=%u (G=%u)\n",
        plan.K_eff, plan.M_max, plan.W_sum, any_sticky_ ? 1u : 0u, G);
    return plan;
  }

  int32_t accumulate(const std::vector<int32_t> &terms) const {
    LOG("[accumulate]\n");
    int32_t acc = 0;
    for (size_t i = 0; i < terms.size(); ++i) {
      int32_t before = acc;
      acc += terms[i];
      LOG("  %02zu: %d + %d -> %d\n", i, before, terms[i], acc);
    }
    LOG("  acc=%d\n", acc);
    return acc;
  }

  uint32_t finalize(int32_t acc32, int32_t max_exp_fine) {
    // Resolve specials (no early exits elsewhere)
    if (has_nv_ ||
        (has_pos_inf_ && has_neg_inf_) ||
        (c_is_inf_ && ((has_pos_inf_ && c_sign_ == 1u) || (has_neg_inf_ && c_sign_ == 0u)))) {
      fflags_ |= FLAG_NV;
      LOG("[finalize] invalid -> qNaN\n");
      return canonicalNaN32();
    }
    if (has_nan_ || c_is_nan_) {
      LOG("[finalize] NaN observed -> qNaN\n");
      return canonicalNaN32();
    }
    if (has_pos_inf_ || has_neg_inf_) {
      const uint32_t s = has_neg_inf_ ? 1u : 0u;
      LOG("[finalize] product Inf -> %sInf\n", s ? "-" : "+");
      return packInf32(s);
    }
    if (c_is_inf_) {
      LOG("[finalize] c is Inf -> pass through\n");
      return packInf32(c_sign_);
    }

    // Finite numeric path
    if (acc32 == 0) {
      if (any_sticky_)
        fflags_ |= (FLAG_NX | FLAG_UF);
      LOG("[finalize] zero, fflags=0x%02x\n", fflags_);
      return packZero32(0);
    }

    const uint32_t sign = (acc32 < 0) ? 1u : 0u;
    uint32_t mag = (acc32 < 0) ? uint32_t(-(int32_t)acc32) : uint32_t(acc32);

    const int nbits = 32 - clz32(mag); // 1..31 by construction
    int32_t e_unb32 = (max_exp_fine - int32_t(TOP_)) + (nbits - 1);

    // Normalize to 24b (1.hidden + 23 frac)
    const int FP_TOP = 23;
    const int sh = (nbits - 1) - FP_TOP;
    uint32_t kept24 = 0, round_bit = 0;
    bool sticky_norm = false;

    if (sh > 0) {
      const uint32_t mask = (sh >= 32) ? 0xFFFFFFFFu : ((1u << sh) - 1u);
      const uint32_t rem = mag & mask;
      round_bit = (sh >= 1) ? ((rem >> (sh - 1)) & 1u) : 0u;
      sticky_norm = (sh >= 2) ? ((rem & ((1u << (sh - 1)) - 1u)) != 0u) : false;
      kept24 = (sh >= 32) ? 0u : (mag >> sh);
    } else {
      kept24 = (sh < -31) ? 0u : (mag << (-sh));
    }
    kept24 &= ((1u << 24) - 1u);

    const bool sticky_any = sticky_norm || any_sticky_;
    if (round_bit || sticky_any)
      fflags_ |= FLAG_NX;

    const uint32_t lsb = kept24 & 1u;
    if (roundInc(sign, lsb, round_bit, sticky_any)) {
      kept24 += 1u;
      if (kept24 >= (1u << 24)) {
        kept24 >>= 1;
        e_unb32 += 1;
      }
    }

    const int32_t e_bias32 = e_unb32 + 127;

    // Overflow → Inf
    if (e_bias32 >= 0xFF) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      const uint32_t out = packInf32(sign);
      LOG("[finalize] overflow -> inf 0x%08x, fflags=0x%02x\n", out, fflags_);
      return out;
    }

    // Subnormal / underflow
    if (e_bias32 <= 0) {
      const int sh2 = 1 - e_bias32;
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
          const uint32_t out = (sign << 31) | (1u << 23);
          if (fflags_ & FLAG_NX)
            fflags_ |= FLAG_UF;
          LOG("[finalize] subnormal->min-normal 0x%08x, fflags=0x%02x\n", out, fflags_);
          return out;
        }
        frac_keep = t;
      }

      if (fflags_ & FLAG_NX)
        fflags_ |= FLAG_UF;
      const uint32_t out = (sign << 31) | frac_keep;
      LOG("[finalize] subnormal 0x%08x, fflags=0x%02x\n", out, fflags_);
      return out;
    }

    // Normal
    const uint32_t exp_out = uint32_t(e_bias32);
    const uint32_t frac_out = kept24 & ((1u << 23) - 1u);
    const uint32_t out = (sign << 31) | (exp_out << 23) | frac_out;
    LOG("[finalize] normal 0x%08x, s=%u, e=0x%x, f=0x%x, fflags=0x%02x\n",
        out, sign, exp_out, frac_out, fflags_);
    return out;
  }

  // ------------------------------ Rounding -----------------------------------
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

  // ------------------------------ Flags/state --------------------------------
  void resetFlags() {
    fflags_ = 0;
    any_sticky_ = false;
    has_nan_ = has_pos_inf_ = has_neg_inf_ = has_nv_ = false;
    c_is_nan_ = c_is_inf_ = false;
    c_sign_ = 0;
  }
  void resetSpecialFlags() {
    has_nan_ = has_pos_inf_ = has_neg_inf_ = has_nv_ = false;
  }

  enum { RNE = 0,
         RTZ = 1,
         RDN = 2,
         RUP = 3,
         RMM = 4 };
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // Config
  unsigned ebits_, sbits_, width_;
  uint32_t bias_, exp_all_ones_, frac_mask_, enc_mask_;
  int frm_;
  uint32_t k_;
  bool packed_ = true;
  uint32_t elems_per_word_ = 1;
  uint32_t guard_bits_ = 1;
  uint32_t TOP_ = 0;

  // State
  uint32_t fflags_ = 0;
  bool any_sticky_ = false;
  bool has_nan_ = false, has_pos_inf_ = false, has_neg_inf_ = false, has_nv_ = false;
  bool c_is_nan_ = false, c_is_inf_ = false;
  uint32_t c_sign_ = 0;
};
