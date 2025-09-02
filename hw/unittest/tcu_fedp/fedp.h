#include <algorithm>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// ---------- Trace Infrastructure ----------
#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

class Logger {
public:
  static void log(const char *format, ...) {
    if constexpr (FEDP_TRACE) {
      va_list args;
      va_start(args, format);
      std::vprintf(format, args);
      va_end(args);
    }
  }
};

#define LOG(...) Logger::log(__VA_ARGS__)

// =============================== FEDP ========================================
class FEDP {
public:
  FEDP(int exp_bits, int sig_bits, int frm)
      : ebits_(exp_bits)
      , sbits_(sig_bits)
      , frm_(frm) {
    assert(ebits_ >= 2 && ebits_ <= 8);
    assert(sbits_ >= 2 && sbits_ <= 28);

    width_ = 1u + ebits_ + sbits_;
    assert(width_ <= 32u && (32 % width_) == 0u);

    bias_ = (1u << (ebits_ - 1)) - 1u;
    exp_all_ones_ = (1u << ebits_) - 1u;
    frac_mask_ = (1u << sbits_) - 1u;
    enc_mask_ = (1u << width_) - 1u;

    LOG("[cfg] ebits=%u, sbits=%u, width=%u, bias=%u\n",
        ebits_, sbits_, width_, bias_);
  }

  // Computes: sum(a[i]*b[i]) + c
  // a/b: packed custom floats in 'n' 32-bit words (same layout for both).
  float operator()(const std::vector<uint32_t> &a,
                   const std::vector<uint32_t> &b,
                   float c,
                   uint32_t n) {
    assert(n <= a.size() && n <= b.size());
    resetFlags();

    // Stage 0: unpack & decode inputs
    auto [adec, bdec] = decodeInputs(a, b, n);

    // Stage 1: elementwise multiply (normalized products in common TOP scale)
    auto prods = multiply(adec, bdec);

    // Early returns (NaN/Inf rules) that do not need datapath
    if (earlyReturnNeeded(c))
      return doEarlyReturn(c);

    // Convert c (fp32) into product domain
    auto cprod = makeProdFromFP32(decodeFp32(bitsFromF32(c)));
    LOG("[cprod] s=%u, e=%d, m=0x%s\n",
        cprod.sign, cprod.exp, hex128(cprod.mnorm).c_str());

    // Stage 2: align to a single fixed-point scale (units = 2^-TOP)
    auto terms = alignTerms(prods, cprod);

    // Stage 3: accumulate wide integer sum
    auto acc = accumulate(terms);

    // Stage 4: round & normalize to fp32
    auto result = finallize(acc);

    return f32FromBits(result);
  }

  uint32_t fflags() const { return fflags_; }

private:
  // ------------------------------- Types -------------------------------------
  struct fp_dec_t {
    uint32_t sign{}, frac{}, exp_field{};
    int32_t exp_unb{}; // unbiased exponent
    bool is_zero{}, is_sub{}, is_inf{}, is_nan{};
  };

  struct fp_prod_t {
    uint32_t sign{};         // 0/1
    int32_t exp{};           // value = mnorm * 2^(exp - TOP)
    __uint128_t mnorm{};     // normalized so MSB sits at TOP
    bool is_zero{};          // exact zero
    bool sticky_pre{};       // lost info when normalizing
    __uint128_t lost_bits{}; // exact low bits dropped by normalization
    uint32_t sh_norm{};      // right-shift used during normalization
  };

  // ------------------------------ Utilities ----------------------------------
  static inline int clz64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#else
    if (!x)
      return 64;
    int n = 0;
    while ((x & (uint64_t(1) << 63)) == 0) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }
  static int clz128(__uint128_t x) {
    const uint64_t hi = static_cast<uint64_t>(x >> 64);
    return hi ? clz64(hi) : 64 + clz64(static_cast<uint64_t>(x));
  }
  static std::string hex128(__uint128_t x) {
    if (!x)
      return "0";
    char buf[33];
    buf[32] = 0;
    int i = 32;
    while (x && i) {
      buf[--i] = "0123456789abcdef"[x & 0xF];
      x >>= 4;
    }
    return std::string(&buf[i]);
  }
  static __uint128_t rshift_sticky(__uint128_t x, uint32_t sh, bool &sticky) {
    if (sh == 0)
      return x;
    if (sh >= 128) {
      sticky = sticky || (x != 0);
      return 0;
    }
    const __uint128_t mask = (static_cast<__uint128_t>(1) << sh) - 1;
    if (x & mask)
      sticky = true;
    return x >> sh;
  }
  static __uint128_t iabs128(__int128 x) {
    return (x < 0) ? static_cast<__uint128_t>(-x) : static_cast<__uint128_t>(x);
  }
  static uint32_t bitsFromF32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
  }
  static float f32FromBits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }

  static uint32_t packZero32(uint32_t sign) { return sign << 31; }
  static uint32_t packInf32(uint32_t sign) { return (sign << 31) | (0xFFu << 23); }
  static uint32_t canonicalNaN32() { return (0xFFu << 23) | (1u << 22); }

  // ------------------------------ Stage 0: decode -----------------------------
  std::pair<std::vector<fp_dec_t>, std::vector<fp_dec_t>>
  decodeInputs(const std::vector<uint32_t> &a,
               const std::vector<uint32_t> &b,
               uint32_t n_words) const {
    const uint32_t elems_per_word = 32u / width_;
    const uint32_t total = n_words * elems_per_word;

    LOG("[decode] words=%u, width=%u, elems/word=%u, total_elems=%u\n",
        n_words, width_, elems_per_word, total);

    std::vector<fp_dec_t> A(total), B(total);

    for (uint32_t i = 0; i < n_words; ++i) {
      uint32_t aw = a[i], bw = b[i];
      for (uint32_t j = 0; j < elems_per_word; ++j, aw >>= width_, bw >>= width_) {
        const uint32_t idx = i * elems_per_word + j;
        A[idx] = decodeCustom(aw & enc_mask_);
        B[idx] = decodeCustom(bw & enc_mask_);
        LOG("  [%u,%u] a(e=%u f=0x%x s=%u) b(e=%u f=0x%x s=%u)\n",
            i, j, A[idx].exp_field, A[idx].frac, A[idx].sign,
            B[idx].exp_field, B[idx].frac, B[idx].sign);
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

    d.exp_unb = (exp == 0) ? (1 - static_cast<int32_t>(bias_))
                           : (exp == exp_all_ones_ ? 0
                                                   : int32_t(exp) - int32_t(bias_));
    return d;
  }

  static fp_dec_t decodeFp32(uint32_t enc) {
    const uint32_t sign = (enc >> 31) & 1u;
    const uint32_t exp = (enc >> 23) & 0xFFu;
    const uint32_t frac = enc & 0x7FFFFFu;

    fp_dec_t d{};
    d.sign = sign;
    d.frac = frac;
    d.exp_field = exp;

    d.is_zero = (exp == 0 && frac == 0);
    d.is_sub = (exp == 0 && frac != 0);
    d.is_inf = (exp == 0xFFu && frac == 0);
    d.is_nan = (exp == 0xFFu && frac != 0);

    d.exp_unb = (exp == 0) ? (1 - 127) : (exp == 0xFFu ? 0 : int32_t(exp) - 127);
    return d;
  }

  // ------------------------------ Stage 1: multiply ---------------------------
  std::vector<fp_prod_t>
  multiply(const std::vector<fp_dec_t> &A,
           const std::vector<fp_dec_t> &B) {
    assert(A.size() == B.size());
    resetSpecialFlags();

    std::vector<fp_prod_t> out;
    out.reserve(A.size());
    const int TOP = 2 * sbits_;

    for (size_t i = 0; i < A.size(); ++i) {
      const auto &a = A[i];
      const auto &b = B[i];

      // Track specials/invalid
      if (a.is_nan || b.is_nan)
        has_nan_ = true;
      if ((a.is_inf && b.is_zero) || (b.is_inf && a.is_zero))
        has_nv_ = true;

      // Infinity or Zero results short-circuit the product value
      if (a.is_inf || b.is_inf) {
        const uint32_t s = a.sign ^ b.sign;
        (s ? has_neg_inf_ : has_pos_inf_) = true;
        out.push_back(fp_prod_t{});
        out.back().is_zero = true;
        continue;
      }
      if (a.is_zero || b.is_zero) {
        out.push_back(fp_prod_t{});
        out.back().is_zero = true;
        continue;
      }

      // Multiply significands (hidden-1 for normals)
      const uint32_t Ma = a.is_sub ? a.frac : ((1u << sbits_) | a.frac);
      const uint32_t Mb = b.is_sub ? b.frac : ((1u << sbits_) | b.frac);
      __uint128_t m = static_cast<__uint128_t>(Ma) * static_cast<__uint128_t>(Mb);
      int32_t E = a.exp_unb + b.exp_unb;

      // Normalize so MSB sits at TOP
      const int pos = 127 - clz128(m);
      const int sh = pos - TOP;

      bool sticky_pre = false;
      __uint128_t lost = 0;
      if (sh > 0) {
        const __uint128_t mask = (static_cast<__uint128_t>(1) << sh) - 1;
        lost = (m & mask);
        sticky_pre = (lost != 0);
        m >>= sh;
        E += sh;
      } else if (sh < 0) {
        m <<= -sh;
        E -= -sh;
      }

      out.push_back(fp_prod_t{
          a.sign ^ b.sign, E, m, false, sticky_pre, lost,
          static_cast<uint32_t>(sh > 0 ? sh : 0)});

      LOG("[multiply] i=%zu s=%u E=%d m=0x%s sh=%d lost=0x%s\n",
          i, out.back().sign, out.back().exp, hex128(out.back().mnorm).c_str(),
          sh, hex128(lost).c_str());
    }

    if (has_nv_)
      LOG("  (invalid 0*Inf)\n");
    if (has_nan_)
      LOG("  (NaN observed)\n");
    if (has_pos_inf_ || has_neg_inf_)
      LOG("  (Inf observed) +Inf=%d -Inf=%d\n", has_pos_inf_, has_neg_inf_);

    return out;
  }

  // --------------------- Early exits (NaN/Inf combination rules) --------------
  bool earlyReturnNeeded(float c) const {
    const auto dc = decodeFp32(bitsFromF32(c));
    return has_nv_ || (has_pos_inf_ && has_neg_inf_) || dc.is_nan || has_nan_ || has_pos_inf_ || has_neg_inf_ || dc.is_inf;
  }

  float doEarlyReturn(float c) {
    const auto dc = decodeFp32(bitsFromF32(c));

    if (has_nv_ || (has_pos_inf_ && has_neg_inf_)) {
      fflags_ |= FLAG_NV;
      return f32FromBits(canonicalNaN32());
    }
    if (dc.is_nan || has_nan_)
      return f32FromBits(canonicalNaN32());

    if (has_pos_inf_ || has_neg_inf_) {
      if (dc.is_inf && ((has_pos_inf_ && dc.sign) || (has_neg_inf_ && !dc.sign))) {
        fflags_ |= FLAG_NV;
        return f32FromBits(canonicalNaN32());
      }
      return f32FromBits(packInf32(has_neg_inf_ ? 1u : 0u));
    }
    if (dc.is_inf)
      return f32FromBits(packInf32(dc.sign));
    return 0.0f; // unreachable
  }

  // -------------------------- c as product (TOP domain) -----------------------
  fp_prod_t makeProdFromFP32(const fp_dec_t &d) const {
    const int TOP = 2 * sbits_;
    if (d.is_zero)
      return fp_prod_t{0, 0, 0, true, false, 0, 0};

    const uint32_t M = d.is_sub ? d.frac : ((1u << 23) | d.frac);
    __uint128_t m = M;
    int32_t E = d.exp_unb - 23;

    const int pos = 127 - clz128(m);
    const int sh = pos - TOP;

    bool sticky_pre = false;
    __uint128_t lost = 0;
    if (sh > 0) {
      const __uint128_t mask = (static_cast<__uint128_t>(1) << sh) - 1;
      lost = (m & mask);
      sticky_pre = (lost != 0);
      m >>= sh;
      E += sh;
    } else if (sh < 0) {
      m <<= -sh;
      E -= -sh;
    }

    // Align exponent semantics with product domain (value = m * 2^(E - TOP))
    E += TOP;

    return fp_prod_t{
        d.sign, E, m, false, sticky_pre, lost,
        static_cast<uint32_t>(sh > 0 ? sh : 0)};
  }

  // --------------------------- Stage 2: alignment -----------------------------
  std::vector<__int128_t> alignTerms(const std::vector<fp_prod_t> &ps,
                                     const fp_prod_t &cprod) {
    LOG("[align]\n");
    std::vector<__int128_t> out;
    out.reserve(ps.size() + 1);

    auto toCommon = [&](const fp_prod_t &p, const char *tag, int idx) {
      if (p.is_zero) {
        out.push_back(0);
        LOG("  %s%02d: zero\n", tag, idx);
        return;
      }

      __uint128_t mag = p.mnorm;
      bool st = p.sticky_pre;
      const int32_t sh = p.exp; // shift to targetE=0

      if (sh >= 0) {
        if (sh < 128)
          mag <<= sh;
        else
          mag = 0;
      } else {
        mag = rshift_sticky(mag, static_cast<uint32_t>(-sh), st);
        if (st) {
          mag |= 1;
          any_sticky_ = true;
        }
      }

      // Re-inject exactly lost bits from product normalization
      if (p.lost_bits && p.sh_norm) {
        __uint128_t extra = 0;
        auto delta = sh - int32_t(p.sh_norm);
        if (delta >= 0) {
          if (delta < 128)
            extra = p.lost_bits << delta;
        } else {
          bool ign = false;
          extra = rshift_sticky(p.lost_bits, static_cast<uint32_t>(-delta), ign);
        }
        mag += extra;
      }

      const __int128_t term = p.sign ? -static_cast<__int128>(mag)
                                     : static_cast<__int128>(mag);

      LOG("  %s%02d: s=%u E=%d -> %c0x%s%s\n",
          tag, idx, p.sign, p.exp,
          (term < 0 ? '-' : '+'), hex128(iabs128(term)).c_str(),
          (st ? " (sticky)" : ""));
      out.push_back(term);
    };

    for (size_t i = 0; i < ps.size(); ++i) {
      toCommon(ps[i], "p", int(i));
    }
    toCommon(cprod, "c", 0);
    return out;
  }

  // --------------------------- Stage 3: accumulation --------------------------
  __int128 accumulate(const std::vector<__int128_t> &terms) const {
    __int128 acc = 0;
    LOG("[accumulate]\n");
    for (size_t i = 0; i < terms.size(); ++i) {
      const __int128 before = acc;
      acc += terms[i];
      LOG("  %02zu: %c0x%s %c 0x%s = %c0x%s\n",
          i,
          (before < 0 ? '-' : '+'), hex128(iabs128(before)).c_str(),
          (terms[i] < 0 ? '-' : '+'), hex128(iabs128(terms[i])).c_str(),
          (acc < 0 ? '-' : '+'), hex128(iabs128(acc)).c_str());
    }
    LOG("  acc=%c0x%s\n", (acc < 0 ? '-' : '+'), hex128(iabs128(acc)).c_str());
    return acc;
  }

  // -------------------- Stage 4: round & normalize to FP32 --------------------
  uint32_t finallize(__int128 acc) {
    if (acc == 0) {
      LOG("[finalize] zero\n");
      return f32FromBits(packZero32(0));
    }

    const int TOP = 2 * sbits_;
    const int FP_TOP = 23;

    const uint32_t sign = (acc < 0) ? 1u : 0u;
    __uint128_t mag = iabs128(acc);

    // Exponent (unbiased, fp32): top_bit_index - TOP
    const int nbits = 128 - clz128(mag);
    int32_t e_unb32 = (nbits - 1) - TOP;

    // Normalize to place MSB at bit 23 (keep 24 bits total: hidden1+23 frac)
    const int sh = (nbits - 1) - FP_TOP;
    __uint128_t norm = mag;
    bool st_from_norm = false;
    if (sh > 0)
      norm = rshift_sticky(norm, static_cast<uint32_t>(sh), st_from_norm);
    else if (sh < 0)
      norm <<= -sh;

    __uint128_t kept = norm & ((static_cast<__uint128_t>(1) << 24) - 1); // 24 LSBs
    const uint32_t lsb = static_cast<uint32_t>(kept & 1u);
    const uint32_t round_b = static_cast<uint32_t>((norm >> 24) & 1u);
    const bool sticky_any = st_from_norm || any_sticky_;

    if (round_b || sticky_any)
      fflags_ |= FLAG_NX;

    if (roundInc(sign, lsb, round_b, sticky_any)) {
      kept += 1;
      if (kept >= (static_cast<__uint128_t>(1) << 24)) {
        kept >>= 1;
        e_unb32 += 1;
      }
    }

    // Pack to fp32
    const int32_t e_bias32 = e_unb32 + 127;

    // Overflow -> Inf
    if (e_bias32 >= 0xFF) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      const uint32_t out = packInf32(sign);
      LOG("[finalize] overflow -> inf 0x%08x fflags=0x%02x\n", out, fflags_);
      return out;
    }

    // Subnormal / underflow
    if (e_bias32 <= 0) {
      // make subnormal: shift right by (1 - e_bias32), no hidden-1
      const int sh2 = 1 - e_bias32;
      bool st2 = false;
      __uint128_t shifted = rshift_sticky(kept, static_cast<uint32_t>(sh2), st2);

      const uint32_t frac_keep = static_cast<uint32_t>(shifted & ((1u << 23) - 1u));
      const uint32_t rb2 = static_cast<uint32_t>((shifted >> 23) & 1u);
      if (rb2 || st2)
        fflags_ |= FLAG_NX;

      const uint32_t lsb2 = frac_keep & 1u;
      uint32_t frac_out = frac_keep;
      if (roundInc(sign, lsb2, rb2, st2)) {
        const uint32_t tmp = frac_out + 1;
        if (tmp >= (1u << 23)) { // bumps to min normal
          const uint32_t out = (sign << 31) | (1u << 23);
          LOG("[finalize] subnormal->min-normal 0x%08x fflags=0x%02x\n", out, fflags_);
          return out;
        }
        frac_out = tmp;
      }

      if (fflags_ & FLAG_NX)
        fflags_ |= FLAG_UF;
      const uint32_t out = (sign << 31) | frac_out;
      LOG("[finalize] subnormal 0x%08x fflags=0x%02x\n", out, fflags_);
      return out;
    }

    // Normal
    const uint32_t exp_out = static_cast<uint32_t>(e_bias32);
    const uint32_t frac_out = static_cast<uint32_t>(kept & ((1u << 23) - 1u));
    const uint32_t out = (sign << 31) | (exp_out << 23) | frac_out;

    LOG("[finalize] normal 0x%08x s=%u e=0x%x f=0x%x fflags=0x%02x\n",
        out, sign, exp_out, frac_out, fflags_);
    return out;
  }

  // ------------------------------ Rounding -----------------------------------
  bool roundInc(uint32_t sign, uint32_t lsb, uint32_t round_bit, bool sticky) const {
    switch (frm_) {
    case RNE: // ties-to-even
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
      return (round_bit || sticky); // ties-to-max-mag
    default:
      return false;
    }
  }

  // ------------------------------ Flags/state --------------------------------
  void resetFlags() {
    fflags_ = 0;
    any_sticky_ = false;
    has_nan_ = has_pos_inf_ = has_neg_inf_ = has_nv_ = false;
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

  // State
  uint32_t fflags_ = 0;
  bool any_sticky_ = false;
  bool has_nan_ = false;
  bool has_pos_inf_ = false;
  bool has_neg_inf_ = false;
  bool has_nv_ = false;
};
