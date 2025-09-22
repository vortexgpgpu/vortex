// fedp_anchor_split.h
// Anchor/windowed FEDP with S1 group reduction, S2 alignment, S3 accumulate (CSA only), S4 final CPA+round.
// Normals + subnormals only; no NaN/Inf/traps. Area-efficient K_WIN=28.
//
// Enable logs with -DFEDP_TRACE=1

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

struct Logger {
  static inline void log(const char *fmt, ...) {
    if constexpr (FEDP_TRACE) {
      va_list a;
      va_start(a, fmt);
      std::vprintf(fmt, a);
      va_end(a);
    }
  }
};
#define LOG(...) Logger::log(__VA_ARGS__)

class FEDP {
public:
  // frm: 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM ; lanes: 1..8
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    assert(frm_ >= 0 && frm_ <= 4);
    assert(lanes_ >= 1 && lanes_ <= 8);
    LOG("[ctor] frm=%d lanes=%u  Wc=%u  KWIN=%u  SCALE_K=%u\n",
        frm_, lanes_, Wc_, K_WIN_, SCALE_K_);
  }

  // Packed inputs (same packing policy as baseline).
  float operator()(const std::vector<uint32_t> &a_words,
                   const std::vector<uint32_t> &b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits) {
    fflags_ = 0;

    const uint32_t width = 1u + (uint32_t)exp_bits + (uint32_t)sig_bits;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t epw = packed ? (32u / width) : 1u;

    LOG("[inputs] fmt=e%dm%d width=%u packed=%u elems/word=%u words=%u\n",
        exp_bits, sig_bits, width, (unsigned)packed, epw, n_words);

    // ---- S1: decode lanes ---------------------------------------------------
    const auto terms = decode_inputs(a_words, b_words, n_words, epw, exp_bits, sig_bits, packed);

    // ---- Decode C (FP32) onto common integer grid (pre-scale <<3) ----------
    const uint32_t c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);
    const CT c_term = decodeC_to_common_scaled(c_dec); // Sc = m24 << 3, Ec in FP32-field scale

    // ---- S1: multiply + per-group local CPA → lanes_ outputs ----------------
    const auto groups = multiply_to_comm(terms, exp_bits, sig_bits);

    // ---- S2: alignment to global anchor (includes C); no CSA here -----------
    const auto al = alignment(groups, c_term);

    // ---- S3: CSA accumulate aligned terms + aligned C (no CPA) --------------
    const auto s3 = accumulate(al);

    // ---- S4: single final CPA + normalize + round ---------------------------
    const norm_in ni{s3.S, s3.C, s3.E_anchor, bool(al.sticky_any)};
    const uint32_t out = normalize_round_pack(ni);
    return f32FromBits(out);
  }

  uint32_t fflags() const { return fflags_; }

private:
  //============================ Types / constants ============================//
  struct dec_t {
    uint32_t sign{0}, frac{0}, exp{0};
    bool is_zero{false}, is_sub{false};
  };
  struct CT {
    int32_t S;
    int32_t E;
    bool zero;
  };

  struct grp_term {
    int32_t V;
    int32_t Eg;
    bool sticky;
  }; // S1 group output: signed int at Eg

  struct align_out {
    std::vector<int32_t> Vals; // size == lanes_ (aligned to E_anchor)
    int32_t Cal{0};            // aligned C at E_anchor
    int32_t E_anchor{INT32_MIN};
    uint32_t sticky_any{0};
  };

  struct s3_out {
    uint32_t S{0}, C{0};
    int32_t E_anchor{INT32_MIN};
  };

  struct norm_in {
    uint32_t S, C;
    int32_t E_anchor;
    bool sticky_any;
  };

  enum { RNE = 0, RTZ = 1, RDN = 2, RUP = 3, RMM = 4 };
  static constexpr uint32_t FLAG_NX = 1u << 0, FLAG_UF = 1u << 1, FLAG_OF = 1u << 2;

  // FP32 common grid sizing
  static constexpr uint32_t Wc_ = 24;                       // hidden+mantissa for FP32
  static constexpr uint32_t G_BITS_ = 3;                    // G/R/S budget
  static constexpr uint32_t SCALE_K_ = (Wc_ - 1) + G_BITS_; // 23 + 3 = 26
  static constexpr uint32_t K_WIN_ = 28;                    // 1+23+3 + 1 headroom
  static constexpr uint32_t C_SHIFT_BASE_ = SCALE_K_ - 23;  // = 3

  const int frm_;
  const uint32_t lanes_;
  uint32_t fflags_{0};

  //============================= Utilities =============================//
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
    while (!(x & 0x80000000u)) {
      x <<= 1;
      ++n;
    }
    return n;
#endif
  }
  static inline std::pair<uint32_t, uint32_t> csa32(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t t = a ^ b;
    return {t ^ c, (a & b) | (b & c) | (a & c)};
  }
  static inline bool any_dropped32_u(uint32_t x, uint32_t k) {
    if (k == 0)
      return false;
    if (k >= 32)
      return x != 0;
    return (x & ((1u << k) - 1u)) != 0u;
  }
  static inline uint32_t ceil_div_u(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

  // Sticky-aware sign-magnitude shifts on 32b integers (for group ints & C)
  static inline int32_t sign_mag_shr32(int32_t v, uint32_t k, bool &st) {
    if (k == 0 || v == 0)
      return v;
    uint32_t m = (v < 0) ? uint32_t(-v) : uint32_t(v);
    if (k >= 31) {
      st |= (m != 0);
      return 0;
    }
    st |= ((m & ((1u << k) - 1u)) != 0u);
    m >>= k;
    return (v < 0) ? -int32_t(m) : int32_t(m);
  }
  static inline int32_t sign_mag_shl32_clipK(int32_t v, uint32_t k, bool &st) {
    if (k == 0 || v == 0)
      return v;
    uint32_t m = (v < 0) ? uint32_t(-v) : uint32_t(v);
    if (k >= 32) {
      st |= (m != 0);
      return 0;
    }
    st |= ((m >> (32u - k)) != 0u);
    m <<= k;
    if (m >> K_WIN_) {
      st |= 1u;
      m &= ((1u << K_WIN_) - 1u);
    } // clip to window
    if (st)
      m |= 1u; // LSB sticky bucket
    return (v < 0) ? -int32_t(m) : int32_t(m);
  }
  static inline bool roundInc(int frm, uint32_t sign, uint32_t lsb, uint32_t guard, bool st) {
    switch (frm) {
    case RNE:
      return guard && (st || (lsb & 1u));
    case RTZ:
      return false;
    case RDN:
      return (guard || st) && (sign == 1u);
    case RUP:
      return (guard || st) && (sign == 0u);
    case RMM:
      return (guard || st);
    default:
      return false;
    }
  }
  static inline uint32_t packInf32(uint32_t s) { return (s << 31) | (0xFFu << 23); }

  //============================== Decode ==============================//
  static inline dec_t decode_input(uint32_t enc, int eb, int sb) {
    const uint32_t fm = (1u << sb) - 1u, em = (1u << eb) - 1u;
    const uint32_t s = (enc >> (eb + sb)) & 1u, e = (enc >> sb) & em, f = enc & fm;
    dec_t d{};
    d.sign = s;
    d.exp = e;
    d.frac = f;
    d.is_zero = (e == 0 && f == 0);
    d.is_sub = (e == 0 && f != 0);
    return d;
  }

  std::vector<std::array<dec_t, 2>>
  decode_inputs(const std::vector<uint32_t> &a_words,
                const std::vector<uint32_t> &b_words,
                uint32_t n_words, uint32_t epw,
                int eb, int sb, bool packed) {
    const uint32_t width = 1u + eb + sb, mask = (width == 32) ? 0xffffffffu : ((1u << width) - 1u);
    std::vector<std::array<dec_t, 2>> out;
    out.reserve(n_words * epw);
    for (uint32_t w = 0; w < n_words; ++w) {
      uint32_t aw = a_words[w], bw = b_words[w];
      for (uint32_t i = 0; i < epw; ++i) {
        const uint32_t aenc = packed ? (aw & mask) : aw, benc = packed ? (bw & mask) : bw;
        auto a = decode_input(aenc, eb, sb);
        auto b = decode_input(benc, eb, sb);
        LOG("[decode] lane=%u A(s=%u e=%u f=0x%x)  B(s=%u e=%u f=0x%x)\n",
            (w * epw) + i, a.sign, a.exp, a.frac, b.sign, b.exp, b.frac);
        out.push_back({a, b});
        if (packed) {
          aw >>= width;
          bw >>= width;
        }
      }
    }
    LOG("[decode_inputs] decoded=%zu\n", out.size());
    return out;
  }

  // Map FP32 C onto the common integer grid when Ec==E_anchor:
  //   int_C = m24 * 2^(SCALE_K - 23)  (SCALE_K-23 = 3)
  CT decodeC_to_common_scaled(const dec_t &d) {
    const uint32_t sb = 23, eb = 8, bias = (1u << (eb - 1)) - 1u;
    const uint32_t M = ((d.exp != 0) << sb) | d.frac;
    const int32_t Ec = d.exp - (int32_t)bias + 127;
    const uint32_t mScaled = (M << C_SHIFT_BASE_);
    const int32_t S = d.sign ? -int32_t(mScaled) : int32_t(mScaled);
    LOG("[decodeC] s=%u Ec=0x%x m24=0x%06x mScaled=0x%06x -> Sc=0x%x\n",
        d.sign, Ec, M, mScaled, S);
    return CT{S, Ec, (M == 0)};
  }

  //======================== S1: multiply_to_comm() ========================//
  // Partition into exactly lanes_ groups (balanced).
  // For each group:
  //   Eg = max Ep in group (FP32-field scale)
  //   Align each product to Eg into K_WIN, per-term sticky bucket; local CSA
  //   Local CPA → Vg (signed int at Eg), store gsticky
  std::vector<grp_term>
  multiply_to_comm(const std::vector<std::array<dec_t, 2>> &terms, int eb, int sb) {
    const uint32_t N = (uint32_t)terms.size();
    const uint32_t G = lanes_;
    const uint32_t gsize = ceil_div_u(N, G);
    LOG("[S1/group ] N=%u lanes=%u gsize=%u\n", N, G, gsize);

    // --- Pass 0: decode products (P, sign, Ep) --------------------------------
    const int32_t bias = (1 << (eb - 1)) - 1;
    std::vector<uint32_t> P(N, 0);
    std::vector<uint32_t> Sgn(N, 0);
    std::vector<int32_t> Ep(N, INT32_MIN);

    for (uint32_t i = 0; i < N; ++i) {
      const auto &a = terms[i][0], &b = terms[i][1];
      const bool a_norm = (a.exp != 0), b_norm = (b.exp != 0);
      const uint32_t mA = (a_norm << sb) | a.frac;
      const uint32_t mB = (b_norm << sb) | b.frac;
      const uint32_t prod = mA * mB; // unsigned product magnitude

      const int32_t ea_unb = a.exp - bias;
      const int32_t eb_unb = b.exp - bias;
      const int32_t ep_unb = ea_unb + eb_unb;
      const int32_t Ep_field = ep_unb + 127;

      P[i] = prod;
      Sgn[i] = a.sign ^ b.sign;
      Ep[i] = Ep_field;

      LOG("[S1/mul ] i=%u s=%u ea=%d eb=%d Ep=0x%x P=0x%x\n",
          i, Sgn[i], ea_unb, eb_unb,
          (unsigned)((Ep_field == INT32_MIN) ? 0 : Ep_field), prod);
    }

    // --- Pass 1: place each product onto the fixed K grid (K_WIN) --------------
    const int PROD_SHIFT_BASE = int(SCALE_K_) - (2 * sb); // fp8:20, bf16:12, fp16:6
    std::vector<uint32_t> Q(N, 0);
    std::vector<uint8_t> Qsticky(N, 0);

    for (uint32_t i = 0; i < N; ++i) {
      if (!P[i])
        continue;

      bool st = false;
      uint32_t tmp = P[i];

      // Fixed placement shift (LEFT) to fill the K window.
      if (PROD_SHIFT_BASE >= 32) {
        st |= (tmp != 0u);
        tmp = 0;
      } else if (PROD_SHIFT_BASE > 0) {
        const uint32_t overflow = (tmp >> (32u - PROD_SHIFT_BASE));
        st |= (overflow != 0u);
        tmp <<= PROD_SHIFT_BASE;
      }

      // Clip to K_WIN and bucket sticky if anything spilled.
      if (tmp >> K_WIN_) {
        st = true;
        tmp &= ((1u << K_WIN_) - 1u);
      }
      if (st)
        tmp |= 1u;

      Q[i] = tmp;
      Qsticky[i] = (uint8_t)st;

      LOG("[S1/place] i=%u base=%d Q=0x%x st=%u\n",
          i, PROD_SHIFT_BASE, Q[i], (unsigned)st);
    }

    // --- Pass 2: per-group reduction (RIGHT shift only to Eg) ------------------
    std::vector<grp_term> out(G, grp_term{0, INT32_MIN, false});

    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t i0 = g * gsize;
      const uint32_t i1 = std::min(i0 + gsize, N);

      // Find group anchor Eg = max Ep in group
      int32_t Eg = INT32_MIN;
      for (uint32_t i = i0; i < i1; ++i)
        if (P[i] != 0u)
          Eg = std::max(Eg, Ep[i]);

      if (Eg == INT32_MIN) {
        LOG("[S1/reduce] g=%u EMPTY\n", g);
        continue;
      }

      uint32_t S = 0, C = 0;
      bool gsticky = false;

      for (uint32_t i = i0; i < i1; ++i) {
        if (!P[i])
          continue;

        const uint32_t Qi = Q[i];
        const uint32_t d = (uint32_t)(Eg - Ep[i]); // >= 0
        uint32_t mag = Qi;
        bool term_sticky = (Qsticky[i] != 0);

        // RIGHT shift only for group alignment; accumulate sticky on dropped bits
        if (d >= 32) {
          term_sticky |= (mag != 0u);
          mag = 0;
        } else if (d) {
          term_sticky |= any_dropped32_u(mag, d);
          mag >>= d;
        }

        if (term_sticky)
          mag |= 1u; // LSB sticky bucket
        gsticky |= term_sticky;

        const uint32_t add = Sgn[i] ? uint32_t(-int32_t(mag)) : mag;
        auto [s1, c1] = csa32(S, (C << 1), add);
        S = s1;
        C = c1;

        LOG("[S1/align] g=%u i=%u delta=%u Qi=0x%x mag=0x%x add=%d  Sg=0x%x Cg=0x%x st=%u\n",
            g, i, d, Qi, mag, int32_t(add), S, C, (unsigned)term_sticky);
      }

      const int32_t Vg = (int32_t)S + ((int32_t)C << 1); // local CPA
      out[g] = grp_term{Vg, Eg, gsticky};
      LOG("[S1/out  ] g=%u Eg=0x%x Vg=0x%x (%d) gsticky=%u\n",
          g, (unsigned)Eg, (uint32_t)Vg, Vg, (unsigned)gsticky);
    }

    return out;
  }

  //======================== S2: alignment() ========================//
  // Compute global anchor E_anchor = max(Eprod, Ec).
  // Align each group Vg@Eg and C@Ec to E_anchor (sticky-aware, window-clipped for left shifts).
  // No CSA here—just produce aligned integers.
  align_out alignment(const std::vector<grp_term> &groups, const CT &c_term) {
    // 1) products' anchor
    int32_t Eprod = INT32_MIN;
    for (const auto &g : groups)
      if (g.V != 0 && g.Eg != INT32_MIN)
        Eprod = std::max(Eprod, g.Eg);

    const int32_t Ec = c_term.E;
    const int32_t E_anchor = (Eprod == INT32_MIN) ? Ec : std::max(Eprod, Ec);
    LOG("[S2] Eprod=0x%x Ec=0x%x -> E_anchor=0x%x\n",
        (unsigned)((Eprod == INT32_MIN) ? 0 : Eprod), (unsigned)Ec, (unsigned)E_anchor);

    align_out out;
    out.Vals.assign(lanes_, 0);
    out.E_anchor = E_anchor;

    // helper: align one integer Vin@Et to E_anchor
    auto align_one = [&](int32_t Vin, int32_t Et, bool term_sticky, const char *tag, int idx) -> int32_t {
      if (Vin == 0 || Et == INT32_MIN)
        return 0;
      int d = int(E_anchor - Et);
      bool st = false;
      int32_t Val = Vin;

      // Val = Vin * 2^(Et - E_anchor)
      Val = sign_mag_shr32(Val, (uint32_t)d, st);

      out.sticky_any |= uint32_t(st | term_sticky);

      LOG("[S2/align] %s i=%d Et=0x%x d=%d Val=0x%x st=%u\n",
          tag, idx, (unsigned)Et, d, (uint32_t)Val, (unsigned)st);
      return Val;
    };

    // 2) align all groups
    for (size_t i = 0; i < groups.size(); ++i)
      out.Vals[i] = align_one(groups[i].V, groups[i].Eg, groups[i].sticky, "G", (int)i);

    // 3) align C at Ec (pre-scaled <<3)
    out.Cal = align_one(c_term.S, Ec, /*term_sticky=*/false, "C", -1);

    return out;
  }

  //======================== S3: accumulate() (CSA only) ========================//
  s3_out accumulate(const align_out &al) {
    uint32_t S = 0, C = 0;
    for (size_t i = 0; i < al.Vals.size(); ++i) {
      const int32_t v = al.Vals[i];
      if (!v)
        continue;
      auto [s1, c1] = csa32(S, (C << 1), (uint32_t)v);
      S = s1;
      C = c1;
      LOG("[S3/accG] i=%zu v=0x%x -> S=0x%x C=0x%x\n", i, (uint32_t)v, S, C);
    }
    if (al.Cal) {
      auto [s1, c1] = csa32(S, (C << 1), (uint32_t)al.Cal);
      S = s1;
      C = c1;
      LOG("[S3/accC] Cal=0x%x -> S=0x%x C=0x%x\n", (uint32_t)al.Cal, S, C);
    }
    LOG("[S3/out ] (S,C) @ E_anchor=0x%x  S=0x%x C=0x%x\n",
        (unsigned)al.E_anchor, S, C);
    return s3_out{S, C, al.E_anchor};
  }

  //======================== S4: final CPA → normalize → round ========================//
  static inline void trim24(uint32_t mag, uint32_t &kept24, uint32_t &guard, bool &sticky) {
    if (mag == 0) {
      kept24 = 0;
      guard = 0;
      return;
    }
    const int nbits = 32 - clz32(mag);
    if (nbits > 24) {
      const int sh = nbits - 24;
      const uint32_t rem = (sh >= 32) ? mag : (mag & ((1u << sh) - 1u));
      kept24 = (sh >= 32) ? 0u : (mag >> sh);
      guard = (sh >= 1) ? ((rem >> (sh - 1)) & 1u) : 0u;
      const bool st2 = (sh >= 2) ? ((rem & ((1u << (sh - 1)) - 1u)) != 0u) : false;
      sticky |= st2;
    } else if (nbits < 24) {
      kept24 = mag << (24 - nbits);
      guard = 0;
    } else {
      kept24 = mag;
      guard = 0;
    }
  }

  uint32_t normalize_round_pack(const norm_in &ni) {
    // The one and only CPA
    const int32_t acc = (int32_t)ni.S + ((int32_t)ni.C << 1);
    const uint32_t sign = (acc < 0) ? 1u : 0u;
    uint32_t mag = (acc < 0) ? uint32_t(-acc) : uint32_t(acc);

    if (mag == 0)
      return sign << 31;

    // Trim to 24 bits, get guard, merge sticky
    bool sticky = ni.sticky_any;
    uint32_t kept24 = 0, guard = 0;
    trim24(mag, kept24, guard, sticky);

    // e_biased = E_anchor + (nbits-1) - SCALE_K
    const int nbits = 32 - clz32(mag);
    int32_t e_biased = ni.E_anchor + (nbits - 1) - int(SCALE_K_);

    LOG("[S4/norm ] E_anchor=0x%x sign=%u nbits=%d kept24=0x%x e_biased=%d guard=%u sticky=%u\n",
        (unsigned)ni.E_anchor, sign, nbits, kept24, e_biased, guard, (unsigned)sticky);

    // Overflow → Inf
    if (e_biased >= 255) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      return packInf32(sign);
    }

    // Normal
    if (e_biased >= 1) {
      const uint32_t frac = kept24 & ((1u << 23) - 1u);
      const uint32_t lsb = frac & 1u;
      if (roundInc(frm_, sign, lsb, guard, sticky)) {
        const uint32_t t = kept24 + 1u;
        if (t >= (1u << 24)) {
          kept24 = t >> 1;
          ++e_biased;
        } else
          kept24 = t;
        fflags_ |= FLAG_NX;
      }
      if (e_biased >= 255) {
        fflags_ |= (FLAG_OF | FLAG_NX);
        return packInf32(sign);
      }
      const uint32_t out = (sign << 31) | (uint32_t(e_biased) << 23) | (kept24 & ((1u << 23) - 1u));
      LOG("[S4/out  ] normal=1 out=0x%08x\n", out);
      return out;
    }

    // Subnormals (e_biased <= 0)
    const int sh2 = 1 - e_biased; // ≥1
    const uint32_t shifted = (sh2 >= 32) ? 0u : (kept24 >> sh2);
    const uint32_t rem2 = (sh2 >= 32) ? kept24 : (kept24 & ((1u << sh2) - 1u));
    const uint32_t guard2 = (sh2 >= 1) ? ((rem2 >> (sh2 - 1)) & 1u) : 0u;
    const bool st2 = (sh2 >= 2) ? ((rem2 & ((1u << (sh2 - 1)) - 1u)) != 0u) : false;

    uint32_t frac_keep = shifted & ((1u << 23) - 1u);
    const uint32_t lsb2 = frac_keep & 1u;

    if (guard2 || st2)
      fflags_ |= FLAG_NX;
    if (roundInc(frm_, sign, lsb2, guard2, st2)) {
      const uint32_t t = frac_keep + 1u;
      if (t >= (1u << 23)) {
        const uint32_t out = (sign << 31) | (1u << 23);
        LOG("[S4/out  ] subnorm->minNorm out=0x%08x\n", out);
        return out;
      }
      const uint32_t out = (sign << 31) | t;
      LOG("[S4/out  ] subnorm(roundUp) out=0x%08x\n", out);
      return out;
    } else {
      const uint32_t out = (sign << 31) | frac_keep;
      LOG("[S4/out  ] subnorm(roundDown) out=0x%08x\n", out);
      return out;
    }
  }

  //============================ Members ============================//
  // (None beyond what's above)
};
