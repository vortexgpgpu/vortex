// fedp_anchor_split.h
// Anchor/windowed FEDP with S1 group reduction, S2 alignment, S3 accumulate (CSA+CPA), S4 normalize+round.
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
    const CT c_term = decodeC_to_common(c_dec); // Sc = m24 << 3, Ec in FP32-field scale

    // ---- S1: multiply + per-group local CPA → lanes_ outputs ----------------
    const auto groups = multiply_to_common(terms, exp_bits, sig_bits);

    // ---- S2: alignment to global anchor (includes C); no CSA here -----------
    const auto al = alignment(groups, c_term);

    // ---- S3: CSA accumulate aligned terms + aligned C + CPA -----------------
    const auto s3 = accumulate(al);

    // ---- S4: normalize + round ---------------------------------------------
    auto no = normalize(s3, al.sticky_any);
    const uint32_t out = rounding(no);
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
    uint32_t sign;
    uint32_t V;
    int32_t E;
    bool zero;
  };

  struct grp_term {
    uint32_t sign;
    uint32_t V;
    int32_t Eg;
    bool sticky;
  }; // S1 group output: unsigned int at Eg

  struct align_out {
    std::vector<int32_t> Vals; // size == lanes_ (aligned to E_anchor)
    int32_t Cal{0};            // aligned C at E_anchor
    int32_t E_anchor{INT32_MIN};
    uint32_t sticky_any{0};
  };

  struct s3_out {
    int32_t acc;               // CPA result
    int32_t E_anchor{INT32_MIN};
  };

  struct norm_in {
    int32_t acc;
    int32_t E_anchor;
    bool sticky_any;
  };

  struct norm_out {
    uint32_t sign;
    uint32_t mag;
    int32_t e_biased;
    uint32_t kept24;
    uint32_t guard;
    bool sticky;
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

  // Sticky-aware magnitude shifts on 32b unsigned integers
  static inline uint32_t mag_shr32(uint32_t m, uint32_t k, bool &st) {
    if (k == 0 || m == 0)
      return m;
    if (k >= 32) {
      st |= (m != 0);
      return 0;
    }
    st |= ((m & ((1u << k) - 1u)) != 0u);
    return m >> k;
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
    std::vector<std::array<dec_t, 2>> out(n_words * epw);
    assert(a_words.size() >= n_words && b_words.size() >= n_words);
    for (uint32_t w = 0; w < n_words; ++w) {
      uint32_t aw = a_words[w], bw = b_words[w];
      for (uint32_t i = 0; i < epw; ++i) {
        const uint32_t aenc = packed ? (aw & mask) : aw, benc = packed ? (bw & mask) : bw;
        auto a = decode_input(aenc, eb, sb);
        auto b = decode_input(benc, eb, sb);
        LOG("[decode] lane=%u A(s=%u e=%u f=0x%x)  B(s=%u e=%u f=0x%x)\n",
            (w * epw) + i, a.sign, a.exp, a.frac, b.sign, b.exp, b.frac);
        out[w * epw + i] = {a, b};
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
  CT decodeC_to_common(const dec_t &d) {
    const uint32_t sb = 23, eb = 8, bias = (1u << (eb - 1)) - 1u;
    const uint32_t M = ((d.exp != 0) << sb) | d.frac;
    const int32_t Ec = d.exp - (int32_t)bias + 127;
    const uint32_t mScaled = (M << C_SHIFT_BASE_);
    CT ct{};
    ct.sign = d.sign;
    ct.V = mScaled;
    ct.E = Ec;
    ct.zero = (M == 0);
    LOG("[decodeC] s=%u Ec=0x%x m24=0x%06x mScaled=0x%06x -> Vc=0x%x\n",
        ct.sign, Ec, M, mScaled, ct.V);
    return ct;
  }

  //======================== S1: multiply_to_common() ========================//
  std::vector<grp_term>
  reduce_terms(const std::vector<uint32_t> &P,
               const std::vector<uint32_t> &Sgn,
               const std::vector<int32_t> &Ep,
               const std::vector<uint32_t> &Q,
               const std::vector<uint8_t> &Qsticky,
               uint32_t N,
               uint32_t G,
               uint32_t gsize) {
    constexpr uint32_t DELTA_BITS = 8;
    constexpr uint32_t K_OVERLAP  = 2;

    std::vector<grp_term> out(G, grp_term{0, 0, INT32_MIN, false});
    assert(G <= 8);

    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t i0 = g * gsize;
      const uint32_t i1 = std::min(i0 + gsize, N);

      // Find group anchor Eg = max Ep in group
      int32_t Eg = INT32_MIN;
      for (uint32_t i = i0; i < i1; ++i) {
        Eg = std::max(Eg, Ep[i]);
      }

      const int32_t Elo = Eg - (int32_t)DELTA_BITS;
      const uint32_t HI_THRESH = DELTA_BITS + K_OVERLAP;

      // Separate pos/neg for unsigned CSAs (HI and LO)
      uint32_t S_hi_pos = 0, C_hi_pos = 0, S_hi_neg = 0, C_hi_neg = 0;
      uint32_t S_lo_pos = 0, C_lo_pos = 0, S_lo_neg = 0, C_lo_neg = 0;
      bool gsticky = false;

      for (uint32_t i = i0; i < i1; ++i) {
        const uint32_t Qi = Q[i];
        const uint32_t d_to_Eg = (uint32_t)(Eg - Ep[i]); // >= 0
        uint32_t mag = Qi;
        bool term_sticky = (Qsticky[i] != 0);

        // Decide bin by distance to Eg (keep K_OVERLAP bits in HI bin boundary)
        const bool to_hi = (d_to_Eg <= HI_THRESH);

        // Align to chosen anchor; accumulate sticky on dropped bits
        if (to_hi) {
          // align to Eg
          if (d_to_Eg < 32) {
            term_sticky |= ((mag & ((1u << d_to_Eg) - 1u)) != 0u);
            mag >>= d_to_Eg;
          } else {
            term_sticky |= (mag != 0u);
            mag = 0;
          }
          if (term_sticky) mag |= 1u;

          if (mag) {
            auto& S_hi = (Sgn[i] == 0) ? S_hi_pos : S_hi_neg;
            auto& C_hi = (Sgn[i] == 0) ? C_hi_pos : C_hi_neg;
            auto [s1, c1] = csa32(S_hi, (C_hi << 1), mag);
            S_hi = s1; C_hi = c1;
          }
        } else {
          // align to Elo = Eg - DELTA_BITS
          const uint32_t d_to_Elo = (uint32_t)(Elo - Ep[i]); // >= 0 and smaller than d_to_Eg by ~DELTA
          if (d_to_Elo < 32) {
            term_sticky |= ((mag & ((1u << d_to_Elo) - 1u)) != 0u);
            mag >>= d_to_Elo;
          } else {
            term_sticky |= (mag != 0u);
            mag = 0;
          }
          if (term_sticky) mag |= 1u;

          if (mag) {
            // NOTE: LO path mag is ≈ (K_WIN - DELTA) bits after alignment.
            // Using the same 32b type here, but in RTL you can narrow this adder.
            auto& S_lo = (Sgn[i] == 0) ? S_lo_pos : S_lo_neg;
            auto& C_lo = (Sgn[i] == 0) ? C_lo_pos : C_lo_neg;
            auto [s1, c1] = csa32(S_lo, (C_lo << 1), mag);
            S_lo = s1; C_lo = c1;
          }
        }

        gsticky |= term_sticky;

        LOG("[S1/align] g=%u i=%u to_%s dEg=%u mag=0x%x Sgn=%u st=%u\n",
            g, i, to_hi ? "HI":"LO", d_to_Eg, mag, (unsigned)Sgn[i], (unsigned)term_sticky);
      }

      // Per-bin FULL ADDERs (unchanged style): local CPAs
      const uint32_t V_hi_pos = (uint32_t)S_hi_pos + ((uint32_t)C_hi_pos << 1);
      const uint32_t V_hi_neg = (uint32_t)S_hi_neg + ((uint32_t)C_hi_neg << 1);
      const uint32_t V_lo_pos = (uint32_t)S_lo_pos + ((uint32_t)C_lo_pos << 1);
      const uint32_t V_lo_neg = (uint32_t)S_lo_neg + ((uint32_t)C_lo_neg << 1);

      // Merge bins at group end: up-shift LO by DELTA_BITS into HI domain.
      // Any bits dropped by this up-shift feed sticky.
      bool merge_sticky_pos = ((V_lo_pos & ((1u << DELTA_BITS) - 1u)) != 0u);
      bool merge_sticky_neg = ((V_lo_neg & ((1u << DELTA_BITS) - 1u)) != 0u);
      uint32_t V_lo_up_pos = V_lo_pos >> DELTA_BITS;
      uint32_t V_lo_up_neg = V_lo_neg >> DELTA_BITS;
      gsticky |= (merge_sticky_pos || merge_sticky_neg);

      // Final group merge: single add in Eg domain
      const uint32_t V_total_pos = V_hi_pos + V_lo_up_pos;
      const uint32_t V_total_neg = V_hi_neg + V_lo_up_neg;
      uint32_t Vg_mag;
      uint32_t Vg_sign;
      if (V_total_pos >= V_total_neg) {
        Vg_mag = V_total_pos - V_total_neg;
        Vg_sign = 0;
      } else {
        Vg_mag = V_total_neg - V_total_pos;
        Vg_sign = 1;
      }

      out[g] = grp_term{Vg_sign, Vg_mag, Eg, gsticky};
      LOG("[S1/out  ] g=%u Eg=0x%x V_hi_pos=0x%x V_hi_neg=0x%x V_lo_pos=0x%x V_lo_neg=0x%x -> Vg_mag=0x%x Vg_sign=%u st=%u\n",
          g, (unsigned)Eg, V_hi_pos, V_hi_neg, V_lo_pos, V_lo_neg, Vg_mag, Vg_sign, (unsigned)gsticky);
    }

    return out;
  }

  std::vector<grp_term>
  multiply_to_common(const std::vector<std::array<dec_t, 2>> &terms, int eb, int sb) {
    const uint32_t N = (uint32_t)terms.size();
    const uint32_t G = lanes_;
    const uint32_t gsize = ceil_div_u(N, G);
    LOG("[S1/group ] N=%u lanes=%u gsize=%u\n", N, G, gsize);

    // --- Pass 0: decode products (P, sign, Ep) --------------------------------
    const int32_t bias = (1 << (eb - 1)) - 1;
    std::vector<uint32_t> P(N, 0);
    std::vector<uint32_t> Sgn(N, 0);
    std::vector<int32_t>  Ep(N, INT32_MIN);

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

      P[i]   = prod;
      Sgn[i] = a.sign ^ b.sign;
      Ep[i]  = Ep_field;

      LOG("[S1/mul ] i=%u s=%u ea=%d eb=%d Ep=0x%x P=0x%x\n",
          i, Sgn[i], ea_unb, eb_unb,
          (unsigned)((Ep_field == INT32_MIN) ? 0 : Ep_field), prod);
    }

    // --- Pass 1: place each product onto the fixed K grid (K_WIN) --------------
    const int PROD_SHIFT_BASE = int(SCALE_K_) - (2 * sb); // fp8:20, bf16:12, fp16:6
    std::vector<uint32_t> Q(N, 0);
    std::vector<uint8_t>  Qsticky(N, 0);

    for (uint32_t i = 0; i < N; ++i) {
      bool st = false;
      uint32_t tmp = P[i];

      // Fixed placement shift (LEFT) to fill the K window.
      if (PROD_SHIFT_BASE < 32) {
        const uint32_t overflow = (tmp >> (32u - PROD_SHIFT_BASE));
        st |= (overflow != 0u);
        tmp <<= PROD_SHIFT_BASE;
      } else {
        st |= (tmp != 0u);
        tmp = 0;
      }

      // Clip to K_WIN and bucket sticky if anything spilled.
      if (tmp >> K_WIN_) {
        st = true;
        tmp &= ((1u << K_WIN_) - 1u);
      }
      if (st) tmp |= 1u;

      Q[i] = tmp;
      Qsticky[i] = (uint8_t)st;

      LOG("[S1/place] i=%u base=%d Q=0x%x st=%u\n",
          i, PROD_SHIFT_BASE, Q[i], (unsigned)st);
    }

    // --- Pass 2: per-group reduction with 2 sub-blocks (block-floating + overlap)
    if (gsize > 1) {
      return reduce_terms(P, Sgn, Ep, Q, Qsticky, N, G, gsize);
    } else {
      // If gsize == 1, each group is one term, no reduction needed
      std::vector<grp_term> out(G, grp_term{0, 0, INT32_MIN, false});
      assert(N <= G);
      for (uint32_t i = 0; i < N; ++i) {
        out[i] = grp_term{Sgn[i], Q[i], Ep[i], Qsticky[i] != 0};
      }
      return out;
    }
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
    auto align_one = [&](uint32_t Vin, uint32_t Vin_sign, int32_t Et, bool term_sticky, const char *tag, int idx) -> int32_t {
      int d = int(E_anchor - Et);
      bool st = false;
      uint32_t Val_mag = mag_shr32(Vin, (uint32_t)d, st);
      out.sticky_any |= uint32_t(st | term_sticky);
      int32_t Val = (Vin_sign ? -int32_t(Val_mag) : int32_t(Val_mag));
      LOG("[S2/align] %s i=%d Et=0x%x d=%d Val_mag=0x%x st=%u\n",
          tag, idx, (unsigned)Et, d, Val_mag, (unsigned)st);
      return Val;
    };

    // 2) align all groups
    for (size_t i = 0; i < groups.size(); ++i)
      out.Vals[i] = align_one(groups[i].V, groups[i].sign, groups[i].Eg, groups[i].sticky, "G", (int)i);

    // 3) align C at Ec (pre-scaled <<3)
    out.Cal = align_one(c_term.V, c_term.sign, Ec, /*term_sticky=*/false, "C", -1);

    return out;
  }

  //======================== S3: accumulate() (CSA only) ========================//
  s3_out accumulate(const align_out &al) {
    uint32_t S = 0, C = 0;
    for (size_t i = 0; i < al.Vals.size(); ++i) {
      const int32_t v = al.Vals[i];
      auto [s1, c1] = csa32(S, (C << 1), (uint32_t)v);
      S = s1;
      C = c1;
      LOG("[S3/accG] i=%zu v=0x%x -> S=0x%x C=0x%x\n", i, (uint32_t)v, S, C);
    }
    auto [s1, c1] = csa32(S, (C << 1), (uint32_t)al.Cal);
    S = s1;
    C = c1;
    LOG("[S3/accC] Cal=0x%x -> S=0x%x C=0x%x\n", (uint32_t)al.Cal, S, C);
    const int32_t acc = (int32_t)S + ((int32_t)C << 1);
    LOG("[S3/out ] (S,C) @ E_anchor=0x%x  S=0x%x C=0x%x acc=0x%x\n",
        (unsigned)al.E_anchor, S, C, (uint32_t)acc);
    return s3_out{acc, al.E_anchor};
  }

  //======================== S4: normalize() ========================//
  norm_out normalize(const s3_out &s3, bool sticky_any) {
    norm_out no;
    no.sign = (s3.acc < 0) ? 1u : 0u;
    no.mag = (s3.acc < 0) ? uint32_t(-s3.acc) : uint32_t(s3.acc);

    if (no.mag == 0) {
      no.e_biased = 0;
      no.kept24 = 0;
      no.guard = 0;
      no.sticky = false;
      return no;
    }

    // Trim to 24 bits, get guard, merge sticky
    no.sticky = sticky_any;
    const int nbits = 32 - clz32(no.mag);
    if (nbits > 24) {
      const int sh = nbits - 24;
      const uint32_t rem = (sh >= 32) ? no.mag : (no.mag & ((1u << sh) - 1u));
      no.kept24 = (sh >= 32) ? 0u : (no.mag >> sh);
      no.guard = (sh >= 1) ? ((rem >> (sh - 1)) & 1u) : 0u;
      const bool st2 = (sh >= 2) ? ((rem & ((1u << (sh - 1)) - 1u)) != 0u) : false;
      no.sticky |= st2;
    } else if (nbits < 24) {
      no.kept24 = no.mag << (24 - nbits);
      no.guard = 0;
    } else {
      no.kept24 = no.mag;
      no.guard = 0;
    }

    // e_biased = E_anchor + (nbits-1) - SCALE_K
    no.e_biased = s3.E_anchor + (nbits - 1) - int(SCALE_K_);

    LOG("[S4/norm ] E_anchor=0x%x sign=%u nbits=%d kept24=0x%x e_biased=%d guard=%u sticky=%u\n",
        (unsigned)s3.E_anchor, no.sign, nbits, no.kept24, no.e_biased, no.guard, (unsigned)no.sticky);

    return no;
  }

  //======================== S4: rounding() ========================//
  uint32_t rounding(norm_out &no) {
    // Handle zero
    if (no.mag == 0)
      return no.sign << 31;

    // Overflow → Inf
    if (no.e_biased >= 255) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      return packInf32(no.sign);
    }

    // Normal
    if (no.e_biased >= 1) {
      uint32_t frac = no.kept24 & ((1u << 23) - 1u);
      const uint32_t lsb = frac & 1u;
      uint32_t e_biased = no.e_biased;
      if (roundInc(frm_, no.sign, lsb, no.guard, no.sticky)) {
        const uint32_t t = no.kept24 + 1u;
        if (t >= (1u << 24)) {
          no.kept24 = t >> 1;
          ++e_biased;
        } else {
          no.kept24 = t;
        }
        fflags_ |= FLAG_NX;
      }
      if (e_biased >= 255) {
        fflags_ |= (FLAG_OF | FLAG_NX);
        return packInf32(no.sign);
      }
      const uint32_t out = (no.sign << 31) | (uint32_t(e_biased) << 23) | (no.kept24 & ((1u << 23) - 1u));
      LOG("[S4/out  ] normal=1 out=0x%08x\n", out);
      return out;
    }

    // Subnormals (e_biased <= 0)
    const int sh2 = 1 - no.e_biased; // ≥1
    const uint32_t shifted = (sh2 >= 32) ? 0u : (no.kept24 >> sh2);
    const uint32_t rem2 = (sh2 >= 32) ? no.kept24 : (no.kept24 & ((1u << sh2) - 1u));
    const uint32_t guard2 = (sh2 >= 1) ? ((rem2 >> (sh2 - 1)) & 1u) : 0u;
    const bool st2 = (sh2 >= 2) ? ((rem2 & ((1u << (sh2 - 1)) - 1u)) != 0u) : false;

    uint32_t frac_keep = shifted & ((1u << 23) - 1u);
    const uint32_t lsb2 = frac_keep & 1u;

    if (guard2 || st2)
      fflags_ |= FLAG_NX;
    if (roundInc(frm_, no.sign, lsb2, guard2, st2)) {
      const uint32_t t = frac_keep + 1u;
      if (t >= (1u << 23)) {
        const uint32_t out = (no.sign << 31) | (1u << 23);
        LOG("[S4/out  ] subnorm->minNorm out=0x%08x\n", out);
        return out;
      }
      const uint32_t out = (no.sign << 31) | t;
      LOG("[S4/out  ] subnorm(roundUp) out=0x%08x\n", out);
      return out;
    } else {
      const uint32_t out = (no.sign << 31) | frac_keep;
      LOG("[S4/out  ] subnorm(roundDown) out=0x%08x\n", out);
      return out;
    }
  }
};