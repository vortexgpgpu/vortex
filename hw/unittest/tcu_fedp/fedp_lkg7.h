// fedp_lkg10.h
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <vector>

#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

struct FEDP_Log {
  static inline void p(const char* fmt, ...) {
    if constexpr (FEDP_TRACE) { va_list a; va_start(a, fmt); std::vprintf(fmt, a); va_end(a); }
  }
};
#define LOG(...) FEDP_Log::p(__VA_ARGS__)

class FEDP {
public:
  // RISC-V rounding modes
  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };

  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    assert(frm_>=0 && frm_<=4);
    assert(lanes_>=1 && lanes_<=8);        // Lanes <= 8
  }

  float operator()(const std::vector<uint32_t>& a_words,
                   const std::vector<uint32_t>& b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits) {
    fflags_ = 0;

    const uint32_t width = 1u + uint32_t(exp_bits) + uint32_t(sig_bits);
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t elems_per_word = packed ? (32u / width) : 1u;

    LOG("[ctor] frm=%d lanes=%u  Wc=%u  Win=%u\n", frm_, lanes_, Wc_, Win_);
    LOG("[inputs] fmt=e%um%u width=%u packed=%u elems/word=%u n_words=%u\n",
        (unsigned)exp_bits, (unsigned)sig_bits, width, (unsigned)packed, elems_per_word, n_words);

    // decode inputs
    const auto ab = decode_inputs(a_words.data(), b_words.data(),
                                  n_words, elems_per_word, exp_bits, sig_bits, packed);

    // decode C (FP32)
    const uint32_t c_bits = bitsFromF32(c);
    const auto c_dec = decode_input(c_bits, 8, 23);

    // S1: multiply & group to single-value terms (UNBIASED E)
    auto groups = multiply_to_common(ab, exp_bits, sig_bits);

    // S2: align product groups to Emax_p (unbiased)
    auto [aligned_vals, Emax_p, sticky_p] = alignment(groups);

    // Map C to internal grid (unbiased)
    const term24_t cterm = decodeC_to_common(c_dec);

    // S3: accumulate with one CPA (dominance-aware Eref)
    auto [Vf, Eref, sticky_all] = accumulate(aligned_vals, Emax_p, cterm, sticky_p);

    // S4: normalize / round / pack to FP32
    const auto norm = normalize(Vf, Eref);
    const uint32_t out_bits = rounding(norm, sticky_all);

    return f32FromBits(out_bits);
  }

private:
  // ---------- config ----------
  static constexpr uint32_t Wc_  = 24;  // kept mantissa (incl hidden 1)
  static constexpr uint32_t Win_ = 25;  // internal adder width (for docs/logs)

  // flags
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // ---------- types ----------
  struct dec_t {
    uint32_t sign{0}, exp{0}, frac{0};
    bool is_zero{false}, is_inf{false}, is_nan{false};
  };
  struct term24_t {
    uint32_t sign{0};
    uint32_t V{0};
    int32_t  E{0};      // unbiased exponent domain
    bool     is_zero{true};
  };
  struct norm_t {
    uint32_t sign{0}, kept24{0}, round_bit{0};
    int32_t  e_unb{0};
    bool sticky{false};
  };

  // ---------- bit utils ----------
  static inline uint32_t bitsFromF32(float f){ uint32_t u; std::memcpy(&u,&f,sizeof u); return u; }
  static inline float f32FromBits(uint32_t u){ float f; std::memcpy(&f,&u,sizeof f); return f; }
  static inline uint32_t packInf32(uint32_t s){ return (s<<31) | (0xFFu<<23); }
  static inline uint32_t packNaN32(){ return (0xFFu<<23) | 0x004000u; }

  // ---------- decode ----------
  static dec_t decode_input(uint32_t enc, int eb, int sb) {
    const uint32_t s = enc >> (eb + sb);
    const uint32_t e = (enc >> sb) & ((1u<<eb) - 1u);
    const uint32_t f = enc & ((1u<<sb) - 1u);
    dec_t d; d.sign=s; d.exp=e; d.frac=f;
    if (e==0 && f==0) { d.is_zero=true; }
    else if (e==((1u<<eb)-1u) && f==0) { d.is_inf=true; }
    else if (e==((1u<<eb)-1u) && f!=0) { d.is_nan=true; }
    return d;
  }

  std::vector<std::array<dec_t,2>>
  decode_inputs(const uint32_t *aw, const uint32_t *bw, uint32_t n_words,
                uint32_t elems_per_word, int eb, int sb, bool packed) {
    std::vector<std::array<dec_t,2>> out;
    out.reserve(n_words * elems_per_word);

    const uint32_t width = 1u + eb + sb;
    const uint32_t mask  = (width==32) ? 0xFFFFFFFFu : ((1u<<width)-1u);

    for (uint32_t w=0; w<n_words; ++w) {
      uint32_t a_enc = aw[w], b_enc = bw[w];
      for (uint32_t i=0; i<elems_per_word; ++i) {
        const uint32_t a_i = packed ? (a_enc & mask) : a_enc;
        const uint32_t b_i = packed ? (b_enc & mask) : b_enc;
        out.push_back({ decode_input(a_i, eb, sb), decode_input(b_i, eb, sb) });
        if (packed) { a_enc >>= width; b_enc >>= width; }
      }
    }
    return out;
  }

  // Map FP32 C onto Wc grid (unbiased). No early returns; mask zero case.
  term24_t decodeC_to_common(const dec_t& c) const {
    term24_t t{};
    const bool z = c.is_zero;
    const uint32_t Mc = ((c.exp!=0)? (1u<<23) : 0u) | c.frac; // 24b
    const int32_t  Eu = int32_t(c.exp) - 127;                 // unbiased
    const uint32_t m  = (Wc_==24) ? Mc : (Mc << (Wc_-24));
    t.sign = z ? 0u : c.sign;
    t.V    = z ? 0u : m;
    t.E    = z ? 0   : Eu;
    t.is_zero = (t.V == 0);
    LOG("[decodeC] s=%u Eu=%d m24=0x%06x -> Vc=0x%08x E(unb)=%d z=%u\n",
        c.sign, Eu, Mc, t.V, t.E, (unsigned)t.is_zero);
    return t;
  }

  // ---------- S1: multiply & group (LOP Ep, single-path) ----------
  std::vector<term24_t>
  multiply_to_common(const std::vector<std::array<dec_t,2>>& terms, int eb, int sb) {
    const int32_t bias = (1<<(eb-1)) - 1;
    const uint32_t Wm_in = uint32_t(sb + 1); // mant bits incl hidden 1
    assert(!terms.empty());

    struct Raw { uint32_t sign; uint32_t mag; int32_t E; bool zero; };
    std::vector<Raw> prods;
    prods.reserve(terms.size());

    for (size_t i=0;i<terms.size();++i) {
      const auto& a = terms[i][0];
      const auto& b = terms[i][1];

      const bool kill = (a.is_zero || b.is_zero || a.is_nan || b.is_nan || a.is_inf || b.is_inf);

      const bool a_norm = (a.exp != 0);
      const bool b_norm = (b.exp != 0);

      const uint32_t Ma = (a_norm ? (1u << sb) : 0u) | a.frac;
      const uint32_t Mb = (b_norm ? (1u << sb) : 0u) | b.frac;

      const int lzc_a = int(lzc_n(Ma, Wm_in));
      const int lzc_b = int(lzc_n(Mb, Wm_in));
      const uint32_t plus1 = (a_norm && b_norm && ((a.frac + b.frac) >= (1u << sb))) ? 1u : 0u;

      const int32_t Ea = int32_t(a.exp) - bias; // unbiased
      const int32_t Eb = int32_t(b.exp) - bias; // unbiased
      const int32_t E_unb = Ea + Eb + int32_t(plus1) - int32_t(lzc_a + lzc_b);

      const uint32_t top_est = 2u*sb - uint32_t(lzc_a + lzc_b) + plus1; // 0..(2*sb+1)

      const uint32_t P = Ma * Mb; // fits 32b for fp8/16/bf16
      const int shift = int(Wc_-1) - int(top_est);
      uint32_t m_wc = 0;
      bool sticky_local = false;

      if (shift >= 0) {
        m_wc = (shift >= 32) ? 0u : (P << uint32_t(shift));
      } else {
        const uint32_t r = uint32_t(-shift);
        if (r >= 32) { sticky_local |= (P != 0u); m_wc = 0u; }
        else { sticky_local |= any_dropped32_u(P, r); m_wc = P >> r; }
      }

      const uint32_t sign = uint32_t(a.sign ^ b.sign);
      Raw out{
        kill ? 0u : sign,
        kill ? 0u : m_wc,
        kill ? 0  : E_unb,
        kill ? true : (m_wc == 0)
      };
      prods.push_back(out);
      LOG("[S1/prod] i=%zu, s=%u, Ea=%d Eb=%d, lzc_a=%d lzc_b=%d, plus1=%u, "
          "E(unb)=%d, top_est=%u, m_wc=0x%08x, st_local=%u, kill=%u\n",
          i, sign, Ea, Eb, lzc_a, lzc_b, plus1,
          E_unb, top_est, m_wc, (unsigned)sticky_local, (unsigned)kill);
    }

    const uint32_t N = (uint32_t)prods.size();
    assert(N > 0);
    const uint32_t gsize = ceil_div_u(N, lanes_);
    LOG("[S1/group ] N=%u lanes=%u gsize=%u\n", N, lanes_, gsize);

    std::vector<term24_t> out;
    out.reserve((N + gsize - 1) / gsize);

    for (size_t g_beg = 0; g_beg < N; g_beg += gsize) {
      const size_t g_end = std::min<size_t>(N, g_beg + gsize);
      const size_t glen  = g_end - g_beg;
      assert(glen >= 1);

      if (glen == 1) {
        const auto &t = prods[g_beg];
        term24_t r{};
        r.sign = t.sign;
        r.V    = t.mag;
        r.E    = t.E;
        r.is_zero = t.zero || (t.mag == 0);
        out.push_back(r);
        LOG("[S1/single] g=%zu E(unb)=%d s=%u V=0x%08x\n", g_beg/gsize, r.E, r.sign, r.V);
      } else {
        int32_t Eg = INT32_MIN;
        for (size_t i=g_beg; i<g_end; ++i) {
          const bool nz = !prods[i].zero;
          Eg = nz ? std::max(Eg, prods[i].E) : Eg;
        }
        term24_t r{};
        if (Eg == INT32_MIN) {
          r = term24_t{0u,0u,0,true};
          LOG("[S1/group ] g=%zu empty(all-zero)\n", g_beg/gsize);
        } else {
          r = reduce_terms(prods.data() + g_beg, glen, Eg);
          LOG("[S1/reduced] g=%zu Eg(unb)=%d s=%u V=0x%08x\n", g_beg/gsize, Eg, r.sign, r.V);
        }
        out.push_back(r);
      }
    }

    LOG("[S1] groups=%zu\n", out.size());
    return out;
  }

  // reduce to one signed/magnitude term at Eg
  term24_t reduce_terms(const void* raw_ptr, size_t glen, int32_t Eg) {
    struct view_t { uint32_t sign; uint32_t mag; int32_t E; bool zero; };
    const view_t* v = reinterpret_cast<const view_t*>(raw_ptr);

    uint32_t S = 0, C = 0;
    for (size_t i=0; i<glen; ++i) {
      const bool skip = v[i].zero;
      bool st=false;
      const uint32_t delta = uint32_t((Eg >= v[i].E) ? (Eg - v[i].E) : 0);
      const uint32_t m_aligned = skip ? 0u : mag_shr(v[i].mag, delta, st);
      (void)st;
      const int32_t add_signed = skip ? 0 : (v[i].sign ? -int32_t(m_aligned) : int32_t(m_aligned));
      auto [s1, c1] = csa32(S, (C<<1), (uint32_t)add_signed);
      S = s1; C = c1;
    }
    const uint32_t Vu = S + (C<<1);
    const bool     neg = (Vu & 0x80000000u) != 0u;
    const uint32_t mag = neg ? (uint32_t)(~Vu + 1u) : Vu;

    term24_t r{}; r.sign = neg ? 1u : 0u; r.V = mag; r.E = Eg; r.is_zero = (mag == 0);
    return r;
  }

  // ---------- S2: align products to Emax_p (renamed to alignment) ----------
  std::tuple<std::vector<int32_t>, int32_t, bool>
  alignment(const std::vector<term24_t>& groups) {
    int32_t Emax_p = INT32_MIN;
    for (const auto& g : groups) {
      const bool nz = !g.is_zero;
      Emax_p = nz ? std::max(Emax_p, g.E) : Emax_p;
    }

    const bool all_zero = (Emax_p == INT32_MIN);
    Emax_p = all_zero ? 0 : Emax_p;

    std::vector<int32_t> out; out.reserve(groups.size());
    bool sticky = false;

    for (size_t i = 0; i < groups.size(); ++i) {
      const auto& t = groups[i];
      const bool use = (!t.is_zero) && (!all_zero);
      const uint32_t delta = uint32_t(Emax_p - t.E);
      bool st = false;
      const uint32_t mag_shift = use ? mag_shr(t.V, delta, st) : 0u;
      sticky |= (use ? st : false);
      const int32_t val = use ? (t.sign ? -int32_t(mag_shift) : int32_t(mag_shift)) : 0;
      if (use) out.push_back(val);

      LOG("[S2/align] i=%zu delta=%u s=%u Vin=0x%08x Vout_mag=0x%08x -> val=0x%08x st=%u use=%u\n",
          i, delta, t.sign, t.V, mag_shift, (uint32_t)val, (unsigned)st, (unsigned)use);
    }

    return {out, Emax_p, sticky};
  }

  // ---------- S3: single-path accumulate ----------
  std::tuple<int32_t,int32_t,bool>
  accumulate(const std::vector<int32_t>& aligned_vals, int32_t Emax_p,
             const term24_t& cterm, bool sticky_prev) {

    const bool use_c = !(cterm.is_zero || cterm.V == 0);
    const int32_t Ec = cterm.E;  // unbiased

    // preview Vp magnitude at Emax_p (exact 32-bit)
    int32_t Vp32 = 0;
    for (int32_t v : aligned_vals) Vp32 += v;
    const uint32_t Vp_mag = (Vp32 < 0) ? uint32_t(-Vp32) : uint32_t(Vp32);
    const uint32_t nbits  = (Vp_mag==0)? 0u : (32u - clz32(Vp_mag));
    const int32_t E_vp_unb = (nbits==0) ? INT32_MIN
                                        : (Emax_p - int32_t(Wc_-1)) + int32_t(nbits-1);

    // dominance-aware reference (no early exit)
    int32_t Eref = Emax_p;
    const bool c_dom = use_c ? ((E_vp_unb==INT32_MIN) ? true : (Ec >= E_vp_unb)) : false;
    if (use_c) {
      if (c_dom) {
        if (Emax_p >= Ec) {
          bool st=false; uint32_t Vp_at_Ec = mag_shr(Vp_mag, uint32_t(Emax_p - Ec), st);
          Eref = (Vp_at_Ec <= 1u) ? Ec : Emax_p;
        } else {
          Eref = Ec;
        }
      } else {
        if (Ec > Emax_p) {
          bool st=false; uint32_t Vc_at_Emax = mag_shr(cterm.V, uint32_t(Ec - Emax_p), st);
          Eref = (Vc_at_Emax <= 1u) ? Emax_p : Ec;
        } else {
          Eref = Emax_p;
        }
      }
    }

    // shift amounts (can be negative -> left shift)
    const int32_t dp = Eref - Emax_p;
    const int32_t dc = use_c ? (Eref - Ec) : 0;

    LOG("[S3/ref   ] Emax_p=%d Ec=%d -> Eref(unb)=%d dp=%d dc=%d |Vp|=0x%08x Vc=0x%08x useC=%u\n",
        Emax_p, Ec, Eref, dp, dc, Vp_mag, cterm.V, (unsigned)use_c);

    uint32_t S = 0, C = 0;
    bool sticky = sticky_prev;

    // fold products at Eref
    for (size_t i = 0; i < aligned_vals.size(); ++i) {
      int32_t Vi = aligned_vals[i];
      Vi = (dp >= 0) ? smag_shr_signed(Vi, uint32_t(dp), sticky)
                     : smag_shl_sat_signed(Vi, uint32_t(-dp), sticky);
      auto [s1, c1] = csa32(S, (C << 1), (uint32_t)Vi);
      S = s1; C = c1;
      LOG("[S3/csa-P] i=%zu add(P>>%u<<%u)=0x%08x S=0x%08x C=0x%08x\n",
          i, (dp>=0)?(uint32_t)dp:0u, (dp<0)?(uint32_t)(-dp):0u, (uint32_t)Vi, S, C);
    }

    // fold C at Eref
    int32_t Cv = use_c ? (cterm.sign ? -int32_t(cterm.V) : int32_t(cterm.V)) : 0;
    Cv = use_c ? ((dc >= 0) ? smag_shr_signed(Cv, uint32_t(dc), sticky)
                            : smag_shl_sat_signed(Cv, uint32_t(-dc), sticky)) : 0;
    auto [s1, c1] = csa32(S, (C << 1), (uint32_t)Cv);
    S = s1; C = c1;
    LOG("[S3/csa-C] add(C>>%u<<%u)=0x%08x S=0x%08x C=0x%08x\n",
        (dc>=0)?(uint32_t)dc:0u, (dc<0)?(uint32_t)(-dc):0u, (uint32_t)Cv, S, C);

    const uint32_t Vu = S + (C << 1);
    const int32_t  Vf = (Vu & 0x80000000u) ? -int32_t((~Vu) + 1u) : int32_t(Vu);
    LOG("[S3/cpa   ] Eref(unb)=%d -> Vf=0x%08x signed=%d st=%u\n",
        Eref, (uint32_t)Vf, Vf, (unsigned)sticky);

    return {Vf, Eref, sticky};
  }

  // ---------- S4: normalize (no early exits) ----------
  norm_t normalize(int32_t V, int32_t Eref) {
    norm_t n{};
    LOG("[S4/norm-in ] V=0x%08x (%d) Eref(unb)=%d\n", (uint32_t)V, V, Eref);

    const bool is_zero = (V == 0);
    const uint32_t sign = (V < 0);
    uint32_t mag = (V < 0) ? uint32_t(-V) : uint32_t(V);
    mag = is_zero ? 0u : mag;

    const uint32_t nbits = is_zero ? 0u : (32u - clz32(mag));
    LOG("[S4/norm-msb] sign=%u mag_in=0x%08x nbits=%u z=%u\n", sign, mag, nbits, (unsigned)is_zero);

    uint32_t kept = 0, rb = 0; bool st=false; int32_t e_unb = INT32_MIN;

    if (!is_zero) {
      if (nbits > Wc_) {
        const uint32_t sh = nbits - Wc_;
        const uint32_t before = mag;
        rb = (mag >> (sh-1)) & 1u;
        st = ((mag & ((1u<<(sh-1))-1u)) != 0u);
        mag >>= sh;
        e_unb = Eref + int32_t((nbits-1) - (Wc_-1));
        LOG("[S4/norm-shr] sh=%u before=0x%08x after=0x%08x rb=%u st+=%u e_unb=%d\n",
            sh, before, mag, rb, (unsigned)st, e_unb);
      } else {
        const uint32_t sh = Wc_ - nbits;
        const uint32_t before = mag;
        mag <<= sh;
        e_unb = Eref - int32_t((Wc_-1) - (nbits-1));
        LOG("[S4/norm-shl] sh=%u before=0x%08x after=0x%08x e_unb=%d\n",
            sh, before, mag, e_unb);
      }
      kept = mag & ((1u<<Wc_) - 1u);
    } else {
      kept = 0; rb=0; st=false; e_unb=INT32_MIN;
      LOG("[S4/norm    ] ZERO path\n");
    }

    n.sign = sign; n.kept24 = kept; n.round_bit = rb; n.sticky = st; n.e_unb = e_unb;
    LOG("[S4/norm-out] kept24=0x%06x rb=%u st=%u e_unb=%d\n", kept, rb, (unsigned)st, e_unb);
    return n;
  }

  // ---------- S4: rounding (no early exits) ----------
  uint32_t rounding(const norm_t& n, bool sticky_in) {
    uint32_t sign = n.sign;
    int32_t e_unb = n.e_unb;      // unbiased
    uint32_t frac = n.kept24 & ((1u<<(Wc_-1))-1u); // drop hidden bit
    uint32_t rb   = n.round_bit;
    uint32_t st   = n.sticky | sticky_in;

    const bool zero_path = (e_unb == INT32_MIN);
    const int32_t e_biased = zero_path ? 0 : (e_unb + 127);
    LOG("[S4/pack-in ] sign=%u e_unb=%d e=%d frac=0x%06x rb=%u st=%u z=%u\n",
        sign, e_unb, e_biased, frac, rb, st, (unsigned)zero_path);

    bool inc=false;
    switch (frm_) {
      case RNE: inc = (rb && (st || (frac & 1u))); break;
      case RTZ: inc = false; break;
      case RDN: inc = (sign && (rb || st)); break;
      case RUP: inc = (!sign && (rb || st)); break;
      case RMM: inc = rb; break;
      default:  inc = false; break;
    }
    LOG("[S4/pack-rnd] mode=%d inc=%u\n", frm_, (unsigned)inc);

    if (!zero_path && inc) {
      const uint32_t before = (1u<<(Wc_-1)) | frac;
      const uint32_t after  = before + 1u;
      const bool overflow   = (after & (1u<<Wc_)) != 0u;
      if (overflow) { frac = 0; e_unb += 1; LOG("[S4/pack-ovf] mant carry -> e_unb=%d frac=0x%06x\n", e_unb, frac); }
      else          { frac = after & ((1u<<(Wc_-1))-1u); LOG("[S4/pack-add] after=0x%08x frac=0x%06x\n", after, frac); }
      fflags_ |= FLAG_NX;
    } else if (!zero_path && (rb || st)) {
      fflags_ |= FLAG_NX;
    }

    // compose result
    uint32_t out = 0;

    const bool is_over = (!zero_path) && ((e_unb + 127) >= 0xFF);
    const bool is_sub  = (!zero_path) && ((e_unb + 127) <= 0);

    if (zero_path) {
      out = 0u;
      LOG("[S4/pack-out] ZERO bits=0x%08x\n", out);
    } else if (is_over) {
      fflags_ |= (FLAG_OF | FLAG_NX);
      out = (sign<<31) | (0xFFu<<23);
      LOG("[S4/pack-out] OVERFLOW -> INF bits=0x%08x\n", out);
    } else if (is_sub) {
      const int32_t e = e_unb + 127;
      const bool tiny_to_zero = (e <= -23);
      if (tiny_to_zero) {
        fflags_ |= (FLAG_UF | FLAG_NX);
        out = (sign<<31);
        LOG("[S4/pack-out] SUBNORMAL -> ZERO bits=0x%08x\n", out);
      } else {
        uint32_t mant = (1u<<(Wc_-1)) | frac;
        const uint32_t sh = uint32_t(1 - e);
        uint32_t lost_mask = (sh>=32)? 0xFFFFFFFFu : ((1u<<sh)-1u);
        if (mant & lost_mask) { fflags_ |= (FLAG_UF | FLAG_NX); }
        uint32_t sub = mant >> sh;
        out = (sign<<31) | sub;
        LOG("[S4/pack-out] SUBNORMAL sh=%u mant=0x%08x sub=0x%08x bits=0x%08x\n",
            sh, mant, sub, out);
      }
    } else {
      const int32_t e = e_unb + 127;
      out = (sign<<31) | (uint32_t(e)<<23) | (frac & ((1u<<23)-1u));
      LOG("[S4/pack-out] NORMAL bits=0x%08x (sign=%u e=%d frac=0x%06x)\n",
          out, sign, e, frac);
    }

    return out;
  }

  // ---------- helpers ----------
  static inline uint32_t ceil_div_u(uint32_t x, uint32_t y) { return (x + y - 1u) / y; }

  static inline uint32_t clz32(uint32_t x) {
  #if defined(__GNUC__)
    return x ? __builtin_clz(x) : 32u;
  #else
    uint32_t n = 0, v = x; if (!v) return 32; while ((v & 0x80000000u) == 0u) { v <<= 1; ++n; } return n;
  #endif
  }

  // leading zeros within W bits (W<=32). Returns W when x==0.
  static inline uint32_t lzc_n(uint32_t x, uint32_t W) {
    if (W >= 32) return clz32(x);
    const uint32_t mask = (W==32)? 0xFFFFFFFFu : ((1u<<W)-1u);
    x &= mask;
    if (!x) return W;
    const uint32_t lz32 = clz32(x);
    return (lz32 - (32u - W));
  }

  // any of the lowest r bits set?
  static inline bool any_dropped32_u(uint32_t x, uint32_t r) {
    if (r == 0) return false;
    if (r >= 32) return x != 0u;
    const uint32_t mask = (1u << r) - 1u;
    return (x & mask) != 0u;
  }

  // 3:2 carry-save adder (unsigned lanes)
  static inline std::pair<uint32_t,uint32_t> csa32(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t s = a ^ b ^ c;
    const uint32_t g = (a & b) | (b & c) | (a & c);
    return {s, g};
  }

  // magnitude >> sh with sticky
  static inline uint32_t mag_shr(uint32_t mag, uint32_t sh, bool &sticky) {
    if (mag == 0 || sh == 0) return mag;
    if (sh >= 32) { sticky |= (mag != 0u); return 0u; }
    const uint32_t lost = (1u << sh) - 1u;
    sticky |= ((mag & lost) != 0u);
    return mag >> sh;
  }

  // signed >> sh with sticky
  static inline int32_t smag_shr_signed(int32_t V, uint32_t sh, bool &sticky) {
    if (V == 0 || sh == 0) return V;
    uint32_t mag = (V < 0) ? uint32_t(-V) : uint32_t(V);
    if (sh >= 31) { sticky |= (mag != 0u); return 0; }
    const uint32_t lost = (1u << sh) - 1u;
    sticky |= ((mag & lost) != 0u);
    mag >>= sh;
    return (V < 0) ? -int32_t(mag) : int32_t(mag);
  }

  // signed << sh with saturation + sticky (aligning up to larger Eref)
  static inline int32_t smag_shl_sat_signed(int32_t V, uint32_t sh, bool &sticky) {
    if (V == 0 || sh == 0) return V;
    uint32_t mag = (V < 0) ? uint32_t(-V) : uint32_t(V);
    if (sh >= 31) { sticky |= (mag != 0u); return (V < 0) ? INT32_MIN : INT32_MAX; }
    sticky |= ((mag >> (31 - sh)) != 0u);
    mag <<= sh;
    return (V < 0) ? -int32_t(mag) : int32_t(mag);
  }

  // ---------- members ----------
  const int frm_;
  const uint32_t lanes_; // Lanes <= 8
  uint32_t fflags_{0};
};
