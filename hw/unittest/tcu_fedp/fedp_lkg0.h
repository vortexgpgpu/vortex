// fedp4_lkg2.h — minimal-change fix: product-only align + dominance-aware final add
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

struct Logger {
  static inline void log(const char* fmt, ...) {
    if constexpr (FEDP_TRACE) {
      va_list args; va_start(args, fmt); std::vprintf(fmt, args); va_end(args);
    }
  }
};
#define LOG(...) Logger::log(__VA_ARGS__)

class FEDP {
public:
  // frm: RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    assert(frm_ >= 0 && frm_ <= 4);
    assert(lanes_ >= 1 && lanes_ <= 8);
    LOG("[ctor] frm=%d, lanes=%u, super=TF32 e8m10, Wc=%u, Win=%u\n",
        frm_, lanes_, Wc_, Win_);
  }

  // Top-level entry: a_words/b_words carry packed inputs (width=1+exp+sig)
  float operator()(const std::vector<uint32_t>& a_words,
                   const std::vector<uint32_t>& b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits) {
    resetFlags();

    const uint32_t width = 1u + exp_bits + sig_bits;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t elems_per_word = packed ? (32u / width) : 1u;
    const uint32_t k = n_words * elems_per_word;

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%u, elems/word=%u, n_words=%u, k=%u\n",
        exp_bits, sig_bits, width, (unsigned)packed, elems_per_word, n_words, k);

    // S1: decode packed inputs
    const auto ab_dec = decode_inputs(a_words, b_words, n_words, elems_per_word, exp_bits, sig_bits, packed);

    // S1: decode C (FP32)
    const uint32_t c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);

    // Specials
    if (const uint32_t fast = decode_special_or_zero(ab_dec, c_dec)) return f32FromBits(fast);

    // S1: map C to common grid
    const auto c_term = decoded_to_common(c_dec, 8, 23);

    // S1: multiply lanes + per-group CSA at each group's Eg (baseline exponent path)
    const auto [prod_groups, mul_sticky] = multiply_to_common(ab_dec, exp_bits, sig_bits);

    // --- NEW S2: align only product groups to Emax_p (collapse CSA→binary, shift sign-mag) ---
    auto [aligned_p, Emax_p, sticky_p] = align_products(prod_groups);

    // --- NEW S3a: CSA-accumulate aligned products; single CPA → Vp (product domain) ---
    const int32_t Vp = accumulate_products(aligned_p);

    // --- NEW S3b: dominance-aware final add with C (LEFT/RIGHT as needed) ---
    auto [V_final, Eref, sticky_final] = add_C_dominance_aware(Vp, Emax_p, c_term, sticky_p);

    // Zero fast-path
    if (V_final == 0) {
      if (mul_sticky || sticky_final) fflags_ |= (FLAG_NX | FLAG_UF);
      LOG("[final-fast] zero=1, fflags=0x%02x\n", fflags_);
      return f32FromBits(0);
    }

    // S4: normalize + round/pack
    const norm_t nrm = normalize(V_final, Eref, (mul_sticky || sticky_final));
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  uint32_t fflags() const { return fflags_; }

private:
  // ----------------------------- Types -----------------------------
  struct dec_t {
    uint32_t sign{0}, frac{0}, exp{0};
    bool is_zero{false}, is_sub{false}, is_inf{false}, is_nan{false};
  };
  struct term24_t {
    int32_t S{0};
    int32_t C{0};
    int32_t E{0};
    bool is_zero{true};
  };
  struct norm_t {
    uint32_t sign{0}, kept24{0}, round_bit{0};
    int32_t  e_unb{0};
    bool sticky{false};
  };

  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };
  static constexpr uint32_t FLAG_NX = 1u << 0;
  static constexpr uint32_t FLAG_UF = 1u << 1;
  static constexpr uint32_t FLAG_OF = 1u << 2;
  static constexpr uint32_t FLAG_NV = 1u << 4;

  // ----------------------------- Utils -----------------------------
  static inline uint32_t bitsFromF32(float f){ uint32_t u; std::memcpy(&u,&f,4); return u; }
  static inline float f32FromBits(uint32_t u){ float f; std::memcpy(&f,&u,4); return f; }
  static inline int clz32(uint32_t x){
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clz(x) : 32;
#else
    if (!x) return 32; int n=0; while(!(x&0x80000000u)){x<<=1; ++n;} return n;
#endif
  }
  static inline std::pair<uint32_t,uint32_t> csa32(uint32_t a,uint32_t b,uint32_t c){
    const uint32_t t=a^b; return {t^c, (a&b)|(b&c)|(a&c)};
  }

  // --- NEW: sign-magnitude shift helpers used for alignment/merge ---
  static inline int32_t sign_mag_shift_right(int32_t v, uint32_t k, bool &sticky){
    if (k==0 || v==0) return v;
    uint32_t mag = (v<0)? uint32_t(-v) : uint32_t(v);
    if (k>=32){ sticky |= (mag!=0); return 0; }
    sticky |= ((mag & ((1u<<k)-1u)) != 0u);
    uint32_t mag2 = mag >> k;
    return (v<0)? -int32_t(mag2) : int32_t(mag2);
  }
  static inline int32_t sign_mag_shift_left(int32_t v, uint32_t k, bool &sticky){
    if (k==0 || v==0) return v;
    uint64_t mag = (v<0)? uint64_t(-(int64_t)v) : uint64_t(v);
    uint64_t magL = mag << k;
    if (magL > 0x7fffffffULL){ sticky = true; magL = 0x7fffffffULL; } // saturate & sticky on overflow
    return (v<0)? -int32_t(magL) : int32_t(magL);
  }

  void resetFlags(){ fflags_=0; }

  // ----------------------------- Decode -----------------------------
  static inline dec_t decode_input(uint32_t enc, int eb, int sb){
    const uint32_t fm = (1u<<sb)-1u, em = (1u<<eb)-1u;
    const uint32_t s = (enc>>(eb+sb)) & 1u;
    const uint32_t e = (enc>>sb) & em;
    const uint32_t f = enc & fm;
    dec_t d{};
    d.sign=s; d.exp=e; d.frac=f;
    d.is_zero=(e==0 && f==0);
    d.is_sub =(e==0 && f!=0);
    d.is_inf =(e==em && f==0);
    d.is_nan =(e==em && f!=0);
    return d;
  }

  std::vector<std::array<dec_t,2>>
  decode_inputs(const std::vector<uint32_t>& a_words,
                const std::vector<uint32_t>& b_words,
                uint32_t n_words, uint32_t epw,
                int eb, int sb, bool packed)
  {
    const uint32_t width = 1u + eb + sb;
    const uint32_t mask = (width==32)?0xffffffffu:((1u<<width)-1u);

    std::vector<std::array<dec_t,2>> out; out.reserve(n_words*epw);

    for (uint32_t w=0; w<n_words; ++w){
      uint32_t aw=a_words[w], bw=b_words[w];
      for (uint32_t i=0;i<epw;++i){
        const uint32_t aenc = packed? (aw & mask) : aw;
        const uint32_t benc = packed? (bw & mask) : bw;
        auto a = decode_input(aenc, eb, sb);
        auto b = decode_input(benc, eb, sb);
        LOG("[decode] idx=%u, A(enc=0x%x, s=%u, e=%u, f=0x%x), B(enc=0x%x, s=%u, e=%u, f=0x%x)\n",
            (w*epw)+i, aenc, a.sign, a.exp, a.frac, benc, b.sign, b.exp, b.frac);
        out.push_back({a,b});
        if (packed){ aw >>= width; bw >>= width; }
      }
    }
    LOG("[decode_inputs] decoded=%zu\n", out.size());
    return out;
  }

  // ------------------ C → common grid (Win/Wc) ------------------
  term24_t decoded_to_common(const dec_t& d, int eb, int sb){
    const uint32_t bias = (1u<<(eb-1)) - 1u;
    const uint32_t s=d.sign, e=d.exp, f=d.frac;
    const int32_t Ec = (int32_t(e) - bias) + 127;
    const uint32_t M = ((e!=0)? (1u<<sb):0u) | f;
    const int dM = int(Wc_-1u) - sb;
    uint32_t m = (dM>=0)? (M<<dM) : (M>>(-dM));

    term24_t t{};
    t.S = s? -int32_t(m) : int32_t(m);
    t.C = 0; t.E = Ec; t.is_zero = (m==0);
    LOG("[decodeC] s=%u, Ec=0x%x, m=0x%x -> add=0x%x\n", s, Ec, m, t.S);
    return t;
  }

  // --------------- specials / zeros (fast path) ----------------
  uint32_t decode_special_or_zero(const std::vector<std::array<dec_t,2>>& ab_dec,
                                  const dec_t& c_dec){
    bool has_nan=false, has_nv=false, has_neg_inf=false, has_pos_inf=false;

    for (size_t i=0;i<ab_dec.size();++i){
      const auto& a=ab_dec[i][0]; const auto& b=ab_dec[i][1];
      if (a.is_nan||b.is_nan) has_nan=true;
      if ((a.is_inf&&b.is_zero)||(b.is_inf&&a.is_zero)) has_nv=true;
      if (a.is_inf||b.is_inf){
        const uint32_t s=a.sign^b.sign;
        if (s) has_neg_inf = true; else has_pos_inf = true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
      }
    }

    if (has_nv
      || (has_pos_inf && has_neg_inf)
      || (c_dec.is_inf && ((has_pos_inf && c_dec.sign==1u) || (has_neg_inf && c_dec.sign==0u)))){
      fflags_ |= FLAG_NV; return canonicalNaN32();
    }
    if (has_nan || c_dec.is_nan) return canonicalNaN32();
    if (has_pos_inf || has_neg_inf) return packInf32(has_neg_inf?1u:0u);
    if (c_dec.is_inf) return packInf32(c_dec.sign);
    return 0;
  }

  // ------------------------- S1: Multiply & group ---------------------------
  std::tuple<std::vector<term24_t>, bool>
  multiply_to_common(const std::vector<std::array<dec_t,2>>& terms, int eb, int sb) {
    const uint32_t width = 1u + eb + sb;
    const int32_t  bias  = (1<<(eb-1)) - 1;
    const uint32_t Wm_in = uint32_t(sb) + 1u;
    const uint32_t Wraw  = 2u * Wm_in;
    const uint32_t L_in  = Wc_ - Wraw; // shift up to Wc grid
    assert(Wraw < Wc_);

    struct Raw { uint32_t sign; uint32_t m_wc; int32_t E; };
    std::vector<Raw> v; v.reserve(terms.size());
    bool sticky=false;

    for (size_t i=0;i<terms.size();++i){
      const auto& a=terms[i][0]; const auto& b=terms[i][1];
      const uint32_t s = a.sign ^ b.sign;

      // **Baseline exponent path**: Ea + Eb + 1 + 127
      const int32_t Ea = int32_t(a.exp) - bias;
      const int32_t Eb = int32_t(b.exp) - bias;
      const int32_t E  = Ea + Eb + 1 + 127;

      const uint32_t Ma = ((a.exp!=0)? (1u<<sb):0u) | a.frac;
      const uint32_t Mb = ((b.exp!=0)? (1u<<sb):0u) | b.frac;
      const uint32_t P  = Ma * Mb;              // Wraw bits
      const uint32_t m  = P << L_in;            // to Wc grid (no loss)

      v.push_back(Raw{s,m,E});
      LOG("[mul-prod] i=%zu, s=%u, E=0x%x, P=0x%x, m_wc=0x%x, Wraw=%u\n",
          i, s, (unsigned)E, P, m, (unsigned)Wraw);
    }

    // Group-reduce into CSA (group size = 16/width)
    const uint32_t n_groups = std::max(1u, 16u/width);
    std::vector<term24_t> out; out.reserve( (terms.size()+n_groups-1)/n_groups );

    for (size_t base=0; base<terms.size(); base+=n_groups){
      const size_t g = base / n_groups;
      const size_t end = std::min(terms.size(), base+n_groups);

      int32_t Eg = INT32_MIN;
      for (size_t i=base;i<end;++i) Eg = std::max(Eg, (int32_t)v[i].E);

      term24_t acc{}; acc.S=0; acc.C=0; acc.E=Eg; acc.is_zero=false;

      for (size_t i=base;i<end;++i){
        const auto& t = v[i];
        const uint32_t delta = uint32_t(Eg - t.E);

        uint32_t m_shift = (delta>=Wc_)? 0u : (t.m_wc >> delta);
        bool st_local = (delta>=Wc_)? (t.m_wc!=0u)
                       : (delta ? ((t.m_wc & ((1u<<delta)-1u)) != 0u) : false);
        sticky |= st_local;

        const int32_t add = t.sign ? -int32_t(m_shift) : int32_t(m_shift);
        auto [s1,c1] = csa32((uint32_t)acc.S, (uint32_t)(acc.C<<1), (uint32_t)add);
        acc.S=(int32_t)s1; acc.C=(int32_t)c1;

        LOG("[s1-csa] g=%zu, i=%zu, s=%u, delta=0x%x, m_adj=0x%x, add=0x%x, S=0x%x, C=0x%x, sticky=%u\n",
            g, i-base, t.sign, delta, m_shift, add, (uint32_t)acc.S, (uint32_t)acc.C, (unsigned)st_local);
      }

      out.push_back(acc);
      LOG("[s1-csa] g=%zu, Eg=0x%x, S=0x%x, C=0x%x, sticky=%u, zero=%u\n",
          g, (unsigned)Eg, (uint32_t)acc.S, (uint32_t)acc.C, (unsigned)sticky, (unsigned)acc.is_zero);
    }

    LOG("[multiply] groups=%zu\n", out.size());
    return {out, sticky};
  }

  // ------------------- (ORIGINAL) S3 accumulate() retained (unused) ---------
  int32_t accumulate(const std::vector<term24_t> &aligned) {
    uint32_t S=0, C=0;
    for (size_t i=0;i<aligned.size();++i){
      auto [s1,c1] = csa32(S, (C<<1), (uint32_t)aligned[i].S);
      S=s1; C=c1;
      auto [s2,c2] = csa32(S, (C<<1), (uint32_t)(aligned[i].C<<1));
      S=s2; C=c2;
      LOG("[acc-csa(orig)] i=%zu, S=0x%x, C=0x%x\n", i, S, C);
    }
    const uint32_t V = S + (C<<1);
    const int32_t  acc = (V & 0x80000000u) ? -int32_t((~V)+1u) : int32_t(V);
    return acc;
  }

  // --- NEW: align products to product Emax using sign-magnitude shift ---
  std::tuple<std::vector<term24_t>, int32_t, bool>
  align_products(const std::vector<term24_t> &groups) {
    int32_t Emax_p = INT32_MIN;
    for (const auto &g : groups) Emax_p = std::max(Emax_p, g.E);

    std::vector<term24_t> out; out.reserve(groups.size());
    bool sticky = false;

    for (size_t i=0; i<groups.size(); ++i) {
      const auto &t = groups[i];
      const uint32_t delta = uint32_t(Emax_p - t.E);
      const int32_t  Vg = t.S + (int32_t(t.C) << 1);
      bool st=false;
      const int32_t Vg_shift = sign_mag_shift_right(Vg, delta, st);
      sticky |= st;

      term24_t a{}; a.S = Vg_shift; a.C = 0; a.E = Emax_p; a.is_zero = (Vg_shift==0);
      out.push_back(a);
      LOG("[align-p] idx=%zu, delta=0x%x, Vg=0x%x, Vg'=0x%x, st=%u\n",
          i, delta, (uint32_t)Vg, (uint32_t)Vg_shift, (unsigned)st);
    }
    return std::tuple{out, Emax_p, sticky};
  }

  // --- NEW: accumulate aligned products; finish with a CPA to scalar ---
  int32_t accumulate_products(const std::vector<term24_t> &aligned_p) {
    uint32_t S=0, C=0;
    for (size_t i=0;i<aligned_p.size();++i){
      auto [s1,c1] = csa32(S, (C<<1), (uint32_t)aligned_p[i].S);
      S=s1; C=c1;
      auto [s2,c2] = csa32(S, (C<<1), (uint32_t)(aligned_p[i].C<<1));
      S=s2; C=c2;
      LOG("[acc-csa] i=%zu, S=0x%x, C=0x%x\n", i, S, C);
    }
    const uint32_t V = S + (C<<1);
    const int32_t  Vp = (V & 0x80000000u) ? -int32_t((~V)+1u) : int32_t(V);
    LOG("[acc-cpa] Vp=0x%x (signed=%d)\n", V, Vp);
    return Vp;
  }

  // --- NEW: dominance-aware final add of C with proper LEFT/RIGHT shifts ---
  std::tuple<int32_t,int32_t,bool>
  add_C_dominance_aware(int32_t Vp, int32_t Emax_p, const term24_t& cterm, bool sticky_prev) {
    const uint32_t magVp = (Vp<0)? uint32_t(-Vp) : uint32_t(Vp);
    const uint32_t nbitsVp = (magVp==0)? 0u : (32u - uint32_t(clz32(magVp)));
    const int32_t  E_vp_unb = (nbitsVp==0) ? INT32_MIN
                              : (Emax_p - int32_t(Wc_-1u)) + int32_t(nbitsVp-1u);
    const int32_t  Ec = cterm.E;

    const bool c_dom = (E_vp_unb==INT32_MIN) ? true : (Ec >= E_vp_unb);
    const int32_t Eref = c_dom ? Ec : Emax_p;

    bool sticky = sticky_prev;
    int32_t V_final = 0;

    if (c_dom){
      if (Ec == Emax_p){
        V_final = Vp + cterm.S;
        LOG("[align-c-final] near(Ec) D=0, Vp stays, C=0x%x\n", (uint32_t)cterm.S);
      } else if (Ec > Emax_p){
        const uint32_t D = uint32_t(Ec - Emax_p);
        bool st=false; const int32_t Vps = sign_mag_shift_right(Vp, D, st); sticky |= st;
        V_final = Vps + cterm.S;
        LOG("[align-c-final] near(Ec) RIGHT D=0x%x, Vp 0x%x→0x%x, C=0x%x, sticky+=%u\n",
            D, (uint32_t)Vp, (uint32_t)Vps, (uint32_t)cterm.S, (unsigned)st);
      } else { // Ec < Emax_p
        const uint32_t D = uint32_t(Emax_p - Ec);
        bool st=false; const int32_t Vps = sign_mag_shift_left(Vp, D, st); sticky |= st;
        V_final = Vps + cterm.S;
        LOG("[align-c-final] near(Ec) LEFT  D=0x%x, Vp 0x%x→0x%x, C=0x%x, sticky+=%u\n",
            D, (uint32_t)Vp, (uint32_t)Vps, (uint32_t)cterm.S, (unsigned)st);
      }
    } else {
      const uint32_t D = uint32_t((Emax_p > Ec)? (Emax_p - Ec) : 0);
      bool st=false; const int32_t Cs = sign_mag_shift_right(cterm.S, D, st); sticky |= st;
      V_final = Vp + Cs;
      LOG("[align-c-final] far(Emax_p) RIGHT D=0x%x, C 0x%x→0x%x, Vp=0x%x, sticky+=%u\n",
          D, (uint32_t)cterm.S, (uint32_t)Cs, (uint32_t)Vp, (unsigned)st);
    }
    return {V_final, Eref, sticky};
  }

  // ------------------------- S2: (helpers from original) --------------------
  static inline int32_t asr32(int32_t x, uint32_t k) {
    if (k == 0) return x;
    if (k >= 31) return (x < 0) ? -1 : 0; // sign-propagate in original impl
    return (x >= 0) ? (x >> k) : ~((~x) >> k);
  }
  static inline bool any_dropped32(int32_t x, uint32_t k) {
    if (k == 0) return false;
    if (k >= 31) return x != 0;
    const uint32_t mask = (1u << k) - 1u;
    return (uint32_t(x) & mask) != 0u;
  }

  // -------------------- S4: normalize/round/pack (FP32) ---------------------
  norm_t normalize(int32_t acc, int32_t Eref, bool sticky_prev){
    norm_t n{};
    n.sign = (acc<0)?1u:0u;
    uint32_t mag = (acc<0)? uint32_t(-acc) : uint32_t(acc);

    const uint32_t nbits = (mag==0)? 1u : (32u - (uint32_t)clz32(mag));
    n.e_unb = (Eref - int32_t(Wc_-1u)) + int32_t(nbits-1u);

    uint32_t kept24=0, round_bit=0;
    bool sticky = sticky_prev;

    if (nbits > 24){
      const uint32_t sh = nbits - 24;
      const uint32_t rem = (sh>=32)? mag : (mag & ((1u<<sh)-1u));
      kept24   = (sh>=32)? 0u  : (mag >> sh);
      round_bit= (sh>=1) ? ((rem>>(sh-1))&1u) : 0u;
      const bool st2 = (sh>=2)? ((rem & ((1u<<(sh-1))-1u)) != 0u) : false;
      sticky |= st2;
    } else if (nbits < 24){
      kept24 = mag << (24-nbits);
    } else {
      kept24 = mag;
    }

    n.kept24 = kept24;
    n.round_bit = round_bit;
    n.sticky = sticky;
    LOG("[normalize] sign=%u, kept24=0x%x, e_unb=%d, round=%u, stickyAny=%d\n",
        n.sign, n.kept24, n.e_unb, n.round_bit, (int)sticky);
    return n;
  }

  uint32_t round_and_pack(const norm_t& n){
    uint32_t kept24 = n.kept24;
    int32_t e_unb = n.e_unb;
    const uint32_t sign = n.sign;

    if (e_unb >= 0xFF){
      fflags_ |= (FLAG_OF | FLAG_NX);
      LOG("[rounding] out=Inf\n");
      return packInf32(sign);
    }

    if (e_unb > 0){
      const uint32_t frac = kept24 & ((1u<<23)-1u);
      const uint32_t lsb = frac & 1u;
      if (roundInc(frm_, sign, lsb, n.round_bit, n.sticky)){
        const uint32_t t = kept24 + 1u;
        if (t >= (1u<<24)){ kept24 = t>>1; ++e_unb; } else kept24 = t;
        fflags_ |= FLAG_NX;
      }
      if (e_unb >= 0xFF){
        fflags_ |= (FLAG_OF | FLAG_NX);
        LOG("[rounding] out=Inf\n");
        return packInf32(sign);
      }
      const uint32_t frac_keep = kept24 & ((1u<<23)-1u);
      const uint32_t out = (sign<<31) | (uint32_t(e_unb)<<23) | frac_keep;
      LOG("[rounding] normal_out=0x%x, fflags=0x%02x\n", out, fflags_);
      return out;
    }

    // subnormals / underflow
    if (e_unb == 0){
      const uint32_t shifted = (kept24>>1);
      const uint32_t rb = kept24 & 1u;
      const bool st = n.sticky | (rb!=0);
      const uint32_t frac_keep = shifted & ((1u<<23)-1u);
      const uint32_t lsb = frac_keep & 1u;
      if (roundInc(frm_, sign, lsb, rb, st)){
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u<<23)){ if (st||rb) fflags_ |= FLAG_NX; return (sign<<31) | (1u<<23); }
        if (st||rb) fflags_ |= FLAG_NX;
        const uint32_t out = (sign<<31) | t;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_);
        return out;
      } else {
        if (st||rb) fflags_ |= FLAG_NX;
        const uint32_t out = (sign<<31) | frac_keep;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_);
        return out;
      }
    } else { // e_unb < 0
      const int sh2 = 1 - e_unb;
      const uint32_t shifted = (sh2>=32)? 0u : (kept24 >> sh2);
      const uint32_t rem2 = (sh2>=32)? kept24 : (kept24 & ((1u<<sh2)-1u));
      const uint32_t rb2 = (sh2>=1)? ((rem2>>(sh2-1))&1u) : 0u;
      const bool st2 = (sh2>=2)? ((rem2 & ((1u<<(sh2-1))-1u)) != 0u) : false;
      uint32_t frac_keep = shifted & ((1u<<23)-1u);
      const uint32_t lsb2 = frac_keep & 1u;

      if (rb2||st2) fflags_ |= FLAG_NX;
      if (roundInc(frm_, sign, lsb2, rb2, st2)){
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u<<23)) return (sign<<31) | (1u<<23);
        const uint32_t out = (sign<<31) | t;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_);
        return out;
      } else {
        const uint32_t out = (sign<<31) | frac_keep;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_);
        return out;
      }
    }
  }

  static inline bool roundInc(int frm, uint32_t sign, uint32_t lsb, uint32_t rb, bool st){
    switch (frm){
      case RNE: return rb && (st || (lsb&1u));
      case RTZ: return false;
      case RDN: return (rb||st) && (sign==1u);
      case RUP: return (rb||st) && (sign==0u);
      case RMM: return (rb||st);
      default:  return false;
    }
  }
  static inline uint32_t packInf32(uint32_t s){ return (s<<31)|(0xFFu<<23); }
  static inline uint32_t canonicalNaN32(){ return 0x7FC00000u; }

  // ----------------------------- Members -----------------------------
  const uint32_t Wc_{24};   // common/grid width
  const uint32_t Win_{25};  // signed add width
  const int frm_;
  const uint32_t lanes_;
  uint32_t fflags_{0};
};
