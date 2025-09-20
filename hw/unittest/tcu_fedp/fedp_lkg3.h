// fedp_final.h
// Final, HW-friendly FEDP for MMA envelope (normals + subnormals)
// - 32-bit throughout
// - S1: decode + operand-LZC norm-adjust (parallel to multiply)
// - S2: reduce product groups to one CSA pair (S,C) @ Emax_p
// - S3: dominant-exponent accumulate (predicated right-shifts, one CSA + one CPA)
// - S4: normalize + round/pack (FP32)
//
// Enable traces with -DFEDP_TRACE=1

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
    if constexpr (FEDP_TRACE) { va_list a; va_start(a, fmt); std::vprintf(fmt, a); va_end(a); }
  }
};
#define LOG(...) Logger::log(__VA_ARGS__)

class FEDP {
public:
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    assert(frm_>=0 && frm_<=4); assert(lanes_>=1 && lanes_<=8);
    LOG("[ctor] frm=%d, lanes=%u, Wc=%u\n", frm_, lanes_, Wc_);
  }

  float operator()(const std::vector<uint32_t>& a_words,
                   const std::vector<uint32_t>& b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits)
  {
    fflags_ = 0;

    // Input packing shape
    const uint32_t width = 1u + exp_bits + sig_bits;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t epw   = packed ? (32u/width) : 1u;

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%u, elems/word=%u, n_words=%u, k=%u\n",
        exp_bits, sig_bits, width, (unsigned)packed, epw, n_words, n_words*epw);

    // ---------------- S1: decode A/B and C, then multiply with LZC adjust ----------------
    const auto ab_dec = decode_inputs(a_words, b_words, n_words, epw, exp_bits, sig_bits, packed);

    const uint32_t c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);
    const CT c_term = decoded_to_common(c_dec, 8, 23); // (S_c @ Ec in FP32-field scale)
    const auto [groups, sticky_mul] = multiply_to_common(ab_dec, exp_bits, sig_bits);

    // ---------------- S2: reduce product groups → one (S,C) @ Emax_p ----------------
    const auto [Sp, Cp, Emax_p, sticky_alignP] = align_and_reduce_products(groups);

    // ---------------- S3: single-CPA accumulate (unified, predicated shifts) ----------
    const auto s3 = accumulate_dom_exponent(Sp, Cp, Emax_p, c_term);

    // ---------------- S4: normalize + round/pack --------------------------------------
    const bool sticky_any = (sticky_mul || sticky_alignP || s3.sticky);
    const norm_t nrm = normalize32(s3.acc, s3.Eref, sticky_any);
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  uint32_t fflags() const { return fflags_; }

private:
  //============================ Types/flags ============================//
  struct dec_t { uint32_t sign{0}, frac{0}, exp{0}; bool is_zero{false}, is_sub{false}; };
  struct grp_t { int32_t S{0}, C{0}, E{0}; }; // CSA rails @Eg
  struct norm_t { uint32_t sign{0}, kept24{0}, round_bit{0}; int32_t e_unb{0}; bool sticky{false}; };
  struct CT { int32_t S; int32_t E; bool zero; };
  struct s3_out { int32_t acc; int32_t Eref; bool sticky; };

  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };
  static constexpr uint32_t FLAG_NX = 1u<<0, FLAG_UF=1u<<1, FLAG_OF=1u<<2;

  // Grid widths (common magnitude domain)
  const uint32_t Wc_{24};   // common mantissa+hidden bits
  const uint32_t Win_{25};  // signed add width (Wc + sign)

  const int frm_;
  const uint32_t lanes_;
  uint32_t fflags_{0};

  //============================= Utils =============================//
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

  static inline bool any_dropped32_u(uint32_t x, uint32_t k){
    if (k==0) return false; if (k>=32) return x!=0; return (x & ((1u<<k)-1u)) != 0u;
  }

  // sign-magnitude RIGHT shift with sticky (32-bit)
  static inline int32_t sign_mag_shr32(int32_t v, uint32_t k, bool &st){
    if (k==0 || v==0) return v;
    uint32_t m = (v<0)? uint32_t(-v) : uint32_t(v);
    if (k>=31){ st |= (m!=0); return 0; }
    st |= ((m & ((1u<<k)-1u)) != 0u);
    uint32_t m2 = m >> k;
    return (v<0)? -int32_t(m2) : int32_t(m2);
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

  //============================= Decode =============================//
  static inline dec_t decode_input(uint32_t enc, int eb, int sb){
    const uint32_t fm=(1u<<sb)-1u, em=(1u<<eb)-1u;
    const uint32_t s=(enc>>(eb+sb))&1u, e=(enc>>sb)&em, f=enc&fm;
    dec_t d{}; d.sign=s; d.exp=e; d.frac=f;
    d.is_zero=(e==0&&f==0); d.is_sub=(e==0&&f!=0);
    return d;
  }

  std::vector<std::array<dec_t,2>>
  decode_inputs(const std::vector<uint32_t>& a_words,
                const std::vector<uint32_t>& b_words,
                uint32_t n_words, uint32_t epw,
                int eb, int sb, bool packed)
  {
    const uint32_t width=1u+eb+sb, mask=(width==32)?0xffffffffu:((1u<<width)-1u);
    std::vector<std::array<dec_t,2>> out; out.reserve(n_words*epw);
    for (uint32_t w=0; w<n_words; ++w){
      uint32_t aw=a_words[w], bw=b_words[w];
      for (uint32_t i=0;i<epw;++i){
        const uint32_t aenc=packed?(aw&mask):aw, benc=packed?(bw&mask):bw;
        auto a=decode_input(aenc,eb,sb); auto b=decode_input(benc,eb,sb);
        LOG("[decode] idx=%u, A(enc=0x%x,s=%u,e=%u,f=0x%x), B(enc=0x%x,s=%u,e=%u,f=0x%x)\n",
            (w*epw)+i, aenc,a.sign,a.exp,a.frac, benc,b.sign,b.exp,b.frac);
        out.push_back({a,b});
        if (packed){ aw>>=width; bw>>=width; }
      }
    }
    LOG("[decode_inputs] decoded=%zu\n", out.size());
    return out;
  }

  // Map C (FP32) onto the common Wc grid (signed magnitude at Ec in FP32 field scale)
  CT decoded_to_common(const dec_t& d, int eb, int sb){
    const uint32_t bias=(1u<<(eb-1))-1u;
    const int32_t Ec = int32_t((d.exp? d.exp:0) - bias) + 127; // FP32-field scale (e_field)
    const uint32_t M  = ((d.exp!=0)? (1u<<sb):0u) | d.frac;    // hidden if normal
    const int shiftM  = int(Wc_-1u) - sb;                      // for FP32 sb==23 → 0
    const uint32_t m  = (shiftM>=0)? (M<<shiftM) : (M>>(-shiftM));
    const int32_t S   = d.sign? -int32_t(m) : int32_t(m);
    LOG("[decodeC] s=%u, Ec=0x%x, m=0x%x -> add=0x%x\n", d.sign, Ec, m, S);
    return CT{S, Ec, (m==0)};
  }

  //================== S1: Multiply + operand-LZC norm-adjust ==================//
  static inline int lzc_n(uint32_t x, uint32_t width){
    if (width==0) return 0;
    if (x==0) return int(width);
    if (width>=32) return clz32(x);
    uint32_t y = x << (32-width);
    return clz32(y);
  }

  std::tuple<std::vector<grp_t>, bool>
  multiply_to_common(const std::vector<std::array<dec_t,2>>& terms,
                         int eb, int sb)
  {
    const int32_t bias=(1<<(eb-1))-1;
    const uint32_t Wm_in=uint32_t(sb)+1u;    // sig+1 (hidden included)
    const uint32_t Wraw = 2u*Wm_in;

    struct Raw { uint32_t sign; uint32_t m_wc; int32_t E; };
    std::vector<Raw> v; v.reserve(terms.size());
    bool sticky=false;

    for (size_t i=0;i<terms.size();++i){
      const auto& a=terms[i][0]; const auto& b=terms[i][1];
      const bool a_norm=(a.exp!=0), b_norm=(b.exp!=0);
      const uint32_t Ma=(a_norm?(1u<<sb):0u)|a.frac;
      const uint32_t Mb=(b_norm?(1u<<sb):0u)|b.frac;

      const int lzc_a = lzc_n(Ma, Wm_in);
      const int lzc_b = lzc_n(Mb, Wm_in);
      const uint32_t plus1 = (a_norm && b_norm && ((a.frac + b.frac) >= (1u<<sb))) ? 1u : 0u;

      const int32_t Ea = int32_t(a.exp) - bias;
      const int32_t Eb = int32_t(b.exp) - bias;
      const int32_t E  = Ea + Eb + 127 + int32_t(plus1) - int32_t(lzc_a + lzc_b);

      const uint32_t top_est = 2u*sb - uint32_t(lzc_a + lzc_b) + plus1; // 0..(2*sb+1)
      const uint32_t P = Ma * Mb; // <=32b for fp16/bf16/fp8

      int shift = int(Wc_-1) - int(top_est);
      uint32_t m_wc = 0;
      if (shift >= 0){
        m_wc = (shift >= 32)? 0u : (P << shift);
      } else {
        const uint32_t r = uint32_t(-shift);
        if (r >= 32){ sticky |= (P != 0); m_wc = 0; }
        else { sticky |= any_dropped32_u(P, r); m_wc = P >> r; }
      }

      v.push_back(Raw{ uint32_t(a.sign ^ b.sign), m_wc, E});
      LOG("[mul-prod] i=%zu, s=%u, Ea=%d Eb=%d, lzc_a=%d lzc_b=%d, plus1=%u, E=0x%x, top_est=%u, m_wc=0x%x, Wraw=%u\n",
          i,(a.sign^b.sign),Ea,Eb,lzc_a,lzc_b,plus1,(unsigned)E,top_est,m_wc,(unsigned)Wraw);
    }

    // Per-format grouping (keeps fan-in bounded)
    const uint32_t width=1u+eb+sb;
    const uint32_t n_groups = std::max(1u, 16u/width);
    std::vector<grp_t> out; out.reserve((terms.size()+n_groups-1)/n_groups);

    for (size_t base=0;base<terms.size();base+=n_groups){
      const size_t end=std::min(terms.size(),base+n_groups);
      int32_t Eg=INT32_MIN; for (size_t i=base;i<end;++i) Eg=std::max(Eg, v[i].E);
      int32_t S=0, C=0;
      for (size_t i=base;i<end;++i){
        const auto& t=v[i]; const uint32_t delta=uint32_t(Eg - t.E);
        const uint32_t m_shift=(delta>=Wc_)?0u:(t.m_wc>>delta);
        const bool st_local=(delta>=Wc_)?(t.m_wc!=0u):(delta?((t.m_wc&((1u<<delta)-1u))!=0u):false);
        sticky |= st_local;
        const int32_t add=(t.sign? -int32_t(m_shift) : int32_t(m_shift));
        auto [s1,c1]=csa32((uint32_t)S,(uint32_t)(C<<1),(uint32_t)add);
        S=(int32_t)s1; C=(int32_t)c1;
        LOG("[s1-csa] g=%zu, i=%zu, s=%u, delta=0x%x, m_adj=0x%x, add=0x%x, S=0x%x, C=0x%x, st=%u\n",
            base/(size_t)n_groups,i-base,t.sign,delta,m_shift,add,(uint32_t)S,(uint32_t)C,(unsigned)st_local);
      }
      out.push_back(grp_t{S,C,Eg});
      LOG("[s1-csa] g=%zu, Eg=0x%x, S=0x%x, C=0x%x\n", base/(size_t)n_groups,(unsigned)Eg,(uint32_t)S,(uint32_t)C);
    }

    LOG("[multiply] groups=%zu\n", out.size());
    return {out, sticky};
  }

  //================== S2: groups → one CSA pair @Emax_p ==================//
  std::tuple<int32_t,int32_t,int32_t,bool>
  align_and_reduce_products(const std::vector<grp_t>& groups){
    int32_t Emax_p=INT32_MIN; for (const auto& g: groups) Emax_p=std::max(Emax_p,g.E);
    bool sticky=false; uint32_t S=0,C=0;
    for (size_t i=0;i<groups.size();++i){
      const auto& t=groups[i]; const uint32_t d=uint32_t(Emax_p - t.E);
      const int32_t Vg = t.S + ((int32_t)t.C << 1); bool st=false;
      const int32_t Vgs = sign_mag_shr32(Vg, d, st); sticky |= st;
      auto [s1,c1]=csa32(S,(C<<1),(uint32_t)Vgs); S=s1; C=c1;
      LOG("[align-p] idx=%zu, d=0x%x, Vg=0x%x, Vg'=0x%x, sticky+=%u\n",
          i, d, (uint32_t)Vg, (uint32_t)Vgs, (unsigned)st);
    }
    LOG("[reduceP] Emax_p=0x%x, S=0x%x, C=0x%x\n", (unsigned)Emax_p, S, C);
    return { (int32_t)S, (int32_t)C, Emax_p, sticky };
  }

  //================== S3: dominant-exponent accumulate (single-CPA) ==================//
  static inline bool rails_zero(int32_t S, int32_t C) {
    const int32_t neg2C = (int32_t)(~(C << 1)) + 1;  // two's complement negate of (C<<1)
    return S == neg2C;
  }
  static inline int32_t shr32_cond(int32_t v, uint32_t d, bool en, bool &st) {
    return en ? sign_mag_shr32(v, d, st) : v;
  }

  s3_out accumulate_dom_exponent(int32_t Sp, int32_t Cp, int32_t Emax_p, const CT& c_term) {
    const int32_t Ec        = c_term.E;
    const bool   prodZero   = rails_zero(Sp, Cp);
    const bool   domInit    = (Emax_p >= Ec);
    const bool   selProd    = domInit & !prodZero;    // product dominates only if non-zero
    const int32_t Eref      = selProd ? Emax_p : Ec;

    const uint32_t d        = (uint32_t)((Emax_p > Ec) ? (Emax_p - Ec) : (Ec - Emax_p));
    const bool     needShift= (d != 0);

    // Only one side shifts
    const bool shP = (!selProd) & needShift;  // shift product rails if C dominates or product==0
    const bool shC = ( selProd) & needShift;  // shift C if product dominates

    bool stP=false, stC=false;
    const int32_t Sx  = shr32_cond(Sp, d, shP, stP);
    const int32_t Cx  = shr32_cond(Cp, d, shP, stP);  // rail shift; (<<1) weight applied in CSA
    const int32_t Cal = shr32_cond(c_term.S, d, shC, stC);

    LOG("[acc-align] selProd=%u, d=%u, Sx=0x%x, Cx=0x%x, Cal=0x%x, stP=%u, stC=%u\n",
        (unsigned)selProd, (unsigned)d, (uint32_t)Sx, (uint32_t)Cx, (uint32_t)Cal,
        (unsigned)stP, (unsigned)stC);

    // One CSA + One CPA (single-CPA design)
    uint32_t S=0, C=0;
    auto [s1,c1] = csa32((uint32_t)Sx, (uint32_t)(Cx << 1), (uint32_t)Cal); S=s1; C=c1;
    const int32_t acc = (int32_t)S + ((int32_t)C << 1);

    LOG("[acc-csa+CPA] Sx=0x%x, Cx=0x%x, Cal=0x%x -> S=0x%x, C=0x%x -> acc=0x%x (signed=%d), Eref=0x%x\n",
        (uint32_t)Sx, (uint32_t)Cx, (uint32_t)Cal, S, C, (uint32_t)acc, acc, (unsigned)Eref);

    return {acc, Eref, (stP | stC)};
  }

  //================== S4: normalize + round (FP32) ==================//
  struct mag_scan32 { uint32_t kept24; uint32_t round_bit; bool sticky; int nbits; };

  static inline mag_scan32 trim_to_24bits(uint32_t mag, bool sticky_prev){
    mag_scan32 r{}; r.sticky = sticky_prev;
    if (mag==0){ r.kept24=0; r.round_bit=0; r.nbits=1; return r; }
    const int nbits = 32 - clz32(mag);
    r.nbits = nbits;
    if (nbits > 24){
      const int sh = nbits - 24;
      const uint32_t rem = (sh>=32)? mag : (mag & ((1u<<sh)-1u));
      r.kept24   = (sh>=32)? 0u : (mag >> sh);
      r.round_bit= (sh>=1) ? ((rem>>(sh-1))&1u) : 0u;
      const bool st2 = (sh>=2)? ((rem & ((1u<<(sh-1))-1u)) != 0u) : false;
      r.sticky |= st2;
    } else if (nbits < 24){
      r.kept24 = mag << (24-nbits);
      r.round_bit = 0;
    } else {
      r.kept24 = mag;
      r.round_bit = 0;
    }
    return r;
  }

  norm_t normalize32(int32_t acc, int32_t Eref, bool sticky_prev){
    norm_t n{}; n.sign = (acc<0)?1u:0u;
    uint32_t mag = (acc<0)? uint32_t(-acc) : uint32_t(acc);

    const auto scan = trim_to_24bits(mag, sticky_prev);
    const int e_unb = (Eref - int32_t(Wc_-1u)) + (scan.nbits - 1);

    n.kept24   = scan.kept24;
    n.round_bit= scan.round_bit;
    n.sticky   = scan.sticky;
    n.e_unb    = e_unb;

    LOG("[normalize] Eref=0x%x, sign=%u, kept24=0x%x, nbits=%d, e_unb=%d, round=%u, stickyAny=%d\n",
        (unsigned)Eref, n.sign, n.kept24, scan.nbits, n.e_unb, n.round_bit, (int)n.sticky);
    return n;
  }

  uint32_t round_and_pack(const norm_t& n){
    uint32_t kept24=n.kept24; int32_t e_unb=n.e_unb; const uint32_t sign=n.sign;

    // Overflow → Inf (for completeness)
    if (e_unb >= 0xFF){ fflags_ |= (FLAG_OF|FLAG_NX); return packInf32(sign); }

    if (e_unb > 0){
      const uint32_t frac = kept24 & ((1u<<23)-1u);
      const uint32_t lsb  = frac & 1u;
      if (roundInc(frm_, sign, lsb, n.round_bit, n.sticky)){
        const uint32_t t = kept24 + 1u;
        if (t >= (1u<<24)){ kept24 = t>>1; ++e_unb; } else kept24 = t;
        fflags_ |= FLAG_NX;
      }
      if (e_unb >= 0xFF){ fflags_ |= (FLAG_OF|FLAG_NX); return packInf32(sign); }
      return (sign<<31) | (uint32_t(e_unb)<<23) | (kept24 & ((1u<<23)-1u));
    }

    // Subnormals / underflow paths (MMA envelope)
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
        return (sign<<31) | t;
      } else {
        if (st||rb) fflags_ |= FLAG_NX;
        return (sign<<31) | frac_keep;
      }
    } else { // e_unb < 0
      const int sh2 = 1 - e_unb;
      const uint32_t shifted = (sh2>=32)? 0u : (kept24 >> sh2);
      const uint32_t rem2    = (sh2>=32)? kept24 : (kept24 & ((1u<<sh2)-1u));
      const uint32_t rb2     = (sh2>=1)? ((rem2>>(sh2-1))&1u) : 0u;
      const bool st2         = (sh2>=2)? ((rem2 & ((1u<<(sh2-1))-1u)) != 0u) : false;
      uint32_t frac_keep = shifted & ((1u<<23)-1u);
      const uint32_t lsb2 = frac_keep & 1u;

      if (rb2||st2) fflags_ |= FLAG_NX;
      if (roundInc(frm_, sign, lsb2, rb2, st2)){
        const uint32_t t = frac_keep + 1u;
        if (t >= (1u<<23)) return (sign<<31) | (1u<<23);
        return (sign<<31) | t;
      } else {
        return (sign<<31) | frac_keep;
      }
    }
  }
};
