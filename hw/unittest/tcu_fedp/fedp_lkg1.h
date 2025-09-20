// fedp4_integrated_Eref_split.h
// FEDP: Operand-LZC norm-adjust; C fully integrated in S2 (alignment + accumulate),
// single final CPA inside accumulate(), then normalize/round.
// Enable debug logs with -DFEDP_TRACE=1

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
  // frm: RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4
  FEDP(int frm, uint32_t lanes) : frm_(frm), lanes_(lanes) {
    assert(frm_>=0 && frm_<=4); assert(lanes_>=1 && lanes_<=8);
    LOG("[ctor] frm=%d, lanes=%u, super=TF32 e8m10, Wc=%u, Win=%u\n", frm_, lanes_, Wc_, Win_);
  }

  float operator()(const std::vector<uint32_t>& a_words,
                   const std::vector<uint32_t>& b_words,
                   float c,
                   uint32_t n_words,
                   int exp_bits,
                   int sig_bits)
  {
    fflags_ = 0; // one reset

    const uint32_t width = 1u + exp_bits + sig_bits;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t epw = packed ? (32u/width) : 1u;

    LOG("[inputs] fmt=e%dm%d, width=%u, packed=%u, elems/word=%u, n_words=%u, k=%u\n",
        exp_bits, sig_bits, width, (unsigned)packed, epw, n_words, n_words*epw);

    // S1: decode + specials
    const auto ab_dec = decode_inputs(a_words, b_words, n_words, epw, exp_bits, sig_bits, packed);

    const uint32_t c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);

    if (const uint32_t fast = decode_special_or_zero(ab_dec, c_dec))
      return f32FromBits(fast);

    const auto c_term = decoded_to_common(c_dec, 8, 23);

    // Multiply + operand-LZC norm-adjust; group reduction to CSA@Eg
    const auto [groups, sticky_mul] = multiply_to_common(ab_dec, exp_bits, sig_bits);

    // S2: alignment() then accumulate() (CPA is inside accumulate)
    const auto al = alignment(groups, c_term);
    const int64_t acc64 = accumulate(al);  // sets NX on overflow if any

    // S4: normalize + round/pack
    const bool sticky_any = (sticky_mul || al.sticky);
    if (acc64 == 0){
      if (sticky_any) fflags_ |= (FLAG_NX | FLAG_UF);
      LOG("[final-fast] zero=1, fflags=0x%02x\n", fflags_);
      return f32FromBits(0);
    }

    const norm_t nrm = normalize64(acc64, al.Eref, sticky_any);
    const uint32_t out = round_and_pack(nrm);
    return f32FromBits(out);
  }

  uint32_t fflags() const { return fflags_; }

private:
  // ----------------------------- Types -----------------------------
  struct dec_t { uint32_t sign{0}, frac{0}, exp{0}; bool is_zero{false}, is_sub{false}, is_inf{false}, is_nan{false}; };
  struct grp_t { int32_t S{0}, C{0}, E{0}; };  // per-group CSA rails @Eg
  struct norm_t { uint32_t sign{0}, kept24{0}, round_bit{0}; int32_t e_unb{0}; bool sticky{false}; };
  struct CT { int32_t S; int32_t E; bool zero; };

  // Output of alignment() for S2
  struct AlignOut {
    std::vector<int64_t> Vgs; // groups aligned to Eref (signed)
    int64_t Sc{0};            // C aligned to Eref (signed)
    int32_t Eref{0};
    bool sticky{false};
  };

  enum { RNE=0, RTZ=1, RDN=2, RUP=3, RMM=4 };
  static constexpr uint32_t FLAG_NX = 1u<<0, FLAG_UF=1u<<1, FLAG_OF=1u<<2, FLAG_NV=1u<<4;

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
  static inline int clz64_u(uint64_t x){
#if defined(__GNUC__) || defined(__clang__)
    return x ? __builtin_clzll(x) : 64;
#else
    if (!x) return 64; int n=0; while(!(x&0x8000000000000000ULL)){x<<=1; ++n;} return n;
#endif
  }

  static inline std::pair<uint32_t,uint32_t> csa32(uint32_t a,uint32_t b,uint32_t c){
    const uint32_t t=a^b; return {t^c, (a&b)|(b&c)|(a&c)};
  }
  static inline std::pair<uint64_t,uint64_t> csa64(uint64_t a,uint64_t b,uint64_t c){
    const uint64_t t=a^b; return {t^c, (a&b)|(b&c)|(a&c)};
  }

  static inline bool any_dropped32_u(uint32_t x, uint32_t k){
    if (k==0) return false; if (k>=32) return x!=0; return (x & ((1u<<k)-1u)) != 0u;
  }

  // sign-magnitude shifts (preserve magnitude; collect sticky on right shifts)
  static inline int32_t sign_mag_shr32(int32_t v, uint32_t k, bool &st){
    if (k==0 || v==0) return v;
    uint32_t m = (v<0)? uint32_t(-v) : uint32_t(v);
    if (k>=32){ st |= (m!=0); return 0; }
    st |= ((m & ((1u<<k)-1u)) != 0u);
    uint32_t m2 = m >> k;
    return (v<0)? -int32_t(m2) : int32_t(m2);
  }
  static inline int64_t sign_mag_shr64(int64_t v, uint32_t k, bool &st){
    if (k==0 || v==0) return v;
    uint64_t m = (v<0)? uint64_t(-v) : uint64_t(v);
    if (k>=64){ st |= (m!=0); return 0; }
    st |= ((m & ((1ULL<<k)-1ULL)) != 0ULL);
    uint64_t m2 = m >> k;
    return (v<0)? -int64_t(m2) : int64_t(m2);
  }
  static inline int64_t sign_mag_shl64(int64_t v, uint32_t k, bool &st){
    if (k==0 || v==0) return v;
    __uint128_t m = (v<0)? __uint128_t(-( __int128)v) : __uint128_t(v);
    __uint128_t mL = m << k;
    if ((mL >> 63) != 0){ st = true; return (v<0)? INT64_MIN+1 : INT64_MAX; } // guard
    return (v<0)? -int64_t((uint64_t)mL) : (int64_t)(uint64_t)mL;
  }

  static inline int lzc_n(uint32_t x, uint32_t width){
    if (width==0) return 0;
    if (x==0) return int(width);
    if (width>=32) return clz32(x);
    uint32_t y = x << (32-width);
    return clz32(y);
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

  // ----------------------------- Decode -----------------------------
  static inline dec_t decode_input(uint32_t enc, int eb, int sb){
    const uint32_t fm=(1u<<sb)-1u, em=(1u<<eb)-1u;
    const uint32_t s=(enc>>(eb+sb))&1u, e=(enc>>sb)&em, f=enc&fm;
    dec_t d{}; d.sign=s; d.exp=e; d.frac=f;
    d.is_zero=(e==0&&f==0); d.is_sub=(e==0&&f!=0); d.is_inf=(e==em&&f==0); d.is_nan=(e==em&&f!=0);
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
        LOG("[decode] idx=%u, A(enc=0x%x, s=%u, e=%u, f=0x%x), B(enc=0x%x, s=%u, e=%u, f=0x%x)\n",
            (w*epw)+i, aenc,a.sign,a.exp,a.frac, benc,b.sign,b.exp,b.frac);
        out.push_back({a,b});
        if (packed){ aw>>=width; bw>>=width; }
      }
    }
    LOG("[decode_inputs] decoded=%zu\n", out.size());
    return out;
  }

  // Map C (FP32) to Wc grid (signed)
  CT decoded_to_common(const dec_t& d, int eb, int sb){
    const uint32_t bias=(1u<<(eb-1))-1u;
    const int32_t Ec = d.exp - bias;
    const uint32_t M  = ((d.exp!=0)? (1u<<sb):0u) | d.frac;
    const int shiftM  = int(Wc_-1u) - sb; // FP32: 23→23 so 0
    const uint32_t m  = (shiftM>=0)? (M<<shiftM) : (M>>(-shiftM));
    const int32_t S   = d.sign? -int32_t(m) : int32_t(m);
    LOG("[decodeC] s=%u, Ec=0x%x, m=0x%x -> add=0x%x\n", d.sign, Ec, m, S);
    return CT{S, Ec, (m==0)};
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
        const uint32_t s=a.sign^b.sign; if (s) has_neg_inf=true; else has_pos_inf=true;
        LOG("[mul-prod] i=%zu, special=Inf/NaN/0*Inf\n", i);
      }
    }
    if (has_nv || (has_pos_inf&&has_neg_inf) ||
        (c_dec.is_inf && ((has_pos_inf&&c_dec.sign==1u)||(has_neg_inf&&c_dec.sign==0u)))){
      fflags_ |= FLAG_NV; return canonicalNaN32();
    }
    if (has_nan||c_dec.is_nan) return canonicalNaN32();
    if (has_pos_inf||has_neg_inf) return packInf32(has_neg_inf?1u:0u);
    if (c_dec.is_inf) return packInf32(c_dec.sign);
    return 0;
  }

  // ------------------------- S1: Multiply + LZC norm-adjust ------------------
  std::tuple<std::vector<grp_t>, bool>
  multiply_to_common(const std::vector<std::array<dec_t,2>>& terms,
                         int eb, int sb)
  {
    const int32_t bias=(1<<(eb-1))-1;
    const uint32_t Wm_in=uint32_t(sb)+1u;
    const uint32_t Wraw=2u*Wm_in;

    struct Raw { uint32_t sign; uint32_t m_wc; int32_t E; };
    std::vector<Raw> v; v.reserve(terms.size());
    bool sticky=false;

    for (size_t i=0;i<terms.size();++i){
      const auto& a=terms[i][0]; const auto& b=terms[i][1];
      const bool a_norm=(a.exp!=0), b_norm=(b.exp!=0);
      const uint32_t Ma=(a_norm?(1u<<sb):0u)|a.frac;
      const uint32_t Mb=(b_norm?(1u<<sb):0u)|b.frac;
      const int32_t lzc_a=lzc_n(Ma, Wm_in);
      const int32_t lzc_b=lzc_n(Mb, Wm_in);
      const uint32_t plus1 = (a_norm && b_norm && ((a.frac + b.frac) >= (1u<<sb)))? 1u:0u;
      const int32_t k = int32_t(lzc_a + lzc_b);
      const int32_t Ea = int32_t(a.exp)-bias;
      const int32_t Eb = int32_t(b.exp)-bias;
      const int32_t E = Ea + Eb + plus1 - k;
      const int32_t top_est = 2 * sb - k + plus1;
      const uint32_t P = Ma * Mb;
      const uint32_t shift = k + int(Wc_-1) - 2 * sb - plus1;

      uint32_t m_wc=0;
      if (shift>=0){ m_wc=(shift>=32)?0u:(P<<shift); }
      else { uint32_t r=uint32_t(-shift);
             if (r>=32){ sticky |= (P!=0); m_wc=0; }
             else { sticky |= any_dropped32_u(P, r); m_wc = P>>r; } }

      v.push_back(Raw{ uint32_t(a.sign^b.sign), m_wc, E});
      LOG("[mul-prod] i=%zu, s=%u, Ea=%d Eb=%d, lzc_a=%d lzc_b=%d, plus1=%u, E=0x%x, shift=%u, m_wc=0x%x, Wraw=%u\n",
          i,(a.sign^b.sign),Ea,Eb,lzc_a,lzc_b,plus1,(unsigned)E,shift,m_wc,(unsigned)Wraw);
    }

    const uint32_t width=1u+eb+sb;
    const uint32_t n_groups = std::max(1u, 16u/width);
    std::vector<grp_t> out; out.reserve((terms.size()+n_groups-1)/n_groups);

    for (size_t base=0;base<terms.size();base+=n_groups){
      const size_t end=std::min(terms.size(),base+n_groups);
      int32_t Eg=INT32_MIN; for (size_t i=base;i<end;++i) Eg=std::max(Eg,v[i].E);
      int32_t S=0,C=0;
      for (size_t i=base;i<end;++i){
        const auto& t=v[i]; const uint32_t delta=uint32_t(Eg - t.E);
        uint32_t m_shift=(delta>=Wc_)?0u:(t.m_wc>>delta);
        const bool st_local=(delta>=Wc_)?(t.m_wc!=0u):(delta?((t.m_wc&((1u<<delta)-1u))!=0u):false);
        sticky |= st_local;
        const int32_t add=(t.sign? -int32_t(m_shift) : int32_t(m_shift));
        auto [s1,c1]=csa32((uint32_t)S,(uint32_t)(C<<1),(uint32_t)add);
        S=(int32_t)s1; C=(int32_t)c1;
        LOG("[s1-csa] g=%zu, i=%zu, s=%u, delta=0x%x, m_adj=0x%x, add=0x%x, S=0x%x, C=0x%x, sticky=%u\n",
            base/(size_t)n_groups,i-base,t.sign,delta,m_shift,add,(uint32_t)S,(uint32_t)C,(unsigned)st_local);
      }
      out.push_back(grp_t{S,C,Eg});
      LOG("[s1-csa] g=%zu, Eg=0x%x, S=0x%x, C=0x%x\n", base/(size_t)n_groups,(unsigned)Eg,(uint32_t)S,(uint32_t)C);
    }
    LOG("[multiply] groups=%zu\n", out.size());
    return {out, sticky};
  }

  // ------------------ S2a: alignment() ------------------
  // Pick Eref once, align each product group Vg=S+(C<<1) to Eref, and align C.
  AlignOut alignment(const std::vector<grp_t>& groups, const CT& c_term){
    // Emax_p across groups
    int32_t Emax_p = INT32_MIN;
    for (const auto& g: groups) Emax_p = std::max(Emax_p, g.E);

    // Reference exponent: bound left-shift of products in later stages
    const int32_t Eref = std::max(c_term.E, Emax_p - int32_t(Wc_-1u));

    AlignOut al; al.Eref = Eref; al.sticky = false;

    // Align each group directly to Eref
    al.Vgs.reserve(groups.size());
    for (size_t i=0;i<groups.size();++i){
      const auto& t = groups[i];
      const int32_t d = Eref - t.E; // +: right, -: left
      bool st=false;
      int64_t Vg  = (int64_t)((int32_t)t.S) + ((int64_t)((int32_t)t.C) << 1);
      int64_t Vgs = (d>=0)? sign_mag_shr64(Vg, (uint32_t)d, st)
                          : sign_mag_shl64(Vg, (uint32_t)(-d), st);
      al.sticky |= st;
      al.Vgs.push_back(Vgs);
      LOG("[align-p/eref] idx=%zu, d=%d, Vg=0x%llx, Vg'=0x%llx, sticky+=%u\n",
          i, d, (unsigned long long)Vg, (unsigned long long)Vgs, (unsigned)st);
    }

    // Align C to Eref (Eref >= Ec by construction)
    {
      const int32_t dC = Eref - c_term.E; // >= 0
      bool stC=false;
      al.Sc = sign_mag_shr64((int64_t)c_term.S, (uint32_t)dC, stC);
      al.sticky |= stC;
      LOG("[align-c/eref] dC=%d, C=0x%x→0x%llx, sticky+=%u\n",
          dC, (unsigned)c_term.S, (unsigned long long)al.Sc, (unsigned)stC);
    }

    LOG("[reduce@Eref-preAcc] Eref=0x%x, sticky=%u, terms=%zu(+C)\n",
        (unsigned)Eref, (unsigned)al.sticky, al.Vgs.size());
    return al;
  }

  // ------------------ S2b/S3: accumulate() (CSA + final CPA) ------------------
  // Consumes aligned contributions (groups and C). CPA is performed inside.
  int64_t accumulate(const AlignOut& al){
    uint64_t S=0, C=0;

    // Accumulate aligned product terms
    for (size_t i=0;i<al.Vgs.size();++i){
      auto [s1, c1] = csa64(S, (C<<1), (uint64_t)al.Vgs[i]);
      S=s1; C=c1;
    }

    // Inject aligned C as the last term, then CPA
    auto [s2, c2] = csa64(S, (C<<1), (uint64_t)al.Sc);
    S=s2; C=c2;
    LOG("[acc-csa-final] S=0x%llx, C=0x%llx\n",
        (unsigned long long)S, (unsigned long long)C);

    __int128 V128s = ( __int128 )( (int64_t)S ) + ( ( (__int128)( (int64_t)C ) ) << 1 );
    int64_t acc64;
    if (V128s > (__int128)INT64_MAX) { fflags_ |= FLAG_NX; acc64 = INT64_MAX; }
    else if (V128s < (__int128)INT64_MIN) { fflags_ |= FLAG_NX; acc64 = INT64_MIN; }
    else acc64 = (int64_t)V128s;

    LOG("[acc-cpa] acc64=0x%llx (signed=%lld)\n",
        (unsigned long long)acc64, (long long)acc64);
    return acc64;
  }

  // -------------------- normalize/round (64b) ---------------------
  norm_t normalize64(int64_t acc, int32_t Eref, bool sticky_prev){
    norm_t n{}; n.sign=(acc<0)?1u:0u; uint64_t mag=(acc<0)? uint64_t(-acc):uint64_t(acc);
    const int nbits=(mag==0)?1:(64 - clz64_u(mag));
    n.e_unb = (Eref - int32_t(Wc_-1u)) + (nbits - 1) + 127;

    uint64_t kept24=0, round_bit=0; bool sticky=sticky_prev;
    if (nbits>24){
      const int sh=nbits-24; const uint64_t rem=(sh>=64)?mag:(mag & ((1ULL<<sh)-1ULL));
      kept24=(sh>=64)?0ULL:(mag>>sh); round_bit=(sh>=1)?((rem>>(sh-1))&1ULL):0ULL;
      const bool st2=(sh>=2)?((rem & ((1ULL<<(sh-1))-1ULL))!=0ULL):false; sticky |= st2;
    } else if (nbits<24){ kept24 = mag << (24-nbits); }
    else { kept24 = mag; }

    n.kept24=(uint32_t)kept24; n.round_bit=(uint32_t)round_bit; n.sticky=sticky;
    LOG("[normalize] sign=%u, kept24=0x%x, e_unb=%d, round=%u, stickyAny=%d\n",
        n.sign,n.kept24,n.e_unb,n.round_bit,(int)sticky);
    return n;
  }

  uint32_t round_and_pack(const norm_t& n){
    uint32_t kept24=n.kept24; int32_t e_unb=n.e_unb; const uint32_t sign=n.sign;

    if (e_unb >= 0xFF){ fflags_ |= (FLAG_OF|FLAG_NX); LOG("[rounding] out=Inf\n"); return packInf32(sign); }

    if (e_unb > 0){
      const uint32_t frac=kept24 & ((1u<<23)-1u); const uint32_t lsb=frac & 1u;
      if (roundInc(frm_, sign, lsb, n.round_bit, n.sticky)){
        const uint32_t t=kept24+1u; if (t >= (1u<<24)){ kept24=t>>1; ++e_unb; } else kept24=t; fflags_ |= FLAG_NX;
      }
      if (e_unb >= 0xFF){ fflags_ |= (FLAG_OF|FLAG_NX); LOG("[rounding] out=Inf\n"); return packInf32(sign); }
      const uint32_t out=(sign<<31) | (uint32_t(e_unb)<<23) | (kept24 & ((1u<<23)-1u));
      LOG("[rounding] normal_out=0x%x, fflags=0x%02x\n", out, fflags_); return out;
    }

    // subnormals / underflow
    if (e_unb == 0){
      const uint32_t shifted=(kept24>>1), rb=kept24&1u; const bool st=n.sticky | (rb!=0);
      const uint32_t frac_keep=shifted & ((1u<<23)-1u); const uint32_t lsb=frac_keep & 1u;
      if (roundInc(frm_, sign, lsb, rb, st)){
        const uint32_t t=frac_keep+1u; if (t >= (1u<<23)){ if (st||rb) fflags_ |= FLAG_NX; return (sign<<31)|(1u<<23); }
        if (st||rb) fflags_ |= FLAG_NX; const uint32_t out=(sign<<31)|t;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_); return out;
      } else {
        if (st||rb) fflags_ |= FLAG_NX; const uint32_t out=(sign<<31)|frac_keep;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_); return out;
      }
    } else {
      const int sh2=1 - e_unb;
      const uint32_t shifted=(sh2>=32)?0u:(kept24>>sh2);
      const uint32_t rem2=(sh2>=32)?kept24:(kept24 & ((1u<<sh2)-1u));
      const uint32_t rb2=(sh2>=1)?((rem2>>(sh2-1))&1u):0u;
      const bool st2=(sh2>=2)?((rem2 & ((1u<<(sh2-1))-1u))!=0u):false;
      uint32_t frac_keep=shifted & ((1u<<23)-1u); const uint32_t lsb2=frac_keep & 1u;

      if (rb2||st2) fflags_ |= FLAG_NX;
      if (roundInc(frm_, sign, lsb2, rb2, st2)){
        const uint32_t t=frac_keep+1u; if (t >= (1u<<23)) return (sign<<31)|(1u<<23);
        const uint32_t out=(sign<<31)|t;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_); return out;
      } else {
        const uint32_t out=(sign<<31)|frac_keep;
        LOG("[rounding] subnormal_out=0x%x, fflags=0x%02x\n", out, fflags_); return out;
      }
    }
  }

  // ----------------------------- Members -----------------------------
  const uint32_t Wc_{24};   // common/grid width
  const uint32_t Win_{25};  // signed add width
  const int frm_;
  const uint32_t lanes_;
  uint32_t fflags_{0};
};
