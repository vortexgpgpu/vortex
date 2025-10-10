// fedp_lkg10.h
// Windowed FEDP (anchor L) with S1 decode+mul, S2 align (pre-shift), S3 accumulate (CSA→CPA), S4 normalize+round (FP32 RNE).
// Debug: detailed LOG traces added to decodeC_to_common, multiply_to_common, alignment, accumulate, rounding.
// Enable logs with -DFEDP_TRACE=1

#pragma once
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

#ifndef FEDP_TRACE
#define FEDP_TRACE 0
#endif

struct FEDP_Log {
  static inline void p(const char *fmt, ...) {
    if constexpr (FEDP_TRACE) {
      va_list a; va_start(a, fmt); std::vprintf(fmt, a); va_end(a);
    }
  }
};
#define LOG(...) FEDP_Log::p(__VA_ARGS__)

class FEDP {
public:
  // ====== DO NOT CHANGE (external interface) ======
  explicit FEDP(int exp_bits, int sig_bits, int frm, int lanes) :
    exp_bits_(exp_bits), sig_bits_(sig_bits), frm_(frm), lanes_(lanes) {
    assert(exp_bits_ > 0 && exp_bits_ < 8);
    assert(sig_bits_ > 0 && sig_bits_ < 10);
    assert(frm_ == 0);
    assert(lanes_ >= 1 && lanes_ <= 8);
    LOG("[ctor] fmt=e%dm%d frm=%d lanes=%u\n", exp_bits_, sig_bits_, frm_, lanes_);
  }

  float operator()(const uint32_t* a, const uint32_t* b, float c, uint32_t n) {
    const auto terms = decode_inputs(a, b, n);
    const auto c_enc = bitsFromF32(c);
    const auto c_dec = decode_input(c_enc, 8, 23);
    const auto c_term = decodeC_to_common(c_dec);
    const auto prods = multiply_to_common(terms);
    const auto aln = alignment(prods, c_term);
    const auto acc = accumulate(aln);
    const auto nrm = normalize(acc);
    const auto out = rounding(nrm);
    return f32FromBits(out);
  }
  // =================================================

private:
  // -------- types --------
  struct dec_t { uint32_t sign{0}, frac{0}, exp{0}; bool is_zero{false}, is_sub{false}; };
  struct term_t { bool s; uint64_t P; int e; };
  struct cterm_t { int sc, Mc, eC, cls; };            // cls: 0 zero,1 sub,2 norm
  struct align_t { std::vector<uint64_t> T; int sticky, L, cc; };   // shifted addends only
  struct acc_t { int64_t V; int sticky, L, cc; };
  struct nrm_t { int s; uint32_t kept; int g; int st; int e; };

  // -------- utils --------
  static inline uint32_t bitsFromF32(float f){ uint32_t u; std::memcpy(&u,&f,4); return u; }
  static inline float f32FromBits(uint32_t u){ float f; std::memcpy(&f,&u,4); return f; }
  static inline int bias(int eb){ return (1<<(eb-1))-1; }
  static inline int lzcN(uint32_t x,int w){ if(!x) return w; return __builtin_clz(x) - (32 - w); }
  static inline int clz64_u(uint64_t x){ return x? __builtin_clzll(x) : 64; }
  static inline std::pair<uint64_t,uint64_t> csa(uint64_t a,uint64_t b,uint64_t c,uint64_t mask){
    uint64_t s=(a^b^c)&mask, k=((a&b)|(a&c)|(b&c))<<1; return {s, k&mask};
  }
  static inline uint64_t cpa(uint64_t s,uint64_t c,uint64_t mask){ return (s+c)&mask; }
  static inline int64_t twos_to_int(uint64_t x,int w){
    return ((x>>(w-1))&1)? (int64_t)(x - (uint64_t(1)<<w)) : (int64_t)x;
  }

  // -------- decode --------
  static dec_t decode_input(uint32_t enc, int eb, int sb) {
    dec_t d{};
    d.sign = (enc>>(eb+sb)) & 1u;
    d.exp  = (enc>>sb) & ((1u<<eb)-1u);
    d.frac = enc & ((1u<<sb)-1u);
    d.is_zero = (d.exp==0 && d.frac==0);
    d.is_sub  = (d.exp==0 && d.frac!=0);
    return d;
  }

  std::vector<std::array<dec_t,2>>
  decode_inputs(const uint32_t* a, const uint32_t* b, uint32_t n) {
    const uint32_t width = 1u + exp_bits_ + sig_bits_;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t epw = packed ? (32u/width) : 1u;
    std::vector<std::array<dec_t,2>> out(n*epw);
    for(uint32_t w=0; w<n; ++w){
      uint32_t aw=a[w], bw=b[w];
      for(uint32_t i=0;i<epw;++i){
        uint32_t aenc = packed ? (aw & ((1u<<width)-1u)) : aw;
        uint32_t benc = packed ? (bw & ((1u<<width)-1u)) : bw;
        auto da = decode_input(aenc, exp_bits_, sig_bits_);
        auto db = decode_input(benc, exp_bits_, sig_bits_);
        LOG("[decode] lane=%u  A(s=%u e=%u f=0x%x)  B(s=%u e=%u f=0x%x)\n",
            (w*epw)+i, da.sign, da.exp, da.frac, db.sign, db.exp, db.frac);
        out[w*epw+i] = {da,db};
        if(packed){ aw >>= width; bw >>= width; }
      }
    }
    return out;
  }

  // -------- S1C: C → common --------
  static cterm_t decodeC_to_common(const dec_t& c){
    const int ebC=8,sbC=23; int cls=0,Mc=0,eC=0;
    LOG("[decodeC_to_common] in: s=%d zero=%d sub=%d exp=%u frac=0x%x\n",
        (int)c.sign, (int)c.is_zero, (int)c.is_sub, (unsigned)c.exp, (unsigned)c.frac);
    if(c.is_zero){ cls=0; Mc=0; eC=0; }
    else if(c.is_sub){ cls=1; Mc=(int)c.frac; eC=(1-bias(ebC))-sbC; }
    else{ cls=2; Mc=(int)((1u<<sbC)|c.frac); eC=((int)c.exp - bias(ebC)) - sbC; }
    LOG("[decodeC_to_common] out: cls=%d Mc=0x%x eC=%d sc=%d\n", cls, (unsigned)Mc, eC, (int)c.sign);
    return { (int)c.sign, Mc, eC, cls };
  }

  // -------- S1: products to common --------
  std::vector<term_t> multiply_to_common(const std::vector<std::array<dec_t,2>>& v){
    const int sb=sig_bits_, eb=exp_bits_, sb1=sb+1, sbmask=(1<<sb1)-1;
    std::unordered_map<int,long long> acc;
    LOG("[multiply_to_common] start sb=%d eb=%d pairs=%zu\n", sb, eb, v.size());
    for(const auto &ab : v){
      auto a=ab[0], b=ab[1];
      bool zA=a.is_zero, zB=b.is_zero;
      bool infA=(a.exp==((1u<<eb)-1u) && a.frac==0), infB=(b.exp==((1u<<eb)-1u) && b.frac==0);
      bool nanA=(a.exp==((1u<<eb)-1u) && a.frac!=0), nanB=(b.exp==((1u<<eb)-1u) && b.frac!=0);
      if(nanA||nanB||infA||infB||zA||zB){ LOG("[S1/drop] special/zero\n"); continue; }
      int Ea = a.is_sub ? (1 - bias(eb)) : ((int)a.exp - bias(eb));
      int Eb = b.is_sub ? (1 - bias(eb)) : ((int)b.exp - bias(eb));
      int Ma = a.is_sub? (int)a.frac : ((1<<sb)| (int)a.frac);
      int Mb = b.is_sub? (int)b.frac : ((1<<sb)| (int)b.frac);
      if(a.is_sub && Ma){ int t=lzcN((uint32_t)Ma,sb)+1; Ma=(Ma<<t)&sbmask; Ea-=t; LOG("[S1/renorm A] t=%d Ma=0x%x Ea=%d\n", t, (unsigned)Ma, Ea); }
      if(b.is_sub && Mb){ int t=lzcN((uint32_t)Mb,sb)+1; Mb=(Mb<<t)&sbmask; Eb-=t; LOG("[S1/renorm B] t=%d Mb=0x%x Eb=%d\n", t, (unsigned)Mb, Eb); }
      if(!Ma || !Mb){ LOG("[S1/drop] zero mantissa\n"); continue; }
      uint64_t P = (uint64_t)Ma * (uint64_t)Mb;
      int e = Ea + Eb - 2*sb;
      bool s = ((a.sign ^ b.sign) != 0);
      acc[e] += s? -(long long)P : (long long)P;
      LOG("[S1/box] s=%d Ea=%d Eb=%d Ma=0x%x Mb=0x%x e=%d P=0x%llx acc[e]=%lld\n",
          (int)s, Ea,Eb,(unsigned)Ma,(unsigned)Mb,e,(unsigned long long)P,(long long)acc[e]);
    }
    std::vector<term_t> out; out.reserve(acc.size());
    for(auto &kv:acc){ long long v = kv.second; if(v) out.push_back({ v<0, (uint64_t)(v<0?-v:v), kv.first }); }
    LOG("[multiply_to_common] terms=%zu\n", out.size());
    return out;
  }

  // -------- S2: align window (pre-shift only; no CSA here) --------
  align_t alignment(const std::vector<term_t>& t, const cterm_t& C){
    const int Ww=W_+HR_; const uint64_t mask = ((uint64_t)1<<Ww)-1;
    std::vector<int> tops; tops.reserve(t.size()+1);
    for(auto &x:t){ if(x.P){ int hi2 = (int)((x.P>>(2*sig_bits_+1)) & 1ULL); tops.push_back(x.e + 2*sig_bits_ + hi2); } }
    if(C.Mc && (C.cls==1 || C.cls==2)) tops.push_back(C.eC + 23);
    if(tops.empty()){ LOG("[alignment] empty tops\n"); return {std::vector<uint64_t>{},0,0,0}; }
    int L = *std::max_element(tops.begin(),tops.end()) - (W_-1);
    LOG("[alignment] L=%d W=%d HR=%d tops=%zu\n", L, W_, HR_, tops.size());

    std::vector<uint64_t> Ts; Ts.reserve(t.size()+1);
    int sticky=0;

    auto emit = [&](long long val,int e){
      int k=e-L; uint64_t T=0;
      if(k>=0){
        uint64_t mag = (val<0)? (uint64_t)(-val):(uint64_t)val;
        uint64_t sh = (mag<<k);
        T = (val<0)? ((~sh + 1ULL)&mask) : (sh&mask);
      }else{
        int sh=-k; uint64_t mag=(val<0)? (uint64_t)(-val):(uint64_t)val;
        uint64_t part = (sh>=64)? 0ULL : (mag>>sh);
        T = (val<0)? ((~part + 1ULL)&mask) : (part&mask);
        sticky |= (sh>=64)? (mag!=0) : ((mag & ((((uint64_t)1)<<sh)-1))!=0);
      }
      Ts.push_back(T);
      LOG("[alignment/emit] k=%d e=%d val=%lld T=0x%llx sticky=%d\n", k, e, (long long)val, (unsigned long long)T, sticky);
    };
    for(auto &x:t) emit( x.s? -(long long)x.P : (long long)x.P, x.e );
    if(C.Mc && (C.cls==1 || C.cls==2)) emit( C.sc? -(long long)C.Mc : (long long)C.Mc, C.eC );

    LOG("[alignment] Ts=%zu sticky=%d L=%d cc=%d\n", Ts.size(), sticky, L, C.cls);
    return {Ts, sticky, L, C.cls};
  }

  // -------- S3: accumulate (CSA → CPA) --------
  acc_t accumulate(const align_t& a){
    const int Ww=W_+HR_; const uint64_t mask=((uint64_t)1<<Ww)-1;
    LOG("[accumulate] count=%zu Ww=%d\n", a.T.size(), Ww);
    uint64_t s_acc=0, c_acc=0; size_t i=0;
    for(uint64_t T : a.T){
      auto sc = csa(s_acc, c_acc, T, mask);
      s_acc = sc.first; c_acc = sc.second;
      LOG("[accumulate/CSA] i=%zu T=0x%llx s=0x%llx c=0x%llx\n", i++, (unsigned long long)T, (unsigned long long)s_acc, (unsigned long long)c_acc);
    }
    uint64_t Vw = cpa(s_acc, c_acc, mask);
    int64_t V = twos_to_int(Vw,Ww);
    LOG("[accumulate/CPA] Vw=0x%llx V=%lld sticky=%d L=%d cc=%d\n", (unsigned long long)Vw, (long long)V, a.sticky, a.L, a.cc);
    return { V, a.sticky, a.L, a.cc };
  }

  // -------- S4: normalize to kept(24), guard, sticky --------
  nrm_t normalize(const acc_t& x){
    const int Ww=W_+HR_;
    if(x.V==0) return {0,0,0,0, -1000000000};
    int s = x.V<0; uint64_t X = (uint64_t)( s? -(x.V) : x.V ) & ((((uint64_t)1)<<Ww)-1);
    int msb = 63 - clz64_u(X);
    int e = x.L + msb;
    int sh = (msb+1) - 24;
    uint32_t kept; int g=0, st=x.sticky;
    if(sh>=0){
      kept = (uint32_t)((X>>sh) & ((1u<<24)-1));
      uint64_t rem = X & ((((uint64_t)1)<<sh)-1);
      g = (sh>0)? (int)((rem>>(sh-1))&1ULL) : 0;
      uint64_t low = (sh>1)? (rem & ((((uint64_t)1)<<(sh-1))-1)) : 0ULL;
      st = (low!=0 || st)? 1:0;
    }else{
      kept = (uint32_t)((X<<(-sh)) & ((1u<<24)-1));
      g=0;
    }
    LOG("[normalize] e_unb=%d sign=%d kept=0x%06x g=%d st=%d\n", e, s, kept, g, st);
    return {s,kept,g,st,e};
  }

  // -------- S5: rounding (merged RNE pack here) --------
  static inline uint32_t rounding(const nrm_t& r){
    if(r.kept==0){ LOG("[rounding] zero path\n"); return 0u; }
    int be = r.e + 127;
    LOG("[rounding] in s=%d kept=0x%06x g=%d st=%d e_unb=%d be=%d\n", r.s, (unsigned)r.kept, r.g, r.st, r.e, be);
    if(be >= 255){ uint32_t u = r.s? 0xff800000u : 0x7f800000u; LOG("[rounding] overflow -> 0x%08x\n", u); return u; }
    if(be <= 0){
      int k = 1 - be;
      uint64_t wide = ((uint64_t)r.kept<<2) | ((uint64_t)r.g<<1) | (r.st?1ULL:0ULL);
      int sh = k + 2;
      uint32_t m = (uint32_t)((sh>=64)? 0ULL : (wide>>sh));
      int rb = (sh>0)? (int)((wide>>(sh-1))&1ULL) : 0;
      uint64_t tail = (sh>1)? (wide & ((((uint64_t)1)<<(sh-1))-1)) : 0ULL;
      uint32_t inc = rb & ( (tail!=0) | ((m&1u)==1u) );
      uint32_t m2 = (uint32_t)(m + inc);
      uint32_t u = (m2 >= (1u<<23)) ? ((r.s?0x80000000u:0u) | (1u<<23))
                                    : ((r.s?0x80000000u:0u) | m2);
      LOG("[rounding] subnormal k=%d sh=%d m=0x%06x inc=%u -> m2=0x%06x out=0x%08x\n", k, sh, m, inc, m2, u);
      return u;
    }
    uint32_t inc = r.g & ( r.st | ((r.kept&1u)==1u) );
    uint32_t m = (r.kept & 0x7fffffu) + inc;
    int be2 = be + ((m >= (1u<<23))? 1:0);
    m &= (1u<<23)-1;
    uint32_t u = (be2 >= 255) ? (r.s? 0xff800000u : 0x7f800000u)
                              : ((r.s?0x80000000u:0u) | ((uint32_t)(be2&0xff)<<23) | m);
    LOG("[rounding] normal inc=%u be2=%d m=0x%06x -> out=0x%08x\n", inc, be2, m, u);
    return u;
  }

  // -------- members --------
  int exp_bits_{0}, sig_bits_{0}, frm_{0}, lanes_{0};
  const int W_{53};   // window bits kept (preserved from baseline)
  const int HR_{4};   // headroom bits
};
