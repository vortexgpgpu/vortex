// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  explicit FEDP(int exp_bits = 5, int sig_bits = 10, int lanes = 4, int frm = 0, int W = 25, int HR = 3, bool renorm = false) :
    exp_bits_(exp_bits), sig_bits_(sig_bits), frm_(frm), lanes_(lanes), W_(W), HR_(HR), renorm_(renorm) {
    assert(exp_bits_ > 0 && exp_bits_ <= 8);
    assert(sig_bits_ > 0 && sig_bits_ <= 10);
    assert(frm_ >= 0 && frm_ <= 4); // 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM
    assert(lanes_ >= 1 && lanes_ <= 8);
    LOG("[ctor] fmt=e%dm%d frm=%d lanes=%u  W=%d HR=%d renorm_=%s\n",
        exp_bits_, sig_bits_, frm_, lanes_, W_, HR_, (renorm_ ? "true": "false"));
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
  enum FRM_TYPE { FRM_RNE=0, FRM_RTZ=1, FRM_RDN=2, FRM_RUP=3, FRM_RMM=4};
  struct dec_t { uint32_t sign{0}, frac{0}, exp{0}; bool is_zero{false}, is_sub{false}; };
  struct term_t { int sign; uint64_t Mp; int Ep; int lzcP; };
  struct cterm_t { int sign, Mc, Ec, cls; };  // cls: 0 zero,1 sub,2 norm
  struct align_t { std::vector<uint64_t> T; int sticky, L, cc; };   // shifted addends only
  struct acc_t { int64_t V; int sticky, L, cc; };
  struct nrm_t { int sign; uint32_t kept; int g; int st; int e; };

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

  // -------- S1: C → common --------
  static cterm_t decodeC_to_common(const dec_t& c){
    const int ebC=8,sbC=23; int cls=0,Mc=0,eC=0;
    LOG("[decodeC_to_common] in: s=%d zero=%d sub=%d exp=%u frac=0x%x\n",
        (int)c.sign, (int)c.is_zero, (int)c.is_sub, c.exp, c.frac);
    if(c.is_zero){ cls=0; Mc=0; eC=0; }
    else if(c.is_sub){ cls=1; Mc=(int)c.frac; eC=(1-bias(ebC))-sbC; }
    else{ cls=2; Mc=(int)((1u<<sbC)|c.frac); eC=((int)c.exp - bias(ebC)) - sbC; }
    LOG("[decodeC_to_common] out: cls=%d Mc=0x%x eC=%d sc=%d\n", cls, Mc, eC, (int)c.sign);
    return { (int)c.sign, Mc, eC, cls };
  }

  // -------- S1: products to common --------
  std::vector<term_t> multiply_to_common(const std::vector<std::array<dec_t,2>>& v){
    const int sb=sig_bits_, eb=exp_bits_;
    std::vector<term_t> out; out.reserve(v.size());
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

      int lzc_prod = 0;
      if(renorm_){
        int lz_a = a.is_sub ? lzcN((uint32_t)Ma, sb) : 0;
        int lz_b = b.is_sub ? lzcN((uint32_t)Mb, sb) : 0;
        lzc_prod = lz_a + lz_b;
        LOG("[S1/renorm] lz_a=%d lz_b=%d -> lzc_p=%d\n", lz_a, lz_b, lzc_prod);
      }

      if(!Ma || !Mb){ LOG("[S1/drop] zero mantissa\n"); continue; }
      uint64_t Mp = (uint64_t)Ma * (uint64_t)Mb;
      int Ep = Ea + Eb - 2*sb;
      bool s = ((a.sign ^ b.sign) != 0);

      out.push_back({s, Mp, Ep, lzc_prod});
      LOG("[S1/box] s=%d Ea=%d Eb=%d Ma=0x%x Mb=0x%x e=%d P=0x%llx lzcp=%d\n",
          (int)s, Ea,Eb,Ma,Mb,Ep,Mp,lzc_prod);
    }
    LOG("[multiply_to_common] terms=%zu\n", out.size());
    return out;
  }

  // -------- S2: align window --------
  align_t alignment(const std::vector<term_t>& t, const cterm_t& C){
    std::vector<term_t> normalized_terms;
    normalized_terms.reserve(t.size());
    for (const auto& term : t) {
      if (term.Mp) {
        uint64_t m_norm = term.Mp << term.lzcP;
        int e_norm = term.Ep - term.lzcP;
        normalized_terms.push_back({term.sign, m_norm, e_norm, 0});
      }
    }

    const int Ww=W_+HR_; const uint64_t mask = ((uint64_t)1<<Ww)-1;
    std::vector<int> tops;
    tops.reserve(normalized_terms.size()+1);

    for(auto &x : normalized_terms){
        if(x.Mp){
            int hi2 = (int)((x.Mp>>(2*sig_bits_+1)) & 1ULL);
            tops.push_back(x.Ep + 2*sig_bits_ + hi2);
        }
    }
    if(C.Mc && (C.cls==1 || C.cls==2)) tops.push_back(C.Ec + 23);
    if(tops.empty()){ LOG("[alignment] empty tops\n"); return {std::vector<uint64_t>{},0,0,0}; }
    int L = *std::max_element(tops.begin(),tops.end()) - (W_-1);
    LOG("[alignment] L=%d W=%d HR=%d tops=%zu\n", L, W_, HR_, tops.size());

    std::vector<uint64_t> Ts;
    Ts.reserve(normalized_terms.size()+1);
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
      LOG("[alignment/emit] k=%d e=%d val=%lld T=0x%llx sticky=%d\n", k, e, val, T, sticky);
    };
    for(auto &x : normalized_terms) emit( x.sign? -x.Mp : x.Mp, x.Ep );
    if(C.Mc && (C.cls==1 || C.cls==2)) emit( C.sign? -C.Mc : C.Mc, C.Ec );

    LOG("[alignment] Ts=%zu sticky=%d L=%d cc=%d\n", Ts.size(), sticky, L, C.cls);
    return {Ts, sticky, L, C.cls};
  }

  // -------- S3: accumulate --------
  acc_t accumulate(const align_t& a){
    const int Ww=W_+HR_; const uint64_t mask=((uint64_t)1<<Ww)-1;
    LOG("[accumulate] count=%zu Ww=%d\n", a.T.size(), Ww);
    uint64_t s_acc=0, c_acc=0; size_t i=0;
    for (uint64_t T : a.T) {
      auto sc = csa(s_acc, c_acc, T, mask);
      s_acc = sc.first; c_acc = sc.second;
      LOG("[accumulate/CSA] i=%zu T=0x%llx s=0x%llx c=0x%llx\n", i++, T, s_acc, c_acc);
    }
    uint64_t Vw = cpa(s_acc, c_acc, mask);
    int64_t V = twos_to_int(Vw,Ww);
    LOG("[accumulate/CPA] Vw=0x%llx V=%lld sticky=%d L=%d cc=%d\n", Vw, V, a.sticky, a.L, a.cc);
    return { V, a.sticky, a.L, a.cc };
  }

  // -------- S4: normalize --------
  nrm_t normalize(const acc_t& x){
    const int Ww=W_+HR_;
    if(x.V==0) return {0,0,0,x.sticky, -1000000000};
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

  // -------- S5: rounding --------
  inline uint32_t rounding(const nrm_t& r) {
    const uint32_t s_bit = (uint32_t)r.sign << 31;

    LOG("[rounding] in s=%d kept=0x%06x g=%d st=%d e_unb=%d frm=%d\n",
        r.sign, r.kept, r.g, r.st, r.e, frm_);

    // Handle exact zero
    if (r.kept == 0 && r.st == 0) {
        LOG("[rounding] zero path\n");
        return 0u; // Canonical +0
    }

    bool discarded = (r.g == 1 || r.st == 1);
    bool round_up = false;

    // 1. Determine round_up condition for normal path
    switch (frm_) {
        case FRM_RNE: round_up = (r.g == 1 && (r.st == 1 || (r.kept & 1) == 1)); break;
        case FRM_RTZ: round_up = false; break;
        case FRM_RDN: round_up = (r.sign == 1 && discarded); break;
        case FRM_RUP: round_up = (r.sign == 0 && discarded); break;
        case FRM_RMM: round_up = (r.g == 1); break;
    }

    uint32_t kept_rounded = r.kept + (round_up ? 1 : 0);
    int e_unb = r.e;

    // 3. Check for mantissa overflow from rounding
    if (kept_rounded & (1u << 24)) {
        kept_rounded >>= 1;
        e_unb += 1;
    }

    int be = e_unb + 127; // Biased exponent

    // 4. Handle Overflow
    if (be >= 255) {
        LOG("[rounding] overflow path\n");
        switch (frm_) {
            case FRM_RTZ: return s_bit | 0x7f7fffffu; // Max Finite
            case FRM_RDN: return (r.sign == 0) ? (s_bit | 0x7f7fffffu) : 0xff800000u; // MaxPosFinite or -Inf
            case FRM_RUP: return (r.sign == 0) ? 0x7f800000u : (s_bit | 0xff7fffffu); // +Inf or MaxNegFinite
            default:      return s_bit | 0x7f800000u; // +/- Inf (FRM_RNE, FRM_RMM)
        }
    }

    // 5. Handle Subnormal / Underflow
    if (be <= 0) {
        LOG("[rounding] subnormal path\n");
        int k = 1 - be; // shift amount

        if (frm_ == FRM_RTZ) {
            uint32_t m = (k < 25) ? (kept_rounded >> k) : 0;
            return s_bit | (m & 0x7FFFFFu);
        }

        // Handle complete underflow for other modes
        if (k >= 25) {
            if (frm_ == FRM_RUP && r.sign == 0 && discarded) return s_bit | 1u;
            if (frm_ == FRM_RDN && r.sign == 1 && discarded) return s_bit | 0x80000001u;
            return s_bit; // signed zero
        }

        // Re-apply rounding at the new subnormal boundary
        uint32_t lsb = (kept_rounded >> k) & 1;
        uint32_t new_g = (kept_rounded >> (k - 1)) & 1;
        uint64_t st_mask = (1ull << (k - 1)) - 1;
        bool new_st = ((kept_rounded & st_mask) != 0 || r.g != 0 || r.st != 0);
        bool new_discarded = (new_g == 1 || new_st == 1);

        uint32_t m_sub = kept_rounded >> k;
        bool round_up_sub = false;

        switch(frm_) {
            case FRM_RNE: round_up_sub = (new_g == 1 && (new_st || lsb == 1)); break;
            case FRM_RDN: round_up_sub = (r.sign == 1 && new_discarded); break;
            case FRM_RUP: round_up_sub = (r.sign == 0 && new_discarded); break;
            case FRM_RMM: round_up_sub = (new_g == 1); break;
            default: break; // FRM_RTZ already handled
        }

        uint32_t m_final = m_sub + (round_up_sub ? 1 : 0);

        if (m_final == (1u << 23)) { // Rounded up to normal
            return s_bit | (1u << 23);
        }

        LOG("[rounding] subnormal k=%d m_sub=0x%x m_final=0x%x\n", k, m_sub, m_final);
        return s_bit | (m_final & 0x7FFFFFu);
    }

    // 6. Normalized
    LOG("[rounding] normal path\n");
    uint32_t m = kept_rounded & 0x7FFFFFu;
    uint32_t u = s_bit | ((uint32_t)(be & 0xff) << 23) | m;
    LOG("[rounding] normal m=0x%06x -> out=0x%08x\n", m, u);
    return u;
  }

  // -------- members --------
  int exp_bits_, sig_bits_, frm_, lanes_;
  int W_, HR_;
  bool renorm_;
};