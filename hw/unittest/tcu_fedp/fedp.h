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

#ifdef FEDP_TRACE
#include <cstdio>
#define LOG(...) std::fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif

class FEDP {
public:
  // ====== DO NOT CHANGE (external interface) ======
  explicit FEDP(int exp_bits = 5, int sig_bits = 10, int lanes = 4, int frm = 0, int W = 25, bool renorm = false) :
    exp_bits_(exp_bits), sig_bits_(sig_bits), frm_(frm), lanes_(lanes), W_(W), renorm_(renorm) {
    HR_ = 32 - lzcN(lanes_, 32); // ceil(log2(lanes + 1))
    assert(exp_bits_ > 0 && exp_bits_ <= 8);
    assert(sig_bits_ > 0 && sig_bits_ <= 10);
    assert(frm_ >= 0 && frm_ <= 4); // 0=RNE, 1=RTZ, 2=RDN, 3=RUP, 4=RMM
    assert(lanes_ >= 1 && lanes_ <= 8);
    int total_inputs = lanes_ + 1;
    HR_ = (total_inputs <= 1) ? 0 : (32 - __builtin_clz(total_inputs - 1));
    LOG("[ctor] fmt=e%dm%d frm=%d lanes=%u W=%d HR=%d renorm_=%s\n",
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

private:
  // -------- types --------
  // sp codes: 0=Norm, 1=+Inf, 2=-Inf, 3=NaN
  enum FRM_TYPE { FRM_RNE=0, FRM_RTZ=1, FRM_RDN=2, FRM_RUP=3, FRM_RMM=4};
  struct dec_t { uint32_t sign{0}, frac{0}, exp{0}; bool is_zero{false}, is_sub{false}; };
  struct term_t { int sign; uint64_t Mp; int Ep; int lzcP; int sp; };
  struct cterm_t { int sign, Mc, Ec, cls; int sp; };
  struct align_t { std::vector<uint64_t> T; int sticky, L, cc; int sp; };
  struct acc_t { int64_t V; int sticky, L, cc; int sp; };
  struct nrm_t { int sign; uint32_t kept; int g; int st; int e; int sp; };

  // -------- utils --------
  static inline uint32_t bitsFromF32(float f) {
    uint32_t u; std::memcpy(&u,&f,4);
    return u;
  }
  static inline float f32FromBits(uint32_t u) {
    float f;
    std::memcpy(&f,&u,4);
    return f;
  }
  static inline int bias(int eb) {
    return (1<<(eb-1))-1;
  }
  static inline int lzcN(uint32_t x,int w) {
    if (!x) {
      return w;
    }
    return __builtin_clz(x) - (32 - w);
  }
  static inline int clz64_u(uint64_t x) {
    return x ? __builtin_clzll(x) : 64;
  }
  static inline std::pair<uint64_t,uint64_t> csa(uint64_t a, uint64_t b, uint64_t c, uint64_t mask) {
    uint64_t s=(a^b^c)&mask, k=((a&b)|(a&c)|(b&c))<<1;
    return {s, k&mask};
  }
  static inline uint64_t cpa(uint64_t s,uint64_t c,uint64_t mask) {
    return (s+c)&mask;
  }
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
    for (uint32_t w=0; w<n; ++w) {
      uint32_t aw=a[w], bw=b[w];
      for (uint32_t i=0;i<epw;++i) {
        uint32_t aenc = packed ? (aw & ((1u<<width)-1u)) : aw;
        uint32_t benc = packed ? (bw & ((1u<<width)-1u)) : bw;
        auto da = decode_input(aenc, exp_bits_, sig_bits_);
        auto db = decode_input(benc, exp_bits_, sig_bits_);
        LOG("[decode] lane=%u  A(s=%u e=%u f=0x%x)  B(s=%u e=%u f=0x%x)\n",
            (w*epw)+i, da.sign, da.exp, da.frac, db.sign, db.exp, db.frac);
        out[w*epw+i] = {da,db};
        if (packed) {
          aw >>= width; bw >>= width;
        }
      }
    }
    return out;
  }

  // -------- S1: C → common --------
  static cterm_t decodeC_to_common(const dec_t& c){
    const int ebC=8,sbC=23; int cls=0,Mc=0,eC=0, sp=0;
    LOG("[decodeC_to_common] in: s=%d zero=%d sub=%d exp=%u frac=0x%x\n",
        (int)c.sign, (int)c.is_zero, (int)c.is_sub, c.exp, c.frac);
    if (c.exp == 255) {
      sp = c.frac ? 3 : (c.sign ? 2 : 1);
    } else if (c.is_zero) {
      cls=0;
      Mc=0;
      eC=0;
    } else if (c.is_sub) {
      cls=1;
      Mc=(int)c.frac;
      eC=(1-bias(ebC))-sbC;
    } else {
      cls=2;
      Mc=(int)((1u<<sbC)|c.frac);
      eC=((int)c.exp - bias(ebC)) - sbC;
    }
    LOG("[decodeC_to_common] out: cls=%d Mc=0x%x eC=%d sc=%d\n", cls, Mc, eC, (int)c.sign);
    return { (int)c.sign, Mc, eC, cls, sp };
  }

  // -------- S1: products to common --------
  std::vector<term_t> multiply_to_common(const std::vector<std::array<dec_t,2>>& v){
    const int sb=sig_bits_, eb=exp_bits_;
    std::vector<term_t> out; out.reserve(v.size());
    for(const auto &ab : v){
      auto a=ab[0], b=ab[1];
      int all1 = (1<<eb)-1;
      bool zA=a.is_zero, zB=b.is_zero;
      bool infA=(a.exp==all1 && !a.frac), infB=(b.exp==all1 && !b.frac);
      bool nanA=(a.exp==all1 && a.frac), nanB=(b.exp==all1 && b.frac);

      // Handle special cases: sp=3(NaN), 2(-Inf), 1(+Inf)
      if (nanA || nanB || (infA && zB) || (infB && zA)) {
         out.push_back({0,0,0,0, 3});
         LOG("[multiply_to_common] skip special cases\n");
         continue;
      }
      if (infA || infB) {
         out.push_back({0,0,0,0, (a.sign^b.sign) ? 2 : 1});
         LOG("[multiply_to_common] skip infiniy\n");
         continue;
      }
      if (zA || zB) {
        LOG("[multiply_to_common] skip zero\n");
        continue;
      }

      int Ea = a.is_sub ? (1 - bias(eb)) : ((int)a.exp - bias(eb));
      int Eb = b.is_sub ? (1 - bias(eb)) : ((int)b.exp - bias(eb));
      int Ma = a.is_sub? (int)a.frac : ((1<<sb)| (int)a.frac);
      int Mb = b.is_sub? (int)b.frac : ((1<<sb)| (int)b.frac);

      int lzc_prod = 0;
      if (renorm_) {
        int lz_a = a.is_sub ? lzcN((uint32_t)Ma, sb) : 0;
        int lz_b = b.is_sub ? lzcN((uint32_t)Mb, sb) : 0;
        lzc_prod = lz_a + lz_b;
        LOG("[multiply_to_common] lz_a=%d lz_b=%d -> lzc_p=%d\n", lz_a, lz_b, lzc_prod);
      }
      uint64_t Mp = (uint64_t)Ma * (uint64_t)Mb;
      int Ep = Ea + Eb - 2*sb;
      bool s = ((a.sign ^ b.sign) != 0);

      out.push_back({(int)s, Mp, Ep, lzc_prod, 0});
      LOG("[multiply_to_common] s=%d Ea=%d Eb=%d P=0x%lx\n", (int)s, Ea, Eb, Mp);
    }
    return out;
  }

  // -------- S2: align window --------
  align_t alignment(const std::vector<term_t>& t, const cterm_t& C){
    std::vector<term_t> normalized_terms;
    bool has_nan = (C.sp == 3);
    bool has_pos = (C.sp == 1);
    bool has_neg = (C.sp == 2);

    for (auto& term : t) {
      if (term.sp == 3) has_nan = true;
      if (term.sp == 1) has_pos = true;
      if (term.sp == 2) has_neg = true;
      if (term.Mp) {
        uint64_t m_norm = term.Mp << term.lzcP;
        int e_norm = term.Ep - term.lzcP;
        normalized_terms.push_back({term.sign, m_norm, e_norm, 0, 0});
      }
    }

    // Resolve special state: NaN takes precedence, then Inf-Inf collision
    int final_sp = 0;
    if (has_nan || (has_pos && has_neg)) {
      final_sp = 3;
    } else if (has_pos) {
      final_sp = 1;
    } else if (has_neg) {
      final_sp = 2;
    }

    const int Ww=W_+HR_; const uint64_t mask = ((uint64_t)1<<Ww)-1;
    std::vector<int> tops;
    for (auto &x : normalized_terms) {
      if (x.Mp) {
        int hi2 = (int)((x.Mp>>(2*sig_bits_+1)) & 1ULL);
        tops.push_back(x.Ep + 2*sig_bits_ + hi2);
      }
    }
    if (C.Mc && C.sp == 0) {
      tops.push_back(C.Ec + 23);
    }
    if (tops.empty()) {
      LOG("[alignment] empty tops\n");
      return {std::vector<uint64_t>{},0,0,C.cls, final_sp};
    }
    int L = *std::max_element(tops.begin(),tops.end()) - (W_-1);
    LOG("[alignment] L=%d W=%d HR=%d tops=%zu\n", L, W_, HR_, tops.size());

    std::vector<uint64_t> Ts; int sticky=0;
    auto emit = [&](long long val,int e){
      int k=e-L;
      if (k>=0) {
        uint64_t mag = (val<0)? -val:val;
        uint64_t sh = (mag<<k);
        Ts.push_back((val<0)? ((~sh + 1ULL)&mask) : (sh&mask));
      } else {
        int sh=-k; uint64_t mag=(val<0)? -val:val;
        uint64_t part = (sh>=64)? 0ULL : (mag>>sh);
        Ts.push_back((val<0)? ((~part + 1ULL)&mask) : (part&mask));
        sticky |= (sh>=64)? (mag!=0) : ((mag & ((1ULL<<sh)-1))!=0);
      }
    };
    for (auto &x : normalized_terms) {
      emit( x.sign? -x.Mp : x.Mp, x.Ep );
    }
    if (C.Mc && C.sp == 0) {
      emit( C.sign? -C.Mc : C.Mc, C.Ec );
    }
    LOG("[alignment] Ts=%zu sticky=%d L=%d cc=%d sp=%d\n", Ts.size(), sticky, L, C.cls, final_sp);
    return {Ts, sticky, L, C.cls, final_sp};
  }

  // -------- S3: accumulate --------
  acc_t accumulate(const align_t& a){
    const int Ww=W_+HR_; const uint64_t mask=((uint64_t)1<<Ww)-1;
    LOG("[accumulate] count=%zu Ww=%d\n", a.T.size(), Ww);
    uint64_t s_acc=0, c_acc=0;
    size_t i=0;
    for (uint64_t T : a.T) {
      auto sc = csa(s_acc, c_acc, T, mask);
      s_acc = sc.first; c_acc = sc.second;
      LOG("[accumulate] i=%zu T=0x%lx s=0x%lx c=0x%lx\n", i++, T, s_acc, c_acc);
    }
    uint64_t Vw = cpa(s_acc, c_acc, mask);
    int64_t V = twos_to_int(Vw,Ww);
    LOG("[accumulate] Vw=0x%lx V=%ld sticky=%d L=%d cc=%d\n", Vw, V, a.sticky, a.L, a.cc);
    return { V, a.sticky, a.L, a.cc, a.sp };
  }

  // -------- S4: normalize --------
  nrm_t normalize(const acc_t& x){
    const int Ww=W_+HR_;
    if (x.V==0) {
      return {0,0,0,x.sticky, -1000000000, x.sp};
    }
    int s = x.V<0; uint64_t X = (uint64_t)( s? -(x.V) : x.V ) & ((((uint64_t)1)<<Ww)-1);
    int msb = 63 - clz64_u(X);
    int e = x.L + msb;
    int sh = (msb+1) - 24;
    uint32_t kept; int g=0, st=x.sticky;
    if (sh>=0) {
      kept = (uint32_t)((X>>sh) & ((1u<<24)-1));
      uint64_t rem = X & ((((uint64_t)1)<<sh)-1);
      g = (sh>0)? (int)((rem>>(sh-1))&1ULL) : 0;
      st = ((rem & ((1ULL<<(sh>1?sh-1:0))-1))!=0 || st)? 1:0;
    } else {
      kept = (uint32_t)((X<<(-sh)) & ((1u<<24)-1));
    }
    LOG("[normalize] e_unb=%d sign=%d kept=0x%06x g=%d st=%d sp=%d\n", e, s, kept, g, st, x.sp);
    return {s,kept,g,st,e, x.sp};
  }

  // -------- S5: rounding --------
  inline uint32_t rounding(const nrm_t& r) {
    if (r.sp == 3) {
      LOG("[rounding] NAN\n");
      return 0x7fc00000; // NaN
    }
    if (r.sp == 1) {
      LOG("[rounding] +Infinity\n");
      return 0x7f800000; // +Inf
    }
    if (r.sp == 2) {
      LOG("[rounding] -Infinity\n");
      return 0xff800000; // -Inf
    }

    const uint32_t s_bit = (uint32_t)r.sign << 31;
    LOG("[rounding] in s=%d kept=0x%06x g=%d st=%d e_unb=%d\n", r.sign, r.kept, r.g, r.st, r.e);

    if (r.kept == 0 && r.st == 0) {
      LOG("[rounding] canonical +0\n");
      return 0u;
    }

    bool discarded = (r.g == 1 || r.st == 1);
    bool round_up = false;
    // Determine round_up condition for normal path
    switch (frm_) {
      case FRM_RNE: round_up = (r.g == 1 && (r.st == 1 || (r.kept & 1) == 1)); break;
      case FRM_RTZ: round_up = false; break;
      case FRM_RDN: round_up = (r.sign == 1 && discarded); break;
      case FRM_RUP: round_up = (r.sign == 0 && discarded); break;
      case FRM_RMM: round_up = (r.g == 1); break;
    }

    uint32_t kept_rounded = r.kept + (round_up ? 1 : 0);
    int e_unb = r.e;
    // Check for mantissa overflow from rounding
    if (kept_rounded & (1u << 24)) {
      kept_rounded >>= 1;
      e_unb += 1;
    }
    int be = e_unb + 127; // Biased exponent

    // Handle Overflow
    if (be >= 255) {
      LOG("[rounding] overflow path\n");
      switch (frm_) {
        case FRM_RTZ: return s_bit | 0x7f7fffffu;
        case FRM_RDN: return (r.sign == 0) ? (s_bit | 0x7f7fffffu) : 0xff800000u;
        case FRM_RUP: return (r.sign == 0) ? 0x7f800000u : (s_bit | 0xff7fffffu);
        default:      return s_bit | 0x7f800000u;
      }
    }

    // Handle Subnormal / Underflow
    if (be <= 0) {
      LOG("[rounding] subnormal path\n");
      int k = 1 - be; // shift amount
      if (frm_ == FRM_RTZ) {
        uint32_t m = (k < 25) ? (kept_rounded >> k) : 0;
        return s_bit | (m & 0x7FFFFFu);
      }
      // Handle complete underflow for other modes
      if (k >= 25) {
        if (frm_ == FRM_RUP && r.sign == 0 && discarded)
          return s_bit | 1u;
        if (frm_ == FRM_RDN && r.sign == 1 && discarded)
          return s_bit | 0x80000001u;
        return s_bit;
      }
      // Re-apply rounding at the new subnormal boundary
      uint32_t lsb = (kept_rounded >> k) & 1;
      uint32_t new_g = (kept_rounded >> (k - 1)) & 1;
      uint64_t st_mask = (1ull << (k - 1)) - 1;
      bool new_st = ((kept_rounded & st_mask) != 0 || r.g || r.st);
      bool new_disc = (new_g || new_st);
      uint32_t m_sub = kept_rounded >> k;
      bool ru_sub = false;
      switch(frm_) {
        case FRM_RNE: ru_sub = (new_g && (new_st || lsb)); break;
        case FRM_RDN: ru_sub = (r.sign && new_disc); break;
        case FRM_RUP: ru_sub = (!r.sign && new_disc); break;
        case FRM_RMM: ru_sub = (new_g); break;
        default: break;
      }
      uint32_t m_final = m_sub + (ru_sub ? 1 : 0);
      if (m_final == (1u << 23)) {
        LOG("[rounding] rounded up to normal\n");
        return s_bit | (1u << 23);
      }
      LOG("[rounding] subnormal k=%d m_sub=0x%x m_final=0x%x\n", k, m_sub, m_final);
      return s_bit | (m_final & 0x7FFFFFu);
    }

    uint32_t m = kept_rounded & 0x7FFFFFu;
    uint32_t u = s_bit | ((uint32_t)(be & 0xff) << 23) | m;
    LOG("[rounding] normal m=0x%06x -> out=0x%08x\n", m, u);
    return u;
  }

  // members --------
  int exp_bits_;  // input format exponent
  int sig_bits_;  // input format significant
  int frm_;       // rounding mode
  int lanes_;     // number of elements
  int W_;         // accumulator width
  int HR_;        // accumulator head room
  bool renorm_;   // renormalize products
};