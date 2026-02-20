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
#include <limits>

#ifdef FEDP_TRACE
#define LOG(...) std::fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif

#ifdef FEDP_RTL_WATCH
#define RTL_WATCH(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define RTL_WATCH(...)
#endif

class FEDP {
public:
  explicit FEDP(int exp_bits = 5, int sig_bits = 10, int lanes = 4, int frm = 0, int W = 25, bool renorm = false, bool no_window = false)
      : exp_bits_(exp_bits), sig_bits_(sig_bits), frm_(frm), lanes_(lanes), W_(W), renorm_(renorm), no_window_(no_window) {
    LOG("[ctor] fmt=e%dm%d frm=%d lanes=%u W=%d renorm_=%s no_window_=%s\n",
        exp_bits_, sig_bits_, frm_, lanes_, W_, (renorm_ ? "true" : "false"), (no_window_ ? "true" : "false"));
    assert(exp_bits_ > 0 && exp_bits_ <= 8);
    assert(sig_bits_ > 0 && sig_bits_ <= 10);
    assert(frm_ >= 0 && frm_ <= 4);
    assert(lanes_ >= 1 && lanes_ <= 16);
  }

  float operator()(const uint32_t *a, const uint32_t *b, float c, uint32_t n) {
    const uint64_t req_id = req_id_++;

    const auto c_enc = bitsFromF32(c);
    rtl_watch_s0(req_id, a, b, c_enc, n);

    const auto terms = decode_inputs(a, b, n);
    const auto c_dec = decode_input(c_enc, 8, 23);
    const auto c_term = decodeC_to_common(c_dec);

    const auto mul_res = multiply_to_common(terms, c_term);

    rtl_watch_s1(req_id, mul_res, c_term, c_enc);

    HR_ = 32 - lzcN(terms.size() + 1, 32);

    const auto aln = alignment(mul_res);

    rtl_watch_s2(req_id, aln, mul_res, c_term, c_enc);

    const auto acc = accumulate(aln);

    rtl_watch_s3(req_id, acc, mul_res, c_term, c_enc);

    const auto nrm = normalize(acc);

    rtl_watch_norm(req_id, nrm, acc.flags, mul_res.L);

    const auto out = rounding(nrm);
    rtl_watch_s4(req_id, out);

    return f32FromBits(out);
  }

private:
  enum FRM_TYPE { FRM_RNE = 0, FRM_RTZ = 1, FRM_RDN = 2, FRM_RUP = 3, FRM_RMM = 4 };

  static const uint8_t FL_NAN  = 1;
  static const uint8_t FL_PINF = 2;
  static const uint8_t FL_NINF = 4;

  struct dec_t {
    int  sign;
    int  frac;
    int  exp;
    bool is_zero;
    bool is_sub;
  };
  struct cterm_t {
    int sign;
    int Mc;
    int Ec;
    int cls;
    int sp;
  };

  struct pterm_t {
    int sign;
    int Mp;
  };

  struct mul_res_t {
    std::vector<pterm_t> terms;
    int L;
    std::vector<int> shifts;
    uint8_t flags;
  };

  struct align_t {
    std::vector<uint32_t> terms;
    int sticky;
    int L;
    uint8_t flags;
    std::vector<uint32_t> dbg_aln_sigs;
    std::vector<int> dbg_sticky_bits;
  };
  struct acc_t {
    uint32_t V;
    int sticky;
    int L;
    uint8_t flags;
  };
  struct nrm_t {
    int sign;
    uint32_t kept;
    int g;
    int st;
    int e;
    uint8_t flags;
    // Debug info
    uint32_t abs_sum;
    int shift_amt;
    bool round_up;
  };

  // -------- utils --------
  static inline uint32_t bitsFromF32(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
  static inline float f32FromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
  static inline int bias(int eb) { return (1 << (eb - 1)) - 1; }
  static inline int lzcN(uint32_t x, int w) { if (!x) return w; return __builtin_clz(x) - (32 - w); }
  static inline uint32_t sign_ext(uint32_t x, uint32_t width) {
    uint32_t mask = 1u << (width - 1);
    x = x & ((1u << width) - 1);
    return (x ^ mask) - mask;
  }

  // -------- [RTL-WATCH] helpers --------
  inline void rtl_watch_s0(uint64_t req_id, const uint32_t* a, const uint32_t* b, uint32_t c_enc, uint32_t n) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S0(%lu): a_row=", req_id);
    RTL_WATCH("{");
    for (int i = (int)n - 1; i >= 0; --i) {
      RTL_WATCH("0x%x", a[i]);
      if (i > 0) RTL_WATCH(", ");
    }
    RTL_WATCH("}, b_col={");
    for (int i = (int)n - 1; i >= 0; --i) {
      RTL_WATCH("0x%x", b[i]);
      if (i > 0) RTL_WATCH(", ");
    }
    RTL_WATCH("}, c_val=0x%x\n", c_enc);
#endif
  }

  inline void rtl_watch_s1(uint64_t req_id, const mul_res_t& res, const cterm_t& c_term, uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S1(%lu): max_exp=0x%x, shift_amt=", req_id, res.L);
    RTL_WATCH("{");
    if (!res.shifts.empty()) {
      RTL_WATCH("0x%x", res.shifts.back());
      if (res.shifts.size() > 1) RTL_WATCH(", ");
      for (int i = (int)res.shifts.size() - 2; i >= 0; --i) {
        RTL_WATCH("0x%x", res.shifts[i]);
        if (i > 0) RTL_WATCH(", ");
      }
    }
    RTL_WATCH("}, raw_sig={");
    uint32_t c_raw = c_term.Mc;
    if (c_term.sign) c_raw |= (1u << 24);
    if (c_term.cls == 2 || c_term.cls == 1) c_raw |= (1u << 23);
    RTL_WATCH("0x%x", c_raw);
    if (res.terms.size() > 1) RTL_WATCH(", ");
    for (int i = (int)res.terms.size() - 2; i >= 0; --i) {
      uint32_t raw_val = res.terms[i].Mp;
      if (res.terms[i].sign) raw_val |= (1u << 24);
      RTL_WATCH("0x%x", raw_val);
      if (i > 0) RTL_WATCH(", ");
    }
    RTL_WATCH("}\n");
#endif
  }

  inline void rtl_watch_s2(uint64_t req_id, const align_t& aln, const mul_res_t& res, const cterm_t& c_term, uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S2(%lu): max_exp=0x%x, aln_sig=", req_id, res.L);
    RTL_WATCH("{");
    if (!aln.dbg_aln_sigs.empty()) {
      uint32_t c_sig = aln.dbg_aln_sigs.back();
      RTL_WATCH("0x%x", c_sig);
      if (aln.dbg_aln_sigs.size() > 1) RTL_WATCH(", ");
      for (int i = (int)aln.dbg_aln_sigs.size() - 2; i >= 0; --i) {
        RTL_WATCH("0x%x", aln.dbg_aln_sigs[i]);
        if (i > 0) RTL_WATCH(", ");
      }
    }
    RTL_WATCH("}, sticky_bits={");
    if (!aln.dbg_sticky_bits.empty()) {
      RTL_WATCH("0x%x", aln.dbg_sticky_bits.back());
      if (aln.dbg_sticky_bits.size() > 1) RTL_WATCH(", ");
      for (int i = (int)aln.dbg_sticky_bits.size() - 2; i >= 0; --i) {
        RTL_WATCH("0x%x", aln.dbg_sticky_bits[i]);
        if (i > 0) RTL_WATCH(", ");
      }
    }
    RTL_WATCH("}\n");
#endif
  }

  inline void rtl_watch_s3(uint64_t req_id, const acc_t& acc, const mul_res_t& res, const cterm_t& c_term, uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    uint32_t sigs_sign = 0;
    for (size_t i = 0; i < res.terms.size() - 1; ++i) {
      if (res.terms[i].sign) sigs_sign |= (1 << i);
    }
    RTL_WATCH("[RTL_WATCH] FEDP-S3(%lu): acc_sig=0x%x, max_exp=0x%x, sigs_sign=0x%x, sticky=%d\n",
        req_id, acc.V, res.L, sigs_sign, acc.sticky ? 1 : 0);
#endif
  }

  inline void rtl_watch_norm(uint64_t req_id, const nrm_t& nrm, int flags, int max_exp) {
#ifdef FEDP_RTL_WATCH
    int lsb = nrm.kept & 1;
    RTL_WATCH("[RTL_WATCH] FEDP-NORM(%lu): abs_sum=0x%x, L=%d, G=%d, R=%d, S=%d, Rup=%d\n",
        req_id, nrm.abs_sum, lsb, nrm.g, 0, nrm.st, nrm.round_up
    );
#endif
  }

  inline void rtl_watch_s4(uint64_t req_id, uint32_t result) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S4(%lu): result=0x%08x\n", req_id, result);
#endif
  }

  // -------- logic --------
  static inline std::pair<uint32_t, uint32_t> csa(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t s = a ^ b ^ c;
    uint32_t k = ((a & b) | (a & c) | (b & c)) << 1;
    return {s, k};
  }
  static inline uint32_t cpa(uint32_t s, uint32_t c) { return s + c; }

  static dec_t decode_input(uint32_t enc, int eb, int sb) {
    dec_t d{};
    d.sign = (enc >> (eb + sb)) & 1u;
    d.exp = (enc >> sb) & ((1u << eb) - 1u);
    d.frac = enc & ((1u << sb) - 1u);
    d.is_zero = (d.exp == 0 && d.frac == 0);
    d.is_sub = (d.exp == 0 && d.frac != 0);
    return d;
  }

  std::vector<std::array<dec_t, 2>> decode_inputs(const uint32_t *a, const uint32_t *b, uint32_t n) {
    const uint32_t width = 1u + exp_bits_ + sig_bits_;
    const bool packed = (width <= 16u) && ((32u % width) == 0u);
    const uint32_t epw = packed ? (32u / width) : 1u;
    std::vector<std::array<dec_t, 2>> out(n * epw);
    for (uint32_t w = 0; w < n; ++w) {
      uint32_t aw = a[w], bw = b[w];
      for (uint32_t i = 0; i < epw; ++i) {
        uint32_t aenc = packed ? (aw & ((1u << width) - 1u)) : aw;
        uint32_t benc = packed ? (bw & ((1u << width) - 1u)) : bw;
        out[w * epw + i] = {decode_input(aenc, exp_bits_, sig_bits_), decode_input(benc, exp_bits_, sig_bits_)};
        if (packed) { aw >>= width; bw >>= width; }
      }
    }
    return out;
  }

  static cterm_t decodeC_to_common(const dec_t &c) {
    const int ebC = 8, sbC = 23;
    int cls = 0, Mc = 0, eC = 0, sp = 0;
    if (c.exp == 255) { sp = c.frac ? 3 : (c.sign ? 2 : 1); }
    else if (c.is_zero) { cls = 0; Mc = 0; eC = 0; }
    else if (c.is_sub) { cls = 1; Mc = c.frac; eC = (1 - bias(ebC)) - sbC; }
    else { cls = 2; Mc = ((1u << sbC) | c.frac); eC = (c.exp - bias(ebC)) - sbC; }
    return {c.sign, Mc, eC, cls, sp};
  }

  mul_res_t multiply_to_common(const std::vector<std::array<dec_t, 2>> &v, const cterm_t &c_term) {
    const int sb = sig_bits_, eb = exp_bits_;
    std::vector<pterm_t> out;
    std::vector<int> eps; // per-term *window LSB exponent* (not raw Ep/Ec)
    out.reserve(v.size() + 1);
    eps.reserve(v.size() + 1);

    const int F32_BIAS = 127 + 128;

    bool c_is_zero = (c_term.cls == 0 && c_term.sp == 0);

    uint8_t c_nan  = (c_term.sp == 3) ? FL_NAN  : 0;
    uint8_t c_pinf = (c_term.sp == 1) ? FL_PINF : 0;
    uint8_t c_ninf = (c_term.sp == 2) ? FL_NINF : 0;
    uint8_t flags  = (c_nan | c_pinf | c_ninf);

    for (const auto &ab : v) {
      auto a = ab[0], b = ab[1];
      int all1 = (1 << eb) - 1;

      bool zA = a.is_zero;
      bool zB = b.is_zero;
      bool p_is_zero = zA || zB;

      bool infA = (a.exp == all1 && !a.frac), infB = (b.exp == all1 && !b.frac);
      bool nanA = (a.exp == all1 && a.frac), nanB = (b.exp == all1 && b.frac);
      bool is_nan = nanA || nanB || (infA && zB) || (infB && zA);
      bool is_inf = (infA || infB) && !is_nan;
      bool sign_xor = ((a.sign ^ b.sign) != 0);

      flags |= is_nan ? FL_NAN : 0;
      flags |= is_inf ? (sign_xor ? FL_NINF : FL_PINF) : 0;

      int Ea = a.is_sub ? (1 - bias(eb)) : (a.exp - bias(eb));
      int Eb = b.is_sub ? (1 - bias(eb)) : (b.exp - bias(eb));
      int Ma = zA ? 0 : (a.is_sub ? a.frac : ((1 << sb) | a.frac));
      int Mb = zB ? 0 : (b.is_sub ? b.frac : ((1 << sb) | b.frac));

      int shift_sf = 22 - (2 * sb);
      int Mp = (Ma * Mb) << shift_sf;
      int Ep = (Ea + Eb) + F32_BIAS - 22;

      if (renorm_) {
        int lzc_prod = (a.is_sub ? lzcN(Ma, sb) : 0) + (b.is_sub ? lzcN(Mb, sb) : 0);
        Mp <<= lzc_prod;
        Ep -= lzc_prod;
      }

      int Ep_w = (Ep + 23 - W_);
      if (p_is_zero) {
        Ep_w = 0;
      }

      out.push_back({sign_xor, Mp});
      eps.push_back(Ep_w);
    }

    int Ec = c_term.Ec + F32_BIAS;
    int Ec_w = Ec + 24 - W_;
    if (c_is_zero) {
      Ec_w = 0;
    }

    out.push_back({c_term.sign, c_term.Mc});
    eps.push_back(Ec_w);

    int L = *std::max_element(eps.begin(), eps.end());

    std::vector<int> shifts;
    shifts.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      shifts.push_back(L - eps[i]);
    }

    return {out, L , shifts, flags};
  }

  align_t alignment(const mul_res_t &res) {
    const auto& t = res.terms;
    int L = res.L;
    const auto& shifts = res.shifts;

    const int WA = W_ + 2;
    const uint32_t mask = (1ULL << WA) - 1;
    std::vector<uint32_t> dbg_aln_sigs;
    std::vector<int> dbg_sticky_bits;
    std::vector<uint32_t> Ts;
    int global_sticky = 0;

    for (size_t i = 0; i < t.size(); ++i) {
      const int MANT = (i + 1 == t.size()) ? (W_ - 24) : (W_ - 23);

      uint32_t mag = t[i].Mp;
      int k = shifts[i];

      uint32_t shifted_mag = 0;
      bool sticky_bit = false;

      if (mag != 0) {
        if (k <= MANT) {
          if (k > (MANT - WA)) {
            shifted_mag = mag << (MANT - k);
          }
        } else {
          if (k < (WA + MANT)) {
            int sh = k - MANT;
            shifted_mag = mag >> sh;
            sticky_bit = (mag & ((1ULL << sh) - 1)) != 0;
          } else {
            sticky_bit = true;
          }
        }
      }

      uint32_t final_val = shifted_mag & mask;
      if (t[i].sign) final_val = (~final_val + 1) & mask;

      int s_bit = sticky_bit ? 1 : 0;
      dbg_aln_sigs.push_back(final_val);
      dbg_sticky_bits.push_back(s_bit);

      Ts.push_back(final_val);
      global_sticky |= s_bit;
    }

    return {Ts, global_sticky, L, res.flags, dbg_aln_sigs, dbg_sticky_bits};
  }

  acc_t accumulate(const align_t &a) {
    const int Wa = W_ + 2;
    const int Ww = W_ + 1 + HR_;
    const uint32_t mask = (1ULL << Ww) - 1;
    uint32_t s_acc = 0, c_acc = 0;
    for (const auto &val : a.terms) {
      uint32_t v = sign_ext(val, Wa) & mask;
      auto sc = csa(s_acc, c_acc, v);
      s_acc = sc.first & mask; c_acc = sc.second & mask;
    }
    return {cpa(s_acc, c_acc) & mask, a.sticky, a.L, a.flags};
  }

  nrm_t normalize(const acc_t &x) {
    const int Ww = W_ + 1 + HR_;
    const uint32_t mask = (1ULL << Ww) - 1;
    if (x.V == 0) return {0, 0, 0, x.sticky, -1000000000, x.flags, 0, 0, 0};

    int s = ((x.V >> (Ww - 1)) & 1);
    uint32_t Q = -x.V & mask;
    uint32_t X = s ? Q : x.V;
    int msb = 31 - __builtin_clz(X);
    int e = x.L + msb - 128;
    int sh = (msb + 1) - 24;

    uint32_t kept;
    int g = 0, st = x.sticky;
    if (sh >= 0) {
      kept = (X >> sh) & ((1u << 24) - 1);
      uint32_t rem = X & ((1 << sh) - 1);
      g = (sh > 0) ? ((rem >> (sh - 1)) & 1ULL) : 0;
      st = ((rem & ((1ULL << (sh > 1 ? sh - 1 : 0)) - 1)) != 0 || st) ? 1 : 0;
    } else {
      kept = (X << (-sh)) & ((1u << 24) - 1);
    }

    bool lsb_bit = (kept & 1);
    bool round_up = false;
    if (frm_ == FRM_RNE) {
      round_up = (g && (st || lsb_bit));
    }
    return {s, kept, g, st, e, x.flags, X, sh, round_up};
  }

  uint32_t rounding(const nrm_t &r) {
    if (r.flags & FL_NAN) return 0x7fc00000;
    if ((r.flags & FL_PINF) && (r.flags & FL_NINF)) return 0x7fc00000;
    if (r.flags & FL_PINF) return 0x7f800000;
    if (r.flags & FL_NINF) return 0xff800000;

    if (r.kept == 0 && r.st == 0) return 0u;

    const uint32_t s_bit = r.sign << 31;
    bool discarded = (r.g == 1 || r.st == 1);
    bool round_up = false;
    switch (frm_) {
    case FRM_RNE: round_up = (r.g == 1 && (r.st == 1 || (r.kept & 1) == 1)); break;
    case FRM_RTZ: round_up = false; break;
    case FRM_RDN: round_up = (r.sign == 1 && discarded); break;
    case FRM_RUP: round_up = (r.sign == 0 && discarded); break;
    case FRM_RMM: round_up = (r.g == 1); break;
    }

    uint32_t kept_rounded = r.kept + (round_up ? 1 : 0);
    int be = r.e;
    if (kept_rounded & (1u << 24)) { kept_rounded >>= 1; be += 1; }
    if (be >= 255) return s_bit | 0x7f800000u;
    if (be <= 0) return s_bit;
    return s_bit | ((be & 0xff) << 23) | (kept_rounded & 0x7FFFFFu);
  }

  // members --------
  int exp_bits_, sig_bits_, frm_, lanes_, W_, HR_;
  bool renorm_, no_window_;
  uint64_t req_id_ = 0;
};