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
#include <utility>

#ifdef FEDP_TRACE
#define FEDP_LOG(...) std::fprintf(stderr, __VA_ARGS__);
#else
#define FEDP_LOG(...)
#endif

#ifdef FEDP_RTL_WATCH
#define RTL_WATCH(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define RTL_WATCH(...)
#endif

class TfrFedp {
public:
  explicit TfrFedp(int exp_bits = 5, int sig_bits = 10, int frm = 0, int W = 25, bool renorm = false, bool no_window = false)
      : exp_bits_(exp_bits), sig_bits_(sig_bits), frm_(frm), W_(W), renorm_(renorm), no_window_(no_window) {
    FEDP_LOG("[ctor] fmt=e%dm%d, frm=%d, W=%d, renorm_=%s, no_window_=%s\n",
        exp_bits_, sig_bits_, frm_, W_, (renorm_ ? "true" : "false"), (no_window_ ? "true" : "false"));
    assert(exp_bits_ > 0 && exp_bits_ <= 8);
    assert(sig_bits_ > 0 && sig_bits_ <= 10);
    assert(frm_ >= 0 && frm_ <= 4);
  }

  // Block-scaled formats: one scale byte per lane for A and B, two lanes per
  // 32-bit register (a lane is 16 bits of it: two fp8 or four fp4 elements).
  //   fp8/bf8 (MXFP8/MXBF8) and fp4 (MXFP4): E8M0 scales;
  //   fp4 with e4m3_scale (NVFP4): E4M3 scales.
  float operator()(const uint32_t *a, const uint32_t *b, float c, uint32_t n,
                   const uint8_t *sf_a, const uint8_t *sf_b, bool e4m3_scale = false) {
    sf_a_ = sf_a;
    sf_b_ = sf_b;
    e4m3_scale_ = e4m3_scale;
    float d = (*this)(a, b, c, n);
    sf_a_ = sf_b_ = nullptr;
    e4m3_scale_ = false;
    return d;
  }

  float operator()(const uint32_t *a, const uint32_t *b, float c, uint32_t n) {
    const auto c_enc = bitsFromF32(c);
    rtl_watch_s0(a, b, c_enc, n);

    const auto c_dec = decode_input(c_enc, 8, 23);
    const auto c_term = decodeC_to_common(c_dec);

    const auto mul_res = (exp_bits_ == 2 && sig_bits_ == 1)
                       ? multiply_f4(a, b, n, c_term)
                       : multiply_to_common(decode_inputs(a, b, n), c_term);

    rtl_watch_s1(mul_res, c_term, c_enc);

    HR_ = 32 - lzcN(mul_res.terms.size() - 1, 32);

    const auto aln = alignment(mul_res);

    rtl_watch_s2(aln, mul_res, c_term, c_enc);

    const auto acc = accumulate(aln);

    rtl_watch_s3(acc, mul_res, c_term, c_enc);

    const auto nrm = normalize(acc);

    rtl_watch_norm(nrm, acc.flags, mul_res.L);

    const auto out = rounding(nrm);
    rtl_watch_s4(out);

    ++req_id_;

    return f32FromBits(out);
  }

private:
  enum FRM_TYPE { FRM_RNE = 0, FRM_RTZ = 1, FRM_RDN = 2, FRM_RUP = 3, FRM_RMM = 4 };

  static const uint8_t FL_NAN  = 1;
  static const uint8_t FL_PINF = 2;
  static const uint8_t FL_NINF = 4;

  static const int F32_BIAS = 127;
  static const int EXP_POS_BIAS = 128;

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
  inline void rtl_watch_s0([[maybe_unused]] const uint32_t* a, [[maybe_unused]] const uint32_t* b, [[maybe_unused]] uint32_t c_enc, [[maybe_unused]] uint32_t n) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S0(%lu): a_row=", req_id_);
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

  inline void rtl_watch_s1([[maybe_unused]] const mul_res_t& res, [[maybe_unused]] const cterm_t& c_term, [[maybe_unused]] uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S1(%lu): max_exp=0x%x, shift_amt=", req_id_, res.L);
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

  inline void rtl_watch_s2([[maybe_unused]] const align_t& aln, [[maybe_unused]] const mul_res_t& res, [[maybe_unused]] const cterm_t& c_term, [[maybe_unused]] uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S2(%lu): max_exp=0x%x, aln_sig=", req_id_, res.L);
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

  inline void rtl_watch_s3([[maybe_unused]] const acc_t& acc, [[maybe_unused]] const mul_res_t& res, [[maybe_unused]] const cterm_t& c_term, [[maybe_unused]] uint32_t c_enc) {
#ifdef FEDP_RTL_WATCH
    uint32_t sigs_sign = 0;
    for (size_t i = 0; i < res.terms.size() - 1; ++i) {
      if (res.terms[i].sign) sigs_sign |= (1 << i);
    }
    RTL_WATCH("[RTL_WATCH] FEDP-S3(%lu): acc_sig=0x%x, max_exp=0x%x, sigs_sign=0x%x, sticky=%d\n",
        req_id_, acc.V, res.L, sigs_sign, acc.sticky ? 1 : 0);
#endif
  }

  inline void rtl_watch_norm([[maybe_unused]] const nrm_t& nrm, [[maybe_unused]] int flags, [[maybe_unused]] int max_exp) {
#ifdef FEDP_RTL_WATCH
    int lsb = nrm.kept & 1;
    RTL_WATCH("[RTL_WATCH] FEDP-NORM(%lu): abs_sum=0x%x, L=%d, G=%d, R=%d, S=%d, Rup=%d\n",
        req_id_, nrm.abs_sum, lsb, nrm.g, 0, nrm.st, nrm.round_up
    );
#endif
  }

  inline void rtl_watch_s4([[maybe_unused]] uint32_t result) {
#ifdef FEDP_RTL_WATCH
    RTL_WATCH("[RTL_WATCH] FEDP-S4(%lu): result=0x%08x\n", req_id_, result);
#endif
  }

  // Exponent of a zero term: it stands for the datapath's exponent 0, which
  // add_c_term places WA - 1 below this model's exponent origin.
  static constexpr int kZeroE = -(1 << 20);

  // The datapath's accumulator width over `lanes` product terms and C.
  int acc_width(uint32_t lanes) const { return W_ + 2 + (32 - lzcN(lanes, 32)); }

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
    std::vector<int> sel_eps;
    std::vector<bool> nonzero;
    out.reserve(v.size() + 1);
    eps.reserve(v.size() + 1);


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

      bool is_e4m3 = (eb == 4 && sb == 3);
      bool infA, infB, nanA, nanB;
      if (is_e4m3) {
        infA = false; infB = false;
        nanA = (a.exp == 15 && a.frac == 7);
        nanB = (b.exp == 15 && b.frac == 7);
      } else {
        infA = (a.exp == all1 && !a.frac); infB = (b.exp == all1 && !b.frac);
        nanA = (a.exp == all1 && a.frac); nanB = (b.exp == all1 && b.frac);
      }

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
      int Ep = (Ea + Eb) + F32_BIAS + EXP_POS_BIAS + 1 - W_;

      if (renorm_) {
        int lzc_prod = (a.is_sub ? lzcN(Ma, sb) : 0) + (b.is_sub ? lzcN(Mb, sb) : 0);
        Mp <<= lzc_prod;
        Ep -= lzc_prod;
      }

      // the fp8 lane pairs its two products by their exponent fields even
      // when one is zero (a zero field counts as 1, as for a subnormal)
      sel_eps.push_back((std::max(a.exp, 1) - bias(eb)) + (std::max(b.exp, 1) - bias(eb))
                        + F32_BIAS + EXP_POS_BIAS + 1 - W_);
      nonzero.push_back(!p_is_zero);

      if (p_is_zero) {
        Ep = kZeroE;
        Mp = 0;
      }

      out.push_back({sign_xor, Mp});
      eps.push_back(Ep);
    }

    // FP8/BF8: reduce product terms pairwise
    if ((1 + eb + sb) <= 8) {
      std::vector<pterm_t> red_out;
      std::vector<int> red_eps;
      red_out.reserve((out.size() + 1) / 2);
      red_eps.reserve((eps.size() + 1) / 2);

      for (size_t j = 0; j < out.size(); j += 2) {
        if (j + 1 < out.size()) {
          auto r = reduce_f8(out[j], sel_eps[j], nonzero[j], out[j + 1], sel_eps[j + 1], nonzero[j + 1]);
          // MX: the lane's E8M0 scales add to its exponent field, which is
          // 10 bits wide and sits WA - 1 above this model's exponents
          if (sf_a_) {
            // an E8M0 scale of 0xFF is NaN, and so is every element it scales
            if (sf_a_[j / 2] == 0xff || sf_b_[j / 2] == 0xff) {
              flags |= FL_NAN;
            }
            // a lane scaled below the exponent range sits at exponent 0,
            // where it only adds to the sticky bit
            if (r.second != kZeroE) {
              const int WA = acc_width(uint32_t(out.size() / 2));
              int lane_exp = r.second + (WA - 1) + int(sf_a_[j / 2]) + int(sf_b_[j / 2]) - 254;
              r.second = std::max(lane_exp, 0) - (WA - 1);
            }
          }
          red_out.push_back(r.first);
          red_eps.push_back(r.second);
        } else {
          red_out.push_back(out[j]);
          red_eps.push_back(eps[j]);
        }
      }

      out.swap(red_out);
      eps.swap(red_eps);
    }

    return add_c_term(std::move(out), std::move(eps), c_term, flags);
  }

  // Append the C term and take every term's shift to the largest exponent.
  mul_res_t add_c_term(std::vector<pterm_t> out, std::vector<int> eps, const cterm_t &c_term, uint8_t flags) {
    bool c_is_zero = (c_term.cls == 0 && c_term.sp == 0);
    int Ec = c_term.Ec + F32_BIAS + EXP_POS_BIAS + 24 - W_;
    if (c_is_zero) {
      Ec = kZeroE;
    }

    out.push_back({c_term.sign, c_term.Mc});
    eps.push_back(Ec);
    const int WA = acc_width(uint32_t(out.size() - 1));
    for (auto &e : eps) {
      if (e == kZeroE) e = -(WA - 1);
    }

    int L = *std::max_element(eps.begin(), eps.end());

    std::vector<int> shifts;
    shifts.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
      shifts.push_back(L - eps[i]);
    }

    return {out, L , shifts, flags};
  }

  // Block-scaled fp4 (E2M1), one lane per 16 bits: its four products are
  // aligned to their largest exponent in a 6-bit lane field, summed in 27-bit
  // two's complement, and leave the datapath as one term. E2M1 has no
  // infinity or NaN. MXFP4 adds the E8M0 scales to the lane exponent; NVFP4
  // folds the E4M3 scales' exponent fields into the 6-bit field (they fit) and
  // multiplies each product by the scales' mantissa product.
  mul_res_t multiply_f4(const uint32_t *a, const uint32_t *b, uint32_t n, const cterm_t &c_term) {
    assert(sf_a_ && sf_b_ && "fp4 needs block scales");
    const int SIG_SHIFT = 11;
    const int EXP_TERM_BIAS = 32;
    const int exp_comp = e4m3_scale_ ? -(2 * 1 + 2 * (7 + 3)) : -(2 * 1 + 10);
    const int exp_adj = (EXP_TERM_BIAS + exp_comp) & 63;
    // the lane exponent the datapath emits is 10 bits wide; this
    // model's lane exponents sit WA - 1 below it
    const int WA = acc_width(2 * n);
    const int bias_base = F32_BIAS + 2 * (23 - 22) - W_ + WA - 1 + EXP_POS_BIAS;
    auto f4_man = [](uint32_t x) -> uint32_t { return ((x >> 1) & 3) == 0 ? 1u : (2u | (x & 1)); };
    auto f4_exp = [](uint32_t x) -> uint32_t {
      uint32_t e2 = (x >> 2) & 1, e1 = (x >> 1) & 1;
      return (e2 & ~e1 & 1) | ((e2 & e1) << 1);
    };
    std::vector<pterm_t> out;
    std::vector<int> eps;
    uint8_t flags = 0;
    if (c_term.sp == 3) flags |= FL_NAN;
    if (c_term.sp == 1) flags |= FL_PINF;
    if (c_term.sp == 2) flags |= FL_NINF;
    for (uint32_t lane = 0; lane < 2 * n; ++lane) {
      const uint32_t word = lane / 2, off = (lane % 2) * 16;
      const uint32_t sfa = sf_a_[lane], sfb = sf_b_[lane];
      // NVFP4 UE4M3 scales (bit 7 is padding): 0x7F is NaN. MXFP4 E8M0
      // scales join the lane exponent; 0xFF is NaN.
      uint32_t sf_man_prod = 1;
      int sf_exp = 0;        // into the 6-bit term exponent (NVFP4)
      int sf_lane_exp = 0;   // into the lane exponent (MXFP4)
      if (e4m3_scale_) {
        sf_man_prod = (8u | (sfa & 7)) * (8u | (sfb & 7));
        sf_exp = int((sfa >> 3) & 15) + int((sfb >> 3) & 15);
        if ((sfa & 0x7f) == 0x7f || (sfb & 0x7f) == 0x7f) {
          flags |= FL_NAN;
        }
      } else {
        sf_lane_exp = int(sfa) + int(sfb) - 254;
        if (sfa == 0xff || sfb == 0xff) {
          flags |= FL_NAN;
        }
      }
      uint32_t texp[4], tprod[4], tsign[4];
      bool tvalid[4];
      uint32_t max_exp = 0;
      for (int j = 0; j < 4; ++j) {
        uint32_t ra = (a[word] >> (off + 4 * j)) & 0xf;
        uint32_t rb = (b[word] >> (off + 4 * j)) & 0xf;
        tvalid[j] = (ra & 7) != 0 && (rb & 7) != 0;
        tsign[j] = ((ra ^ rb) >> 3) & 1;
        uint32_t prod = f4_man(ra) * f4_man(rb);
        tprod[j] = e4m3_scale_ ? ((prod * sf_man_prod) & 0x7ff) : prod;
        texp[j] = tvalid[j] ? (uint32_t(exp_adj + sf_exp + int(f4_exp(ra)) + int(f4_exp(rb))) & 63) : 0;
        max_exp = std::max(max_exp, texp[j]);
      }
      uint32_t sum = 0; // 27-bit two's complement
      for (int j = 0; j < 4; ++j) {
        uint32_t shift = (max_exp - texp[j]) & 63;
        uint32_t mag = tvalid[j] ? ((tprod[j] << SIG_SHIFT) & 0xffffff) : 0;
        uint32_t aligned = (shift >= 24) ? 0 : (mag >> shift);
        sum = (sum + (tsign[j] ? (0u - aligned) : aligned)) & 0x7ffffff;
      }
      uint32_t sum_sign = (sum >> 26) & 1;
      uint32_t abs_sum = (sum_sign ? (0u - sum) : sum) & 0x3ffffff;
      if (abs_sum == 0) {
        out.push_back({0, 0});
        eps.push_back(kZeroE);
        continue;
      }
      const int lane_exp = (int(max_exp) + bias_base + exp_comp + sf_lane_exp) & 1023;
      out.push_back({int(sum_sign), int(abs_sum & 0xffffff)});
      eps.push_back(lane_exp - (WA - 1));
    }
    return add_c_term(std::move(out), std::move(eps), c_term, flags);
  }

  // One fp8 lane: its two products, aligned to the one the datapath picks as
  // larger. A zero product still competes by its exponent fields when the
  // other is nonzero and listed second, as the datapath compares them.
  std::pair<pterm_t, int> reduce_f8(const pterm_t& t1, int e1, bool v1, const pterm_t& t2, int e2, bool v2) {
    // Pick max exponent term
    int max_E, max_s, min_E, min_s;
    int64_t max_P, min_P;

    if (!v1 && !v2) {
      return {{0, 0}, kZeroE};
    }
    if ((v1 && !v2) || (v2 && e1 >= e2)) {
      max_E = e1; max_P = t1.Mp; max_s = t1.sign;
      min_E = e2; min_P = t2.Mp; min_s = t2.sign;
    } else {
      max_E = e2; max_P = t2.Mp; max_s = t2.sign;
      min_E = e1; min_P = t1.Mp; min_s = t1.sign;
    }

    int diff = max_E - min_E;

    // Align smaller term
    int64_t aligned = 0;
    if (diff >= 32) {
      aligned = 0;
    } else if (diff == 0) {
      aligned = min_P;
    } else {
      aligned = (min_P >> diff);
    }

    // Signed add
    int64_t val1 = max_s ? -max_P : max_P;
    int64_t val2 = min_s ? -aligned : aligned;
    int64_t val  = val1 + val2;

    if (val == 0) {
      // Only an exact cancellation of two equal-exponent products clears the
      // lane exponent; a product aligned out of the lane keeps it, so the
      // empty lane still sets the largest exponent.
      bool cancel = (max_s != min_s) && (e1 == e2) && (t1.Mp == t2.Mp);
      return {{0, 0}, cancel ? kZeroE : (max_E + 1)};
    }

    int s = (val < 0) ? 1 : 0;
    int64_t P = (val < 0) ? -val : val;
    int E = max_E;

    // scaling by 1 to avoid renormalization
    P >>= 1;
    E += 1;

    FEDP_LOG("[reduce_f8 (%ld)] t1=(%d, %d, %d), t2=(%d, %d, %d) | diff=%d max_P=0x%lx min_P=0x%lx aligned=0x%lx val=0x%lx -> s=%d P=0x%lx E=%d\n",
        req_id_, t1.sign, t1.Mp, e1, t2.sign, t2.Mp, e2, diff, max_P, min_P, aligned, val, s, P, E);

    return {{s, (int)P}, E};
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
    const int Ww = Wa + HR_;
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
    const int Ww = (W_ + 2) + HR_;
    const uint32_t mask = (1ULL << Ww) - 1;
    if (x.V == 0) return {0, 0, 0, x.sticky, -1000000000, x.flags, 0, 0, 0};

    int s = ((x.V >> (Ww - 1)) & 1);
    uint32_t Q = -x.V & mask;
    uint32_t X = s ? Q : x.V;
    int msb = 31 - __builtin_clz(X);
    int e = x.L + msb - EXP_POS_BIAS;
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
    // the datapath's result exponent is a signed 10-bit quantity
    be = ((be & 1023) ^ 512) - 512;
    if (be >= 255) return s_bit | 0x7f800000u;
    if (be <= 0) return s_bit;
    return s_bit | ((be & 0xff) << 23) | (kept_rounded & 0x7FFFFFu);
  }

  // members --------
  int exp_bits_, sig_bits_, frm_, W_, HR_;
  bool renorm_, no_window_;
  const uint8_t *sf_a_ = nullptr; // per-lane block scales, null when unscaled
  const uint8_t *sf_b_ = nullptr;
  bool e4m3_scale_ = false;
  uint64_t req_id_ = 0;
};