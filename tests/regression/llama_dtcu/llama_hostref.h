#ifndef _LLAMA_DTCU_HOSTREF_H_
#define _LLAMA_DTCU_HOSTREF_H_

// Host-side pieces shared by the prefill forward (llama_forward.cpp) and the batched decode
// step (llama_decode.cpp): device buffer helper, fp16 weight conversion, the fp16-emulating
// host reference (HostRef), the plain fp32 llama2.c forward (the numerical truth) and the
// comparison helper. Header-only; moved here unchanged from llama_forward.cpp except for
// HostRef's optional KV recording (record_kv) that the decode step uses to build its cache.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "llama_dev.h"
#include "llama_model.h"
#include "llama_passes.h"

#define RT_CHECK(_expr)                                                   \
  do {                                                                    \
    int _ret = (int)(_expr);                                              \
    if (0 != _ret) {                                                      \
      std::cerr << "llama_dtcu: " << #_expr << " returned " << _ret       \
                << std::endl;                                             \
      return _ret;                                                        \
    }                                                                     \
  } while (false)




struct DevBuf {
  vx_buffer_h buf = nullptr; uint64_t addr = 0; uint64_t bytes = 0;
};

inline int make_buf(LlamaDevice& dev, DevBuf& b, uint64_t bytes, int flags) {
  b.bytes = bytes;
  return dev.alloc(bytes, flags, &b.buf, &b.addr);
}

// Host-side fp16 helpers over whole arrays.
inline void to_fp16(const float* src, size_t n, std::vector<uint16_t>& dst) {
  dst.resize(n);
  for (size_t i = 0; i < n; ++i) dst[i] = llama_f2h(src[i]);
}

// ---------------------------------------------------------------------------------------
// Host reference with the device's fp16 rounding points (see file header).
struct HostRef {
  const LlamaModel& m;
  uint32_t T, dim, hidden, H, hs, hp;
  std::vector<std::vector<uint16_t>> wqkv16, wo16, w13_16, w2_16;   // per layer
  std::vector<float> rope;                                          // [T][hs/2][2]
  // per-layer outputs kept for comparison
  std::vector<float> x, qkv, d13, S, Dh;
  std::vector<uint16_t> a16, Qp, Kp, VpT, P, att16, hg16;
  // decode support: when record_kv, layer() also stores the RoPE'd k and the v of every
  // row as fp16 [T x dim] per layer (the KV cache rows the decode step attends to).
  bool record_kv = false;
  std::vector<std::vector<uint16_t>> Kc, Vc;

  HostRef(const LlamaModel& model, uint32_t T_, uint32_t hp_) : m(model), T(T_) {
    dim = m.cfg.dim; hidden = m.cfg.hidden_dim; H = m.cfg.n_heads; hs = dim / H; hp = hp_;
  }

  static float sum_sq(const float* v, uint32_t n) { float s = 0; for (uint32_t i = 0; i < n; ++i) s += v[i] * v[i]; return s; }

  void rmsnorm(const float* xin, const float* w, uint16_t* out) {
    for (uint32_t r = 0; r < T; ++r) {
      float inv = sum_sq(xin + (size_t)r * dim, dim) / (float)dim;
      inv += 1e-5f; inv = 1.0f / sqrtf(inv);
      for (uint32_t j = 0; j < dim; ++j) out[(size_t)r * dim + j] = llama_f2h(w[j] * (inv * xin[(size_t)r * dim + j]));
    }
  }
  // D[MxN] = C + A[MxK](fp16) * B(fp16 col-major, W[NxK])
  static void gemm(uint32_t M, uint32_t N, uint32_t K, const uint16_t* A, const uint16_t* B,
                   const float* C, float* D) {
    for (uint32_t i = 0; i < M; ++i)
      for (uint32_t j = 0; j < N; ++j) {
        float acc = C ? C[(size_t)i * N + j] : 0.0f;
        const uint16_t* a = A + (size_t)i * K; const uint16_t* b = B + (size_t)j * K;
        for (uint32_t k = 0; k < K; ++k) acc += llama_h2f(a[k]) * llama_h2f(b[k]);
        D[(size_t)i * N + j] = acc;
      }
  }
  void forward(const std::vector<int>& tokens, std::vector<uint16_t>& final16) {
    x.assign((size_t)T * dim, 0.0f);
    for (uint32_t r = 0; r < T; ++r)
      std::memcpy(&x[(size_t)r * dim], m.w.token_embedding_table + (size_t)tokens[r] * dim, dim * sizeof(float));
    a16.resize((size_t)T * dim); qkv.resize((size_t)T * 3 * dim);
    Qp.resize((size_t)H * T * hp); Kp.resize((size_t)H * T * hp); VpT.resize((size_t)H * hp * T);
    S.resize((size_t)H * T * T); P.resize((size_t)H * T * T); Dh.resize((size_t)H * T * hp);
    att16.resize((size_t)T * dim); d13.resize((size_t)T * 2 * hidden); hg16.resize((size_t)T * hidden);
    if (record_kv) { Kc.assign(m.cfg.n_layers, {}); Vc.assign(m.cfg.n_layers, {}); }
    for (int l = 0; l < m.cfg.n_layers; ++l) layer(l);
    final16.resize((size_t)T * dim);
    rmsnorm(x.data(), m.w.rms_final_weight, final16.data());
  }
  void layer(int l) {
    rmsnorm(x.data(), m.w.rms_att_weight + (size_t)l * dim, a16.data());
    gemm(T, 3 * dim, dim, a16.data(), wqkv16[l].data(), nullptr, qkv.data());
    // rope + head pack
    std::fill(Qp.begin(), Qp.end(), 0); std::fill(Kp.begin(), Kp.end(), 0); std::fill(VpT.begin(), VpT.end(), 0);
    for (uint32_t r = 0; r < T; ++r) {
      const float* q = &qkv[(size_t)r * 3 * dim];
      const float* rp = &rope[(size_t)r * (hs / 2) * 2];
      for (uint32_t i = 0; i < dim; i += 2) {
        const uint32_t h = i / hs, c = i % hs;
        const float fcr = rp[(c / 2) * 2], fci = rp[(c / 2) * 2 + 1];
        const float q0 = q[i], q1 = q[i + 1], k0 = q[dim + i], k1 = q[dim + i + 1];
        Qp[(size_t)h * T * hp + (size_t)r * hp + c]     = llama_f2h(q0 * fcr - q1 * fci);
        Qp[(size_t)h * T * hp + (size_t)r * hp + c + 1] = llama_f2h(q0 * fci + q1 * fcr);
        Kp[(size_t)h * T * hp + (size_t)r * hp + c]     = llama_f2h(k0 * fcr - k1 * fci);
        Kp[(size_t)h * T * hp + (size_t)r * hp + c + 1] = llama_f2h(k0 * fci + k1 * fcr);
      }
      for (uint32_t i = 0; i < dim; ++i) {
        const uint32_t h = i / hs, c = i % hs;
        VpT[(size_t)h * T * hp + (size_t)c * T + r] = llama_f2h(q[2 * dim + i]);
      }
    }
    if (record_kv) {
      Kc[l].resize((size_t)T * dim); Vc[l].resize((size_t)T * dim);
      for (uint32_t r = 0; r < T; ++r)
        for (uint32_t i = 0; i < dim; ++i) {
          const uint32_t h = i / hs, c = i % hs;
          Kc[l][(size_t)r * dim + i] = Kp[(size_t)h * T * hp + (size_t)r * hp + c];
          Vc[l][(size_t)r * dim + i] = VpT[(size_t)h * T * hp + (size_t)c * T + r];
        }
    }
    const float scale = 1.0f / sqrtf((float)hs);
    for (uint32_t h = 0; h < H; ++h) {
      gemm(T, T, hp, &Qp[(size_t)h * T * hp], &Kp[(size_t)h * T * hp], nullptr, &S[(size_t)h * T * T]);
      for (uint32_t r = 0; r < T; ++r) {
        const float* s = &S[(size_t)h * T * T + (size_t)r * T];
        uint16_t* p = &P[(size_t)h * T * T + (size_t)r * T];
        const uint32_t valid = r + 1;
        float mx = -3.4028235e38f;
        for (uint32_t j = 0; j < valid; ++j) { const float v = s[j] * scale; mx = v > mx ? v : mx; }
        float sum = 0.0f;
        for (uint32_t j = 0; j < valid; ++j) sum += llama_fast_exp(s[j] * scale - mx);
        const float inv = 1.0f / sum;
        for (uint32_t j = 0; j < T; ++j) p[j] = (j < valid) ? llama_f2h(llama_fast_exp(s[j] * scale - mx) * inv) : 0;
      }
      gemm(T, hp, T, &P[(size_t)h * T * T], &VpT[(size_t)h * T * hp], nullptr, &Dh[(size_t)h * T * hp]);
    }
    for (uint32_t r = 0; r < T; ++r)
      for (uint32_t i = 0; i < dim; ++i) {
        const uint32_t h = i / hs, c = i % hs;
        att16[(size_t)r * dim + i] = llama_f2h(Dh[(size_t)h * T * hp + (size_t)r * hp + c]);
      }
    std::vector<float> xin(x);
    gemm(T, dim, dim, att16.data(), wo16[l].data(), xin.data(), x.data());
    rmsnorm(x.data(), m.w.rms_ffn_weight + (size_t)l * dim, a16.data());
    gemm(T, 2 * hidden, dim, a16.data(), w13_16[l].data(), nullptr, d13.data());
    for (uint32_t r = 0; r < T; ++r)
      for (uint32_t j = 0; j < hidden; ++j) {
        const float a = d13[(size_t)r * 2 * hidden + j], b = d13[(size_t)r * 2 * hidden + hidden + j];
        hg16[(size_t)r * hidden + j] = llama_f2h(a * (1.0f / (1.0f + llama_fast_exp(-a))) * b);
      }
    xin = x;
    gemm(T, dim, hidden, hg16.data(), w2_16[l].data(), xin.data(), x.data());
  }
};

// Plain fp32 llama2.c forward for all T positions (no fp16 anywhere): the numerical truth.
inline void forward_fp32(const LlamaModel& m, const std::vector<int>& tokens, uint32_t T, std::vector<float>& final_x) {
  const uint32_t dim = m.cfg.dim, hidden = m.cfg.hidden_dim, H = m.cfg.n_heads, hs = dim / H;
  std::vector<float> x((size_t)T * dim), xb((size_t)T * dim), q((size_t)T * dim), k((size_t)T * dim), v((size_t)T * dim),
      att((size_t)T * dim), hb((size_t)T * hidden), hb2((size_t)T * hidden);
  for (uint32_t r = 0; r < T; ++r) std::memcpy(&x[(size_t)r * dim], m.w.token_embedding_table + (size_t)tokens[r] * dim, dim * 4);
  auto matmul = [](float* out, const float* in, const float* w, uint32_t n, uint32_t d) {
    for (uint32_t i = 0; i < d; ++i) { float s = 0; for (uint32_t j = 0; j < n; ++j) s += w[(size_t)i * n + j] * in[j]; out[i] = s; }
  };
  auto rms = [&](float* o, const float* xi, const float* w) {
    float ss = 0; for (uint32_t j = 0; j < dim; ++j) ss += xi[j] * xi[j];
    ss /= dim; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
    for (uint32_t j = 0; j < dim; ++j) o[j] = w[j] * (ss * xi[j]);
  };
  for (int l = 0; l < m.cfg.n_layers; ++l) {
    for (uint32_t r = 0; r < T; ++r) {
      rms(&xb[(size_t)r * dim], &x[(size_t)r * dim], m.w.rms_att_weight + (size_t)l * dim);
      matmul(&q[(size_t)r * dim], &xb[(size_t)r * dim], m.w.wq + (size_t)l * dim * dim, dim, dim);
      matmul(&k[(size_t)r * dim], &xb[(size_t)r * dim], m.w.wk + (size_t)l * dim * dim, dim, dim);
      matmul(&v[(size_t)r * dim], &xb[(size_t)r * dim], m.w.wv + (size_t)l * dim * dim, dim, dim);
      for (uint32_t i = 0; i < dim; i += 2) {
        const int head_dim = i % hs;
        const float freq = 1.0f / powf(10000.0f, head_dim / (float)hs);
        const float val = (float)r * freq, fcr = cosf(val), fci = sinf(val);
        float* qq = &q[(size_t)r * dim + i]; float* kk = &k[(size_t)r * dim + i];
        const float q0 = qq[0], q1 = qq[1], k0 = kk[0], k1 = kk[1];
        qq[0] = q0 * fcr - q1 * fci; qq[1] = q0 * fci + q1 * fcr;
        kk[0] = k0 * fcr - k1 * fci; kk[1] = k0 * fci + k1 * fcr;
      }
    }
    std::vector<float> sc(T);
    for (uint32_t r = 0; r < T; ++r)
      for (uint32_t h = 0; h < H; ++h) {
        const float* qh = &q[(size_t)r * dim + h * hs];
        for (uint32_t t = 0; t <= r; ++t) {
          const float* kh = &k[(size_t)t * dim + h * hs];
          float s = 0; for (uint32_t c = 0; c < hs; ++c) s += qh[c] * kh[c];
          sc[t] = s / sqrtf((float)hs);
        }
        float mx = sc[0]; for (uint32_t t = 1; t <= r; ++t) mx = sc[t] > mx ? sc[t] : mx;
        float sum = 0; for (uint32_t t = 0; t <= r; ++t) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        for (uint32_t t = 0; t <= r; ++t) sc[t] /= sum;
        float* out = &att[(size_t)r * dim + h * hs];
        for (uint32_t c = 0; c < hs; ++c) out[c] = 0;
        for (uint32_t t = 0; t <= r; ++t) { const float* vh = &v[(size_t)t * dim + h * hs]; for (uint32_t c = 0; c < hs; ++c) out[c] += sc[t] * vh[c]; }
      }
    for (uint32_t r = 0; r < T; ++r) {
      matmul(&xb[(size_t)r * dim], &att[(size_t)r * dim], m.w.wo + (size_t)l * dim * dim, dim, dim);
      for (uint32_t j = 0; j < dim; ++j) x[(size_t)r * dim + j] += xb[(size_t)r * dim + j];
      rms(&xb[(size_t)r * dim], &x[(size_t)r * dim], m.w.rms_ffn_weight + (size_t)l * dim);
      matmul(&hb[(size_t)r * hidden], &xb[(size_t)r * dim], m.w.w1 + (size_t)l * dim * hidden, dim, hidden);
      matmul(&hb2[(size_t)r * hidden], &xb[(size_t)r * dim], m.w.w3 + (size_t)l * dim * hidden, dim, hidden);
      for (uint32_t j = 0; j < hidden; ++j) { float a = hb[(size_t)r * hidden + j]; a *= 1.0f / (1.0f + expf(-a)); hb[(size_t)r * hidden + j] = a * hb2[(size_t)r * hidden + j]; }
      matmul(&xb[(size_t)r * dim], &hb[(size_t)r * hidden], m.w.w2 + (size_t)l * dim * hidden, hidden, dim);
      for (uint32_t j = 0; j < dim; ++j) x[(size_t)r * dim + j] += xb[(size_t)r * dim + j];
    }
  }
  final_x.resize((size_t)T * dim);
  for (uint32_t r = 0; r < T; ++r) rms(&final_x[(size_t)r * dim], &x[(size_t)r * dim], m.w.rms_final_weight);
}

struct Compare {
  double max_rel = 0.0; size_t n_bad = 0; size_t n = 0;
  void add(double got, double ref, double ref_scale, double tol) {
    const double err = std::fabs(got - ref) / (ref_scale > 0 ? ref_scale : 1.0);
    if (err > max_rel) max_rel = err;
    if (err > tol) ++n_bad;
    ++n;
  }
};
inline double max_abs(const float* v, size_t n) { double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(v[i])); return m; }
inline double max_abs16(const uint16_t* v, size_t n) { double m = 0; for (size_t i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(llama_h2f(v[i]))); return m; }

// Weights of every layer to fp16 (RNE), concatenated the way the GEMM sites consume them:
// Wqkv = [Wq; Wk; Wv] ([3*dim x dim]), W13 = [W1; W3] ([2*hidden x dim]).
inline void weights_to_fp16(const LlamaModel& model, HostRef& ref) {
  const uint32_t dim = model.cfg.dim, hidden = model.cfg.hidden_dim;
  const int L = model.cfg.n_layers;
  ref.wqkv16.resize(L); ref.wo16.resize(L); ref.w13_16.resize(L); ref.w2_16.resize(L);
  for (int l = 0; l < L; ++l) {
    std::vector<float> tmp((size_t)3 * dim * dim);
    std::memcpy(&tmp[0],                     model.w.wq + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    std::memcpy(&tmp[(size_t)dim * dim],     model.w.wk + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    std::memcpy(&tmp[(size_t)2 * dim * dim], model.w.wv + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    to_fp16(tmp.data(), tmp.size(), ref.wqkv16[l]);
    to_fp16(model.w.wo + (size_t)l * dim * dim, (size_t)dim * dim, ref.wo16[l]);
    tmp.resize((size_t)2 * hidden * dim);
    std::memcpy(&tmp[0],                    model.w.w1 + (size_t)l * dim * hidden, (size_t)dim * hidden * 4);
    std::memcpy(&tmp[(size_t)hidden * dim], model.w.w3 + (size_t)l * dim * hidden, (size_t)dim * hidden * 4);
    to_fp16(tmp.data(), tmp.size(), ref.w13_16[l]);
    to_fp16(model.w.w2 + (size_t)l * dim * hidden, (size_t)dim * hidden, ref.w2_16[l]);
  }
}

// llama2.c's RoPE table for positions 0..n_pos-1: [pos][head/2][2] = (cos, sin).
inline void build_rope(uint32_t n_pos, uint32_t hs, std::vector<float>& rope) {
  rope.resize((size_t)n_pos * (hs / 2) * 2);
  for (uint32_t pos = 0; pos < n_pos; ++pos)
    for (uint32_t c2 = 0; c2 < hs / 2; ++c2) {
      const int head_dim = 2 * c2;
      const float freq = 1.0f / powf(10000.0f, head_dim / (float)hs);
      const float val = (float)pos * freq;
      rope[((size_t)pos * (hs / 2) + c2) * 2 + 0] = cosf(val);
      rope[((size_t)pos * (hs / 2) + c2) * 2 + 1] = sinf(val);
    }
}

#endif // _LLAMA_DTCU_HOSTREF_H_
