// The batched decode step on the device (plan D1, "batched decode"): B sequences, each
// with the same prompt already in its KV cache, each generating one token at position P
// (P = prompt length). M = B for every linear layer, so qkv / wo / w13 / w2 and the
// classifier are GEMM sites; attention is a per-(sequence, head) SIMT pass -- a GEMV over
// the sequence's own cache, since no GEMM batches across sequences that do not share a
// cache -- like every other non-GEMM op. B copies of one sequence cost the same cycles as
// B different sequences (no unit here has data-dependent timing) and let one host row be
// the reference for every device row.
//
// Per layer (9 launches):
//   RMSNorm(x) -> a16 ; qkv = a16 * Wqkv (C = 0) ; RoPE(q, k), k and v -> cache row P ;
//   attention -> att16 ; x += att16 * Wo ; RMSNorm(x) -> a16 ; d13 = a16 * W13 (C = 0) ;
//   hg16 = silu gate ; x += hg16 * W2
// then RMSNorm(x) -> a16 ; logits = a16 * Wcls (C = 0, fp32 [B x vocab]) ; argmax on the host.
//
// The KV cache of the prompt comes from the host fp16-emulating reference (HostRef with
// record_kv), not from a device prefill: the prefill is not what is measured here, and the
// host build has no shape restrictions. The token decoded is the fp32 forward's next token
// after the prompt; the fp32 forward over prompt + that token gives the decode step's truth.

#include "llama_forward.h"
#include "llama_dev.h"
#include "llama_model.h"
#include "llama_passes.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#include "common.h"
#include "host_modes.h"

#include "llama_hostref.h"

namespace {

enum DSite { D_QKV, D_WO, D_W13, D_W2, D_CLS, D_NSITES };
const char* kDSiteName[D_NSITES] = { "qkv", "wo", "w13", "w2", "cls" };
enum DPass { DP_RMSNORM, DP_ROPEKV, DP_ATTN, DP_SILU, DP_NPASSES };
const char* kDPassName[DP_NPASSES]  = { "pass_rmsnorm", "pass_rope_kv", "pass_attn_decode", "pass_silu" };
const char* kDPassEntry[DP_NPASSES] = { "llama_rmsnorm_f2h", "llama_rope_kvappend", "llama_attn_decode", "llama_silu_gate" };

// One row of the decode step on the host with the device's rounding points; every device
// row is this row (B copies of one sequence).
struct HostDecodeRef {
  const LlamaModel& m; const HostRef& w;       // fp16 weights and the RoPE table (P+1 rows)
  uint32_t dim, hidden, H, hs, P;
  std::vector<uint16_t> wcls16;
  std::vector<std::vector<uint16_t>> Kc, Vc;    // per layer [(P+1) x dim]; rows 0..P-1 = the prompt
  std::vector<float> x, qkv, d13, logits;
  std::vector<uint16_t> a16, q16, att16, hg16, final16;

  HostDecodeRef(const LlamaModel& model, const HostRef& pre, uint32_t P_) : m(model), w(pre), P(P_) {
    dim = m.cfg.dim; hidden = m.cfg.hidden_dim; H = m.cfg.n_heads; hs = dim / H;
    to_fp16(m.w.wcls, (size_t)m.cfg.vocab_size * dim, wcls16);
    Kc.resize(m.cfg.n_layers); Vc.resize(m.cfg.n_layers);
    for (int l = 0; l < m.cfg.n_layers; ++l) {
      Kc[l].assign((size_t)(P + 1) * dim, 0); Vc[l].assign((size_t)(P + 1) * dim, 0);
      std::memcpy(Kc[l].data(), pre.Kc[l].data(), (size_t)P * dim * 2);
      std::memcpy(Vc[l].data(), pre.Vc[l].data(), (size_t)P * dim * 2);
    }
    a16.resize(dim); q16.resize(dim); att16.resize(dim); hg16.resize(hidden); final16.resize(dim);
    qkv.resize((size_t)3 * dim); d13.resize((size_t)2 * hidden); logits.resize(m.cfg.vocab_size);
  }
  void rmsnorm(const float* xin, const float* wt, uint16_t* out) {
    float inv = HostRef::sum_sq(xin, dim) / (float)dim; inv += 1e-5f; inv = 1.0f / sqrtf(inv);
    for (uint32_t j = 0; j < dim; ++j) out[j] = llama_f2h(wt[j] * (inv * xin[j]));
  }
  void start(int token) {
    x.assign(dim, 0.0f);
    std::memcpy(x.data(), m.w.token_embedding_table + (size_t)token * dim, dim * 4);
  }
  void layer(int l) {
    rmsnorm(x.data(), m.w.rms_att_weight + (size_t)l * dim, a16.data());
    HostRef::gemm(1, 3 * dim, dim, a16.data(), w.wqkv16[l].data(), nullptr, qkv.data());
    // RoPE(q, k) at position P; k, v -> cache row P  (mirrors llama_rope_kvappend)
    const float* rp = &w.rope[(size_t)P * (hs / 2) * 2];
    uint16_t* kc = &Kc[l][(size_t)P * dim]; uint16_t* vc = &Vc[l][(size_t)P * dim];
    for (uint32_t i = 0; i < dim; i += 2) {
      const uint32_t c = i % hs;
      const float fcr = rp[(c / 2) * 2], fci = rp[(c / 2) * 2 + 1];
      const float q0 = qkv[i], q1 = qkv[i + 1], k0 = qkv[dim + i], k1 = qkv[dim + i + 1];
      q16[i] = llama_f2h(q0 * fcr - q1 * fci); q16[i + 1] = llama_f2h(q0 * fci + q1 * fcr);
      kc[i]  = llama_f2h(k0 * fcr - k1 * fci); kc[i + 1]  = llama_f2h(k0 * fci + k1 * fcr);
      vc[i]  = llama_f2h(qkv[2 * dim + i]);    vc[i + 1]  = llama_f2h(qkv[2 * dim + i + 1]);
    }
    // attention per head over the P+1 cached keys  (mirrors llama_attn_decode)
    const float scale = 1.0f / sqrtf((float)hs);
    const uint32_t L = P + 1;
    std::vector<float> sc(L);
    for (uint32_t h = 0; h < H; ++h) {
      float mx = -3.4028235e38f;
      for (uint32_t j = 0; j < L; ++j) {
        float s = 0.0f;
        for (uint32_t c = 0; c < hs; ++c) s += llama_h2f(q16[h * hs + c]) * llama_h2f(Kc[l][(size_t)j * dim + h * hs + c]);
        s *= scale; sc[j] = s; mx = s > mx ? s : mx;
      }
      float sum = 0.0f;
      for (uint32_t j = 0; j < L; ++j) { sc[j] = llama_fast_exp(sc[j] - mx); sum += sc[j]; }
      const float inv = 1.0f / sum;
      for (uint32_t c = 0; c < hs; ++c) {
        float o = 0.0f;
        for (uint32_t j = 0; j < L; ++j) o += sc[j] * llama_h2f(Vc[l][(size_t)j * dim + h * hs + c]);
        att16[h * hs + c] = llama_f2h(o * inv);
      }
    }
    std::vector<float> xin(x);
    HostRef::gemm(1, dim, dim, att16.data(), w.wo16[l].data(), xin.data(), x.data());
    rmsnorm(x.data(), m.w.rms_ffn_weight + (size_t)l * dim, a16.data());
    HostRef::gemm(1, 2 * hidden, dim, a16.data(), w.w13_16[l].data(), nullptr, d13.data());
    for (uint32_t j = 0; j < hidden; ++j) {
      const float a = d13[j], b = d13[hidden + j];
      hg16[j] = llama_f2h(a * (1.0f / (1.0f + llama_fast_exp(-a))) * b);
    }
    xin = x;
    HostRef::gemm(1, dim, hidden, hg16.data(), w.w2_16[l].data(), xin.data(), x.data());
  }
  void finish() {
    rmsnorm(x.data(), m.w.rms_final_weight, final16.data());
    HostRef::gemm(1, (uint32_t)m.cfg.vocab_size, dim, final16.data(), wcls16.data(), nullptr, logits.data());
  }
};

int argmax(const float* v, size_t n) {
  size_t b = 0; for (size_t i = 1; i < n; ++i) if (v[i] > v[b]) b = i; return (int)b;
}

}  // namespace

int run_decode(const RunOptions& opt) {
  LlamaModel model;
  if (!model.load(opt.model.c_str())) return -1;
  LlamaTokenizer tok;
  if (!tok.load(opt.tokenizer.c_str(), model.cfg.vocab_size)) return -1;
  const LlamaConfig& cfg = model.cfg;
  const uint32_t dim = cfg.dim, hidden = cfg.hidden_dim, H = cfg.n_heads, hs = dim / H, B = opt.B;
  const uint32_t vocab = (uint32_t)cfg.vocab_size;
  const int L = cfg.n_layers;
  if (cfg.n_kv_heads != cfg.n_heads) { std::cerr << "llama_dtcu: n_kv_heads != n_heads not supported" << std::endl; return -1; }
  if (B == 0) { std::cerr << "llama_dtcu: B must be >= 1" << std::endl; return -1; }

  // ---- tokens: BOS + prompt = the cached context; the step decodes position P ----
  std::vector<int> tokens;
  tok.encode(opt.prompt.c_str(), true, false, tokens);
  const uint32_t P = (uint32_t)tokens.size();
  if (P + 1 > (uint32_t)cfg.seq_len) { std::cerr << "llama_dtcu: prompt + 1 exceeds seq_len" << std::endl; return -1; }
  const uint32_t S = P + 1;   // KV cache rows per sequence

  // ---- host: prompt prefill with the device's rounding points -> KV cache; token to decode ----
  HostRef pre(model, P, 64);
  weights_to_fp16(model, pre);
  build_rope(P + 1, hs, pre.rope);
  pre.record_kv = true;
  std::vector<uint16_t> pre_final16;
  pre.forward(tokens, pre_final16);
  auto argmax_fp32 = [&](const float* act) {
    std::vector<float> logits(vocab);
    for (uint32_t i = 0; i < vocab; ++i) {
      const float* wr = model.w.wcls + (size_t)i * dim; float s = 0;
      for (uint32_t j = 0; j < dim; ++j) s += wr[j] * act[j];
      logits[i] = s;
    }
    return argmax(logits.data(), vocab);
  };
  std::vector<float> fp32_prompt_final;
  forward_fp32(model, tokens, P, fp32_prompt_final);
  const int token_in = argmax_fp32(&fp32_prompt_final[(size_t)(P - 1) * dim]);   // decoded at position P
  std::vector<int> tokens1(tokens); tokens1.push_back(token_in);
  std::vector<float> fp32_final;
  forward_fp32(model, tokens1, P + 1, fp32_final);
  const int next_fp32 = argmax_fp32(&fp32_final[(size_t)P * dim]);

  // ---- mode per site ----
  int site_mode[D_NSITES];
  for (int s = 0; s < D_NSITES; ++s) site_mode[s] = opt.mode;
  if (!opt.modemap.empty()) {
    std::ifstream f(opt.modemap);
    if (!f) { std::cerr << "llama_dtcu: cannot open modemap " << opt.modemap << std::endl; return -1; }
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream is(line); std::string site; int m;
      if (!(is >> site >> m)) continue;
      bool hit = false;
      for (int s = 0; s < D_NSITES; ++s) if (site == kDSiteName[s]) { site_mode[s] = m; hit = true; }
      if (site == "lin") { site_mode[D_QKV] = site_mode[D_WO] = site_mode[D_W13] = site_mode[D_W2] = m; hit = true; }
      if (!hit) { std::cerr << "llama_dtcu: unknown decode site in modemap: " << site << " (qkv wo w13 w2 cls lin)" << std::endl; return -1; }
    }
  }
  const uint32_t site_M[D_NSITES] = { B, B, B, B, B };
  const uint32_t site_N[D_NSITES] = { 3 * dim, dim, 2 * hidden, dim, vocab };
  const uint32_t site_K[D_NSITES] = { dim, dim, dim, hidden, dim };

  LlamaDevice dev;
  RT_CHECK(dev.open());
  for (int s = 0; s < D_NSITES; ++s) {
    std::string why = dev.mode_illegal_reason(site_mode[s], site_M[s], site_N[s], site_K[s]);
    if (!why.empty()) {
      if (opt.fallback >= 0 && dev.mode_illegal_reason(opt.fallback, site_M[s], site_N[s], site_K[s]).empty()) {
        std::cerr << "llama_dtcu: site " << kDSiteName[s] << ": " << why << " -> fallback mode " << opt.fallback << std::endl;
        site_mode[s] = opt.fallback;
      } else {
        std::cerr << "llama_dtcu: site " << kDSiteName[s] << " (" << site_M[s] << "x" << site_N[s] << "x" << site_K[s]
                  << "): " << why << std::endl;
        return -1;
      }
    }
  }
  std::string run_tag;
  {
    bool uniform = true;
    for (int s = 1; s < D_NSITES; ++s) if (site_mode[s] != site_mode[0]) uniform = false;
    std::ostringstream os;
    if (uniform) os << "uniform:" << site_mode[0];
    else { os << "map"; for (int s = 0; s < D_NSITES; ++s) os << ":" << kDSiteName[s] << "=" << site_mode[s]; }
    run_tag = os.str();
  }
  {
    std::vector<std::string> entries;
    for (int p = 0; p < DP_NPASSES; ++p) entries.push_back(kDPassEntry[p]);
    RT_CHECK(dev.register_passes("llama_passes.vxbin", entries));
  }
  const uint32_t attn_lmem = (S + NUM_THREADS) * sizeof(float);   // scores, reduction scratch
  if (attn_lmem > dev.lmem_size()) { std::cerr << "llama_dtcu: decode attention needs " << attn_lmem << " B of Local Memory" << std::endl; return -1; }

  HostDecodeRef ref(model, pre, P);
  ref.start(token_in);

  // ---- device weights, KV cache, activations ----
  std::vector<DevBuf> d_wqkv(L), d_wo(L), d_w13(L), d_w2(L), d_rms_att(L), d_rms_ffn(L), d_Kc(L), d_Vc(L);
  for (int l = 0; l < L; ++l) {
    RT_CHECK(make_buf(dev, d_wqkv[l], pre.wqkv16[l].size() * 2, VX_MEM_READ));  RT_CHECK(dev.upload(d_wqkv[l].buf, 0, pre.wqkv16[l].data(), d_wqkv[l].bytes));
    RT_CHECK(make_buf(dev, d_wo[l],   pre.wo16[l].size() * 2,   VX_MEM_READ));  RT_CHECK(dev.upload(d_wo[l].buf,   0, pre.wo16[l].data(),   d_wo[l].bytes));
    RT_CHECK(make_buf(dev, d_w13[l],  pre.w13_16[l].size() * 2, VX_MEM_READ));  RT_CHECK(dev.upload(d_w13[l].buf,  0, pre.w13_16[l].data(), d_w13[l].bytes));
    RT_CHECK(make_buf(dev, d_w2[l],   pre.w2_16[l].size() * 2,  VX_MEM_READ));  RT_CHECK(dev.upload(d_w2[l].buf,   0, pre.w2_16[l].data(),  d_w2[l].bytes));
    RT_CHECK(make_buf(dev, d_rms_att[l], (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_att[l].buf, 0, model.w.rms_att_weight + (size_t)l * dim, dim * 4));
    RT_CHECK(make_buf(dev, d_rms_ffn[l], (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_ffn[l].buf, 0, model.w.rms_ffn_weight + (size_t)l * dim, dim * 4));
    // B copies of the prompt's cache rows; row P is written by the RoPE pass
    std::vector<uint16_t> kv((size_t)B * S * dim, 0);
    for (uint32_t b = 0; b < B; ++b) std::memcpy(&kv[(size_t)b * S * dim], pre.Kc[l].data(), (size_t)P * dim * 2);
    RT_CHECK(make_buf(dev, d_Kc[l], kv.size() * 2, VX_MEM_READ_WRITE)); RT_CHECK(dev.upload(d_Kc[l].buf, 0, kv.data(), d_Kc[l].bytes));
    for (uint32_t b = 0; b < B; ++b) std::memcpy(&kv[(size_t)b * S * dim], pre.Vc[l].data(), (size_t)P * dim * 2);
    RT_CHECK(make_buf(dev, d_Vc[l], kv.size() * 2, VX_MEM_READ_WRITE)); RT_CHECK(dev.upload(d_Vc[l].buf, 0, kv.data(), d_Vc[l].bytes));
  }
  DevBuf d_wcls; RT_CHECK(make_buf(dev, d_wcls, ref.wcls16.size() * 2, VX_MEM_READ)); RT_CHECK(dev.upload(d_wcls.buf, 0, ref.wcls16.data(), d_wcls.bytes));
  DevBuf d_rms_final; RT_CHECK(make_buf(dev, d_rms_final, (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_final.buf, 0, model.w.rms_final_weight, dim * 4));
  DevBuf d_rope; RT_CHECK(make_buf(dev, d_rope, pre.rope.size() * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rope.buf, 0, pre.rope.data(), d_rope.bytes));

  DevBuf d_x, d_a16, d_qkv, d_q16, d_att16, d_d13, d_hg16, d_logits, d_zero;
  RT_CHECK(make_buf(dev, d_x,      (size_t)B * dim * 4,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_a16,    (size_t)B * dim * 2,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_qkv,    (size_t)B * 3 * dim * 4,    VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_q16,    (size_t)B * dim * 2,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_att16,  (size_t)B * dim * 2,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_d13,    (size_t)B * 2 * hidden * 4, VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_hg16,   (size_t)B * hidden * 2,     VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_logits, (size_t)B * vocab * 4,      VX_MEM_READ_WRITE));
  const size_t zero_elems = std::max<size_t>((size_t)B * 2 * hidden, (size_t)B * vocab);
  RT_CHECK(make_buf(dev, d_zero,   zero_elems * 4,             VX_MEM_READ));
  {
    std::vector<float> zeros(zero_elems, 0.0f);
    RT_CHECK(dev.upload(d_zero.buf, 0, zeros.data(), d_zero.bytes));
    std::vector<float> x0((size_t)B * dim);
    for (uint32_t b = 0; b < B; ++b) std::memcpy(&x0[(size_t)b * dim], model.w.token_embedding_table + (size_t)token_in * dim, dim * 4);
    RT_CHECK(dev.upload(d_x.buf, 0, x0.data(), d_x.bytes));
  }

  // ---- the device decode step ----
  LaunchStats site_st[D_NSITES], pass_st[DP_NPASSES];
  llama_pass_arg_t pa{};
  pa.T = B; pa.dim = dim; pa.hidden = hidden; pa.n_heads = H; pa.head = hs; pa.head_pad = 64;
  pa.pos0 = P; pa.kv_stride = S; pa.eps = 1e-5f; pa.scale = 1.0f / sqrtf((float)hs);
  const uint32_t red_lmem = NUM_THREADS * sizeof(float);
  int verify_failures = 0;
  const double tol = 4e-3;   // as in the prefill: fp32 summation-order noise only

  // every device row against the one host row
  auto check_rows = [&](const char* what, int l, const std::vector<float>& got, const std::vector<float>& want) {
    if (!opt.verify) return;
    Compare c; const double sc = max_abs(want.data(), want.size());
    for (uint32_t b = 0; b < B; ++b)
      for (size_t i = 0; i < want.size(); ++i) c.add(got[(size_t)b * want.size() + i], want[i], sc, tol);
    std::printf("[LLAMA-VERIFY] layer=%d site=%s n=%zu max_rel=%.3e n_bad=%zu%s\n", l, what, c.n, c.max_rel, c.n_bad, c.n_bad ? "  <-- FAIL" : "");
    if (c.n_bad) ++verify_failures;
  };
  auto check_rows16 = [&](const char* what, int l, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
    if (!opt.verify) return;
    Compare c; const double sc = max_abs16(want.data(), want.size());
    for (uint32_t b = 0; b < B; ++b)
      for (size_t i = 0; i < want.size(); ++i) c.add(llama_h2f(got[(size_t)b * want.size() + i]), llama_h2f(want[i]), sc, tol);
    std::printf("[LLAMA-VERIFY] layer=%d site=%s n=%zu max_rel=%.3e n_bad=%zu%s\n", l, what, c.n, c.max_rel, c.n_bad, c.n_bad ? "  <-- FAIL" : "");
    if (c.n_bad) ++verify_failures;
  };
  std::vector<float> got_f; std::vector<uint16_t> got_h;

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int l = 0; l < L; ++l) {
    if (opt.verify) ref.layer(l);
    // RMSNorm(att) -> a16
    pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_att[l].addr;
    RT_CHECK(dev.launch_pass(DP_RMSNORM, &pa, sizeof(pa), B, 1, red_lmem, &pass_st[DP_RMSNORM]));
    // qkv
    RT_CHECK(dev.gemm(site_mode[D_QKV], B, 3 * dim, dim, d_a16.addr, d_wqkv[l].addr, d_zero.addr, d_qkv.addr, &site_st[D_QKV]));
    if (opt.verify) { got_f.resize((size_t)B * 3 * dim); RT_CHECK(dev.download(got_f.data(), d_qkv.buf, 0, d_qkv.bytes)); check_rows("qkv", l, got_f, ref.qkv); }
    // RoPE(q, k); k, v -> cache row P
    pa.src = d_qkv.addr; pa.dst = d_q16.addr; pa.aux0 = d_Kc[l].addr; pa.aux1 = d_Vc[l].addr; pa.aux2 = d_rope.addr;
    RT_CHECK(dev.launch_pass(DP_ROPEKV, &pa, sizeof(pa), (dim / 2 + NUM_THREADS - 1) / NUM_THREADS, B, 0, &pass_st[DP_ROPEKV]));
    // attention over the cache -> att16
    pa.src = d_q16.addr; pa.dst = d_att16.addr; pa.aux0 = d_Kc[l].addr; pa.aux1 = d_Vc[l].addr;
    RT_CHECK(dev.launch_pass(DP_ATTN, &pa, sizeof(pa), H, B, attn_lmem, &pass_st[DP_ATTN]));
    if (opt.verify) { got_h.resize((size_t)B * dim); RT_CHECK(dev.download(got_h.data(), d_att16.buf, 0, d_att16.bytes)); check_rows16("attn", l, got_h, ref.att16); }
    // wo, residual as the C preload, in place
    RT_CHECK(dev.gemm(site_mode[D_WO], B, dim, dim, d_att16.addr, d_wo[l].addr, d_x.addr, d_x.addr, &site_st[D_WO]));
    // RMSNorm(ffn) -> a16
    pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_ffn[l].addr;
    RT_CHECK(dev.launch_pass(DP_RMSNORM, &pa, sizeof(pa), B, 1, red_lmem, &pass_st[DP_RMSNORM]));
    // w1|w3
    RT_CHECK(dev.gemm(site_mode[D_W13], B, 2 * hidden, dim, d_a16.addr, d_w13[l].addr, d_zero.addr, d_d13.addr, &site_st[D_W13]));
    if (opt.verify) { got_f.resize((size_t)B * 2 * hidden); RT_CHECK(dev.download(got_f.data(), d_d13.buf, 0, d_d13.bytes)); check_rows("w13", l, got_f, ref.d13); }
    // SiLU gate -> hg16
    pa.src = d_d13.addr; pa.dst = d_hg16.addr;
    RT_CHECK(dev.launch_pass(DP_SILU, &pa, sizeof(pa), (hidden + NUM_THREADS - 1) / NUM_THREADS, B, 0, &pass_st[DP_SILU]));
    // w2, residual as the C preload, in place
    RT_CHECK(dev.gemm(site_mode[D_W2], B, dim, hidden, d_hg16.addr, d_w2[l].addr, d_x.addr, d_x.addr, &site_st[D_W2]));
    if (opt.verify) { got_f.resize((size_t)B * dim); RT_CHECK(dev.download(got_f.data(), d_x.buf, 0, d_x.bytes)); check_rows("x", l, got_f, ref.x); }
  }
  // final RMSNorm -> a16 ; classifier GEMM -> logits
  pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_final.addr;
  RT_CHECK(dev.launch_pass(DP_RMSNORM, &pa, sizeof(pa), B, 1, red_lmem, &pass_st[DP_RMSNORM]));
  if (opt.verify) { ref.finish(); got_h.resize((size_t)B * dim); RT_CHECK(dev.download(got_h.data(), d_a16.buf, 0, d_a16.bytes)); check_rows16("final", L, got_h, ref.final16); }
  RT_CHECK(dev.gemm(site_mode[D_CLS], B, vocab, dim, d_a16.addr, d_wcls.addr, d_zero.addr, d_logits.addr, &site_st[D_CLS]));
  auto t_end = std::chrono::high_resolution_clock::now();
  std::vector<float> logits((size_t)B * vocab);
  RT_CHECK(dev.download(logits.data(), d_logits.buf, 0, d_logits.bytes));
  if (opt.verify) check_rows("cls", L, logits, ref.logits);

  const int next_dev = argmax(logits.data(), vocab);
  uint32_t rows_agree = 0;
  for (uint32_t b = 0; b < B; ++b) if (argmax(&logits[(size_t)b * vocab], vocab) == next_dev) ++rows_agree;
  const int next_ref = opt.verify ? argmax(ref.logits.data(), vocab) : -1;

  // ---- report ----
  uint64_t total_cycles = 0, gemm_cycles = 0, pass_cycles = 0; uint32_t total_launches = 0;
  for (int s = 0; s < D_NSITES; ++s) {
    std::printf("[LLAMA] run=%s B=%u pos=%u site=%s mode=%d launches=%u cycles=%llu\n", run_tag.c_str(), B, P,
                kDSiteName[s], site_mode[s], site_st[s].launches, (unsigned long long)site_st[s].cycles);
    total_cycles += site_st[s].cycles; gemm_cycles += site_st[s].cycles; total_launches += site_st[s].launches;
  }
  for (int p = 0; p < DP_NPASSES; ++p) {
    std::printf("[LLAMA] run=%s B=%u pos=%u site=%s mode=simt launches=%u cycles=%llu\n", run_tag.c_str(), B, P,
                kDPassName[p], pass_st[p].launches, (unsigned long long)pass_st[p].cycles);
    total_cycles += pass_st[p].cycles; pass_cycles += pass_st[p].cycles; total_launches += pass_st[p].launches;
  }
  const double cpt = (double)total_cycles / (double)B;
  const double wall_s = std::chrono::duration<double>(t_end - t_start).count();
  std::printf("[LLAMA] run=%s B=%u pos=%u total_launches=%u total_cycles=%llu gemm_cycles=%llu pass_cycles=%llu "
              "cycles_per_token=%.0f tok_per_s_400MHz=%.2f verify_fail_sites=%d next_dev=%d next_ref=%d next_fp32=%d "
              "rows_agree=%u program_swaps=%u wall_s=%.1f\n",
              run_tag.c_str(), B, P, total_launches, (unsigned long long)total_cycles,
              (unsigned long long)gemm_cycles, (unsigned long long)pass_cycles, cpt, 400e6 / cpt,
              verify_failures, next_dev, next_ref, next_fp32, rows_agree, dev.program_swaps(), wall_s);
  if (opt.text) {
    std::printf("context (%u tokens): ", P);
    for (uint32_t i = 0; i < P; ++i) std::printf("%s", tok.decode(i ? tokens[i - 1] : 1, tokens[i]));
    std::printf("\ntoken decoded at position %u (fp32 host's next token): '%s'\n", P, tok.decode(tokens[P - 1], token_in));
    std::printf("next token  device: '%s'   fp32 host: '%s'   (%u/%u device rows agree)%s\n",
                tok.decode(token_in, next_dev), tok.decode(token_in, next_fp32), rows_agree, B,
                opt.verify ? (next_ref == next_dev ? "   [fp16 reference agrees with device]" : "   [fp16 reference DIFFERS from device]") : "");
  }
  return verify_failures;
}
