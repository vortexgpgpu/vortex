// The llama prefill forward on the device, one GEMM mode per site.
//
// Data path per layer (T rows):
//   x     fp32 [T x dim]        residual stream (lives on the device for the whole forward)
//   a16   fp16 [T x dim]        RMSNorm(x) -> A operand of the QKV GEMM
//   qkv   fp32 [T x 3*dim]      QKV GEMM output (C = 0)
//   Qp,Kp fp16 [H][T x 64]      RoPE'd q/k, head-packed, head 48 -> 64 zero padded
//   VpT   fp16 [H][64 x T]      v, head-packed and transposed (B operand of PV)
//   S     fp32 [H][T x T]       QK^T per head (C = 0)
//   P     fp16 [H][T x T]       causal softmax(S / sqrt(head))
//   Dh    fp32 [H][T x 64]      P V per head (C = 0)
//   att16 fp16 [T x dim]        heads gathered, pad dropped -> A of the wo GEMM
//   x     <- x + att16 * wo     residual as the C preload, in place
//   a16   <- RMSNorm(x)
//   d13   fp32 [T x 2*hidden]   (w1 | w3) GEMM (C = 0)
//   hg16  fp16 [T x hidden]     silu(d13[:, :hidden]) * d13[:, hidden:]
//   x     <- x + hg16 * w2      residual as the C preload, in place
// Then RMSNorm(x) -> a16 once more; the classifier for the last real row runs on the host
// from that fp16 activation and is reported separately (see the plan, D7).
//
// Weights are converted to fp16 once at load (RNE) and stay resident. Every non-GEMM op
// is a pass in llama_passes.vxbin that all modes run identically. The host reference
// repeats the whole computation with the same fp16 rounding points and plain fp32
// accumulation, so only summation order separates it from the device.

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
#include <map>
#include <sstream>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#include "common.h"
#include "host_modes.h"

#include "llama_hostref.h"

namespace {

enum Site { S_QKV, S_QK, S_PV, S_WO, S_W13, S_W2, S_NSITES };
const char* kSiteName[S_NSITES] = { "qkv", "qk", "pv", "wo", "w13", "w2" };
enum Pass { P_RMSNORM, P_ROPE, P_SOFTMAX, P_GATHER, P_SILU, P_NPASSES };
const char* kPassName[P_NPASSES]  = { "pass_rmsnorm", "pass_rope", "pass_softmax", "pass_gather", "pass_silu" };
const char* kPassEntry[P_NPASSES] = { "llama_rmsnorm_f2h", "llama_rope_headpack", "llama_softmax_causal",
                                      "llama_attn_gather", "llama_silu_gate" };

}  // namespace

int run_llama(const RunOptions& opt) {
  LlamaModel model;
  if (!model.load(opt.model.c_str())) return -1;
  LlamaTokenizer tok;
  if (!tok.load(opt.tokenizer.c_str(), model.cfg.vocab_size)) return -1;
  const LlamaConfig& cfg = model.cfg;
  const uint32_t dim = cfg.dim, hidden = cfg.hidden_dim, H = cfg.n_heads, hs = dim / H, hp = 64;
  const uint32_t T = opt.T;
  if (cfg.n_kv_heads != cfg.n_heads) { std::cerr << "llama_dtcu: n_kv_heads != n_heads not supported" << std::endl; return -1; }
  if (hs > hp) { std::cerr << "llama_dtcu: head size " << hs << " > padded head " << hp << std::endl; return -1; }
  if (T > (uint32_t)cfg.seq_len) { std::cerr << "llama_dtcu: T exceeds seq_len" << std::endl; return -1; }

  // ---- tokens: BOS + prompt, padded to T rows (padding rows are never attended by real ones) ----
  std::vector<int> tokens;
  tok.encode(opt.prompt.c_str(), true, false, tokens);
  const uint32_t n_real = (uint32_t)std::min<size_t>(tokens.size(), T);
  if (tokens.size() > T) std::cerr << "llama_dtcu: prompt has " << tokens.size() << " tokens, truncated to T=" << T << std::endl;
  tokens.resize(T, 0);

  // ---- mode per site ----
  int site_mode[S_NSITES];
  for (int s = 0; s < S_NSITES; ++s) site_mode[s] = opt.mode;
  if (!opt.modemap.empty()) {
    std::ifstream f(opt.modemap);
    if (!f) { std::cerr << "llama_dtcu: cannot open modemap " << opt.modemap << std::endl; return -1; }
    std::string line;
    while (std::getline(f, line)) {
      std::istringstream is(line); std::string site; int m;
      if (!(is >> site >> m)) continue;
      bool hit = false;
      for (int s = 0; s < S_NSITES; ++s) if (site == kSiteName[s]) { site_mode[s] = m; hit = true; }
      if (site == "attn") { site_mode[S_QK] = m; site_mode[S_PV] = m; hit = true; }
      if (!hit) { std::cerr << "llama_dtcu: unknown site in modemap: " << site << std::endl; return -1; }
    }
  }
  const uint32_t site_M[S_NSITES] = { T, T, T, T, T, T };
  const uint32_t site_N[S_NSITES] = { 3 * dim, T, hp, dim, 2 * hidden, dim };
  const uint32_t site_K[S_NSITES] = { dim, hp, T, dim, dim, hidden };

  LlamaDevice dev;
  RT_CHECK(dev.open());
  for (int s = 0; s < S_NSITES; ++s) {
    std::string why = dev.mode_illegal_reason(site_mode[s], site_M[s], site_N[s], site_K[s]);
    if (!why.empty()) {
      if (opt.fallback >= 0 && dev.mode_illegal_reason(opt.fallback, site_M[s], site_N[s], site_K[s]).empty()) {
        std::cerr << "llama_dtcu: site " << kSiteName[s] << ": " << why << " -> fallback mode " << opt.fallback << std::endl;
        site_mode[s] = opt.fallback;
      } else {
        std::cerr << "llama_dtcu: site " << kSiteName[s] << " (" << site_M[s] << "x" << site_N[s] << "x" << site_K[s]
                  << "): " << why << std::endl;
        return -1;
      }
    }
  }
  std::string run_tag;
  {
    bool uniform = true;
    for (int s = 1; s < S_NSITES; ++s) if (site_mode[s] != site_mode[0]) uniform = false;
    std::ostringstream os;
    if (uniform) os << "uniform:" << site_mode[0];
    else { os << "map"; for (int s = 0; s < S_NSITES; ++s) os << ":" << kSiteName[s] << "=" << site_mode[s]; }
    run_tag = os.str();
  }
  {
    std::vector<std::string> entries;
    for (int p = 0; p < P_NPASSES; ++p) entries.push_back(kPassEntry[p]);
    RT_CHECK(dev.register_passes("llama_passes.vxbin", entries));
  }

  // ---- weights -> fp16, resident ----
  HostRef ref(model, T, hp);
  ref.wqkv16.resize(cfg.n_layers); ref.wo16.resize(cfg.n_layers); ref.w13_16.resize(cfg.n_layers); ref.w2_16.resize(cfg.n_layers);
  std::vector<DevBuf> d_wqkv(cfg.n_layers), d_wo(cfg.n_layers), d_w13(cfg.n_layers), d_w2(cfg.n_layers), d_rms_att(cfg.n_layers), d_rms_ffn(cfg.n_layers);
  for (int l = 0; l < cfg.n_layers; ++l) {
    std::vector<float> tmp((size_t)3 * dim * dim);
    std::memcpy(&tmp[0],                       model.w.wq + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    std::memcpy(&tmp[(size_t)dim * dim],       model.w.wk + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    std::memcpy(&tmp[(size_t)2 * dim * dim],   model.w.wv + (size_t)l * dim * dim, (size_t)dim * dim * 4);
    to_fp16(tmp.data(), tmp.size(), ref.wqkv16[l]);
    to_fp16(model.w.wo + (size_t)l * dim * dim, (size_t)dim * dim, ref.wo16[l]);
    tmp.resize((size_t)2 * hidden * dim);
    std::memcpy(&tmp[0],                    model.w.w1 + (size_t)l * dim * hidden, (size_t)dim * hidden * 4);
    std::memcpy(&tmp[(size_t)hidden * dim], model.w.w3 + (size_t)l * dim * hidden, (size_t)dim * hidden * 4);
    to_fp16(tmp.data(), tmp.size(), ref.w13_16[l]);
    to_fp16(model.w.w2 + (size_t)l * dim * hidden, (size_t)dim * hidden, ref.w2_16[l]);
    RT_CHECK(make_buf(dev, d_wqkv[l], ref.wqkv16[l].size() * 2, VX_MEM_READ));  RT_CHECK(dev.upload(d_wqkv[l].buf, 0, ref.wqkv16[l].data(), d_wqkv[l].bytes));
    RT_CHECK(make_buf(dev, d_wo[l],   ref.wo16[l].size() * 2,   VX_MEM_READ));  RT_CHECK(dev.upload(d_wo[l].buf,   0, ref.wo16[l].data(),   d_wo[l].bytes));
    RT_CHECK(make_buf(dev, d_w13[l],  ref.w13_16[l].size() * 2, VX_MEM_READ));  RT_CHECK(dev.upload(d_w13[l].buf,  0, ref.w13_16[l].data(), d_w13[l].bytes));
    RT_CHECK(make_buf(dev, d_w2[l],   ref.w2_16[l].size() * 2,  VX_MEM_READ));  RT_CHECK(dev.upload(d_w2[l].buf,   0, ref.w2_16[l].data(),  d_w2[l].bytes));
    RT_CHECK(make_buf(dev, d_rms_att[l], (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_att[l].buf, 0, model.w.rms_att_weight + (size_t)l * dim, dim * 4));
    RT_CHECK(make_buf(dev, d_rms_ffn[l], (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_ffn[l].buf, 0, model.w.rms_ffn_weight + (size_t)l * dim, dim * 4));
  }
  DevBuf d_rms_final; RT_CHECK(make_buf(dev, d_rms_final, (size_t)dim * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rms_final.buf, 0, model.w.rms_final_weight, dim * 4));
  // RoPE table, llama2.c's formula, shared by device and reference.
  ref.rope.resize((size_t)T * (hs / 2) * 2);
  for (uint32_t pos = 0; pos < T; ++pos)
    for (uint32_t c2 = 0; c2 < hs / 2; ++c2) {
      const int head_dim = 2 * c2;
      const float freq = 1.0f / powf(10000.0f, head_dim / (float)hs);
      const float val = (float)pos * freq;
      ref.rope[((size_t)pos * (hs / 2) + c2) * 2 + 0] = cosf(val);
      ref.rope[((size_t)pos * (hs / 2) + c2) * 2 + 1] = sinf(val);
    }
  DevBuf d_rope; RT_CHECK(make_buf(dev, d_rope, ref.rope.size() * 4, VX_MEM_READ)); RT_CHECK(dev.upload(d_rope.buf, 0, ref.rope.data(), d_rope.bytes));

  // ---- activations ----
  DevBuf d_x, d_a16, d_qkv, d_Qp, d_Kp, d_VpT, d_S, d_P, d_Dh, d_att16, d_d13, d_hg16, d_zero;
  RT_CHECK(make_buf(dev, d_x,     (size_t)T * dim * 4,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_a16,   (size_t)T * dim * 2,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_qkv,   (size_t)T * 3 * dim * 4,    VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_Qp,    (size_t)H * T * hp * 2,     VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_Kp,    (size_t)H * T * hp * 2,     VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_VpT,   (size_t)H * hp * T * 2,     VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_S,     (size_t)H * T * T * 4,      VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_P,     (size_t)H * T * T * 2,      VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_Dh,    (size_t)H * T * hp * 4,     VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_att16, (size_t)T * dim * 2,        VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_d13,   (size_t)T * 2 * hidden * 4, VX_MEM_READ_WRITE));
  RT_CHECK(make_buf(dev, d_hg16,  (size_t)T * hidden * 2,     VX_MEM_READ_WRITE));
  const size_t zero_elems = std::max<size_t>((size_t)T * T, (size_t)T * 2 * hidden);
  RT_CHECK(make_buf(dev, d_zero,  zero_elems * 4,             VX_MEM_READ));
  {
    std::vector<float> zeros(zero_elems, 0.0f);
    RT_CHECK(dev.upload(d_zero.buf, 0, zeros.data(), d_zero.bytes));
    std::vector<float> x0((size_t)T * dim);
    for (uint32_t r = 0; r < T; ++r) std::memcpy(&x0[(size_t)r * dim], model.w.token_embedding_table + (size_t)tokens[r] * dim, dim * 4);
    RT_CHECK(dev.upload(d_x.buf, 0, x0.data(), d_x.bytes));
  }

  // ---- host references (before the device run, so a device hang still leaves them) ----
  std::vector<uint16_t> ref_final16;
  if (opt.verify) ref.forward(tokens, ref_final16);
  std::vector<float> fp32_final;
  forward_fp32(model, tokens, T, fp32_final);

  // ---- the device forward ----
  LaunchStats site_st[S_NSITES], pass_st[P_NPASSES];
  llama_pass_arg_t pa{};
  pa.T = T; pa.dim = dim; pa.hidden = hidden; pa.n_heads = H; pa.head = hs; pa.head_pad = hp;
  pa.pos0 = 0; pa.eps = 1e-5f; pa.scale = 1.0f / sqrtf((float)hs);
  const uint32_t red_lmem = NUM_THREADS * sizeof(float);
  // The RoPE pass zeroes the head padding with the lanes past the last (q,k) pair, so the
  // grid must have at least head_pad - head of them.
  if (((dim / 2 + NUM_THREADS - 1) / NUM_THREADS) * NUM_THREADS - dim / 2 < hp - hs) {
    std::cerr << "llama_dtcu: RoPE grid has too few spare lanes for the head padding" << std::endl;
    return -1;
  }
  int verify_failures = 0;
  const double tol = 4e-3;   // relative to the site's max |ref|: fp32 summation-order noise only (2.2e-3 seen)

  auto check = [&](const char* what, int l, const std::vector<float>& got, const std::vector<float>& want) {
    if (!opt.verify) return;
    Compare c; const double sc = max_abs(want.data(), want.size());
    for (size_t i = 0; i < got.size(); ++i) c.add(got[i], want[i], sc, tol);
    std::printf("[LLAMA-VERIFY] layer=%d site=%s n=%zu max_rel=%.3e n_bad=%zu%s\n", l, what, c.n, c.max_rel, c.n_bad, c.n_bad ? "  <-- FAIL" : "");
    if (c.n_bad) ++verify_failures;
  };
  auto check16 = [&](const char* what, int l, const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
    if (!opt.verify) return;
    Compare c; const double sc = max_abs16(want.data(), want.size());
    for (size_t i = 0; i < got.size(); ++i) c.add(llama_h2f(got[i]), llama_h2f(want[i]), sc, tol);
    std::printf("[LLAMA-VERIFY] layer=%d site=%s n=%zu max_rel=%.3e n_bad=%zu%s\n", l, what, c.n, c.max_rel, c.n_bad, c.n_bad ? "  <-- FAIL" : "");
    if (c.n_bad) ++verify_failures;
  };
  // Reference layer-by-layer replay for comparison: recompute the reference per layer in
  // lockstep with the device so intermediate buffers are available.
  HostRef rep(model, T, hp);
  rep.wqkv16 = ref.wqkv16; rep.wo16 = ref.wo16; rep.w13_16 = ref.w13_16; rep.w2_16 = ref.w2_16; rep.rope = ref.rope;
  if (opt.verify) {
    rep.x.assign((size_t)T * dim, 0.0f);
    for (uint32_t r = 0; r < T; ++r) std::memcpy(&rep.x[(size_t)r * dim], model.w.token_embedding_table + (size_t)tokens[r] * dim, dim * 4);
    rep.a16.resize((size_t)T * dim); rep.qkv.resize((size_t)T * 3 * dim);
    rep.Qp.resize((size_t)H * T * hp); rep.Kp.resize((size_t)H * T * hp); rep.VpT.resize((size_t)H * hp * T);
    rep.S.resize((size_t)H * T * T); rep.P.resize((size_t)H * T * T); rep.Dh.resize((size_t)H * T * hp);
    rep.att16.resize((size_t)T * dim); rep.d13.resize((size_t)T * 2 * hidden); rep.hg16.resize((size_t)T * hidden);
  }
  std::vector<float> got_f; std::vector<uint16_t> got_h;

  auto t_start = std::chrono::high_resolution_clock::now();
  for (int l = 0; l < cfg.n_layers; ++l) {
    if (opt.verify) rep.layer(l);
    // RMSNorm(att) -> a16
    pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_att[l].addr;
    RT_CHECK(dev.launch_pass(P_RMSNORM, &pa, sizeof(pa), T, 1, red_lmem, &pass_st[P_RMSNORM]));
    // QKV
    RT_CHECK(dev.gemm(site_mode[S_QKV], T, 3 * dim, dim, d_a16.addr, d_wqkv[l].addr, d_zero.addr, d_qkv.addr, &site_st[S_QKV]));
    if (opt.verify) { got_f.resize((size_t)T * 3 * dim); RT_CHECK(dev.download(got_f.data(), d_qkv.buf, 0, d_qkv.bytes)); check("qkv", l, got_f, rep.qkv); }
    // RoPE + head pack
    pa.src = d_qkv.addr; pa.dst = d_Qp.addr; pa.aux0 = d_Kp.addr; pa.aux1 = d_VpT.addr; pa.aux2 = d_rope.addr;
    RT_CHECK(dev.launch_pass(P_ROPE, &pa, sizeof(pa), (dim / 2 + NUM_THREADS - 1) / NUM_THREADS, T, 0, &pass_st[P_ROPE]));
    // QK^T per head
    for (uint32_t h = 0; h < H; ++h)
      RT_CHECK(dev.gemm(site_mode[S_QK], T, T, hp, d_Qp.addr + (uint64_t)h * T * hp * 2, d_Kp.addr + (uint64_t)h * T * hp * 2,
                        d_zero.addr, d_S.addr + (uint64_t)h * T * T * 4, &site_st[S_QK]));
    if (opt.verify) { got_f.resize((size_t)H * T * T); RT_CHECK(dev.download(got_f.data(), d_S.buf, 0, d_S.bytes)); check("qk", l, got_f, rep.S); }
    // causal softmax -> P
    pa.src = d_S.addr; pa.dst = d_P.addr;
    RT_CHECK(dev.launch_pass(P_SOFTMAX, &pa, sizeof(pa), H * T, 1, red_lmem, &pass_st[P_SOFTMAX]));
    // P V per head
    for (uint32_t h = 0; h < H; ++h)
      RT_CHECK(dev.gemm(site_mode[S_PV], T, hp, T, d_P.addr + (uint64_t)h * T * T * 2, d_VpT.addr + (uint64_t)h * hp * T * 2,
                        d_zero.addr, d_Dh.addr + (uint64_t)h * T * hp * 4, &site_st[S_PV]));
    if (opt.verify) { got_f.resize((size_t)H * T * hp); RT_CHECK(dev.download(got_f.data(), d_Dh.buf, 0, d_Dh.bytes)); check("pv", l, got_f, rep.Dh); }
    // gather heads -> att16
    pa.src = d_Dh.addr; pa.dst = d_att16.addr;
    RT_CHECK(dev.launch_pass(P_GATHER, &pa, sizeof(pa), (dim + NUM_THREADS - 1) / NUM_THREADS, T, 0, &pass_st[P_GATHER]));
    // wo, residual as C preload, in place
    RT_CHECK(dev.gemm(site_mode[S_WO], T, dim, dim, d_att16.addr, d_wo[l].addr, d_x.addr, d_x.addr, &site_st[S_WO]));
    if (opt.verify) {
      // rep.x now holds x after the whole layer; compare after w2 instead. Here compare a16 of the FFN norm below.
    }
    // RMSNorm(ffn) -> a16
    pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_ffn[l].addr;
    RT_CHECK(dev.launch_pass(P_RMSNORM, &pa, sizeof(pa), T, 1, red_lmem, &pass_st[P_RMSNORM]));
    // w1|w3
    RT_CHECK(dev.gemm(site_mode[S_W13], T, 2 * hidden, dim, d_a16.addr, d_w13[l].addr, d_zero.addr, d_d13.addr, &site_st[S_W13]));
    if (opt.verify) { got_f.resize((size_t)T * 2 * hidden); RT_CHECK(dev.download(got_f.data(), d_d13.buf, 0, d_d13.bytes)); check("w13", l, got_f, rep.d13); }
    // SiLU gate -> hg16
    pa.src = d_d13.addr; pa.dst = d_hg16.addr;
    RT_CHECK(dev.launch_pass(P_SILU, &pa, sizeof(pa), (hidden + NUM_THREADS - 1) / NUM_THREADS, T, 0, &pass_st[P_SILU]));
    if (opt.verify) { got_h.resize((size_t)T * hidden); RT_CHECK(dev.download(got_h.data(), d_hg16.buf, 0, d_hg16.bytes)); check16("silu", l, got_h, rep.hg16); }
    // w2, residual as C preload, in place
    RT_CHECK(dev.gemm(site_mode[S_W2], T, dim, hidden, d_hg16.addr, d_w2[l].addr, d_x.addr, d_x.addr, &site_st[S_W2]));
    if (opt.verify) { got_f.resize((size_t)T * dim); RT_CHECK(dev.download(got_f.data(), d_x.buf, 0, d_x.bytes)); check("x", l, got_f, rep.x); }
  }
  // final RMSNorm -> a16
  pa.src = d_x.addr; pa.dst = d_a16.addr; pa.aux0 = d_rms_final.addr;
  RT_CHECK(dev.launch_pass(P_RMSNORM, &pa, sizeof(pa), T, 1, red_lmem, &pass_st[P_RMSNORM]));
  auto t_end = std::chrono::high_resolution_clock::now();
  std::vector<uint16_t> dev_final16((size_t)T * dim);
  RT_CHECK(dev.download(dev_final16.data(), d_a16.buf, 0, d_a16.bytes));
  if (opt.verify) check16("final", cfg.n_layers, dev_final16, ref_final16);

  // ---- classifier for the last real row, on the host (see D7), three ways ----
  auto argmax_logits = [&](const float* act, int* best, float* margin) {
    std::vector<float> logits(cfg.vocab_size);
    for (int i = 0; i < cfg.vocab_size; ++i) {
      const float* w = model.w.wcls + (size_t)i * dim; float s = 0;
      for (uint32_t j = 0; j < dim; ++j) s += w[j] * act[j];
      logits[i] = s;
    }
    int b = 0; for (int i = 1; i < cfg.vocab_size; ++i) if (logits[i] > logits[b]) b = i;
    float second = -3.4e38f; for (int i = 0; i < cfg.vocab_size; ++i) if (i != b && logits[i] > second) second = logits[i];
    *best = b; *margin = logits[b] - second;
  };
  const uint32_t last = n_real - 1;
  std::vector<float> act(dim);
  int next_dev = 0, next_ref = 0, next_fp32 = 0; float m_dev = 0, m_ref = 0, m_fp32 = 0;
  for (uint32_t j = 0; j < dim; ++j) act[j] = llama_h2f(dev_final16[(size_t)last * dim + j]);
  argmax_logits(act.data(), &next_dev, &m_dev);
  if (opt.verify) {
    for (uint32_t j = 0; j < dim; ++j) act[j] = llama_h2f(ref_final16[(size_t)last * dim + j]);
    argmax_logits(act.data(), &next_ref, &m_ref);
  }
  argmax_logits(&fp32_final[(size_t)last * dim], &next_fp32, &m_fp32);

  // ---- report ----
  uint64_t total_cycles = 0, gemm_cycles = 0, pass_cycles = 0; uint32_t total_launches = 0;
  for (int s = 0; s < S_NSITES; ++s) {
    std::printf("[LLAMA] run=%s T=%u real=%u site=%s mode=%d launches=%u cycles=%llu\n", run_tag.c_str(), T, n_real,
                kSiteName[s], site_mode[s], site_st[s].launches, (unsigned long long)site_st[s].cycles);
    total_cycles += site_st[s].cycles; gemm_cycles += site_st[s].cycles; total_launches += site_st[s].launches;
  }
  for (int p = 0; p < P_NPASSES; ++p) {
    std::printf("[LLAMA] run=%s T=%u real=%u site=%s mode=simt launches=%u cycles=%llu\n", run_tag.c_str(), T, n_real,
                kPassName[p], pass_st[p].launches, (unsigned long long)pass_st[p].cycles);
    total_cycles += pass_st[p].cycles; pass_cycles += pass_st[p].cycles; total_launches += pass_st[p].launches;
  }
  const double cpt = (double)total_cycles / (double)T;
  const double wall_s = std::chrono::duration<double>(t_end - t_start).count();
  std::printf("[LLAMA] run=%s T=%u real=%u total_launches=%u total_cycles=%llu gemm_cycles=%llu pass_cycles=%llu "
              "cycles_per_token=%.0f tok_per_s_400MHz=%.2f verify_fail_sites=%d next_dev=%d next_ref=%d next_fp32=%d "
              "program_swaps=%u wall_s=%.1f\n",
              run_tag.c_str(), T, n_real, total_launches, (unsigned long long)total_cycles,
              (unsigned long long)gemm_cycles, (unsigned long long)pass_cycles, cpt, 400e6 / cpt,
              verify_failures, next_dev, opt.verify ? next_ref : -1, next_fp32, dev.program_swaps(), wall_s);
  if (opt.text) {
    std::printf("prompt tokens (%u): ", n_real);
    for (uint32_t i = 0; i < n_real; ++i) std::printf("%s", tok.decode(i ? tokens[i - 1] : 1, tokens[i]));
    std::printf("\nnext token  device: '%s' (margin %.3f)   fp32 host: '%s' (margin %.3f)%s\n",
                tok.decode(tokens[last], next_dev), m_dev, tok.decode(tokens[last], next_fp32), m_fp32,
                opt.verify ? (next_ref == next_dev ? "   [fp16 reference agrees with device]" : "   [fp16 reference DIFFERS from device]") : "");
  }
  return verify_failures;
}
