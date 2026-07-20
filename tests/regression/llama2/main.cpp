// Llama-2 end-to-end decode on Vortex -- batched, WGMMA matmuls.
//
// LLAMA_BATCH sequences are decoded in lockstep. Activations live as
// [dim][BATCH], which turns every projection into a GEMM the tensor cores can
// eat directly (see kernel.cpp). Weights are converted to fp16 once at load
// and pinned on device; the KV cache is per-sequence and also resident.
//
// Correctness check: with greedy sampling and all sequences seeded from BOS,
// every sequence must produce the *same* text, and that text must match
//     ./run <checkpoint> -t 0 -n <steps>
// So one run tests both the batching (all lanes agree) and the math (matches
// the reference). Any lane disagreeing points at a batch-indexing bug; all
// lanes agreeing but diverging from the reference points at precision.
//
// Usage: llama2 <checkpoint> [-z tokenizer.bin] [-n steps] [-k kernel.vxbin]
//               [--scalar]   use the fp32 scalar matmul instead of WGMMA

#include <vortex2.h>
#include <rvfloats.h>
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#define CHECK(x)                                                             \
  do {                                                                       \
    int _e = (x);                                                            \
    if (_e != VX_SUCCESS) {                                                  \
      printf("ERROR: %s returned %d (%s:%d)\n", #x, _e, __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

#define FAIL(msg) do { printf("ERROR: %s\n", (msg)); exit(1); } while (0)

static inline uint16_t f32_to_f16(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(bits));
  uint32_t fflags = 0;
  return rv_ftoh_s(bits, 0 /*RNE*/, &fflags);
}

// ---------------------------------------------------------------------------

struct WeightOffsets {
  size_t token_embedding, rms_att, wq, wk, wv, wo;
  size_t rms_ffn, w1, w2, w3, rms_final, wcls, total;
};

static WeightOffsets compute_offsets(const llama_config_t& c, int shared) {
  const size_t hs = c.dim / c.n_heads, L = c.n_layers;
  WeightOffsets o{}; size_t p = 0;
  o.token_embedding = p; p += (size_t)c.vocab_size * c.dim;
  o.rms_att         = p; p += L * c.dim;
  o.wq              = p; p += L * (size_t)c.dim * c.n_heads * hs;
  o.wk              = p; p += L * (size_t)c.dim * c.n_kv_heads * hs;
  o.wv              = p; p += L * (size_t)c.dim * c.n_kv_heads * hs;
  o.wo              = p; p += L * (size_t)c.n_heads * hs * c.dim;
  o.rms_ffn         = p; p += L * c.dim;
  o.w1              = p; p += L * (size_t)c.hidden_dim * c.dim;
  o.w2              = p; p += L * (size_t)c.dim * c.hidden_dim;
  o.w3              = p; p += L * (size_t)c.hidden_dim * c.dim;
  o.rms_final       = p; p += c.dim;
  p += (size_t)c.seq_len * hs / 2;   // unused freq_cis_real
  p += (size_t)c.seq_len * hs / 2;   // unused freq_cis_imag
  o.wcls = shared ? o.token_embedding : p;
  if (!shared) p += (size_t)c.vocab_size * c.dim;
  o.total = p;
  return o;
}

struct Tokenizer {
  std::vector<std::string> vocab;
  bool load(const char* path, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int mtl = 0;
    if (fread(&mtl, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      float score; int len;
      if (fread(&score, sizeof(float), 1, f) != 1) { fclose(f); return false; }
      if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
      std::string s(len, '\0');
      if (len && fread(&s[0], 1, len, f) != (size_t)len) { fclose(f); return false; }
      vocab[i] = std::move(s);
    }
    fclose(f);
    return true;
  }
  std::string decode(int prev, int token) const {
    if (token < 0 || token >= (int)vocab.size()) return "";
    const char* p = vocab[token].c_str();
    if (prev == 1 && p[0] == ' ') ++p;
    unsigned char bv;
    if (sscanf(p, "<0x%02hhX>", &bv) == 1) return std::string(1, (char)bv);
    return std::string(p);
  }
};

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  const char* checkpoint = nullptr;
  const char* tokenizer_path = "tokenizer.bin";
  const char* kernel_file = "kernel.vxbin";
  int steps = 32;
  bool use_scalar = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--scalar")) { use_scalar = true; continue; }
    if (argv[i][0] != '-') { checkpoint = argv[i]; continue; }
    if (i + 1 >= argc) FAIL("missing value for option");
    switch (argv[i][1]) {
      case 'z': tokenizer_path = argv[++i]; break;
      case 'n': steps = atoi(argv[++i]); break;
      case 'k': kernel_file = argv[++i]; break;
      default: FAIL("unknown option");
    }
  }
  if (!checkpoint) {
    printf("usage: %s <checkpoint.bin> [-z tok.bin] [-n steps] [-k kernel.vxbin] [--scalar]\n", argv[0]);
    return 1;
  }

  FILE* f = fopen(checkpoint, "rb");
  if (!f) FAIL("cannot open checkpoint");
  llama_config_t cfg{};
  if (fread(&cfg, sizeof(int32_t), 7, f) != 7) FAIL("bad checkpoint header");
  int shared = cfg.vocab_size > 0 ? 1 : 0;
  cfg.vocab_size = abs(cfg.vocab_size);

  const int head_size = cfg.dim / cfg.n_heads;
  const int kv_dim    = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
  const int kv_mul    = cfg.n_heads / cfg.n_kv_heads;
  const int B         = LLAMA_BATCH;

  printf("model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d\n",
         cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
         cfg.vocab_size, cfg.seq_len);
  printf("batch=%d  matmul=%s\n", B, use_scalar ? "scalar-fp32" : "WGMMA-fp16");

  WeightOffsets off = compute_offsets(cfg, shared);
  std::vector<float> weights(off.total);
  if (fread(weights.data(), sizeof(float), off.total, f) != off.total)
    FAIL("checkpoint shorter than header implies");
  fclose(f);

  Tokenizer tok;
  if (!tok.load(tokenizer_path, cfg.vocab_size)) FAIL("cannot load tokenizer");
  if (steps > cfg.seq_len) steps = cfg.seq_len;

  // Narrow the weights once, on the host. WGMMA takes fp16 operands with fp32
  // accumulation; the elementwise ops keep their fp32 copy.
  std::vector<uint16_t> weights_h(off.total);
  for (size_t i = 0; i < off.total; ++i) weights_h[i] = f32_to_f16(weights[i]);
  printf("weights: %.1f MB fp32 -> %.1f MB fp16\n",
         off.total * 4.0 / (1 << 20), off.total * 2.0 / (1 << 20));

  // ---- device ----
  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));
  vx_queue_info_t qi = {}; qi.struct_size = sizeof(qi);
  qi.priority = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  auto mk = [&](uint64_t bytes, uint32_t flags) {
    vx_buffer_h b = nullptr;
    CHECK(vx_buffer_create(dev, bytes, flags, &b));
    return b;
  };
  auto addr_of = [&](vx_buffer_h b) {
    uint64_t a = 0; CHECK(vx_buffer_address(b, &a)); return a;
  };

  const size_t Fdim = (size_t)cfg.dim * B;
  const size_t Fhid = (size_t)cfg.hidden_dim * B;
  const size_t Fkv  = (size_t)kv_dim * B;
  const size_t kv_seq_stride = (size_t)cfg.n_layers * cfg.seq_len * kv_dim;
  const size_t kv_total = kv_seq_stride * B;
  const size_t stage_elems = (Fdim > Fhid ? Fdim : Fhid);

  vx_buffer_h w32_buf = mk(off.total * 4, VX_MEM_READ);
  vx_buffer_h w16_buf = mk(off.total * 2, VX_MEM_READ);
  vx_buffer_h x_buf   = mk(Fdim * 4, VX_MEM_READ_WRITE);
  vx_buffer_h xb_buf  = mk(Fdim * 4, VX_MEM_READ_WRITE);
  vx_buffer_h xb2_buf = mk(Fdim * 4, VX_MEM_READ_WRITE);
  vx_buffer_h qq_buf  = mk(Fdim * 4, VX_MEM_READ_WRITE);
  vx_buffer_h kt_buf  = mk(Fkv * 4,  VX_MEM_READ_WRITE);
  vx_buffer_h vt_buf  = mk(Fkv * 4,  VX_MEM_READ_WRITE);
  vx_buffer_h hb_buf  = mk(Fhid * 4, VX_MEM_READ_WRITE);
  vx_buffer_h hb2_buf = mk(Fhid * 4, VX_MEM_READ_WRITE);
  vx_buffer_h stg_buf = mk(stage_elems * 2, VX_MEM_READ_WRITE);   // fp16 staging
  vx_buffer_h att_buf = mk((size_t)B * cfg.n_heads * cfg.seq_len * 4, VX_MEM_READ_WRITE);
  vx_buffer_h log_buf = mk((size_t)cfg.vocab_size * B * 4, VX_MEM_READ_WRITE);
  vx_buffer_h kc_buf  = mk(kv_total * 4, VX_MEM_READ_WRITE);
  vx_buffer_h vc_buf  = mk(kv_total * 4, VX_MEM_READ_WRITE);
  vx_buffer_h tok_buf = mk((size_t)B * 4, VX_MEM_READ_WRITE);
  vx_buffer_h pos_buf = mk((size_t)B * 4, VX_MEM_READ_WRITE);
  vx_buffer_h ss_buf  = mk((size_t)B * 4, VX_MEM_READ_WRITE);

  printf("kv cache: %.1f MB x2 (%d sequences)\n", kv_total * 4.0 / (1 << 20), B);

  CHECK(vx_enqueue_write(q, w32_buf, 0, weights.data(),  off.total * 4, 0, nullptr, nullptr));
  CHECK(vx_enqueue_write(q, w16_buf, 0, weights_h.data(), off.total * 2, 0, nullptr, nullptr));

  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h k_mm = nullptr, k_embed = nullptr, k_rsum = nullptr, k_rscale = nullptr;
  vx_kernel_h k_f16 = nullptr, k_rope = nullptr, k_attn = nullptr;
  vx_kernel_h k_swiglu = nullptr, k_resid = nullptr, k_kv = nullptr;
  CHECK(vx_module_get_kernel(mod, use_scalar ? "matmul_scalar_k" : "matmul_wgmma_k", &k_mm));
  CHECK(vx_module_get_kernel(mod, "embed_k",         &k_embed));
  CHECK(vx_module_get_kernel(mod, "rmsnorm_sum_k",   &k_rsum));
  CHECK(vx_module_get_kernel(mod, "rmsnorm_scale_k", &k_rscale));
  CHECK(vx_module_get_kernel(mod, "to_fp16_k",       &k_f16));
  CHECK(vx_module_get_kernel(mod, "rope_k",          &k_rope));
  CHECK(vx_module_get_kernel(mod, "attn_k",          &k_attn));
  CHECK(vx_module_get_kernel(mod, "swiglu_k",        &k_swiglu));
  CHECK(vx_module_get_kernel(mod, "residual_k",      &k_resid));
  CHECK(vx_module_get_kernel(mod, "kv_store_k",      &k_kv));

  const uint64_t W32 = addr_of(w32_buf), W16 = addr_of(w16_buf);
  const uint64_t A_x = addr_of(x_buf),  A_xb = addr_of(xb_buf), A_xb2 = addr_of(xb2_buf);
  const uint64_t A_q = addr_of(qq_buf), A_kt = addr_of(kt_buf), A_vt = addr_of(vt_buf);
  const uint64_t A_hb = addr_of(hb_buf), A_hb2 = addr_of(hb2_buf);
  const uint64_t A_stg = addr_of(stg_buf), A_att = addr_of(att_buf);
  const uint64_t A_log = addr_of(log_buf);
  const uint64_t A_kc = addr_of(kc_buf), A_vc = addr_of(vc_buf);
  const uint64_t A_tok = addr_of(tok_buf), A_pos = addr_of(pos_buf), A_ss = addr_of(ss_buf);

  auto w32p = [&](size_t o_) { return W32 + o_ * 4; };
  auto w16p = [&](size_t o_) { return W16 + o_ * 2; };

  // 1-D launch, sized to n work items.
  auto launch1 = [&](vx_kernel_h k, kernel_arg_t& a, uint32_t n) {
    uint32_t grid[1] = {1}, block[1] = {1};
    const uint32_t gd[1] = { n };
    CHECK(vx_device_max_occupancy_grid(dev, 1, gd, grid, block));
    vx_launch_info_t li = {}; li.struct_size = sizeof(li);
    li.kernel = k; li.args_host = &a; li.args_size = sizeof(a);
    li.ndim = 1; li.grid_dim[0] = grid[0]; li.block_dim[0] = block[0];
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, nullptr));
    CHECK(vx_queue_finish(q, UINT64_MAX));
  };

  // 2-D launch for the WGMMA GEMM: one CTA per (N-tile, M-tile). Block is a
  // full warpgroup so the cooperative smem staging in the kernel is valid.
  // Mirror wgmma_context's geometry rather than hardcoding it. In VX_tcu_pkg:
  //   LG = log2(NUM_THREADS), EN = LG/2, EM = LG-EN, tcM = 1<<EM
  //   xtileM = m_steps * tcM, and WG_TILE_M = 2*tcM so m_steps = 2
  // giving xtileM = 2*tcM: 16 at NT=32, 8 at NT=16. A literal 16 is right only
  // for NT=32 and silently mis-sizes the grid on any other core.
  constexpr uint32_t LG_NT = (VX_CFG_NUM_THREADS >= 64) ? 6 :
                             (VX_CFG_NUM_THREADS >= 32) ? 5 :
                             (VX_CFG_NUM_THREADS >= 16) ? 4 :
                             (VX_CFG_NUM_THREADS >= 8)  ? 3 : 2;
  constexpr uint32_t TC_M    = 1u << (LG_NT - LG_NT / 2);
  const uint32_t WARPS   = 8;
  const uint32_t XTILE_M = 2 * TC_M;
  const uint32_t XTILE_N = 2 * WGMMA_NRC;
  auto launch_gemm = [&](kernel_arg_t& a) {
    uint32_t cta_m = WARPS * XTILE_M;
    vx_launch_info_t li = {}; li.struct_size = sizeof(li);
    li.kernel = k_mm; li.args_host = &a; li.args_size = sizeof(a);
    li.ndim = 2;
    li.grid_dim[0]  = ((uint32_t)a.batch + XTILE_N - 1) / XTILE_N;
    li.grid_dim[1]  = ((uint32_t)a.d + cta_m - 1) / cta_m;
    li.block_dim[0] = WARPS * VX_CFG_NUM_THREADS;
    li.block_dim[1] = 1;
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, nullptr));
    CHECK(vx_queue_finish(q, UINT64_MAX));
  };

  // out[d][B] = W[d][n] . src[n][B].  Narrows src to fp16 into the staging
  // buffer first, then runs the GEMM (or the fp32 scalar path for --scalar).
  auto matmul = [&](kernel_arg_t& a, uint64_t out, uint64_t src,
                    size_t woff, int n, int d) {
    a.n = n; a.d = d; a.batch = B;
    if (use_scalar) {
      a.out_addr = out; a.x_addr = src; a.w_addr = w32p(woff);
      launch1(k_mm, a, (uint32_t)(d * B));
    } else {
      kernel_arg_t c = a;
      c.n = n; c.batch = B; c.x_addr = src; c.out_addr = A_stg;
      launch1(k_f16, c, (uint32_t)(n * B));
      a.out_addr = out; a.x_addr = A_stg; a.w_addr = w16p(woff);
      launch_gemm(a);
    }
  };

  // ---- generation ----
  std::vector<int32_t> h_tok(B, 1), h_pos(B, 0);   // all sequences start at BOS
  std::vector<float>   logits((size_t)cfg.vocab_size * B);
  std::vector<std::string> text(B);

  printf("\n--- generating %d tokens x %d sequences (greedy) ---\n", steps, B);

  for (int pos = 0; pos < steps; ++pos) {
    for (int b = 0; b < B; ++b) h_pos[b] = pos;
    CHECK(vx_enqueue_write(q, tok_buf, 0, h_tok.data(), B * 4, 0, nullptr, nullptr));
    CHECK(vx_enqueue_write(q, pos_buf, 0, h_pos.data(), B * 4, 0, nullptr, nullptr));

    kernel_arg_t a{};
    a.dim = cfg.dim; a.kv_dim = kv_dim; a.head_size = head_size;
    a.n_heads = cfg.n_heads; a.kv_mul = kv_mul; a.seq_len = cfg.seq_len;
    a.batch = B; a.kv_seq_stride = (int32_t)kv_seq_stride;
    a.tokens_addr = A_tok; a.pos_addr = A_pos; a.scratch_addr = A_ss;
    a.key_cache_addr = A_kc; a.value_cache_addr = A_vc; a.att_addr = A_att;

    a.out_addr = A_x; a.w_addr = w32p(off.token_embedding);
    launch1(k_embed, a, (uint32_t)(cfg.dim * B));

    for (int l = 0; l < cfg.n_layers; ++l) {
      a.loff = l * cfg.seq_len * kv_dim;

      // attention rmsnorm -> xb
      a.n = cfg.dim; a.x_addr = A_x;
      launch1(k_rsum, a, (uint32_t)B);
      a.out_addr = A_xb; a.w_addr = w32p(off.rms_att + (size_t)l * cfg.dim);
      launch1(k_rscale, a, (uint32_t)(cfg.dim * B));

      // Q, K, V projections
      matmul(a, A_q,  A_xb, off.wq + (size_t)l * cfg.dim * cfg.dim,   cfg.dim, cfg.dim);
      matmul(a, A_kt, A_xb, off.wk + (size_t)l * cfg.dim * kv_dim,    cfg.dim, kv_dim);
      matmul(a, A_vt, A_xb, off.wv + (size_t)l * cfg.dim * kv_dim,    cfg.dim, kv_dim);

      // RoPE on q and the new k, then scatter k/v into the per-sequence cache
      a.q_addr = A_q; a.k_addr = A_kt;
      launch1(k_rope, a, (uint32_t)((cfg.dim / 2) * B));
      a.x_addr = A_kt; a.key_cache_addr = A_kc;
      launch1(k_kv, a, (uint32_t)(kv_dim * B));
      a.x_addr = A_vt; a.key_cache_addr = A_vc;
      launch1(k_kv, a, (uint32_t)(kv_dim * B));
      a.key_cache_addr = A_kc;

      // attention -> xb ; xb2 = Wo xb ; x += xb2
      a.out_addr = A_xb;
      launch1(k_attn, a, (uint32_t)(cfg.n_heads * B));
      matmul(a, A_xb2, A_xb, off.wo + (size_t)l * cfg.dim * cfg.dim, cfg.dim, cfg.dim);
      a.n = cfg.dim; a.x_addr = A_x; a.out_addr = A_xb2;
      launch1(k_resid, a, (uint32_t)(cfg.dim * B));

      // ffn rmsnorm -> xb ; w1/w3 ; swiglu ; w2 ; residual
      a.n = cfg.dim; a.x_addr = A_x;
      launch1(k_rsum, a, (uint32_t)B);
      a.out_addr = A_xb; a.w_addr = w32p(off.rms_ffn + (size_t)l * cfg.dim);
      launch1(k_rscale, a, (uint32_t)(cfg.dim * B));

      matmul(a, A_hb,  A_xb, off.w1 + (size_t)l * cfg.hidden_dim * cfg.dim,
             cfg.dim, cfg.hidden_dim);
      matmul(a, A_hb2, A_xb, off.w3 + (size_t)l * cfg.hidden_dim * cfg.dim,
             cfg.dim, cfg.hidden_dim);

      a.n = cfg.hidden_dim; a.hb_addr = A_hb; a.hb2_addr = A_hb2;
      launch1(k_swiglu, a, (uint32_t)(cfg.hidden_dim * B));

      matmul(a, A_xb2, A_hb, off.w2 + (size_t)l * cfg.dim * cfg.hidden_dim,
             cfg.hidden_dim, cfg.dim);
      a.n = cfg.dim; a.x_addr = A_x; a.out_addr = A_xb2;
      launch1(k_resid, a, (uint32_t)(cfg.dim * B));
    }

    // final norm (into xb -- never in place, rmsnorm_scale reads x) + classifier
    a.n = cfg.dim; a.x_addr = A_x;
    launch1(k_rsum, a, (uint32_t)B);
    a.out_addr = A_xb; a.w_addr = w32p(off.rms_final);
    launch1(k_rscale, a, (uint32_t)(cfg.dim * B));
    matmul(a, A_log, A_xb, off.wcls, cfg.dim, cfg.vocab_size);

    CHECK(vx_enqueue_read(q, logits.data(), log_buf, 0,
                          (size_t)cfg.vocab_size * B * 4, 0, nullptr, nullptr));
    CHECK(vx_queue_finish(q, UINT64_MAX));

    // greedy argmax per sequence; logits are [vocab][B]
    for (int b = 0; b < B; ++b) {
      int best = 0; float bv = logits[(size_t)0 * B + b];
      for (int v = 1; v < cfg.vocab_size; ++v) {
        float s = logits[(size_t)v * B + b];
        if (s > bv) { bv = s; best = v; }
      }
      std::string piece = tok.decode(h_tok[b], best);
      text[b] += piece;
      if (b == 0) { printf("%s", piece.c_str()); fflush(stdout); }
      h_tok[b] = best;
    }
  }

  printf("\n\n--- sequence 0 ---\n%s\n", text[0].c_str());
  int mismatch = 0;
  for (int b = 1; b < B; ++b) if (text[b] != text[0]) ++mismatch;
  if (mismatch) {
    printf("\nFAILED: %d/%d sequences diverged from lane 0 (batch indexing bug)\n",
           mismatch, B - 1);
  } else {
    printf("\nall %d sequences identical (batch lanes agree)\n", B);
  }

  vx_module_release(mod);
  for (auto b : { w32_buf, w16_buf, x_buf, xb_buf, xb2_buf, qq_buf, kt_buf, vt_buf,
                  hb_buf, hb2_buf, stg_buf, att_buf, log_buf, kc_buf, vc_buf,
                  tok_buf, pos_buf, ss_buf })
    vx_buffer_release(b);
  vx_queue_release(q);
  vx_device_release(dev);
  return mismatch ? 1 : 0;
}
