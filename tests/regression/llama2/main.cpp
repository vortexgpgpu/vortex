// Llama-2 end-to-end decode on Vortex (Stage 1: fp32, correctness-first).
//
// Loads a llama2.c checkpoint, pins the weights and KV cache in device memory,
// and runs the decoder one token at a time. Every op is a kernel launch from a
// single multi-entry .vxbin; the only host round-trip per token is reading the
// logits back to sample.
//
// Greedy (argmax) sampling with an empty prompt makes a run bit-deterministic,
// so the output is directly comparable against the reference:
//     ./run stories15M.bin -t 0 -n <steps>
//
// Usage: llama2 <checkpoint> [-z tokenizer.bin] [-n steps] [-k kernel.vxbin]

#include <vortex2.h>
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

#define FAIL(msg)                        \
  do {                                   \
    printf("ERROR: %s\n", (msg));        \
    exit(1);                             \
  } while (0)

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

// Byte offsets (in floats) of each tensor inside the flat weight blob, in the
// order llama2.c writes them. Keeping one contiguous device buffer and indexing
// by offset mirrors the file exactly, so there is no per-tensor upload.
struct WeightOffsets {
  size_t token_embedding;
  size_t rms_att;
  size_t wq, wk, wv, wo;
  size_t rms_ffn;
  size_t w1, w2, w3;
  size_t rms_final;
  size_t wcls;
  size_t total;  // floats in the blob, excluding the header
};

static WeightOffsets compute_offsets(const llama_config_t& c, int shared_weights) {
  const size_t head_size = c.dim / c.n_heads;
  const size_t L = c.n_layers;
  WeightOffsets o{};
  size_t p = 0;
  o.token_embedding = p; p += (size_t)c.vocab_size * c.dim;
  o.rms_att         = p; p += L * c.dim;
  o.wq              = p; p += L * (size_t)c.dim * c.n_heads * head_size;
  o.wk              = p; p += L * (size_t)c.dim * c.n_kv_heads * head_size;
  o.wv              = p; p += L * (size_t)c.dim * c.n_kv_heads * head_size;
  o.wo              = p; p += L * (size_t)c.n_heads * head_size * c.dim;
  o.rms_ffn         = p; p += L * c.dim;
  o.w1              = p; p += L * (size_t)c.hidden_dim * c.dim;
  o.w2              = p; p += L * (size_t)c.dim * c.hidden_dim;
  o.w3              = p; p += L * (size_t)c.hidden_dim * c.dim;
  o.rms_final       = p; p += c.dim;
  // Two unused RoPE tables (freq_cis_real/imag) sit here in the file format;
  // we recompute the angles in rope_k instead.
  p += (size_t)c.seq_len * head_size / 2;
  p += (size_t)c.seq_len * head_size / 2;
  o.wcls = shared_weights ? o.token_embedding : p;
  if (!shared_weights) p += (size_t)c.vocab_size * c.dim;
  o.total = p;
  return o;
}

// ---------------------------------------------------------------------------
// Tokenizer (decode only -- enough for greedy generation from BOS)
// ---------------------------------------------------------------------------

struct Tokenizer {
  std::vector<std::string> vocab;

  bool load(const char* path, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int max_token_length = 0;
    if (fread(&max_token_length, sizeof(int), 1, f) != 1) { fclose(f); return false; }
    vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      float score;
      int len;
      if (fread(&score, sizeof(float), 1, f) != 1) { fclose(f); return false; }
      if (fread(&len, sizeof(int), 1, f) != 1) { fclose(f); return false; }
      std::string s(len, '\0');
      if (len && fread(&s[0], 1, len, f) != (size_t)len) { fclose(f); return false; }
      vocab[i] = std::move(s);
    }
    fclose(f);
    return true;
  }

  // Mirrors llama2.c's decode(): strip the leading space that follows BOS, and
  // expand raw-byte tokens of the form <0xXX>.
  std::string decode(int prev_token, int token) const {
    if (token < 0 || token >= (int)vocab.size()) return "";
    const std::string& piece = vocab[token];
    const char* p = piece.c_str();
    if (prev_token == 1 && p[0] == ' ') ++p;
    unsigned char byte_val;
    if (sscanf(p, "<0x%02hhX>", &byte_val) == 1) {
      return std::string(1, (char)byte_val);
    }
    return std::string(p);
  }
};

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
  const char* checkpoint = nullptr;
  const char* tokenizer_path = "tokenizer.bin";
  const char* kernel_file = "kernel.vxbin";
  int steps = 32;

  for (int i = 1; i < argc; ++i) {
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
    printf("usage: %s <checkpoint.bin> [-z tokenizer.bin] [-n steps] [-k kernel.vxbin]\n", argv[0]);
    return 1;
  }

  // ---- load checkpoint -----------------------------------------------------
  FILE* f = fopen(checkpoint, "rb");
  if (!f) FAIL("cannot open checkpoint");

  llama_config_t cfg{};
  if (fread(&cfg, sizeof(int32_t), 7, f) != 7) FAIL("bad checkpoint header");
  // llama2.c signals unshared classifier weights with a negative vocab_size.
  int shared_weights = cfg.vocab_size > 0 ? 1 : 0;
  cfg.vocab_size = abs(cfg.vocab_size);

  const int head_size = cfg.dim / cfg.n_heads;
  const int kv_dim    = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
  const int kv_mul    = cfg.n_heads / cfg.n_kv_heads;

  printf("model: dim=%d hidden_dim=%d n_layers=%d n_heads=%d n_kv_heads=%d vocab=%d seq_len=%d\n",
         cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
         cfg.vocab_size, cfg.seq_len);

  WeightOffsets off = compute_offsets(cfg, shared_weights);
  std::vector<float> weights(off.total);
  if (fread(weights.data(), sizeof(float), off.total, f) != off.total)
    FAIL("checkpoint shorter than its header implies");
  fclose(f);
  printf("weights: %.1f MB (%zu floats), shared_classifier=%d\n",
         off.total * 4.0 / (1 << 20), off.total, shared_weights);

  Tokenizer tok;
  if (!tok.load(tokenizer_path, cfg.vocab_size))
    FAIL("cannot load tokenizer");

  if (steps > cfg.seq_len) steps = cfg.seq_len;

  // ---- device setup --------------------------------------------------------
  vx_device_h dev = nullptr;
  CHECK(vx_device_open(0, &dev));

  vx_queue_info_t qi = {};
  qi.struct_size = sizeof(qi);
  qi.priority    = VX_QUEUE_PRIORITY_NORMAL;
  vx_queue_h q = nullptr;
  CHECK(vx_queue_create(dev, &qi, &q));

  // Weights: one contiguous read-only blob, uploaded once and never moved.
  vx_buffer_h w_buf = nullptr;
  CHECK(vx_buffer_create(dev, off.total * sizeof(float), VX_MEM_READ, &w_buf));
  CHECK(vx_enqueue_write(q, w_buf, 0, weights.data(), off.total * sizeof(float),
                         0, nullptr, nullptr));

  // Activations and KV cache stay resident for the whole generation.
  const size_t kv_elems = (size_t)cfg.n_layers * cfg.seq_len * kv_dim;
  vx_buffer_h x_buf = nullptr, xb_buf = nullptr, xb2_buf = nullptr;
  vx_buffer_h hb_buf = nullptr, hb2_buf = nullptr;
  vx_buffer_h qq_buf = nullptr, att_buf = nullptr, logits_buf = nullptr;
  vx_buffer_h kc_buf = nullptr, vc_buf = nullptr;
  CHECK(vx_buffer_create(dev, cfg.dim * sizeof(float),        VX_MEM_READ_WRITE, &x_buf));
  CHECK(vx_buffer_create(dev, cfg.dim * sizeof(float),        VX_MEM_READ_WRITE, &xb_buf));
  CHECK(vx_buffer_create(dev, cfg.dim * sizeof(float),        VX_MEM_READ_WRITE, &xb2_buf));
  CHECK(vx_buffer_create(dev, cfg.hidden_dim * sizeof(float), VX_MEM_READ_WRITE, &hb_buf));
  CHECK(vx_buffer_create(dev, cfg.hidden_dim * sizeof(float), VX_MEM_READ_WRITE, &hb2_buf));
  CHECK(vx_buffer_create(dev, cfg.dim * sizeof(float),        VX_MEM_READ_WRITE, &qq_buf));
  CHECK(vx_buffer_create(dev, (size_t)cfg.n_heads * cfg.seq_len * sizeof(float),
                         VX_MEM_READ_WRITE, &att_buf));
  CHECK(vx_buffer_create(dev, (size_t)cfg.vocab_size * sizeof(float),
                         VX_MEM_READ_WRITE, &logits_buf));
  CHECK(vx_buffer_create(dev, kv_elems * sizeof(float), VX_MEM_READ_WRITE, &kc_buf));
  CHECK(vx_buffer_create(dev, kv_elems * sizeof(float), VX_MEM_READ_WRITE, &vc_buf));
  printf("kv cache: %.1f MB per tensor\n", kv_elems * 4.0 / (1 << 20));

  // ---- resolve kernels -----------------------------------------------------
  vx_module_h mod = nullptr;
  CHECK(vx_module_load_file(dev, kernel_file, &mod));
  vx_kernel_h k_embed = nullptr, k_rmsnorm = nullptr, k_matmul = nullptr;
  vx_kernel_h k_rope = nullptr, k_attn = nullptr, k_swiglu = nullptr, k_residual = nullptr;
  CHECK(vx_module_get_kernel(mod, "embed_k",    &k_embed));
  CHECK(vx_module_get_kernel(mod, "rmsnorm_k",  &k_rmsnorm));
  CHECK(vx_module_get_kernel(mod, "matmul_k",   &k_matmul));
  CHECK(vx_module_get_kernel(mod, "rope_k",     &k_rope));
  CHECK(vx_module_get_kernel(mod, "attn_k",     &k_attn));
  CHECK(vx_module_get_kernel(mod, "swiglu_k",   &k_swiglu));
  CHECK(vx_module_get_kernel(mod, "residual_k", &k_residual));

  uint64_t w_addr = 0, x_addr = 0, xb_addr = 0, xb2_addr = 0;
  uint64_t hb_addr = 0, hb2_addr = 0, q_addr = 0, att_addr = 0, logits_addr = 0;
  uint64_t kc_addr = 0, vc_addr = 0;
  CHECK(vx_buffer_address(w_buf, &w_addr));
  CHECK(vx_buffer_address(x_buf, &x_addr));
  CHECK(vx_buffer_address(xb_buf, &xb_addr));
  CHECK(vx_buffer_address(xb2_buf, &xb2_addr));
  CHECK(vx_buffer_address(hb_buf, &hb_addr));
  CHECK(vx_buffer_address(hb2_buf, &hb2_addr));
  CHECK(vx_buffer_address(qq_buf, &q_addr));
  CHECK(vx_buffer_address(att_buf, &att_addr));
  CHECK(vx_buffer_address(logits_buf, &logits_addr));
  CHECK(vx_buffer_address(kc_buf, &kc_addr));
  CHECK(vx_buffer_address(vc_buf, &vc_addr));

  const uint64_t W = w_addr;  // float-offset helper below indexes from here
  auto wptr = [&](size_t float_off) { return W + float_off * sizeof(float); };

  // One launch, sized to cover `n` work items. Launches are enqueued in order
  // on a single queue, which is what orders the dependent ops.
  auto launch = [&](vx_kernel_h k, kernel_arg_t& arg, uint32_t n) {
    uint32_t grid[1] = {1}, block[1] = {1};
    const uint32_t global_dim[1] = { n };
    CHECK(vx_device_max_occupancy_grid(dev, 1, global_dim, grid, block));
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = k;
    li.args_host    = &arg;
    li.args_size    = sizeof(arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid[0];
    li.block_dim[0] = block[0];
    CHECK(vx_enqueue_launch(q, &li, 0, nullptr, nullptr));
    // The arg block is consumed at enqueue time, so the caller may reuse it.
    CHECK(vx_queue_finish(q, UINT64_MAX));
  };

  // ---- generation loop -----------------------------------------------------
  std::vector<float> logits(cfg.vocab_size);
  int token = 1;  // BOS
  int prev  = 0;
  std::string out_text;

  printf("\n--- generating %d tokens (greedy) ---\n", steps);

  for (int pos = 0; pos < steps; ++pos) {
    kernel_arg_t a{};
    a.dim = cfg.dim; a.kv_dim = kv_dim; a.head_size = head_size;
    a.n_heads = cfg.n_heads; a.kv_mul = kv_mul; a.seq_len = cfg.seq_len;
    a.pos = pos;

    // x = token_embedding_table[token]
    a.out_addr = x_addr;
    a.w_addr   = wptr(off.token_embedding);
    a.token    = token;
    launch(k_embed, a, cfg.dim);

    for (int l = 0; l < cfg.n_layers; ++l) {
      const int loff = l * cfg.seq_len * kv_dim;
      a.loff = loff;

      // --- attention block ---
      // xb = rmsnorm(x) * rms_att_weight[l]
      a.n = cfg.dim; a.out_addr = xb_addr; a.x_addr = x_addr;
      a.w_addr = wptr(off.rms_att + (size_t)l * cfg.dim);
      launch(k_rmsnorm, a, cfg.dim);

      // q, k, v = Wq xb, Wk xb, Wv xb   (k and v write straight into the cache)
      const uint64_t k_dst = kc_addr + (size_t)(loff + pos * kv_dim) * sizeof(float);
      const uint64_t v_dst = vc_addr + (size_t)(loff + pos * kv_dim) * sizeof(float);

      a.x_addr = xb_addr; a.n = cfg.dim;
      a.d = cfg.dim;    a.out_addr = q_addr;
      a.w_addr = wptr(off.wq + (size_t)l * cfg.dim * cfg.dim);
      launch(k_matmul, a, cfg.dim);

      a.d = kv_dim;     a.out_addr = k_dst;
      a.w_addr = wptr(off.wk + (size_t)l * cfg.dim * kv_dim);
      launch(k_matmul, a, kv_dim);

      a.d = kv_dim;     a.out_addr = v_dst;
      a.w_addr = wptr(off.wv + (size_t)l * cfg.dim * kv_dim);
      launch(k_matmul, a, kv_dim);

      // RoPE on q and the freshly written k
      a.q_addr = q_addr; a.k_addr = k_dst;
      launch(k_rope, a, cfg.dim / 2);

      // multi-head attention over the cache -> xb
      a.out_addr = xb_addr;
      a.key_cache_addr = kc_addr; a.value_cache_addr = vc_addr;
      a.att_addr = att_addr;
      launch(k_attn, a, cfg.n_heads);

      // xb2 = Wo xb ; x += xb2
      a.x_addr = xb_addr; a.n = cfg.dim; a.d = cfg.dim; a.out_addr = xb2_addr;
      a.w_addr = wptr(off.wo + (size_t)l * cfg.dim * cfg.dim);
      launch(k_matmul, a, cfg.dim);

      a.n = cfg.dim; a.x_addr = x_addr; a.out_addr = xb2_addr;
      launch(k_residual, a, cfg.dim);

      // --- FFN block ---
      a.n = cfg.dim; a.out_addr = xb_addr; a.x_addr = x_addr;
      a.w_addr = wptr(off.rms_ffn + (size_t)l * cfg.dim);
      launch(k_rmsnorm, a, cfg.dim);

      a.x_addr = xb_addr; a.n = cfg.dim; a.d = cfg.hidden_dim;
      a.out_addr = hb_addr;
      a.w_addr = wptr(off.w1 + (size_t)l * cfg.hidden_dim * cfg.dim);
      launch(k_matmul, a, cfg.hidden_dim);

      a.out_addr = hb2_addr;
      a.w_addr = wptr(off.w3 + (size_t)l * cfg.hidden_dim * cfg.dim);
      launch(k_matmul, a, cfg.hidden_dim);

      // hb = silu(hb) * hb2
      a.n = cfg.hidden_dim; a.hb_addr = hb_addr; a.hb2_addr = hb2_addr;
      launch(k_swiglu, a, cfg.hidden_dim);

      // xb2 = W2 hb ; x += xb2
      a.x_addr = hb_addr; a.n = cfg.hidden_dim; a.d = cfg.dim;
      a.out_addr = xb2_addr;
      a.w_addr = wptr(off.w2 + (size_t)l * cfg.dim * cfg.hidden_dim);
      launch(k_matmul, a, cfg.dim);

      a.n = cfg.dim; a.x_addr = x_addr; a.out_addr = xb2_addr;
      launch(k_residual, a, cfg.dim);
    }

    // Final norm + classifier. Must NOT be in-place: rmsnorm_k has every thread
    // read the whole vector for its reduction while writing one element, so
    // aliasing in and out would race. Land it in xb instead.
    a.n = cfg.dim; a.out_addr = xb_addr; a.x_addr = x_addr;
    a.w_addr = wptr(off.rms_final);
    launch(k_rmsnorm, a, cfg.dim);

    a.x_addr = xb_addr; a.n = cfg.dim; a.d = cfg.vocab_size;
    a.out_addr = logits_addr;
    a.w_addr = wptr(off.wcls);
    launch(k_matmul, a, cfg.vocab_size);

    // sample (greedy) on the host
    CHECK(vx_enqueue_read(q, logits.data(), logits_buf, 0,
                          cfg.vocab_size * sizeof(float), 0, nullptr, nullptr));
    CHECK(vx_queue_finish(q, UINT64_MAX));

    int next = 0;
    float best = logits[0];
    for (int i = 1; i < cfg.vocab_size; ++i) {
      if (logits[i] > best) { best = logits[i]; next = i; }
    }

    std::string piece = tok.decode(token, next);
    out_text += piece;
    printf("%s", piece.c_str());
    fflush(stdout);

    prev  = token;
    token = next;
    if (token == 1) break;  // BOS again -> stop
  }
  (void)prev;

  printf("\n--- done ---\n");

  // ---- teardown ------------------------------------------------------------
  vx_module_release(mod);
  vx_buffer_release(kc_buf); vx_buffer_release(vc_buf);
  vx_buffer_release(logits_buf); vx_buffer_release(att_buf); vx_buffer_release(qq_buf);
  vx_buffer_release(hb2_buf); vx_buffer_release(hb_buf);
  vx_buffer_release(xb2_buf); vx_buffer_release(xb_buf); vx_buffer_release(x_buf);
  vx_buffer_release(w_buf);
  vx_queue_release(q);
  vx_device_release(dev);

  return 0;
}
