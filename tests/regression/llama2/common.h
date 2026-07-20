#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Llama-2 decoder, one token per forward pass (llama2.c semantics).
//
// Stage 1 contract: fp32 throughout, weights and KV cache resident on device,
// one kernel launch per op. The host only seeds the token id and reads back
// logits to sample. Everything else stays in device memory across the ~13
// launches per layer, so there is no per-op host round-trip.
//
// Device-side kernels are resolved by name from a single multi-entry .vxbin
// (see multikernel): embed_k, rmsnorm_k, matmul_k, rope_k, attn_k, swiglu_k,
// residual_k.

// Model hyperparameters, mirrored from the llama2.c checkpoint header.
typedef struct {
  int32_t dim;         // transformer width
  int32_t hidden_dim;  // FFN inner width
  int32_t n_layers;
  int32_t n_heads;     // query heads
  int32_t n_kv_heads;  // key/value heads (< n_heads for GQA/MQA)
  int32_t vocab_size;
  int32_t seq_len;     // max context, sizes the KV cache
} llama_config_t;

// One argument block shared by every kernel. Each kernel reads only the
// fields it needs; unused fields stay zero. Keeping a single struct means the
// host can build one arg buffer per launch without a per-op type zoo.
typedef struct {
  // --- geometry (as needed per op) ---
  int32_t  n;            // matmul: input length   / elementwise: length
  int32_t  d;            // matmul: output length  (rows of W)
  int32_t  pos;          // current token position (RoPE, attention)
  int32_t  dim;
  int32_t  kv_dim;       // dim * n_kv_heads / n_heads
  int32_t  head_size;    // dim / n_heads
  int32_t  n_heads;
  int32_t  kv_mul;       // n_heads / n_kv_heads (GQA replication factor)
  int32_t  seq_len;
  int32_t  loff;         // KV-cache offset of the current layer, in floats
  int32_t  token;        // embed_k: token id to look up
  int32_t  _pad;

  // --- device pointers ---
  uint64_t out_addr;     // matmul/rmsnorm/embed destination
  uint64_t x_addr;       // primary input activation
  uint64_t w_addr;       // weight matrix / rmsnorm gain / embedding table
  uint64_t q_addr;       // RoPE + attention: query
  uint64_t k_addr;       // RoPE: key (writes into the KV cache in place)
  uint64_t key_cache_addr;
  uint64_t value_cache_addr;
  uint64_t att_addr;     // per-head score scratch (n_heads * seq_len)
  uint64_t hb_addr;      // FFN branch 1 (SwiGLU gate)
  uint64_t hb2_addr;     // FFN branch 2 (SwiGLU value)
} kernel_arg_t;

#endif
