#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef WGMMA_NRC
  #define WGMMA_NRC 8
#endif

// Number of sequences decoded concurrently. This is the GEMM N dimension.
// 16 is chosen so that (a) it fills the WGMMA CTA tile, xtileN = 2*WGMMA_NRC,
// and (b) arithmetic intensity reaches ~16 FLOP/byte, past the ~6-12 machine
// balance -- below that the tensor cores stall on weight fetches regardless of
// how well the tile is packed.
#ifndef LLAMA_BATCH
  #define LLAMA_BATCH 16
#endif

// Llama-2 decoder, batched, WGMMA matmuls.
//
// Activations are [dim][BATCH] (dim-major, batch-minor), which makes every
// projection a GEMM: OUT[d][B] = W[d][n] . X[n][B]. W's row-major (d,n) file
// layout is already the WGMMA A operand, so no weight transpose is needed.
//
// Weights and KV cache stay resident on device; the host only seeds tokens
// and reads logits back to sample.

typedef struct {
  int32_t dim;
  int32_t hidden_dim;
  int32_t n_layers;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t vocab_size;
  int32_t seq_len;
} llama_config_t;

typedef struct {
  // --- geometry ---
  int32_t  n;              // matmul K / elementwise length (per batch column)
  int32_t  d;              // matmul M (output rows)
  int32_t  batch;          // matmul N -- concurrent sequences
  int32_t  dim;
  int32_t  kv_dim;
  int32_t  head_size;
  int32_t  n_heads;
  int32_t  kv_mul;         // n_heads / n_kv_heads (GQA replication)
  int32_t  seq_len;
  int32_t  loff;           // KV offset of the current layer, in floats
  int32_t  kv_seq_stride;  // floats per sequence in the KV cache
  int32_t  _pad;

  // --- device pointers ---
  uint64_t out_addr;
  uint64_t x_addr;
  uint64_t w_addr;
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t key_cache_addr;
  uint64_t value_cache_addr;
  uint64_t att_addr;
  uint64_t hb_addr;
  uint64_t hb2_addr;
  uint64_t tokens_addr;    // int32[batch] -- current token per sequence
  uint64_t pos_addr;       // int32[batch] -- current position per sequence
  uint64_t scratch_addr;   // float[batch] -- rmsnorm reciprocal norms
} kernel_arg_t;

#endif
