#ifndef _LLAMA_DTCU_FORWARD_H_
#define _LLAMA_DTCU_FORWARD_H_

#include <cstdint>
#include <string>

struct RunOptions {
  std::string model;        // stories15M.bin
  std::string tokenizer;    // tokenizer.bin
  std::string prompt;       // prompt text (BOS is prepended)
  uint32_t    T = 128;      // rows per forward: prompt tokens, padded up to T
  uint32_t    B = 32;       // decode: sequences per step (M of every linear layer)
  int         mode = 1;     // uniform mode for every GEMM site
  std::string modemap;      // optional file: "<site> <mode>" per line, overrides `mode`
  int         fallback = -1;// mode to use for a site the chosen mode cannot run (-1 = abort)
  bool        verify = true;// compare every device output against the host reference
  bool        text = true;  // print the next-token pieces
};

// Prefill forward of one prompt batch through the device, one GEMM mode per site.
// Prints [LLAMA] lines (per site and total) and [LLAMA-VERIFY] lines when verifying.
// Returns 0 on success, the number of failed verification sites, or a negative error.
int run_llama(const RunOptions& opt);

// One batched decode step (B sequences with the prompt in their KV cache, one new token each)
// through the device, one GEMM mode per site (qkv wo w13 w2 cls). Same outputs and return.
int run_decode(const RunOptions& opt);

#endif // _LLAMA_DTCU_FORWARD_H_
