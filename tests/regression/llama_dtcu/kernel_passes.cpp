// GPU-side program holding the llama forward's non-GEMM passes. One vxbin, separate from
// the per-mode GEMM programs so their code placement (and therefore their cycle counts)
// is untouched. Every pass is launched with one warp (NUM_THREADS lanes) per block.
//
// Two shapes of pass:
//   * row reductions (RMSNorm, softmax): one warp per row, lanes stride the row, the
//     reduction goes through Local Memory with vx_fence between steps (the warp issues in
//     lockstep, so no barrier is needed) -- the microbenchmark's moti_softmax pattern.
//   * elementwise (RoPE/head packing, head gather, SiLU gate): one lane per element on a
//     2D grid (x = element blocks, y = row), so the whole machine's lanes are busy. A
//     one-warp-per-row version of these was 10-20x slower: 128 warps for 64 warp slots,
//     each looping over its whole row.
// exp() is the shared llama_fast_exp (no libm), the same code on the host reference.

#include <math.h>
#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn2.h>

#include "llama_passes.h"

// RMSNorm over each row of src (fp32 [T x dim]) with weight aux0 (fp32 [dim]), written as
// fp16 to dst ([T x dim]) -- the A operand of the next GEMM. Mirrors llama2.c rmsnorm():
//   ss = sum x^2 / dim + eps ; o = w * (x / sqrt(ss))
// grid (T), one warp per row.
__kernel __attribute__((aligned(256))) void llama_rmsnorm_f2h(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t dim = arg->dim;
  const uint32_t row = blockIdx.x;
  const uint32_t t   = threadIdx.x;
  const uint32_t nt  = blockDim.x;
  auto x   = reinterpret_cast<const float*>(arg->src) + (size_t)row * dim;
  auto w   = reinterpret_cast<const float*>(arg->aux0);
  auto o   = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)row * dim;
  auto red = reinterpret_cast<float*>(__local_mem());

  float ss = 0.0f;
  for (uint32_t j = t; j < dim; j += nt) ss += x[j] * x[j];
  red[t] = ss;
  vx_fence();
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    vx_fence();
  }
  float inv = red[0] / (float)dim;
  inv += arg->eps;
  inv = 1.0f / sqrtf(inv);
  for (uint32_t j = t; j < dim; j += nt)
    o[j] = llama_f2h(w[j] * (inv * x[j]));
}

// RoPE on q and k, then head packing for the attention GEMMs.
//   src : qkv fp32 [T x 3*dim]  (q | k | v per row, kv_dim == dim here)
//   aux2: RoPE table fp32 [T][head/2][2] = (cos, sin) per position and pair, host-built
//   dst : Qp  fp16 [n_heads][T x head_pad]   (A of QK^T; pad columns zero)
//   aux0: Kp  fp16 [n_heads][T x head_pad]   (B of QK^T, col-major [K=head_pad x N=T])
//   aux1: VpT fp16 [n_heads][head_pad x T]   (B of PV,   col-major [K=T x N=head_pad])
// grid (ceil(dim/2 / 32), T): lane p < dim/2 owns pair (2p, 2p+1) of q, k and v; the lanes
// past the last pair zero the padding (head_pad - head of them per head, all heads).
__kernel __attribute__((aligned(256))) void llama_rope_headpack(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t dim = arg->dim, T = arg->T, H = arg->head, HP = arg->head_pad;
  const uint32_t row = blockIdx.y;
  const uint32_t p   = blockIdx.x * blockDim.x + threadIdx.x;
  auto qkv = reinterpret_cast<const float*>(arg->src) + (size_t)row * (3 * dim);
  auto Qp  = reinterpret_cast<uint16_t*>(arg->dst);
  auto Kp  = reinterpret_cast<uint16_t*>(arg->aux0);
  auto VpT = reinterpret_cast<uint16_t*>(arg->aux1);
  const size_t head_stride = (size_t)T * HP;

  if (p < dim / 2) {
    auto rope = reinterpret_cast<const float*>(arg->aux2) + (size_t)(arg->pos0 + row) * (H / 2) * 2;
    const uint32_t i  = 2 * p;
    const uint32_t h  = i / H, c = i % H;
    const float fcr = rope[(c / 2) * 2 + 0], fci = rope[(c / 2) * 2 + 1];
    const float q0 = qkv[i], q1 = qkv[i + 1];
    const float k0 = qkv[dim + i], k1 = qkv[dim + i + 1];
    uint16_t* qd = Qp + h * head_stride + (size_t)row * HP + c;
    uint16_t* kd = Kp + h * head_stride + (size_t)row * HP + c;
    qd[0] = llama_f2h(q0 * fcr - q1 * fci); qd[1] = llama_f2h(q0 * fci + q1 * fcr);
    kd[0] = llama_f2h(k0 * fcr - k1 * fci); kd[1] = llama_f2h(k0 * fci + k1 * fcr);
    uint16_t* vd = VpT + h * head_stride + (size_t)c * T + row;   // transposed: [c][row]
    vd[0] = llama_f2h(qkv[2 * dim + i]);
    vd[T] = llama_f2h(qkv[2 * dim + i + 1]);
  } else {
    const uint32_t idx = p - dim / 2;        // 0 .. (lanes past the pairs)
    if (idx < HP - H) {
      const uint32_t c = H + idx;
      for (uint32_t h = 0; h < arg->n_heads; ++h) {
        Qp [h * head_stride + (size_t)row * HP + c] = 0;
        Kp [h * head_stride + (size_t)row * HP + c] = 0;
        VpT[h * head_stride + (size_t)c * T + row]  = 0;
      }
    }
  }
}

// Causal softmax over attention scores, per head and row.
//   src: S fp32 [n_heads][T x T] (raw QK^T; overwritten with exp values), dst: P fp16 [n_heads][T x T]
//   P[r][j] = softmax_j( scale * S[r][j] ) for j <= pos0 + r, else 0.
// grid (n_heads * T), one warp per (head, row): blockIdx.x = head * T + row.
__kernel __attribute__((aligned(256))) void llama_softmax_causal(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t T = arg->T;
  const uint32_t hr  = blockIdx.x;
  const uint32_t row = hr % T;
  const uint32_t t   = threadIdx.x;
  const uint32_t nt  = blockDim.x;
  const uint32_t valid = arg->pos0 + row + 1;   // keys 0 .. pos0+row
  auto S   = reinterpret_cast<float*>(arg->src) + (size_t)hr * T;
  auto P   = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)hr * T;
  auto red = reinterpret_cast<float*>(__local_mem());
  const float scale = arg->scale;

  float m = -3.4028235e38f;
  for (uint32_t j = t; j < valid; j += nt) { const float v = S[j] * scale; m = v > m ? v : m; }
  red[t] = m;
  vx_fence();
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] = red[t] > red[t + s] ? red[t] : red[t + s];
    vx_fence();
  }
  const float row_max = red[0];
  vx_fence();

  float acc = 0.0f;
  for (uint32_t j = t; j < valid; j += nt) { const float e = llama_fast_exp(S[j] * scale - row_max); S[j] = e; acc += e; }
  red[t] = acc;
  vx_fence();
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    vx_fence();
  }
  const float inv = 1.0f / red[0];

  for (uint32_t j = t; j < T; j += nt)
    P[j] = (j < valid) ? llama_f2h(S[j] * inv) : (uint16_t)0;
}

// Gather the per-head PV outputs (fp32 [n_heads][T x head_pad]) into the attention output
// fp16 [T x dim] (pad columns dropped) -- the A operand of the wo GEMM.
// grid (dim / 32, T), one lane per element.
__kernel __attribute__((aligned(256))) void llama_attn_gather(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t dim = arg->dim, T = arg->T, H = arg->head, HP = arg->head_pad;
  const uint32_t row = blockIdx.y;
  const uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  auto Dh = reinterpret_cast<const float*>(arg->src);
  auto o  = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)row * dim;
  const uint32_t h = i / H, c = i % H;
  o[i] = llama_f2h(Dh[(size_t)h * T * HP + (size_t)row * HP + c]);
}

// SwiGLU gate: src fp32 [T x 2*hidden] = (w1 x | w3 x) -> dst fp16 [T x hidden]
//   o = silu(a) * b,  silu(a) = a / (1 + exp(-a))   (llama2.c)
// grid (hidden / 32, T), one lane per element.
__kernel __attribute__((aligned(256))) void llama_silu_gate(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t hidden = arg->hidden;
  const uint32_t row = blockIdx.y;
  const uint32_t j   = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= hidden) return;
  auto in = reinterpret_cast<const float*>(arg->src) + (size_t)row * (2 * hidden);
  auto o  = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)row * hidden;
  const float a = in[j], b = in[hidden + j];
  const float s = a * (1.0f / (1.0f + llama_fast_exp(-a)));
  o[j] = llama_f2h(s * b);
}

// ======================= batched decode (B sequences, one new token each) =======================
// Both passes below serve the decode step (llama_decode.cpp): every sequence b has its own
// KV cache [kv_stride x dim] per layer (fp16, RoPE'd k and plain v, the same rounding points
// as the prefill's Kp / VpT), and all B new tokens sit at position pos0.

// RoPE on the new token's q and k; k and v appended to the cache at row pos0.
//   src : qkv fp32 [B x 3*dim]
//   aux2: RoPE table fp32 [pos][head/2][2], row pos0 is used
//   dst : q16 fp16 [B x dim]                (RoPE'd query, read by llama_attn_decode)
//   aux0: Kc  fp16 [B][kv_stride x dim]     (row pos0 written)
//   aux1: Vc  fp16 [B][kv_stride x dim]
// grid (ceil(dim/2 / 32), B): lane p < dim/2 owns pair (2p, 2p+1) of q, k and v.
__kernel __attribute__((aligned(256))) void llama_rope_kvappend(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t dim = arg->dim, H = arg->head, S = arg->kv_stride, pos = arg->pos0;
  const uint32_t b = blockIdx.y;
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= dim / 2) return;
  auto qkv  = reinterpret_cast<const float*>(arg->src) + (size_t)b * (3 * dim);
  auto rope = reinterpret_cast<const float*>(arg->aux2) + (size_t)pos * (H / 2) * 2;
  auto q16  = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)b * dim;
  auto kc   = reinterpret_cast<uint16_t*>(arg->aux0) + ((size_t)b * S + pos) * dim;
  auto vc   = reinterpret_cast<uint16_t*>(arg->aux1) + ((size_t)b * S + pos) * dim;
  const uint32_t i = 2 * p, c = i % H;
  const float fcr = rope[(c / 2) * 2 + 0], fci = rope[(c / 2) * 2 + 1];
  const float q0 = qkv[i], q1 = qkv[i + 1];
  const float k0 = qkv[dim + i], k1 = qkv[dim + i + 1];
  q16[i] = llama_f2h(q0 * fcr - q1 * fci); q16[i + 1] = llama_f2h(q0 * fci + q1 * fcr);
  kc[i]  = llama_f2h(k0 * fcr - k1 * fci); kc[i + 1]  = llama_f2h(k0 * fci + k1 * fcr);
  vc[i]  = llama_f2h(qkv[2 * dim + i]);    vc[i + 1]  = llama_f2h(qkv[2 * dim + i + 1]);
}

// Decode attention, one warp per (sequence, head): scores of the new query against the
// pos0+1 cached keys (lanes stride the keys), softmax over them, then the weighted sum of
// the cached values (lanes stride the head dims). Output att16 fp16 [B x dim], the A operand
// of the wo GEMM. The probabilities stay fp32 here (the prefill rounds P to fp16 because it
// is a GEMM operand there); the host reference does the same.
// q is re-read from global memory (coalesced, cache-resident) inside the dot product on
// purpose: keeping it in Local Memory made every lane read the same word and the pass ran
// 3x slower (B=16: 3.77 M vs 1.23 M cycles), and a lanes-over-dims variant with a warp
// reduction per key was slower still.
//   src: q16 fp16 [B x dim]; aux0: Kc; aux1: Vc (layouts as above); dst: att16
//   lmem: (kv_stride + blockDim.x) floats = scores[kv_stride] then the reduction scratch
// grid (n_heads, B).
__kernel __attribute__((aligned(256))) void llama_attn_decode(llama_pass_arg_t* __UNIFORM__ arg) {
  const uint32_t dim = arg->dim, hs = arg->head, S = arg->kv_stride;
  const uint32_t L  = arg->pos0 + 1;          // cached keys incl. the new one
  const uint32_t h  = blockIdx.x, b = blockIdx.y;
  const uint32_t t  = threadIdx.x, nt = blockDim.x;
  auto q   = reinterpret_cast<const uint16_t*>(arg->src) + (size_t)b * dim + (size_t)h * hs;
  auto K   = reinterpret_cast<const uint16_t*>(arg->aux0) + (size_t)b * S * dim + (size_t)h * hs;
  auto V   = reinterpret_cast<const uint16_t*>(arg->aux1) + (size_t)b * S * dim + (size_t)h * hs;
  auto o16 = reinterpret_cast<uint16_t*>(arg->dst) + (size_t)b * dim + (size_t)h * hs;
  auto sc  = reinterpret_cast<float*>(__local_mem());
  auto red = sc + S;
  const float scale = arg->scale;

  float m = -3.4028235e38f;
  for (uint32_t j = t; j < L; j += nt) {
    const uint16_t* kr = K + (size_t)j * dim;
    float s = 0.0f;
    for (uint32_t c = 0; c < hs; ++c) s += llama_h2f(q[c]) * llama_h2f(kr[c]);
    s *= scale;
    sc[j] = s;
    m = s > m ? s : m;
  }
  red[t] = m;
  vx_fence();
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] = red[t] > red[t + s] ? red[t] : red[t + s];
    vx_fence();
  }
  const float row_max = red[0];
  vx_fence();

  float acc = 0.0f;
  for (uint32_t j = t; j < L; j += nt) { const float e = llama_fast_exp(sc[j] - row_max); sc[j] = e; acc += e; }
  red[t] = acc;
  vx_fence();
  for (uint32_t s = nt >> 1; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    vx_fence();
  }
  const float inv = 1.0f / red[0];
  vx_fence();

  for (uint32_t c = t; c < hs; c += nt) {
    float o = 0.0f;
    for (uint32_t j = 0; j < L; ++j) o += sc[j] * llama_h2f(V[(size_t)j * dim + c]);
    o16[c] = llama_f2h(o * inv);
  }
}
