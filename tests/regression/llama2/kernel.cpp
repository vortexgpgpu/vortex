// Llama-2 decoder kernels (Stage 1: fp32, correctness-first).
//
// Multi-entry .vxbin: each __kernel below is its own KMU entry point, resolved
// by name on the host via vx_module_get_kernel. There is deliberately no
// kernel_main -- see tests/regression/multikernel for the mechanism.
//
// Stage-1 simplifications, all marked OPT: below. They trade throughput for
// having no cross-thread synchronization anywhere, so a numerical mismatch is
// always an arithmetic bug and never a race:
//   - rmsnorm recomputes its reduction redundantly per thread (dim is 288).
//   - attention runs one thread per head (n_heads is 6).
// Both become real bottlenecks only once the matmuls are on the TCU.

#include <vx_spawn2.h>
#include <math.h>
#include "common.h"

// Copy one row of the token-embedding table into the activation vector.
__kernel void embed_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out   = reinterpret_cast<float*>(arg->out_addr);
  auto table = reinterpret_cast<const float*>(arg->w_addr);
  int  dim   = arg->dim;
  int  i     = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim)
    return;
  out[i] = table[(size_t)arg->token * dim + i];
}

// out = (x / rms(x)) * weight,  rms(x) = sqrt(mean(x^2) + 1e-5)
__kernel void rmsnorm_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<float*>(arg->out_addr);
  auto x   = reinterpret_cast<const float*>(arg->x_addr);
  auto w   = reinterpret_cast<const float*>(arg->w_addr);
  int  n   = arg->n;
  int  i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  // OPT: every thread recomputes the sum of squares so the kernel needs no
  // barrier. Replace with a block reduction once this shows up in a profile.
  float ss = 0.0f;
  for (int j = 0; j < n; ++j) {
    ss += x[j] * x[j];
  }
  ss = 1.0f / sqrtf(ss / n + 1e-5f);

  out[i] = w[i] * (ss * x[i]);
}

// out[i] = dot(W[i, :], x)   -- W is (d, n) row-major, x is (n,), out is (d,).
// This is the workhorse: QKV, attention output, all three FFN matrices, and
// the classifier all land here. One thread per output row.
__kernel void matmul_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<float*>(arg->out_addr);
  auto x   = reinterpret_cast<const float*>(arg->x_addr);
  auto w   = reinterpret_cast<const float*>(arg->w_addr);
  int  n   = arg->n;
  int  d   = arg->d;
  int  i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d)
    return;

  const float* row = w + (size_t)i * n;
  float sum = 0.0f;
  for (int j = 0; j < n; ++j) {
    sum += row[j] * x[j];
  }
  out[i] = sum;
}

// RoPE: rotate (q, k) pairwise within each head by the position-dependent
// angle. Thread i handles the complex pair at element 2*i.
__kernel void rope_k(kernel_arg_t* __UNIFORM__ arg) {
  auto q = reinterpret_cast<float*>(arg->q_addr);
  auto k = reinterpret_cast<float*>(arg->k_addr);
  int  dim       = arg->dim;
  int  kv_dim    = arg->kv_dim;
  int  head_size = arg->head_size;
  int  pos       = arg->pos;

  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  int i    = pair * 2;
  if (i >= dim)
    return;

  int   head_dim = i % head_size;
  float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
  float val  = pos * freq;
  float fcr  = cosf(val);
  float fci  = sinf(val);

  // Rotate the query always; rotate the key only while inside kv_dim (the
  // tail of q beyond kv_dim has no matching k under GQA).
  float q0 = q[i], q1 = q[i + 1];
  q[i]     = q0 * fcr - q1 * fci;
  q[i + 1] = q0 * fci + q1 * fcr;
  if (i < kv_dim) {
    float k0 = k[i], k1 = k[i + 1];
    k[i]     = k0 * fcr - k1 * fci;
    k[i + 1] = k0 * fci + k1 * fcr;
  }
}

// Causal multi-head attention over the KV cache, for the current position.
// One thread per head: scores -> softmax -> value-weighted sum.
__kernel void attn_k(kernel_arg_t* __UNIFORM__ arg) {
  auto xb  = reinterpret_cast<float*>(arg->out_addr);
  auto q   = reinterpret_cast<const float*>(arg->q_addr);
  auto kc  = reinterpret_cast<const float*>(arg->key_cache_addr);
  auto vc  = reinterpret_cast<const float*>(arg->value_cache_addr);
  auto att = reinterpret_cast<float*>(arg->att_addr);

  int head_size = arg->head_size;
  int kv_dim    = arg->kv_dim;
  int kv_mul    = arg->kv_mul;
  int pos       = arg->pos;
  int loff      = arg->loff;
  int seq_len   = arg->seq_len;

  int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= arg->n_heads)
    return;

  const float* qh    = q + h * head_size;
  float*       att_h = att + (size_t)h * seq_len;
  // Under GQA several query heads share one KV head.
  int          kvh   = (h / kv_mul) * head_size;

  // 1. scores[t] = dot(q_h, k_h(t)) / sqrt(head_size), for t <= pos (causal)
  float scale = 1.0f / sqrtf((float)head_size);
  for (int t = 0; t <= pos; ++t) {
    const float* kt = kc + loff + (size_t)t * kv_dim + kvh;
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i) {
      score += qh[i] * kt[i];
    }
    att_h[t] = score * scale;
  }

  // 2. softmax over 0..pos, max-subtracted for numerical stability
  float max_val = att_h[0];
  for (int t = 1; t <= pos; ++t) {
    if (att_h[t] > max_val) max_val = att_h[t];
  }
  float sum = 0.0f;
  for (int t = 0; t <= pos; ++t) {
    att_h[t] = expf(att_h[t] - max_val);
    sum += att_h[t];
  }
  float inv = 1.0f / sum;

  // 3. xb_h = sum_t softmax_t * v_h(t)
  float* xb_h = xb + h * head_size;
  for (int i = 0; i < head_size; ++i) {
    xb_h[i] = 0.0f;
  }
  for (int t = 0; t <= pos; ++t) {
    const float* vt = vc + loff + (size_t)t * kv_dim + kvh;
    float a = att_h[t] * inv;
    for (int i = 0; i < head_size; ++i) {
      xb_h[i] += a * vt[i];
    }
  }
}

// SwiGLU: hb = silu(hb) * hb2, where silu(v) = v * sigmoid(v).
__kernel void swiglu_k(kernel_arg_t* __UNIFORM__ arg) {
  auto hb  = reinterpret_cast<float*>(arg->hb_addr);
  auto hb2 = reinterpret_cast<const float*>(arg->hb2_addr);
  int  n   = arg->n;
  int  i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = hb[i];
  v *= 1.0f / (1.0f + expf(-v));
  hb[i] = v * hb2[i];
}

// Residual: x += xb2
__kernel void residual_k(kernel_arg_t* __UNIFORM__ arg) {
  auto x   = reinterpret_cast<float*>(arg->x_addr);
  auto xb2 = reinterpret_cast<const float*>(arg->out_addr);
  int  n   = arg->n;
  int  i   = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  x[i] += xb2[i];
}
