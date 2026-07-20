// Llama-2 decoder kernels -- batched, WGMMA matmuls (Stage 2).
//
// Multi-entry .vxbin: each __kernel is its own KMU entry point, resolved by
// name on the host. There is deliberately no kernel_main.
//
// BATCH LAYOUT is the whole trick. Activations are stored [dim][BATCH] --
// dim-major, batch-minor -- so every projection becomes a plain GEMM:
//
//     OUT[d][B] = W[d][n] . X[n][B]        M=d, K=n, N=BATCH
//
// W's row-major (d, n) file layout is already exactly the A operand WGMMA
// wants, and X's batch-minor layout is exactly the B operand. So the matmul
// below is the sgemm_tcu_wg kernel with N=BATCH, verified at Llama's real
// shapes (M=288, N=16, K=288) before this file was written.
//
// Why BATCH=16 and not 4: filling the WGMMA tile only needs batch >= xtileN
// (= 2*WGMMA_NRC), but decode re-reads every weight each step, so at batch=4
// the arithmetic intensity is ~4 FLOP/byte against a machine balance of
// 6-12 -- the tensor cores would stall on memory. Batch 16 clears both.
//
// Precision: fp16 operands, fp32 accumulate (WGMMA's native mode). The
// elementwise ops stay fp32; only the GEMM operands are narrowed.

#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <math.h>
#include "common.h"

namespace vt = vortex::tensor;

using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, WGMMA_NRC>;

// ---------------------------------------------------------------------------
// WGMMA matmul: C[M][N] = A[M][K] . B[K][N],  N = BATCH
//   A = weights   (M=d rows, K=n cols) row-major, fp16
//   B = activations (K=n rows, N=batch cols)     fp16
//   C = output    (M=d rows, N=batch cols)       fp32
// ---------------------------------------------------------------------------
__kernel void matmul_wgmma_k(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t*>(arg->w_addr);   // weights
  auto pB = reinterpret_cast<ctx::input_t*>(arg->x_addr);   // activations
  auto pC = reinterpret_cast<ctx::output_t*>(arg->out_addr);

  uint32_t M = arg->d;      // output rows
  uint32_t N = arg->batch;  // batch  (the GEMM N dimension)
  uint32_t K = arg->n;      // reduction length

  uint32_t tid         = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank   = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps   = num_threads / VX_CFG_NUM_THREADS;

  uint32_t cta_M    = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // smem: A [cta_M x tileK] then B [tileK x xtileN]
  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperatively stage A (block-major) and B into smem. Rows/cols past the
    // real extents are zero-filled so partial tiles contribute nothing --
    // Llama's dims (288, 768, 32000 x batch) are not tile multiples.
    uint32_t a_size = cta_M * ctx::tileK;
    for (uint32_t i = 0; i < a_size; i += num_threads) {
      uint32_t idx = i + tid;
      if (idx >= a_size) break;
      uint32_t r = idx / ctx::tileK;
      uint32_t c = idx % ctx::tileK;
      uint32_t gr = tile_row + r, gc = k + c;
      A_smem[ctx::a_blockmajor_idx(r, c)] =
          (gr < M && gc < K) ? pA[(size_t)gr * K + gc] : ctx::input_t(0);
    }

    uint32_t b_size = ctx::tileK * ctx::xtileN;
    for (uint32_t i = 0; i < b_size; i += num_threads) {
      uint32_t idx = i + tid;
      if (idx >= b_size) break;
      uint32_t r = idx / ctx::xtileN;
      uint32_t c = idx % ctx::xtileN;
      uint32_t gr = k + r, gc = tile_col + c;
      B_smem[ctx::b_blockmajor_idx(r, c)] =
          (gr < K && gc < N) ? pB[(size_t)gr * N + gc] : ctx::input_t(0);
    }

    __syncthreads();

    auto A_warp = A_smem + warp_rank * ctx::a_warp_elems;
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0);
  #if (WGMMA_NRC <= 16)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, 0);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    auto desc_a = vt::vx_make_smem_desc(A_warp, 0);
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    __syncthreads();
  }

  uint32_t out_row = tile_row + warp_rank * ctx::xtileM;
  if (out_row < M) {
    ctx::store_matrix_sync(pC + (size_t)out_row * N + tile_col, fragC, N);
  }
}

// ---------------------------------------------------------------------------
// Scalar fp32 matmul, kept as the numerical reference for A/B against WGMMA.
// Same [dim][batch] layout.
// ---------------------------------------------------------------------------
__kernel void matmul_scalar_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<float*>(arg->out_addr);
  auto x   = reinterpret_cast<const float*>(arg->x_addr);
  auto w   = reinterpret_cast<const float*>(arg->w_addr);
  int n = arg->n, d = arg->d, B = arg->batch;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d * B) return;
  int row = idx / B, b = idx % B;
  const float* wr = w + (size_t)row * n;
  float sum = 0.0f;
  for (int j = 0; j < n; ++j) sum += wr[j] * x[(size_t)j * B + b];
  out[(size_t)row * B + b] = sum;
}

// ---------------------------------------------------------------------------
// Elementwise / attention kernels. All operate on the [dim][batch] layout.
// ---------------------------------------------------------------------------

// Per-batch-element embedding lookup: x[d][b] = table[token[b]][d]
__kernel void embed_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out    = reinterpret_cast<float*>(arg->out_addr);
  auto table  = reinterpret_cast<const float*>(arg->w_addr);
  auto tokens = reinterpret_cast<const int32_t*>(arg->tokens_addr);
  int dim = arg->dim, B = arg->batch;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= dim * B) return;
  int d = idx / B, b = idx % B;
  out[(size_t)d * B + b] = table[(size_t)tokens[b] * dim + d];
}

// Pass 1: ss[b] = sum_d x[d][b]^2. One thread per batch element -- this
// replaces the Stage-1 scheme where every thread redundantly reduced.
__kernel void rmsnorm_sum_k(kernel_arg_t* __UNIFORM__ arg) {
  auto ss = reinterpret_cast<float*>(arg->scratch_addr);
  auto x  = reinterpret_cast<const float*>(arg->x_addr);
  int n = arg->n, B = arg->batch;
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  float acc = 0.0f;
  for (int d = 0; d < n; ++d) {
    float v = x[(size_t)d * B + b];
    acc += v * v;
  }
  ss[b] = 1.0f / sqrtf(acc / n + 1e-5f);
}

// Pass 2: out[d][b] = w[d] * ss[b] * x[d][b]
__kernel void rmsnorm_scale_k(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<float*>(arg->out_addr);
  auto x   = reinterpret_cast<const float*>(arg->x_addr);
  auto w   = reinterpret_cast<const float*>(arg->w_addr);
  auto ss  = reinterpret_cast<const float*>(arg->scratch_addr);
  int n = arg->n, B = arg->batch;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n * B) return;
  int d = idx / B, b = idx % B;
  out[(size_t)d * B + b] = w[d] * (ss[b] * x[(size_t)d * B + b]);
}

// fp32 -> fp16 (IEEE binary16) bit conversion, round-to-nearest-even.
//
// ctx::input_t is It::dtype == uint16_t -- a STORAGE type, not an arithmetic
// one. Writing `ctx::input_t(f)` is a numeric conversion that truncates 0.37f
// to integer 0, which silently destroys every activation. There is no Zfh on
// this core and no device-side helper, so do the bit twiddling explicitly.
// Verified exhaustively against rv_ftoh_s (8.3M values, 0 mismatches).
static inline uint16_t f32_to_f16_bits(float f) {
  uint32_t x;
  __builtin_memcpy(&x, &f, 4);
  uint32_t sign = (x & 0x80000000u) >> 16;
  x &= 0x7FFFFFFFu;
  if (x >= 0x47800000u)                       // >= 65536, or Inf/NaN
    return (uint16_t)(sign | (x > 0x7F800000u ? 0x7E00u : 0x7C00u));
  if (x < 0x38800000u) {                      // < 2^-14 : subnormal or zero
    if (x < 0x33000000u) return (uint16_t)sign;
    uint32_t e  = x >> 23;
    uint32_t m  = (x & 0x7FFFFFu) | 0x800000u;
    uint32_t sh = 126u - e;                   // >= 14
    uint32_t v  = m >> sh;
    uint32_t r  = m & ((1u << sh) - 1u);
    uint32_t hf = 1u << (sh - 1);
    if (r > hf || (r == hf && (v & 1u))) ++v;
    return (uint16_t)(sign | v);
  }
  uint32_t v = (x - 0x38000000u) >> 13;       // rebias 127 -> 15
  uint32_t r = x & 0x1FFFu;
  if (r > 0x1000u || (r == 0x1000u && (v & 1u))) ++v;
  return (uint16_t)(sign | v);
}

// Narrow fp32 activations to fp16 for the WGMMA operands.
__kernel void to_fp16_k(kernel_arg_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<const float*>(arg->x_addr);
  auto dst = reinterpret_cast<uint16_t*>(arg->out_addr);
  int total = arg->n * arg->batch;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  dst[i] = f32_to_f16_bits(src[i]);
}

// RoPE. Each batch element sits at its own position (pos_addr[b]); thread
// handles one (pair, batch) slot.
__kernel void rope_k(kernel_arg_t* __UNIFORM__ arg) {
  auto q   = reinterpret_cast<float*>(arg->q_addr);
  auto k   = reinterpret_cast<float*>(arg->k_addr);
  auto pos = reinterpret_cast<const int32_t*>(arg->pos_addr);
  int dim = arg->dim, kv_dim = arg->kv_dim, head_size = arg->head_size;
  int B = arg->batch, kvB = arg->batch;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int npairs = dim / 2;
  if (idx >= npairs * B) return;
  int pair = idx / B, b = idx % B;
  int i = pair * 2;

  int   head_dim = i % head_size;
  float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)head_size);
  float val  = pos[b] * freq;
  float fcr = cosf(val), fci = sinf(val);

  float q0 = q[(size_t)i * B + b], q1 = q[(size_t)(i + 1) * B + b];
  q[(size_t)i * B + b]       = q0 * fcr - q1 * fci;
  q[(size_t)(i + 1) * B + b] = q0 * fci + q1 * fcr;
  if (i < kv_dim) {
    float k0 = k[(size_t)i * kvB + b], k1 = k[(size_t)(i + 1) * kvB + b];
    k[(size_t)i * kvB + b]       = k0 * fcr - k1 * fci;
    k[(size_t)(i + 1) * kvB + b] = k0 * fci + k1 * fcr;
  }
}

// Causal attention. One thread per (batch, head). Each sequence owns a
// private KV cache slice and attends only up to its own position.
__kernel void attn_k(kernel_arg_t* __UNIFORM__ arg) {
  auto xb  = reinterpret_cast<float*>(arg->out_addr);
  auto q   = reinterpret_cast<const float*>(arg->q_addr);
  auto kc  = reinterpret_cast<const float*>(arg->key_cache_addr);
  auto vc  = reinterpret_cast<const float*>(arg->value_cache_addr);
  auto att = reinterpret_cast<float*>(arg->att_addr);
  auto pos = reinterpret_cast<const int32_t*>(arg->pos_addr);

  int head_size = arg->head_size, kv_dim = arg->kv_dim, kv_mul = arg->kv_mul;
  int seq_len = arg->seq_len, loff = arg->loff;
  int n_heads = arg->n_heads, B = arg->batch;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_heads * B) return;
  int h = idx / B, b = idx % B;

  int   p        = pos[b];
  // Per-sequence KV cache: sequence b owns a [n_layers*seq_len*kv_dim] slab.
  size_t seq_kv  = (size_t)arg->kv_seq_stride * b;
  float* att_h   = att + ((size_t)b * n_heads + h) * seq_len;
  int    kvh     = (h / kv_mul) * head_size;
  float  scale   = 1.0f / sqrtf((float)head_size);

  for (int t = 0; t <= p; ++t) {
    const float* kt = kc + seq_kv + loff + (size_t)t * kv_dim + kvh;
    float score = 0.0f;
    for (int i = 0; i < head_size; ++i)
      score += q[(size_t)(h * head_size + i) * B + b] * kt[i];
    att_h[t] = score * scale;
  }

  float mx = att_h[0];
  for (int t = 1; t <= p; ++t) if (att_h[t] > mx) mx = att_h[t];
  float sum = 0.0f;
  for (int t = 0; t <= p; ++t) { att_h[t] = expf(att_h[t] - mx); sum += att_h[t]; }
  float inv = 1.0f / sum;

  for (int i = 0; i < head_size; ++i)
    xb[(size_t)(h * head_size + i) * B + b] = 0.0f;
  for (int t = 0; t <= p; ++t) {
    const float* vt = vc + seq_kv + loff + (size_t)t * kv_dim + kvh;
    float a = att_h[t] * inv;
    for (int i = 0; i < head_size; ++i)
      xb[(size_t)(h * head_size + i) * B + b] += a * vt[i];
  }
}

// SwiGLU: hb = silu(hb) * hb2
__kernel void swiglu_k(kernel_arg_t* __UNIFORM__ arg) {
  auto hb  = reinterpret_cast<float*>(arg->hb_addr);
  auto hb2 = reinterpret_cast<const float*>(arg->hb2_addr);
  int total = arg->n * arg->batch;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = hb[i];
  v *= 1.0f / (1.0f + expf(-v));
  hb[i] = v * hb2[i];
}

// Residual: x += xb2
__kernel void residual_k(kernel_arg_t* __UNIFORM__ arg) {
  auto x   = reinterpret_cast<float*>(arg->x_addr);
  auto xb2 = reinterpret_cast<const float*>(arg->out_addr);
  int total = arg->n * arg->batch;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  x[i] += xb2[i];
}

// Scatter the freshly computed k/v columns into each sequence's KV cache.
// The projections write [kv_dim][batch]; the cache is per-sequence contiguous.
__kernel void kv_store_k(kernel_arg_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<const float*>(arg->x_addr);   // [kv_dim][batch]
  auto kc  = reinterpret_cast<float*>(arg->key_cache_addr);
  auto pos = reinterpret_cast<const int32_t*>(arg->pos_addr);
  int kv_dim = arg->kv_dim, B = arg->batch, loff = arg->loff;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_dim * B) return;
  int d = idx / B, b = idx % B;
  size_t seq_kv = (size_t)arg->kv_seq_stride * b;
  kc[seq_kv + loff + (size_t)pos[b] * kv_dim + d] = src[(size_t)d * B + b];
}
