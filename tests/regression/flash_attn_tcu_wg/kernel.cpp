#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <cmath>
#include <type_traits>
#include "common.h"
#include "fpcvt.h"
#include <flash_attn_device.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
using input_t  = ctx::input_t;
using output_t = ctx::output_t;

static_assert(std::is_same_v<output_t, float>, "OTYPE must be fp32");

using flash_attn::warp_reduce_max;
using flash_attn::warp_reduce_sum;
using flash_attn::expf_fast;

// Lanes per softmax row, when a warp's xtileM rows spread evenly over its
// lanes in power-of-two groups; 0 falls back to one row at a time.
constexpr uint32_t kSmLanesPerRowRaw =
    (VX_CFG_NUM_THREADS % ctx::xtileM == 0) ? (VX_CFG_NUM_THREADS / ctx::xtileM) : 0;
constexpr uint32_t kSmLanesPerRow =
    (kSmLanesPerRowRaw != 0 && (kSmLanesPerRowRaw & (kSmLanesPerRowRaw - 1)) == 0) ? kSmLanesPerRowRaw : 0;
// 4-byte copies for staging 16-bit operands.
inline uint32_t ld32(const void* p) {
  uint32_t v;
  __builtin_memcpy(&v, __builtin_assume_aligned(p, 4), 4);
  return v;
}
inline void st32(void* p, uint32_t v) {
  __builtin_memcpy(__builtin_assume_aligned(p, 4), &v, 4);
}

// Element (k, n) of a B operand staged as block-major tiles of tileK x xtileN.
// Inside a block K runs fastest, so k and k + 1 (k even) share one word.
inline uint32_t b_off(uint32_t k, uint32_t n, uint32_t ntiles) {
  return ((k / ctx::tileK) * ntiles + (n / ctx::xtileN)) * (ctx::tileK * ctx::xtileN)
       + ctx::b_blockmajor_idx(k % ctx::tileK, n % ctx::xtileN);
}

// Wide staging moves 16-bit operands as 32-bit words.
constexpr bool kWideStage = (sizeof(input_t) == 2) && (ctx::b_blk_k % 2 == 0);

// fp32 -> ITYPE pack
template <typename T> struct always_false : std::false_type {};
template <typename It = vt::ITYPE>
static inline input_t pack_input(output_t v) {
  if constexpr (std::is_same_v<It, vt::tf32>) {
    return (input_t)fpcvt::f32_to_tf32(v);
  } else if constexpr (std::is_same_v<It, vt::fp16>) {
    return (input_t)fpcvt::f32_to_fp16(v);
  } else if constexpr (std::is_same_v<It, vt::bf16>) {
    return (input_t)fpcvt::f32_to_bf16(v);
  } else {
    static_assert(always_false<It>::value, "no fpcvt pack for this ITYPE");
  }
}

// Pack a softmax weight p (0 <= p <= 1, finite) into ITYPE, rounding to
// nearest even like the host reference. The general fpcvt routines also handle
// signs, NaN, infinity and overflow, which p never has, at about 50 integer
// operations per value. These do the same rounding in a handful.
template <typename It = vt::ITYPE>
static inline input_t pack_p(output_t v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, 4);
  if constexpr (std::is_same_v<It, vt::fp16>) {
#ifdef VX_CFG_EXT_ZFHMIN_ENABLE
    // Hardware conversion (fcvt.h.s), round to nearest even.
    _Float16 h = (_Float16)v;
    uint16_t hb;
    __builtin_memcpy(&hb, &h, 2);
    return (input_t)hb;
#else
    // Normal fp16 (p >= 2^-14): drop 13 mantissa bits with round-to-nearest-
    // even, then move the exponent bias from 127 to 15.
    const uint32_t norm = ((b + 0xFFFu + ((b >> 13) & 1u)) >> 13) - (112u << 10);
    // Below 2^-14 the fp16 value is a multiple of 2^-24: round p * 2^24 to an
    // integer with the float adder (1024 rounds up to the smallest normal).
    const float t = v * 16777216.0f + 8388608.0f;
    uint32_t tb;
    __builtin_memcpy(&tb, &t, 4);
    const uint32_t sub = tb - 0x4B000000u;
    const uint32_t is_sub = (b < 0x38800000u);
    return (input_t)((sub & (0u - is_sub)) | (norm & (is_sub - 1u)));
#endif
  } else if constexpr (std::is_same_v<It, vt::bf16>) {
    // bf16 keeps fp32's exponent range: round the low 16 bits away.
    return (input_t)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
  } else {
    // tf32 keeps the exact generic routine.
    return pack_input<It>(v);
  }
}

// Inverse of ctx::store_matrix_sync, with per-row scale applied
template <bool Scaled, typename Frag>
static inline void load_acc_scaled(Frag &dst, const output_t *tile, uint32_t ldm,
                                   const output_t *w_row) {
  auto lane = threadIdx.x;
  auto base_row = lane / ctx::tcN;
  auto base_col = lane % ctx::tcN;
  vt::detail::unroll_for<Frag::NR>([&](auto r) {
    auto bm  = r % ctx::m_steps;
    auto bn  = r / ctx::m_steps;
    auto row = base_row + bm * ctx::tcM;
    output_t v = tile[row * ldm + bn * ctx::tcN + base_col];
    if constexpr (Scaled) {
      v *= w_row[row];
    }
    dst.data[r] = v;
  });
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto Q = reinterpret_cast<const input_t*>(arg->Q_addr);
  auto K = reinterpret_cast<const input_t*>(arg->K_addr);
  auto V = reinterpret_cast<const input_t*>(arg->V_addr);
  auto O = reinterpret_cast<output_t*>(arg->O_addr);

  const auto seq_len  = arg->seq_len;
  const auto head_dim = arg->head_dim;
  const auto dt       = arg->head_dim_tile;
  const auto Br       = arg->block_size_r;
  const auto Bc       = arg->block_size_c;

  const auto lane       = threadIdx.x;
  const auto nt_lanes   = blockDim.x;
  const auto warp_rank  = threadIdx.y;
  const auto cta_warps  = blockDim.y;
  const auto cta_thread = warp_rank * nt_lanes + lane;
  const auto cta_size   = cta_warps * nt_lanes;

  const auto row0 = blockIdx.x * Br;

  // LMEM layout
  //
  // K and V share one dt*Bc buffer. They are never live at the same time,
  // so alias them.
  auto lm       = __local_mem();
  auto local_Q  = reinterpret_cast<input_t*>(lm);
  auto local_K  = local_Q + Br * head_dim;
  auto local_V  = local_K;                                        // aliased
  auto local_P  = local_K + dt * Bc;                              // holds packed P (input_t)
  auto local_S  = reinterpret_cast<output_t*>(local_P + Br * Bc); // holds S (output_t)
  auto local_O  = local_S + Br * Bc;
  auto local_m  = local_O + Br * head_dim;
  auto local_l  = local_m + Br;
  auto local_w  = local_l + Br;

  // B-operand tile counts (block-major tiles of tileK x xtileN)
  constexpr auto b_tile_elems = ctx::tileK * ctx::xtileN;
  const auto s_ntiles = Bc / ctx::xtileN;
  const auto o_ntiles = dt / ctx::xtileN;
  const auto s_ktiles = dt / ctx::tileK;
  const auto o_ktiles = Bc / ctx::tileK;

  // Q is staged as dense per-k-tile tiles [kt][Br][tileK] so the A-operand
  // descriptor can use ldm == tileK
  for (uint32_t i = cta_thread; i < Br * head_dim; i += cta_size) {
    auto r = i / head_dim, c = i % head_dim;
    auto kt = c / ctx::tileK, cin = c % ctx::tileK;
    local_Q[kt * (Br * ctx::tileK) + r * ctx::tileK + cin] = Q[(row0 + r) * head_dim + c];
    local_O[r * head_dim + c] = output_t(0);
  }
  for (uint32_t r = cta_thread; r < Br; r += cta_size) {
    local_m[r] = -INFINITY;
    local_l[r] = output_t(0);
  }
  __syncthreads();

  // this warp's slice of the A operands / output rows
  const auto warp_row = warp_rank * ctx::xtileM;

  // KV-block loop
  for (uint32_t j = 0; j < seq_len; j += Bc) {

    // S = Q * K^T
    for (uint32_t h = 0; h < head_dim; h += dt) {
      // Stage K^T: B[k][n] = K[(j+n)][h+k]
      if constexpr (kWideStage) {
        // Threads walk along K's rows a word (two k) at a time, so a warp
        // reads contiguous memory, and each word lands whole in the B layout.
        // (n, p) steps by cta_size words without a division per element.
        const uint32_t pairs = dt / 2;
        const uint32_t dn = cta_size / pairs, dp = cta_size % pairs;
        uint32_t n = cta_thread / pairs, p = cta_thread % pairs;
        for (uint32_t i = cta_thread; i < Bc * pairs; i += cta_size) {
          const uint32_t k = 2 * p;
          st32(local_K + b_off(k, n, s_ntiles), ld32(K + (j + n) * head_dim + (h + k)));
          p += dp;
          n += dn;
          const uint32_t wrap = (p >= pairs);
          p -= pairs & (0u - wrap);
          n += wrap;
        }
      } else {
      for (uint32_t i = cta_thread; i < dt * Bc; i += cta_size) {
        auto k = i / Bc, n = i % Bc;
        auto off = ((k / ctx::tileK) * s_ntiles + (n / ctx::xtileN)) * b_tile_elems
                 + ctx::b_blockmajor_idx(k % ctx::tileK, n % ctx::xtileN);
        local_K[off] = K[(j + n) * head_dim + (h + k)];
      }
      }
      __syncthreads();

      for (uint32_t n = 0; n < s_ntiles; ++n) {
        ctx::fragment_acc frag;
        // First h-chunk starts accumulation, later ones resume the partial S
        // already in LMEM
        if (h == 0) {
          ctx::fill_fragment(frag, output_t(0));
        } else {
          load_acc_scaled<false>(frag, local_S + warp_row * Bc + n * ctx::xtileN, Bc, nullptr);
        }

        for (uint32_t k = 0; k < s_ktiles; ++k) {
          auto desc_a = vt::vx_make_smem_desc(
              local_Q + ((h / ctx::tileK) + k) * (Br * ctx::tileK) + warp_row * ctx::tileK,
              ctx::tileK * sizeof(input_t));
          auto desc_b = vt::vx_make_smem_desc(local_K + (k * s_ntiles + n) * b_tile_elems, 0);
          ctx::wgmma_sync(frag, desc_a, desc_b, frag);
          // Required because warps share one B buffer
          __syncthreads();
        }
        ctx::store_matrix_sync(local_S + warp_row * Bc + n * ctx::xtileN, frag, Bc);
      }
      __syncthreads();
    }

    // softmax, S -> P
    if (kSmLanesPerRow != 0 && Bc % (kSmLanesPerRow ? kSmLanesPerRow : 1u) == 0) {
      // Each row of this warp's slice belongs to kSmLanesPerRow neighbouring
      // lanes, which split its columns. All of the warp's rows run at once, and
      // a row's max and sum take log2(kSmLanesPerRow) shuffles instead of
      // log2(warp size) for each row in turn.
      const uint32_t rr  = lane / kSmLanesPerRow;
      const uint32_t sub = lane % kSmLanesPerRow;
      const uint32_t r   = warp_row + rr;
      const uint32_t per_lane = Bc / kSmLanesPerRow;
      const output_t* S_row = local_S + r * Bc;
      input_t* P_base = local_P + r * ctx::tileK;

      output_t lmax = -INFINITY;
      for (uint32_t i = 0; i < per_lane; ++i) {
        lmax = __builtin_fmaxf(lmax, S_row[i * kSmLanesPerRow + sub]);
      }
      const output_t rowmax = warp_reduce_max(lmax, kSmLanesPerRow);

      const output_t m_old = local_m[r];
      const output_t new_m = __builtin_fmaxf(m_old, rowmax);
      const output_t w_old = expf_fast(m_old - new_m);

      output_t p_sum = output_t(0);
      for (uint32_t i = 0; i < per_lane; ++i) {
        const uint32_t c = i * kSmLanesPerRow + sub;
        output_t p = expf_fast(S_row[c] - new_m);
        p_sum += p;
        P_base[(c / ctx::tileK) * (Br * ctx::tileK) + (c % ctx::tileK)] = pack_p(p);
      }
      const output_t rowsum = warp_reduce_sum(p_sum, kSmLanesPerRow);

      // The row's lanes hold the same values; they all store them.
      local_m[r] = new_m;
      local_l[r] = w_old * local_l[r] + rowsum;
      local_w[r] = w_old;
    } else {
    for (uint32_t rr = 0; rr < ctx::xtileM; ++rr) {
      const auto r = warp_row + rr;
      const output_t* S_row = local_S + r * Bc;
      // P in same layout as Q: [kt][Br][tileK].
      input_t* P_base = local_P + r * ctx::tileK;

      output_t lmax = -INFINITY;
      for (uint32_t c = lane; c < Bc; c += nt_lanes) {
        output_t v = S_row[c];
        lmax = __builtin_fmaxf(lmax, v);
      }
      const output_t rowmax = warp_reduce_max(lmax, nt_lanes);

      const output_t m_old = local_m[r];
      const output_t new_m = __builtin_fmaxf(m_old, rowmax);
      const output_t w_old = expf_fast(m_old - new_m);

      output_t p_sum = output_t(0);
      for (uint32_t c = lane; c < Bc; c += nt_lanes) {
        output_t p = expf_fast(S_row[c] - new_m);
        p_sum += p;
        P_base[(c / ctx::tileK) * (Br * ctx::tileK) + (c % ctx::tileK)] = pack_p(p);
      }
      const output_t rowsum = warp_reduce_sum(p_sum, nt_lanes);

      // After butterfly reduces, rowmax/rowsum are warp-uniform, so every
      // lane computes and stores same value
      local_m[r] = new_m;
      local_l[r] = w_old * local_l[r] + rowsum;
      local_w[r] = w_old;
    }
    }
    __syncthreads();

    // O = diag(w_old)*O + P*V
    for (uint32_t h = 0; h < head_dim; h += dt) {
      // Stage V: B[k][n] = V[(j+k)][h+n]
      if constexpr (kWideStage) {
        // 2x2 blocks: one word from each of rows k and k+1 of V (columns n,
        // n+1), regrouped into the B layout's (k, k+1) words for n and n+1.
        const uint32_t pairs = dt / 2;
        const uint32_t dk = cta_size / pairs, dp = cta_size % pairs;
        uint32_t kp = cta_thread / pairs, p = cta_thread % pairs;
        for (uint32_t i = cta_thread; i < (Bc / 2) * pairs; i += cta_size) {
          const uint32_t k = 2 * kp, n = 2 * p;
          const input_t* src = V + (j + k) * head_dim + (h + n);
          const uint32_t r0 = ld32(src);             // V[k][n],   V[k][n+1]
          const uint32_t r1 = ld32(src + head_dim);  // V[k+1][n], V[k+1][n+1]
          st32(local_V + b_off(k, n, o_ntiles),     (r0 & 0xFFFFu) | (r1 << 16));
          st32(local_V + b_off(k, n + 1, o_ntiles), (r0 >> 16) | (r1 & 0xFFFF0000u));
          p += dp;
          kp += dk;
          const uint32_t wrap = (p >= pairs);
          p -= pairs & (0u - wrap);
          kp += wrap;
        }
      } else {
      for (uint32_t i = cta_thread; i < Bc * dt; i += cta_size) {
        auto k = i / dt, n = i % dt;
        auto off = ((k / ctx::tileK) * o_ntiles + (n / ctx::xtileN)) * b_tile_elems
                     + ctx::b_blockmajor_idx(k % ctx::tileK, n % ctx::xtileN);
        local_V[off] = V[(j + k) * head_dim + (h + n)];
      }
      }
      __syncthreads();

      for (uint32_t n = 0; n < o_ntiles; ++n) {
        ctx::fragment_acc frag;
        load_acc_scaled<true>(frag, local_O + warp_row * head_dim + h + n * ctx::xtileN,
                        head_dim, local_w + warp_row);

        for (uint32_t k = 0; k < o_ktiles; ++k) {
          auto desc_a = vt::vx_make_smem_desc(
              local_P + k * (Br * ctx::tileK) + warp_row * ctx::tileK,
              ctx::tileK * sizeof(input_t));
          auto desc_b = vt::vx_make_smem_desc(local_V + (k * o_ntiles + n) * b_tile_elems, 0);
          ctx::wgmma_sync(frag, desc_a, desc_b, frag);
          // Required because warps share one B buffer
          __syncthreads();
        }
        ctx::store_matrix_sync(local_O + warp_row * head_dim + h + n * ctx::xtileN, frag, head_dim);
      }
      __syncthreads();
    }
  }

  // Normalize by softmax denominator and write back
  for (uint32_t i = cta_thread; i < Br * head_dim; i += cta_size) {
    auto r = i / head_dim, c = i % head_dim;
    O[(row0 + r) * head_dim + c] = local_O[r * head_dim + c] / local_l[r];
  }
}
