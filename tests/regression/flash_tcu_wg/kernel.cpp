#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <cmath>
#include <type_traits>
#include "common.h"
#include "fpcvt.h"

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
using input_t  = ctx::input_t;
using output_t = ctx::output_t;

static_assert(std::is_same_v<output_t, float>, "OTYPE must be fp32");

static inline size_t bits_from(output_t x) { size_t u = 0; __builtin_memcpy(&u, &x, sizeof(x)); return u; }
static inline output_t from_bits(size_t u) { output_t x; __builtin_memcpy(&x, &u, sizeof(x)); return x; }

static inline output_t warp_reduce_max(output_t x, uint32_t warp_size) {
  int clamp = warp_size - 1, segmask = ~clamp & 0x3f;
  for (uint32_t off = warp_size >> 1; off > 0; off >>= 1) {
    output_t y = from_bits(vx_shfl_bfly(bits_from(x), off, clamp, segmask));
    x = (y > x) ? y : x;
  }
  return x;
}

static inline output_t warp_reduce_sum(output_t x, uint32_t warp_size) {
  int clamp = warp_size - 1, segmask = ~clamp & 0x3f;
  for (uint32_t off = warp_size >> 1; off > 0; off >>= 1)
    x += from_bits(vx_shfl_bfly(bits_from(x), off, clamp, segmask));
  return x;
}

// fp32 -> ITYPE pack
template <typename T> struct always_false : std::false_type {};
template <typename It = vt::ITYPE>
static inline input_t pack_input(output_t v) {
  if      constexpr (std::is_same_v<It, vt::tf32>) return (input_t)fpcvt::f32_to_tf32(v);
  else if constexpr (std::is_same_v<It, vt::fp16>) return (input_t)fpcvt::f32_to_fp16(v);
  else if constexpr (std::is_same_v<It, vt::bf16>) return (input_t)fpcvt::f32_to_bf16(v);
  else static_assert(always_false<It>::value, "no fpcvt pack for this ITYPE");
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
    if constexpr (Scaled) v *= w_row[row];
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
  auto lm       = __local_mem();
  auto local_Q  = reinterpret_cast<input_t*>(lm);
  auto local_K  = local_Q + Br * head_dim;
  auto local_V  = local_K + dt * Bc;
  auto local_P  = local_V + Bc * dt;                              // holds packed P (input_t)
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
      for (uint32_t i = cta_thread; i < dt * Bc; i += cta_size) {
        auto k = i / Bc, n = i % Bc;
        auto off = ((k / ctx::tileK) * s_ntiles + (n / ctx::xtileN)) * b_tile_elems
                 + ctx::b_blockmajor_idx(k % ctx::tileK, n % ctx::xtileN);
        local_K[off] = K[(j + n) * head_dim + (h + k)];
      }
      __syncthreads();

      for (uint32_t n = 0; n < s_ntiles; ++n) {
        ctx::fragment_acc frag;
        // First h-chunk starts accumulation, later ones resume the partial S
        // already in LMEM
        if (h == 0) ctx::fill_fragment(frag, output_t(0));
        else        load_acc_scaled<false>(frag, local_S + warp_row * Bc + n * ctx::xtileN, Bc, nullptr);

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
    for (uint32_t rr = 0; rr < ctx::xtileM; ++rr) {
      const auto r = warp_row + rr;
      const output_t* S_row = local_S + r * Bc;
      // P in same layout as Q: [kt][Br][tileK].
      input_t* P_base = local_P + r * ctx::tileK;

      output_t lmax = -INFINITY;
      for (uint32_t c = lane; c < Bc; c += nt_lanes) {
        output_t v = S_row[c];
        lmax = (v > lmax) ? v : lmax;
      }
      const output_t rowmax = warp_reduce_max(lmax, nt_lanes);

      const output_t m_old = local_m[r];
      const output_t new_m = (m_old > rowmax) ? m_old : rowmax;
      const output_t w_old = expf(m_old - new_m);

      output_t p_sum = output_t(0);
      for (uint32_t c = lane; c < Bc; c += nt_lanes) {
        output_t p = expf(S_row[c] - new_m);
        p_sum += p;
        P_base[(c / ctx::tileK) * (Br * ctx::tileK) + (c % ctx::tileK)] = pack_input(p);
      }
      const output_t rowsum = warp_reduce_sum(p_sum, nt_lanes);

      // After butterfly reduces, rowmax/rowsum are warp-uniform, so every
      // lane computes and stores same value
      local_m[r] = new_m;
      local_l[r] = w_old * local_l[r] + rowsum;
      local_w[r] = w_old;
    }
    __syncthreads();

    // O = diag(w_old)*O + P*V
    for (uint32_t h = 0; h < head_dim; h += dt) {
      // Stage V: B[k][n] = V[(j+k)][h+n]
      for (uint32_t i = cta_thread; i < Bc * dt; i += cta_size) {
        auto k = i / dt, n = i % dt;
        auto off = ((k / ctx::tileK) * o_ntiles + (n / ctx::xtileN)) * b_tile_elems
                     + ctx::b_blockmajor_idx(k % ctx::tileK, n % ctx::xtileN);
        local_V[off] = V[(j + k) * head_dim + (h + n)];
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
