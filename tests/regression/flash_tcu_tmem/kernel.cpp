#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <cmath>
#include <type_traits>
#include "common.h"
#include "fpcvt.h"

namespace vt = vortex::tensor;
using ctx = vt::umma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, UMMA_NRC>;
using input_t  = ctx::input_t;
using output_t = ctx::output_t;

static_assert(std::is_same_v<output_t, float>, "OTYPE must be fp32");

#ifndef UMMA_A_TMEM
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
#endif

// fp32 -> ITYPE pack
template <typename T> struct always_false : std::false_type {};
template <typename It = vt::ITYPE>
static inline input_t pack_input(output_t v) {
  if      constexpr (std::is_same_v<It, vt::tf32>) return (input_t)fpcvt::f32_to_tf32(v);
  else if constexpr (std::is_same_v<It, vt::fp16>) return (input_t)fpcvt::f32_to_fp16(v);
  else if constexpr (std::is_same_v<It, vt::bf16>) return (input_t)fpcvt::f32_to_bf16(v);
  else static_assert(always_false<It>::value, "no fpcvt pack for this ITYPE");
}

// This warp's accumulator tile is TMEM lanes [rank*xtileM, +xtileM) x columns
// [handle, +xtileN). Stride by tmem_active_lanes, not NT: above the
// NT<=xtileM crossover only the first xtileM lanes are ever live, and the
// row this pass covers must follow the same stride the masking uses.
static inline uint32_t tmem_lane_base(uint32_t rank, uint32_t r) {
  return rank * ctx::xtileM + r * ctx::tmem_active_lanes;
}

#ifndef UMMA_A_TMEM
// LMEM tile -> TMEM, with the online-softmax rescale folded in on the way
template <bool Scaled>
static inline void tmem_load_tile(uint32_t handle, uint32_t rank, const output_t *tile,
                                  uint32_t ldm, const output_t *w_row) {
  auto lane = threadIdx.x;
  size_t saved_mask;
  if (!ctx::tmem_lanes_enter(saved_mask))
    return;
  for (uint32_t r = 0; r < ctx::tmem_rows_per_thread; ++r) {
    auto row = r * ctx::tmem_active_lanes + lane;
    auto src = tile + row * ldm;
    auto lane_base = tmem_lane_base(rank, r);
    for (uint32_t c = 0; c < ctx::xtileN; ++c) {
      output_t v = src[c];
      if constexpr (Scaled) v *= w_row[row];
      vt::vx_tmem_st(vt::vx_make_tmem_addr(lane_base, handle + c),
                     vt::detail::bit_cast<uint32_t>(v));
    }
  }
  ctx::tmem_lanes_exit(saved_mask);
}

// TMEM -> LMEM tile
static inline void tmem_store_tile(uint32_t handle, uint32_t rank, output_t *tile, uint32_t ldm) {
  auto lane = threadIdx.x;
  size_t saved_mask;
  if (!ctx::tmem_lanes_enter(saved_mask))
    return;
  for (uint32_t r = 0; r < ctx::tmem_rows_per_thread; ++r) {
    auto row = r * ctx::tmem_active_lanes + lane;
    auto dst = tile + row * ldm;
    auto lane_base = tmem_lane_base(rank, r);
    for (uint32_t c = 0; c < ctx::xtileN; ++c)
      dst[c] = vt::detail::bit_cast<output_t>(
          vt::vx_tmem_ld(vt::vx_make_tmem_addr(lane_base, handle + c)));
  }
  ctx::tmem_lanes_exit(saved_mask);
}
#else
// Online softmax over this warp's S rows, read from TMEM, with P packed back
// over S in place as the TS A operand: word j holds keys [epw*j, epw*j+epw).
// Word j is written only after keys up to epw*j+epw-1 have been read, so the
// overwrite never clobbers an S value still to be read. The thread owns the
// row, so it also applies the diag(w_old) rescale to O's row in TMEM.
static inline void softmax_tmem(uint32_t s_col, uint32_t o_col, uint32_t rank,
                                uint32_t Bc, uint32_t head_dim,
                                output_t *m, output_t *l) {
  using raw_t = std::conditional_t<sizeof(input_t) == 2, uint16_t, uint32_t>;
  constexpr uint32_t epw    = ctx::a_elems_per_word;
  constexpr uint32_t e_bits = 8 * sizeof(input_t);
  auto lane = threadIdx.x;
  size_t saved_mask;
  if (!ctx::tmem_lanes_enter(saved_mask))
    return;
  for (uint32_t r = 0; r < ctx::tmem_rows_per_thread; ++r) {
    auto row = r * ctx::tmem_active_lanes + lane;
    auto lane_base = tmem_lane_base(rank, r);

    output_t rowmax = -INFINITY;
    for (uint32_t c = 0; c < Bc; ++c) {
      output_t v = vt::detail::bit_cast<output_t>(
          vt::vx_tmem_ld(vt::vx_make_tmem_addr(lane_base, s_col + c)));
      rowmax = (v > rowmax) ? v : rowmax;
    }
    const output_t m_old = m[row];
    const output_t new_m = (m_old > rowmax) ? m_old : rowmax;
    const output_t w_old = expf(m_old - new_m);

    output_t rowsum = output_t(0);
    for (uint32_t j = 0; j < Bc / epw; ++j) {
      uint32_t word = 0;
      for (uint32_t e = 0; e < epw; ++e) {
        output_t s = vt::detail::bit_cast<output_t>(
            vt::vx_tmem_ld(vt::vx_make_tmem_addr(lane_base, s_col + j * epw + e)));
        output_t p = expf(s - new_m);
        rowsum += p;
        word |= uint32_t(vt::detail::bit_cast<raw_t>(pack_input(p))) << (e * e_bits);
      }
      vt::vx_tmem_st(vt::vx_make_tmem_addr(lane_base, s_col + j), word);
    }

    for (uint32_t c = 0; c < head_dim; ++c) {
      auto addr = vt::vx_make_tmem_addr(lane_base, o_col + c);
      output_t o = vt::detail::bit_cast<output_t>(vt::vx_tmem_ld(addr));
      vt::vx_tmem_st(addr, vt::detail::bit_cast<uint32_t>(o * w_old));
    }

    m[row] = new_m;
    l[row] = w_old * l[row] + rowsum;
  }
  ctx::tmem_lanes_exit(saved_mask);
}

// Normalize this warp's O rows by the softmax denominator, TMEM -> global
static inline void store_o_tmem(uint32_t o_col, uint32_t rank, uint32_t head_dim,
                                const output_t *l, output_t *O) {
  auto lane = threadIdx.x;
  size_t saved_mask;
  if (!ctx::tmem_lanes_enter(saved_mask))
    return;
  for (uint32_t r = 0; r < ctx::tmem_rows_per_thread; ++r) {
    auto row = r * ctx::tmem_active_lanes + lane;
    auto lane_base = tmem_lane_base(rank, r);
    auto dst = O + row * head_dim;
    for (uint32_t c = 0; c < head_dim; ++c)
      dst[c] = vt::detail::bit_cast<output_t>(
          vt::vx_tmem_ld(vt::vx_make_tmem_addr(lane_base, o_col + c))) / l[row];
  }
  ctx::tmem_lanes_exit(saved_mask);
}
#endif

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
  // umma_sync derives its TMEM lane band from the CTA rank
  const auto warp_rank  = (uint32_t)vx_cta_rank();
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
#ifdef UMMA_A_TMEM
  // S, P and O live in TMEM
  auto local_m  = reinterpret_cast<output_t*>(local_K + dt * Bc);
  auto local_l  = local_m + Br;
#else
  auto local_P  = local_K + dt * Bc;                              // holds packed P (input_t)
  auto local_S  = reinterpret_cast<output_t*>(local_P + Br * Bc); // holds S (output_t)
  auto local_O  = local_S + Br * Bc;
  auto local_m  = local_O + Br * head_dim;
  auto local_l  = local_m + Br;
  auto local_w  = local_l + Br;
#endif

  // B-operand tile counts (block-major tiles of tileK x xtileN)
  constexpr auto b_tile_elems = ctx::tileK * ctx::xtileN;
  const auto s_ntiles = Bc / ctx::xtileN;
  const auto o_ntiles = dt / ctx::xtileN;
  const auto s_ktiles = dt / ctx::tileK;
  const auto o_ktiles = Bc / ctx::tileK;

#ifdef UMMA_A_TMEM
  // One allocation, since a CTA holds one live range: S, whose n-tiles
  // accumulate in place with P packed over them, then O, resident for the
  // whole KV loop.
  auto handle = vt::vx_tmem_alloc(Bc + head_dim);
  const auto s_col = handle;
  const auto o_col = handle + Bc;
#else
  // One xtileN-wide accumulator tile, reused by every n-tile of both GEMMs.
  auto handle = vt::vx_tmem_alloc(ctx::xtileN);
#endif

  // Q is staged as dense per-k-tile tiles [kt][Br][tileK] so the A-operand
  // descriptor can use ldm == tileK
  for (uint32_t i = cta_thread; i < Br * head_dim; i += cta_size) {
    auto r = i / head_dim, c = i % head_dim;
    auto kt = c / ctx::tileK, cin = c % ctx::tileK;
    local_Q[kt * (Br * ctx::tileK) + r * ctx::tileK + cin] = Q[(row0 + r) * head_dim + c];
#ifndef UMMA_A_TMEM
    local_O[r * head_dim + c] = output_t(0);
#endif
  }
#ifdef UMMA_A_TMEM
  for (uint32_t n = 0; n < head_dim / ctx::xtileN; ++n)
    ctx::fill_tmem(o_col + n * ctx::xtileN, output_t(0));
#endif
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
  #ifdef UMMA_A_TMEM
    // S stays resident across h-chunks, so it is seeded once per KV block.
    // Seeding is per-warp and unsynchronized.
    for (uint32_t n = 0; n < s_ntiles; ++n)
      ctx::fill_tmem(s_col + n * ctx::xtileN, output_t(0));
    __syncthreads();
  #endif
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
      #ifdef UMMA_A_TMEM
        const auto s_acc = s_col + n * ctx::xtileN;
      #else
        const auto s_acc = handle;
        // umma_sync accumulates in place, so the tile is seeded
        if (h == 0) ctx::fill_tmem(handle, output_t(0));
        else        tmem_load_tile<false>(handle, warp_rank,
                                          local_S + warp_row * Bc + n * ctx::xtileN, Bc, nullptr);
        // Seeding is per-warp and unsynchronized
        __syncthreads();
      #endif

        for (uint32_t k = 0; k < s_ktiles; ++k) {
          auto desc_a = vt::vx_make_smem_desc(
              local_Q + ((h / ctx::tileK) + k) * (Br * ctx::tileK) + warp_row * ctx::tileK,
              ctx::tileK * sizeof(input_t));
          auto desc_b = vt::vx_make_smem_desc(local_K + (k * s_ntiles + n) * b_tile_elems, 0);
          ctx::umma_sync(desc_a, desc_b, s_acc);
          __syncthreads();
        }
      #ifndef UMMA_A_TMEM
        tmem_store_tile(handle, warp_rank, local_S + warp_row * Bc + n * ctx::xtileN, Bc);
      #endif
      }
      __syncthreads();
    }

    // softmax, S -> P
  #ifdef UMMA_A_TMEM
    // Thread-per-row: TMEM_LD maps thread t to this warp's lane t
    softmax_tmem(s_col, o_col, warp_rank, Bc, head_dim, local_m + warp_row, local_l + warp_row);
  #else
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

      // After butterfly reduces, rowmax/rowsum are warp-uniform
      local_m[r] = new_m;
      local_l[r] = w_old * local_l[r] + rowsum;
      local_w[r] = w_old;
    }
  #endif
    // The P and rescaled-O TMEM_STs must land before umma_sync reads them
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
      #ifdef UMMA_A_TMEM
        const auto o_acc = o_col + h + n * ctx::xtileN;
      #else
        const auto o_acc = handle;
        tmem_load_tile<true>(handle, warp_rank,
                             local_O + warp_row * head_dim + h + n * ctx::xtileN,
                             head_dim, local_w + warp_row);
        __syncthreads();
      #endif

        for (uint32_t k = 0; k < o_ktiles; ++k) {
        #ifdef UMMA_A_TMEM
          auto desc_a = vt::vx_make_tmem_ref(s_col + k * ctx::a_tmem_words);
        #else
          auto desc_a = vt::vx_make_smem_desc(
              local_P + k * (Br * ctx::tileK) + warp_row * ctx::tileK,
              ctx::tileK * sizeof(input_t));
        #endif
          auto desc_b = vt::vx_make_smem_desc(local_V + (k * o_ntiles + n) * b_tile_elems, 0);
          ctx::umma_sync(desc_a, desc_b, o_acc);
          __syncthreads();
        }
      #ifndef UMMA_A_TMEM
        tmem_store_tile(handle, warp_rank,
                        local_O + warp_row * head_dim + h + n * ctx::xtileN, head_dim);
      #endif
      }
      __syncthreads();
    }
  }

  // Normalize by softmax denominator and write back
#ifdef UMMA_A_TMEM
  store_o_tmem(o_col, warp_rank, head_dim, local_l + warp_row, O + (row0 + warp_row) * head_dim);
#else
  for (uint32_t i = cta_thread; i < Br * head_dim; i += cta_size) {
    auto r = i / head_dim, c = i % head_dim;
    O[(row0 + r) * head_dim + c] = local_O[r * head_dim + c] / local_l[r];
  }
#endif

  __syncthreads();
  vt::vx_tmem_dealloc(handle);
}
