// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tensor_cfg.h>
#include <vx_intrinsics.h>

namespace vortex {
namespace tensor {

enum mem_layout {
  row_major,
  col_major
};

// Shared-memory matrix descriptor (32-bit packed):
//   bits[31:16] = row stride in bytes (leading dimension)
//   bits[15:0]  = byte offset from local memory base (max 64 KB)
struct smem_matrix_desc {
  uint32_t value;
};

// Build a smem descriptor from a pointer and row stride in bytes.
static __attribute__((always_inline)) smem_matrix_desc vx_make_smem_desc(const void* ptr, uint32_t leading_bytes) {
  size_t lmem_base = csr_read(VX_CSR_LOCAL_MEM_BASE);
  uint32_t offset = static_cast<uint32_t>(static_cast<size_t>(reinterpret_cast<uintptr_t>(ptr)) - lmem_base);
  return {((leading_bytes << 16) | offset)};
}

namespace detail {

  template <typename F, std::size_t... Is>
  __attribute__((always_inline))
  constexpr void unroll_for_impl(std::index_sequence<Is...>, F&& f) {
    (f(std::integral_constant<std::size_t, Is>{}), ...);
  }

  template <std::size_t N, typename F>
  __attribute__((always_inline))
  constexpr void unroll_for(F&& f) {
    unroll_for_impl(std::make_index_sequence<N>{}, std::forward<F>(f));
  }

  template <typename T>
  struct raw_unsigned {
    using type = std::conditional_t<(sizeof(T) == 1), uint8_t,
      std::conditional_t<(sizeof(T) == 2), uint16_t,
        std::conditional_t<(sizeof(T) == 4), uint32_t,
          uint64_t>>>;
  };
  template <typename T>
  using raw_unsigned_t = typename raw_unsigned<T>::type;

  template <typename T, typename D>
  struct data_accessor_t {
    using Type = typename T::dtype;

    static __attribute__((always_inline)) D bit_fill(Type src) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      if constexpr (std::is_same_v<Type, D>) {
        return src; // passthrough
      } else {
        constexpr uint32_t count = sizeof(D) / sizeof(Type);
        constexpr uint32_t bits = 8 * sizeof(Type);
        using US = raw_unsigned_t<Type>;
        using UD = raw_unsigned_t<D>;
        auto src_u = *reinterpret_cast<const US*>(&src); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        UD result_u(0);
        detail::unroll_for<count>([&](auto i) {
          result_u |= (src_d << (i * bits));
        });
        return *reinterpret_cast<const D*>(&result_u);
      }
    }

    static __attribute__((always_inline)) D pack_row(const Type *base, uint32_t ldm) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      if constexpr (sizeof(Type) == 1 && sizeof(D) == 4) {
        // 4 × 1-byte strided loads → single pack-load byte instruction
        return vx_packlb_f(base, ldm);
      } else if constexpr (sizeof(Type) == 2 && sizeof(D) == 4) {
        // 2 × 2-byte strided loads → single pack-load halfword instruction
        return vx_packlh_f(base, ldm * 2u);
      } else {
      constexpr uint32_t count = sizeof(D) / sizeof(Type);
      constexpr uint32_t bits = 8 * sizeof(Type);
      using US = raw_unsigned_t<Type>;
      using UD = raw_unsigned_t<D>;
      UD result_u(0);
      detail::unroll_for<count>([&](auto i) {
        auto src_u = *reinterpret_cast<const US*>(base); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        result_u |= (src_d << (i * bits));
        base += ldm; // next row
      });
      return *reinterpret_cast<const D*>(&result_u);
    }
    }
  };

  template <typename D>
  struct data_accessor_t<int4, D> {

    static __attribute__((always_inline)) D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      detail::unroll_for<count>([&](auto i) {
        result_u |= (src_d << (i * 8));
      });
      return *reinterpret_cast<const D*>(&result_u);
    }
  };

  template <typename D>
  struct data_accessor_t<uint4, D> {

    static __attribute__((always_inline)) D bit_fill(uint8_t src) {
      constexpr uint32_t count = sizeof(D);
      assert((src & 0xf0) == 0 && "src must be a 4-bit value");
      using UD = raw_unsigned_t<D>;
      uint8_t src_u8 = (src << 4) | src; // pack 2 nibbles
      auto src_d = static_cast<UD>(src_u8); // zero-extend
      UD result_u(0);
      detail::unroll_for<count>([&](auto i) {
        result_u |= (src_d << (i * 8));
      });
      return *reinterpret_cast<const D*>(&result_u);
    }
  };
}

template <uint32_t NT, // number of threads per warp
          typename It, // input type (A,B)
          typename Ot, // output type (C,D)
          bool is_sparse = false,   // sparse mode flag
          uint32_t NR_ = 8,         // registers per C/D fragment
          uint32_t DK_ = 0          // K dimension of the tile
          >
struct wmma_context {
private:
  using cfg = wmma_config_t<NT, fp32, fp32, NR_, DK_>;
  enum frag_use_t { matrix_a, matrix_b, accumulator };

  using vreg_t = float;

  static __attribute__((always_inline)) vreg_t mx_word_as_f32(uint32_t value) {
    union {
      uint32_t u;
      vreg_t f;
    } cvt{value};
    return cvt.f;
  }

  template <frag_use_t U, typename T, uint32_t N>
  struct fragment_t {
    using Type = T;
    static constexpr frag_use_t Use = U;
    static constexpr uint32_t NR = N;
    std::array<vreg_t, N> data;
  };

  static __attribute__((always_inline)) uint32_t pack_mx_scale_row4(const uint8_t* base,
                                                                     uint32_t stride,
                                                                     uint32_t scale_col) {
    uint32_t packed = 0;
    detail::unroll_for<4>([&](auto i) {
      uint32_t byte = base[i * stride + scale_col];
      packed |= (byte << (i * 8));
    });
    return packed;
  }

  static __attribute__((always_inline)) uint32_t pack_mx_scale_col4(const uint8_t* base,
                                                                     uint32_t stride,
                                                                     uint32_t scale_row,
                                                                     uint32_t col_start) {
    uint32_t packed = 0;
    uint32_t row_off = scale_row * stride;
    detail::unroll_for<4>([&](auto i) {
      uint32_t col = col_start + i;
      uint32_t byte = (col < tileN) ? base[row_off + col] : 0;
      packed |= (byte << (i * 8));
    });
    return packed;
  }

public:

  using input_t  = typename It::dtype;
  using output_t = typename Ot::dtype;

  using input_acessor_t = detail::data_accessor_t<It, vreg_t>;
  using output_acessor_t = detail::data_accessor_t<Ot, vreg_t>;

  static constexpr uint32_t input_is_subbyte = (It::bits < 8);

  static constexpr bool is_mx = std::is_same_v<It, mxfp8>;

  static constexpr uint32_t i_ratio = sizeof(vreg_t) / sizeof(input_t);
  static constexpr uint32_t tileM = cfg::tileM;
  static constexpr uint32_t tileN = cfg::tileN;
  static constexpr uint32_t tileK = cfg::tileK * i_ratio;

  // Sparse metadata constants (using actual It, not cfg's fp32 default)
  static constexpr uint32_t sp_rtl_i_ratio = 32 / It::bits;
  static constexpr uint32_t sp_meta_cols = (NT * 2 * sp_rtl_i_ratio + 31) / 32;
  static constexpr uint32_t sp_per_warp_depth = cfg::m_steps * (cfg::k_steps / 2);
  static constexpr uint32_t sp_cols_per_load = (NT >= sp_per_warp_depth) ? (NT / sp_per_warp_depth) : 1;
  static constexpr uint32_t sp_num_meta_loads = (sp_per_warp_depth * sp_meta_cols + NT - 1) / NT;
  static constexpr uint32_t meta_stride = sp_num_meta_loads * NT;
  static constexpr uint32_t sparse_k_steps = cfg::k_steps / 2;
  static constexpr uint32_t sparse_regs = cfg::m_steps * sparse_k_steps;
  static constexpr uint32_t a_k_stride_sp = tileK / 2;

  // WGMMA_SP smem metadata layout constants (smem stored immediately after compressed A)
  // meta_row_w: bits per tcM row = tcK * 2 * (32/It::bits)
  // wg_meta_stride_bytes: bytes per (step_m, step_k) bank = ceil(tcM * meta_row_w / 32) * 4
  static constexpr uint32_t wg_meta_banks        = cfg::m_steps * (cfg::k_steps / 2);
  static constexpr uint32_t wg_meta_row_bits      = cfg::tcK * 2 * sp_rtl_i_ratio;
  static constexpr uint32_t wg_meta_stride_words  = (cfg::tcM * wg_meta_row_bits + 31) / 32;
  static constexpr uint32_t wg_meta_stride_bytes  = wg_meta_stride_words * 4;
  static constexpr uint32_t wg_meta_total_bytes   = wg_meta_banks * wg_meta_stride_bytes;

  using fragment_a   = fragment_t<matrix_a, input_t, cfg::NRA>;
  using fragment_b   = fragment_t<matrix_b, input_t, cfg::NRB>;
  using fragment_acc = fragment_t<accumulator, output_t, cfg::NRC>;

  // Per-thread metadata words used only for mxfp8 register plumbing.
  // DSE2 dense-MX ABI uses two packed words for A and two packed words for B.
  static inline vreg_t mx_meta_a[2] = {0.0f, 0.0f};
  static inline vreg_t mx_meta_b[2] = {0.0f, 0.0f};

  template <typename Frag>
  static __attribute__((always_inline)) void load_mx_metadata(const void* meta_mx_ptr) {
    if constexpr (is_mx) {
      if (nullptr == meta_mx_ptr) {
        if constexpr (Frag::Use == matrix_a) {
          mx_meta_a[0] = 0.0f;
          mx_meta_a[1] = 0.0f;
        } else if constexpr (Frag::Use == matrix_b) {
          mx_meta_b[0] = 0.0f;
          mx_meta_b[1] = 0.0f;
        }
      } else {
        auto meta_base = reinterpret_cast<const uint32_t*>(meta_mx_ptr);
        if constexpr (Frag::Use == matrix_a) {
          mx_meta_a[0] = mx_word_as_f32(meta_base[0]);
          mx_meta_a[1] = mx_word_as_f32(meta_base[1]);
        } else if constexpr (Frag::Use == matrix_b) {
          mx_meta_b[0] = mx_word_as_f32(meta_base[2]);
          mx_meta_b[1] = mx_word_as_f32(meta_base[3]);
        }
      }
    } else {
      (void)meta_mx_ptr;
    }
  }

  template <typename Frag, typename T>
  static __attribute__((always_inline)) void fill_fragment(Frag &dst, T value) {
    vreg_t fill_data;
    if constexpr (Frag::Use == accumulator) {
      fill_data = output_acessor_t::bit_fill(value);
    } else {
      fill_data = input_acessor_t::bit_fill(value);
    }
    detail::unroll_for<Frag::NR>([&](auto r) {
      vreg_t tmp;
      __asm__ volatile("fmv.s %0, %1" : "=f"(tmp): "f"(fill_data));
      dst.data[r] = tmp;
    });
  }

  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm) {
    uint32_t lane = vx_thread_id();
    if constexpr (Frag::Use == matrix_a) {
      // Load row-major matrix A
      uint32_t block_idx = (cfg::a_block_size == NT) ? 0 : (lane / cfg::a_block_size);
      uint32_t lane_in_blk = (cfg::a_block_size == NT) ? lane : (lane % cfg::a_block_size);
      uint32_t block_row = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcM);
      uint32_t block_col = (lane_in_blk % cfg::tcK) * i_ratio;
      uint32_t m_stride  = cfg::a_sub_blocks * cfg::tcM;
      uint32_t k_stride  = cfg::tcK * i_ratio;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      if constexpr (is_sparse) {
        // Sparse A load: only load half the K-steps (compressed A)
        constexpr uint32_t sparse_k_steps = cfg::k_steps / 2;
        constexpr uint32_t sparse_regs = cfg::m_steps * sparse_k_steps;
        auto base = reinterpret_cast<const input_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<sparse_regs>([&](auto r) {
          uint32_t block_m  = r / sparse_k_steps;
          uint32_t block_k  = r % sparse_k_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_k * k_stride;
          if constexpr (src_layout == col_major) {
            static_assert(input_is_subbyte == false, "col_major layout is not supported for sub-byte matrix_a");
            std::swap(elem_row, elem_col);
            auto ptr = base + elem_row * ldm + elem_col;
            if constexpr (sizeof(vreg_t) == sizeof(input_t) && !input_is_subbyte) {
              dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
            } else {
              dst.data[r] = input_acessor_t::pack_row(ptr, ldm);
            }
          } else {
            // row_major layout
            auto ptr = base + elem_row * ldm + elem_col;
            assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
          }
        });
        // Load metadata into tail registers (fragA.data[sparse_regs..sparse_regs+num_loads-1])
        // meta_ptr is always non-null in sparse path — no runtime check needed
        {
          constexpr uint32_t rtl_i_ratio = 32 / It::bits;
          constexpr uint32_t num_cols = (NT * 2 * rtl_i_ratio) / 32;
          constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
          constexpr uint32_t cols_per_load = NT / PD;
          constexpr uint32_t num_loads = (num_cols + cols_per_load - 1) / cols_per_load;
          auto meta_base = reinterpret_cast<const float*>(meta_ptr);
          dst.data[sparse_regs] = meta_base[lane];
          if constexpr (num_loads == 2) {
            dst.data[sparse_regs + 1] = meta_base[NT + lane];
          }
        }
      } else {
        // Dense A load: load all K-steps
        auto base = reinterpret_cast<const input_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_m  = r / cfg::k_steps;
          uint32_t block_k  = r % cfg::k_steps;
          uint32_t elem_row = block_m * m_stride;
          uint32_t elem_col = block_k * k_stride;
          if constexpr (src_layout == col_major) {
            static_assert(input_is_subbyte == false, "col_major layout is not supported for sub-byte matrix_a");
            std::swap(elem_row, elem_col);
            auto ptr = base + elem_row * ldm + elem_col;
            if constexpr (sizeof(vreg_t) == sizeof(input_t) && !input_is_subbyte) {
              dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
            } else {
              dst.data[r] = input_acessor_t::pack_row(ptr, ldm);
            }
          } else {
            // raw_major layout
            auto ptr = base + elem_row * ldm + elem_col;
            assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
          }
        });
      }

    } else if constexpr (Frag::Use == matrix_b) {
      if constexpr (is_sparse) {
        // Sparse B load: uses 2x tcK for B block
        constexpr uint32_t b_tcK = cfg::tcK * 2;
        uint32_t block_idx = (cfg::b_block_size_sp == NT) ? 0 : (lane / cfg::b_block_size_sp);
        uint32_t lane_in_blk = (cfg::b_block_size_sp == NT) ? lane : (lane % cfg::b_block_size_sp);
        uint32_t block_col = (lane_in_blk / b_tcK) + (block_idx * cfg::tcN);
        uint32_t block_row = (lane_in_blk % b_tcK) * i_ratio;
        // NT=16 sparse: each register = 2 columns × full K (n_stride=2, no K iteration)
        // NT=8/32 sparse: standard interleaved layout
        uint32_t n_stride  = cfg::sym_sparse ? (cfg::tcN / 2) : (cfg::b_sub_blocks_sp * cfg::tcN);
        uint32_t k_stride  = b_tcK * i_ratio;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const input_t*>(src) + block_row * ldm + block_col;

          if constexpr (src_layout == row_major) {
            static_assert(input_is_subbyte == false, "row_major layout is not supported for sub-byte matrix_b");
          // Pre-compute k-group base pointers (elem_row * ldm varies by k-group)
          constexpr uint32_t num_k_groups = cfg::sym_sparse ? 1 : (Frag::NR / cfg::b_sub_steps_sp);
          const input_t* k_bases[num_k_groups];
          k_bases[0] = base;
          if constexpr (num_k_groups >= 2) {
            asm volatile("" : "+r"(k_bases[0])); // prevent reverse strength reduction
            auto k_ldm_step = k_stride * (uint32_t)ldm;
            detail::unroll_for<num_k_groups - 1>([&](auto i) {
              k_bases[i + 1] = k_bases[i] + k_ldm_step;
            });
          }
          detail::unroll_for<Frag::NR>([&](auto r) {
            uint32_t block_k, block_n;
            if constexpr (cfg::sym_sparse) { block_k = 0; block_n = r; }
            else { block_k = r / cfg::b_sub_steps_sp; block_n = r % cfg::b_sub_steps_sp; }
            uint32_t elem_col = block_n * n_stride;
            auto ptr = k_bases[block_k] + elem_col;
            if constexpr (sizeof(vreg_t) == sizeof(input_t) && !input_is_subbyte) {
              dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
            } else {
              dst.data[r] = input_acessor_t::pack_row(ptr, ldm);
            }
          });
          } else {
          // col_major: after swap, elem_row = block_n * n_stride, elem_col = block_k * k_stride
          // NT=16 sparse: each register = separate column group, so num_n_groups = NRB
          constexpr uint32_t num_n_groups = cfg::sym_sparse ? Frag::NR : cfg::b_sub_steps_sp;
          const input_t* n_bases[num_n_groups];
          n_bases[0] = base;
          if constexpr (num_n_groups >= 2) {
            asm volatile("" : "+r"(n_bases[0])); // prevent reverse strength reduction
            auto n_ldm_step = n_stride * (uint32_t)ldm;
            detail::unroll_for<num_n_groups - 1>([&](auto i) {
              n_bases[i + 1] = n_bases[i] + n_ldm_step;
            });
          }
          detail::unroll_for<Frag::NR>([&](auto r) {
            uint32_t block_k, block_n;
            if constexpr (cfg::sym_sparse) { block_k = 0; block_n = r; }
            else { block_k = r / cfg::b_sub_steps_sp; block_n = r % cfg::b_sub_steps_sp; }
            uint32_t elem_col = block_k * k_stride;
            auto ptr = n_bases[block_n] + elem_col;
            assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        });
        }
      } else {
        // Dense B load
        uint32_t block_idx = (cfg::b_block_size == NT) ? 0 : (lane / cfg::b_block_size);
        uint32_t lane_in_blk = (cfg::b_block_size == NT) ? lane : (lane % cfg::b_block_size);
        uint32_t block_col = (lane_in_blk / cfg::tcK) + (block_idx * cfg::tcN);
        uint32_t block_row = (lane_in_blk % cfg::tcK) * i_ratio;
        uint32_t n_stride  = cfg::b_sub_blocks * cfg::tcN;
        uint32_t k_stride  = cfg::tcK * i_ratio;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const input_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_k = r / cfg::b_sub_steps;
          uint32_t block_n = r % cfg::b_sub_steps;
          uint32_t elem_row = block_k * k_stride;
          uint32_t elem_col = block_n * n_stride;
          if constexpr (src_layout == row_major) {
            static_assert(input_is_subbyte == false, "row_major layout is not supported for sub-byte matrix_b");
            auto ptr = base + elem_row * ldm + elem_col;
            if constexpr (sizeof(vreg_t) == sizeof(input_t) && !input_is_subbyte) {
              dst.data[r] = *reinterpret_cast<const vreg_t*>(ptr);
            } else {
              dst.data[r] = input_acessor_t::pack_row(ptr, ldm);
            }
          } else {
            // col_major layout
            std::swap(elem_row, elem_col);
            auto ptr = base + elem_row * ldm + elem_col;
            assert(reinterpret_cast<uintptr_t>(ptr) % alignof(vreg_t) == 0 && "pointer must be aligned to 4 bytes");
            dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
          }
        });
      }

    } else {
      // Load accumulator matrix C
      uint32_t block_row = lane / cfg::tcN;
      uint32_t block_col = lane % cfg::tcN;
      uint32_t m_stride = cfg::tcM;
      uint32_t n_stride = cfg::tcN;
      if constexpr (src_layout == col_major) {
        std::swap(block_row, block_col);
      }
      auto base = reinterpret_cast<const output_t*>(src) + block_row * ldm + block_col;
      detail::unroll_for<Frag::NR>([&](auto r) {
        uint32_t block_m  = r / cfg::n_steps;
        uint32_t block_n  = r % cfg::n_steps;
        uint32_t elem_row = block_m * m_stride;
        uint32_t elem_col = block_n * n_stride;
        if constexpr (src_layout == col_major) {
          std::swap(elem_row, elem_col);
        }
        auto ptr = base + elem_row * ldm + elem_col;
        if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
          dst.data[r] = *reinterpret_cast<const vreg_t *>(ptr);
        } else {
          vreg_t tmp(0);
          *reinterpret_cast<output_t*>(&tmp) = *ptr;
          dst.data[r] = tmp;
        }
      });
    }
  }

  // Dense-MX overload (4 args): matrix data + packed MX metadata pointer.
  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm,
                                                               const void *meta_mx_ptr) {
    static_assert(!is_sparse, "4-arg load_matrix_sync is dense-MX only");
    load_matrix_sync<src_layout, Frag>(dst, src, ldm);
    load_mx_metadata<Frag>(meta_mx_ptr);
  }

  // Sparse overload (5 args): MX pointer first, then SP metadata pointer.
  // MX is optional in sparse mode (pass nullptr when unused).
  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm,
                                                               const void *meta_mx_ptr,
                                                               const void *meta_sp_ptr) {
    static_assert(is_sparse, "5-arg load_matrix_sync is sparse only");

    // Reuse base matrix load
    load_matrix_sync<src_layout, Frag>(dst, src, ldm);

    // Sparse metadata (A fragment only)
    if constexpr (Frag::Use == matrix_a) {
      auto meta_base = reinterpret_cast<const float*>(meta_sp_ptr);
    uint32_t lane_id = vx_thread_id();
    dst.data[sparse_regs] = meta_base[lane_id];
    if constexpr (sp_num_meta_loads == 2) {
      dst.data[sparse_regs + 1] = meta_base[NT + lane_id];
    }
  }

    // Optional MX metadata sideband
    load_mx_metadata<Frag>(meta_mx_ptr);
  }

  template <mem_layout dst_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void store_matrix_sync(void *dst, const Frag &src, size_t ldm) {
    static_assert(Frag::Use == accumulator, "only accumulator fragment can be stored");
    uint32_t lane = vx_thread_id();
    uint32_t block_row = lane / cfg::tcN;
    uint32_t block_col = lane % cfg::tcN;
    uint32_t m_stride  = cfg::tcM;
    uint32_t n_stride  = cfg::tcN;
    if constexpr (dst_layout == col_major) {
      std::swap(block_row, block_col);
    }
    auto base = reinterpret_cast<output_t*>(dst) + block_row * ldm + block_col;
    detail::unroll_for<Frag::NR>([&](auto r) {
      uint32_t block_m  = r / cfg::n_steps;
      uint32_t block_n  = r % cfg::n_steps;
      uint32_t elem_row = block_m * m_stride;
      uint32_t elem_col = block_n * n_stride;
      if constexpr (dst_layout == col_major) {
        std::swap(elem_row, elem_col);
      }
      auto ptr = base + elem_row * ldm + elem_col;
      if constexpr (sizeof(vreg_t) == sizeof(output_t)) {
        *reinterpret_cast<vreg_t*>(ptr) = src.data[r];
      } else {
        vreg_t tmp(src.data[r]);
        *ptr = *reinterpret_cast<const output_t*>(&tmp);
      }
    });
  }

  template <typename FragD, typename FragA, typename FragB, typename FragC>
  static __attribute__((always_inline)) void mma_sync(FragD &frag_d, const FragA &frag_a, const FragB &frag_b, const FragC &frag_c) {
    constexpr int flags = is_sparse ? 1 : 0;
    static_assert(FragA::Use == matrix_a, "A must be matrix_a");
    static_assert(FragB::Use == matrix_b, "B must be matrix_b");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");

    // frag_c initialized into accumulator registers (f0-f7)
    register float fd0 __asm__("f0") = frag_c.data[0];
    register float fd1 __asm__("f1") = frag_c.data[1];
    register float fd2 __asm__("f2") = frag_c.data[2];
    register float fd3 __asm__("f3") = frag_c.data[3];
    register float fd4 __asm__("f4") = frag_c.data[4];
    register float fd5 __asm__("f5") = frag_c.data[5];
    register float fd6 __asm__("f6") = frag_c.data[6];
    register float fd7 __asm__("f7") = frag_c.data[7];

    // frag_a: caller-saved registers (f10-f17)
    register float fa0 __asm__("f10") = frag_a.data[0];
    register float fa1 __asm__("f11") = frag_a.data[1];
    register float fa2 __asm__("f12") = frag_a.data[2];
    register float fa3 __asm__("f13") = frag_a.data[3];
    register float fa4 __asm__("f14") = frag_a.data[4];
    register float fa5 __asm__("f15") = frag_a.data[5];
    register float fa6 __asm__("f16") = frag_a.data[6];
    register float fa7 __asm__("f17") = frag_a.data[7];

    // frag_meta_mx: caller-saved registers (f8-f9 (A), f18-f19 (B))
    register float fma0 __asm__("f8")  = is_mx ? mx_meta_a[0] : 0.0f;
    register float fma1 __asm__("f9")  = is_mx ? mx_meta_a[1] : 0.0f;
    register float fmb0 __asm__("f18") = is_mx ? mx_meta_b[0] : 0.0f;
    register float fmb1 __asm__("f19") = is_mx ? mx_meta_b[1] : 0.0f;

    if constexpr (FragB::NR == 8) {

      // frag_b: caller-saved registers (f24-f31)
      register float fb0 __asm__("f24")  = frag_b.data[0];
      register float fb1 __asm__("f25")  = frag_b.data[1];
      register float fb2 __asm__("f26")  = frag_b.data[2];
      register float fb3 __asm__("f27")  = frag_b.data[3];
      register float fb4 __asm__("f28")  = frag_b.data[4];
      register float fb5 __asm__("f29")  = frag_b.data[5];
      register float fb6 __asm__("f30")  = frag_b.data[6];
      register float fb7 __asm__("f31")  = frag_b.data[7];

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7)
      );
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");

      // frag_b: caller-saved registers (f28-f31)
      register float fb0 __asm__("f28") = frag_b.data[0];
      register float fb1 __asm__("f29") = frag_b.data[1];
      register float fb2 __asm__("f30") = frag_b.data[2];
      register float fb3 __asm__("f31") = frag_b.data[3];

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3)
      );
    }

    // Write results to frag_d
    frag_d.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
    }

  // WGMMA (Warp Group Matrix Multiply-Accumulate) — shared-memory source mode.
  // Both A and B tiles are read directly from shared memory via 32-bit base addresses.
  // The full 32-element accumulator occupies f0..f31; A and B addresses are passed
  // in integer registers a0 and a1 respectively.
  // Instruction encoding: CUSTOM0, funct3=1, funct7=2; smem descriptors in a0/a1.
  // Requires NRC == 32 (instantiate wmma_context with NR=32).
  //
  // Dense (is_sparse=false):  wgmma_sync(frag_d, desc_a, desc_b, frag_c)
  // Sparse (is_sparse=true):  wgmma_sync(frag_d, desc_a, desc_b, frag_c)
  //   Sparse mode is selected by instantiating wmma_context with is_sparse=true.
  //   Metadata is located implicitly at desc_a + tileM*ldm in shared memory.

  template <typename FragD, typename FragC>
  static __attribute__((always_inline)) void wgmma_sync(FragD &frag_d,
                                                         smem_matrix_desc desc_a,
                                                         smem_matrix_desc desc_b,
                                                         const FragC &frag_c) {
    static_assert(FragC::NR == 32, "wgmma_sync requires NRC=32; use wmma_context<NT,It,Ot,false,32>");
    static_assert(FragD::NR == 32, "wgmma_sync requires NRC=32; use wmma_context<NT,It,Ot,false,32>");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");
    constexpr int flags = is_sparse ? 1 : 0;

    register float fd0  __asm__("f0")  = frag_c.data[0];
    register float fd1  __asm__("f1")  = frag_c.data[1];
    register float fd2  __asm__("f2")  = frag_c.data[2];
    register float fd3  __asm__("f3")  = frag_c.data[3];
    register float fd4  __asm__("f4")  = frag_c.data[4];
    register float fd5  __asm__("f5")  = frag_c.data[5];
    register float fd6  __asm__("f6")  = frag_c.data[6];
    register float fd7  __asm__("f7")  = frag_c.data[7];
    register float fd8  __asm__("f8")  = frag_c.data[8];
    register float fd9  __asm__("f9")  = frag_c.data[9];
    register float fd10 __asm__("f10") = frag_c.data[10];
    register float fd11 __asm__("f11") = frag_c.data[11];
    register float fd12 __asm__("f12") = frag_c.data[12];
    register float fd13 __asm__("f13") = frag_c.data[13];
    register float fd14 __asm__("f14") = frag_c.data[14];
    register float fd15 __asm__("f15") = frag_c.data[15];
    register float fd16 __asm__("f16") = frag_c.data[16];
    register float fd17 __asm__("f17") = frag_c.data[17];
    register float fd18 __asm__("f18") = frag_c.data[18];
    register float fd19 __asm__("f19") = frag_c.data[19];
    register float fd20 __asm__("f20") = frag_c.data[20];
    register float fd21 __asm__("f21") = frag_c.data[21];
    register float fd22 __asm__("f22") = frag_c.data[22];
    register float fd23 __asm__("f23") = frag_c.data[23];
    register float fd24 __asm__("f24") = frag_c.data[24];
    register float fd25 __asm__("f25") = frag_c.data[25];
    register float fd26 __asm__("f26") = frag_c.data[26];
    register float fd27 __asm__("f27") = frag_c.data[27];
    register float fd28 __asm__("f28") = frag_c.data[28];
    register float fd29 __asm__("f29") = frag_c.data[29];
    register float fd30 __asm__("f30") = frag_c.data[30];
    register float fd31 __asm__("f31") = frag_c.data[31];

    register uint32_t ra __asm__("a0") = desc_a.value;
    register uint32_t rb __asm__("a1") = desc_b.value;

    __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
      : "+f"(fd0),  "+f"(fd1),  "+f"(fd2),  "+f"(fd3),
        "+f"(fd4),  "+f"(fd5),  "+f"(fd6),  "+f"(fd7),
        "+f"(fd8),  "+f"(fd9),  "+f"(fd10), "+f"(fd11),
        "+f"(fd12), "+f"(fd13), "+f"(fd14), "+f"(fd15),
        "+f"(fd16), "+f"(fd17), "+f"(fd18), "+f"(fd19),
        "+f"(fd20), "+f"(fd21), "+f"(fd22), "+f"(fd23),
        "+f"(fd24), "+f"(fd25), "+f"(fd26), "+f"(fd27),
        "+f"(fd28), "+f"(fd29), "+f"(fd30), "+f"(fd31)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
        "r"(ra), "r"(rb)
      );

    frag_d.data = {
      fd0,  fd1,  fd2,  fd3,  fd4,  fd5,  fd6,  fd7,
      fd8,  fd9,  fd10, fd11, fd12, fd13, fd14, fd15,
      fd16, fd17, fd18, fd19, fd20, fd21, fd22, fd23,
      fd24, fd25, fd26, fd27, fd28, fd29, fd30, fd31
    };
  }
};

} // namespace tensor
} // namespace vortex
