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

  template <bool HasMxMeta>
  struct mx_meta_storage_t {};

  template <>
  struct mx_meta_storage_t<true> {
    std::array<vreg_t, 4> mx_meta;
  };

  template <frag_use_t U, typename T, uint32_t N>
  struct fragment_t : mx_meta_storage_t<(U == matrix_a || U == matrix_b)> {
    using Type = T;
    static constexpr frag_use_t Use = U;
    static constexpr bool HasMxMeta = (U == matrix_a || U == matrix_b);
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

  static constexpr bool is_mxfp8 = std::is_same_v<It, mxfp8>;
  static constexpr bool is_mxint8 = std::is_same_v<It, mxint8>;
  static constexpr bool is_nvfp4 = std::is_same_v<It, nvfp4>;
  static constexpr bool is_mx = is_mxfp8 || is_mxint8 || is_nvfp4;

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

  // Per-thread metadata words used for MX register plumbing.
  // mxfp8/mxint8 use 2 words per axis; nvfp4 uses 4 words per axis.
  static constexpr uint32_t mx_meta_slots = is_nvfp4 ? 16 : 8;
  static constexpr uint32_t mx_meta_words_per_axis = mx_meta_slots / 4;
  static constexpr uint32_t mx_meta_words = 2 * mx_meta_words_per_axis;

  template <typename Frag>
  static __attribute__((always_inline)) void load_mx_metadata(Frag& frag, const void* meta_mx_ptr) {
    static_assert(Frag::HasMxMeta, "MX metadata is only valid for matrix_a/matrix_b fragments");

        auto meta_base = reinterpret_cast<const uint32_t*>(meta_mx_ptr);
        if constexpr (Frag::Use == matrix_a) {
      frag.mx_meta[0] = mx_word_as_f32(meta_base[0]);
      frag.mx_meta[1] = mx_word_as_f32(meta_base[1]);
      if constexpr (is_nvfp4) {
        frag.mx_meta[2] = mx_word_as_f32(meta_base[2]);
        frag.mx_meta[3] = mx_word_as_f32(meta_base[3]);
      }
    } else {
      static_assert(Frag::Use == matrix_b, "Unsupported MX fragment use");
      if constexpr (is_nvfp4) {
        frag.mx_meta[0] = mx_word_as_f32(meta_base[4]);
        frag.mx_meta[1] = mx_word_as_f32(meta_base[5]);
        frag.mx_meta[2] = mx_word_as_f32(meta_base[6]);
        frag.mx_meta[3] = mx_word_as_f32(meta_base[7]);
      } else {
        frag.mx_meta[0] = mx_word_as_f32(meta_base[2]);
        frag.mx_meta[1] = mx_word_as_f32(meta_base[3]);
      }
    }
  }

  template <typename Frag>
  static __attribute__((always_inline)) void load_sp_metadata(Frag& frag, const void* meta_sp_ptr) {
    static_assert(is_sparse, "load_sp_metadata requires sparse configuration");
    static_assert(Frag::Use == matrix_a, "sparse metadata load is only valid for matrix_a fragment");

    auto meta_base = reinterpret_cast<const float*>(meta_sp_ptr);
    uint32_t lane_id = vx_thread_id();
    frag.data[sparse_regs] = meta_base[lane_id];
    if constexpr (sp_num_meta_loads == 2) {
      frag.data[sparse_regs + 1] = meta_base[NT + lane_id];
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

    // Bank-conflict-free register offset permutations (0 stalls).
    // SW must place fragment data into registers matching the HW's
    // permuted offset order.  These are the INVERSE of the forward
    // formulas in VX_tcu_uops.sv, mapping physical register offset
    // back to the logical fragment index.
    constexpr uint32_t b_sub_eff = is_sparse ? cfg::b_sub_blocks_sp : cfg::b_sub_blocks;
    constexpr bool bcfree_sp = is_sparse && !cfg::sym_sparse;      // sparse non-sym
    constexpr bool bcfree_a  = !is_sparse && (b_sub_eff == 1);     // dense pattern A
    constexpr bool bcfree_b  = !is_sparse && (b_sub_eff > 1);      // dense pattern B

    // A inverse: physical offset → logical fragment index
    constexpr auto ra_idx = [](uint32_t off) constexpr -> uint32_t {
      if constexpr (bcfree_sp) {
        // Sparse non-sym: A is identity (off = m*k_count+k)
        return off;
      } else if constexpr (bcfree_a) {
        // Dense Pattern A (b_sub==1): forward A={m[0],~m[hi],k}
        uint32_t m = ((1 - ((off >> 1) & 1)) << 1) | (off >> 2);
        uint32_t k = off & 1;
        return m * cfg::k_steps + k;
      } else if constexpr (bcfree_b) {
        // Dense Pattern B (b_sub==2): forward A={k[0],~m,m^k[hi]}
        uint32_t m = 1 - ((off >> 1) & 1);
        uint32_t k_hi = m ^ (off & 1);
        uint32_t k_lo = off >> 2;
        return m * cfg::k_steps + (k_hi << 1 | k_lo);
      } else {
        return off;
      }
    };

    // B inverse: physical offset → logical fragment index
    constexpr auto rb_idx = [](uint32_t off) constexpr -> uint32_t {
      if constexpr (bcfree_sp) {
        // Sparse non-sym: forward B={n[hi], ~(n[0]^k), ~k}
        // Inverse: k = 1-off[0], n[0] = (1-off[1])^k, n[hi] = off[2]
        uint32_t k = 1 - (off & 1);
        uint32_t n_lo = (1 - ((off >> 1) & 1)) ^ k;
        uint32_t n_hi = off >> 2;
        return k * cfg::n_steps + (n_hi << 1 | n_lo);
      } else if constexpr (bcfree_b) {
        // Dense Pattern B (b_sub==2): forward B={k[0],k[hi]^np,~np}
        uint32_t n_pair = 1 - (off & 1);
        uint32_t k_hi = ((off >> 1) & 1) ^ n_pair;
        uint32_t k_lo = off >> 2;
        return (k_hi << 1 | k_lo) * cfg::b_sub_blocks + n_pair;
      } else if constexpr (bcfree_a) {
        // Dense Pattern A (b_sub==1): forward B={n^k,~k}
        uint32_t k = 1 - (off & 1);
        uint32_t n = ((off >> 1) & 1) ^ k;
        return k * cfg::n_steps + n;
      } else {
        return off;
      }
    };

    // C inverse: physical offset → logical fragment index
    constexpr auto rc_idx = [](uint32_t off) constexpr -> uint32_t {
      if constexpr (bcfree_sp) {
        // Sparse non-sym: forward C={n[hi], m, ~(m^n[0])}
        // Inverse: m = off[1], n[0] = (1-off[0])^m, n[hi] = off[2]
        uint32_t m = (off >> 1) & 1;
        uint32_t n_lo = (1 - (off & 1)) ^ m;
        uint32_t n_hi = off >> 2;
        return m * cfg::n_steps + (n_hi << 1 | n_lo);
      } else if constexpr (bcfree_a) {
        // Dense Pattern A (b_sub==1): forward C={m[0],~m[hi],XNOR(m[hi],n)}
        uint32_t m_hi = 1 - ((off >> 1) & 1);
        uint32_t m = (m_hi << 1) | (off >> 2);
        uint32_t n = 1 - (m_hi ^ (off & 1));
        return m * cfg::n_steps + n;
      } else if constexpr (bcfree_b) {
        // Dense Pattern B (b_sub==2): forward C={n[0],~m,n[hi]}
        uint32_t m = 1 - ((off >> 1) & 1);
        uint32_t n = ((off & 1) << 1) | (off >> 2);
        return m * cfg::n_steps + n;
      } else {
        return off;
      }
    };

    // frag_c initialized into accumulator registers (f0-f7)
    register float fd0 __asm__("f0") = frag_c.data[rc_idx(0)];
    register float fd1 __asm__("f1") = frag_c.data[rc_idx(1)];
    register float fd2 __asm__("f2") = frag_c.data[rc_idx(2)];
    register float fd3 __asm__("f3") = frag_c.data[rc_idx(3)];
    register float fd4 __asm__("f4") = frag_c.data[rc_idx(4)];
    register float fd5 __asm__("f5") = frag_c.data[rc_idx(5)];
    register float fd6 __asm__("f6") = frag_c.data[rc_idx(6)];
    register float fd7 __asm__("f7") = frag_c.data[rc_idx(7)];

    // frag_a: caller-saved registers (f10-f17)
    register float fa0 __asm__("f10") = frag_a.data[ra_idx(0)];
    register float fa1 __asm__("f11") = frag_a.data[ra_idx(1)];
    register float fa2 __asm__("f12") = frag_a.data[ra_idx(2)];
    register float fa3 __asm__("f13") = frag_a.data[ra_idx(3)];
    register float fa4 __asm__("f14") = frag_a.data[ra_idx(4)];
    register float fa5 __asm__("f15") = frag_a.data[ra_idx(5)];
    register float fa6 __asm__("f16") = frag_a.data[ra_idx(6)];
    register float fa7 __asm__("f17") = frag_a.data[ra_idx(7)];

    if constexpr (FragB::NR == 8) {

      // frag_b: caller-saved registers (f24-f31)
      register float fb0 __asm__("f24")  = frag_b.data[rb_idx(0)];
      register float fb1 __asm__("f25")  = frag_b.data[rb_idx(1)];
      register float fb2 __asm__("f26")  = frag_b.data[rb_idx(2)];
      register float fb3 __asm__("f27")  = frag_b.data[rb_idx(3)];
      register float fb4 __asm__("f28")  = frag_b.data[rb_idx(4)];
      register float fb5 __asm__("f29")  = frag_b.data[rb_idx(5)];
      register float fb6 __asm__("f30")  = frag_b.data[rb_idx(6)];
      register float fb7 __asm__("f31")  = frag_b.data[rb_idx(7)];

      if constexpr (is_mx) {
        register float fma0 __asm__("f8")  = frag_a.mx_meta[0];
        register float fma1 __asm__("f9")  = frag_a.mx_meta[1];
        register float fmb0 __asm__("f18") = frag_b.mx_meta[0];
        register float fmb1 __asm__("f19") = frag_b.mx_meta[1];

        if constexpr (is_nvfp4) {
          register float fma2 __asm__("f20") = frag_a.mx_meta[2];
          register float fma3 __asm__("f21") = frag_a.mx_meta[3];
          register float fmb2 __asm__("f22") = frag_b.mx_meta[2];
          register float fmb3 __asm__("f23") = frag_b.mx_meta[3];

          __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
            : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
            : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
              "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1), "f"(fma2), "f"(fma3), "f"(fmb2), "f"(fmb3),
              "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
              "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7)
          );
        } else {
      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7)
      );
        }
      } else {
        __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
            "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7)
        );
      }
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");

      // frag_b: caller-saved registers (f28-f31)
      register float fb0 __asm__("f28") = frag_b.data[rb_idx(0)];
      register float fb1 __asm__("f29") = frag_b.data[rb_idx(1)];
      register float fb2 __asm__("f30") = frag_b.data[rb_idx(2)];
      register float fb3 __asm__("f31") = frag_b.data[rb_idx(3)];

      if constexpr (is_mx) {
        register float fma0 __asm__("f8")  = frag_a.mx_meta[0];
        register float fma1 __asm__("f9")  = frag_a.mx_meta[1];
        register float fmb0 __asm__("f18") = frag_b.mx_meta[0];
        register float fmb1 __asm__("f19") = frag_b.mx_meta[1];

        if constexpr (is_nvfp4) {
          register float fma2 __asm__("f20") = frag_a.mx_meta[2];
          register float fma3 __asm__("f21") = frag_a.mx_meta[3];
          register float fmb2 __asm__("f22") = frag_b.mx_meta[2];
          register float fmb3 __asm__("f23") = frag_b.mx_meta[3];

          __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
            : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
            : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
              "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1), "f"(fma2), "f"(fma3), "f"(fmb2), "f"(fmb3),
              "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
              "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3)
          );
        } else {
      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fma0), "f"(fma1), "f"(fmb0), "f"(fmb1),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3)
      );
    }
      } else {
        __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
            "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3)
        );
      }
    }

    // Write results to frag_d (inverse-permute back to logical order)
    frag_d.data[rc_idx(0)] = fd0;
    frag_d.data[rc_idx(1)] = fd1;
    frag_d.data[rc_idx(2)] = fd2;
    frag_d.data[rc_idx(3)] = fd3;
    frag_d.data[rc_idx(4)] = fd4;
    frag_d.data[rc_idx(5)] = fd5;
    frag_d.data[rc_idx(6)] = fd6;
    frag_d.data[rc_idx(7)] = fd7;
    }
};

// =============================================================================
// WGMMA context — Warp-Group MMA with per-warp tiles larger than WMMA.
//
// Composes two wmma_context instantiations:
//   ctx_c: NR=NRC (accumulator/C/D config)
//   ctx_a: NR=NRA=4 (A register load config for RS mode)
//
// Provides: geometry constants, fragment types, fill, load, store, wgmma_sync.
// =============================================================================

template <uint32_t NT,
          typename It,
          typename Ot,
          bool is_sparse = false,
          uint32_t NRC_ = 8>
struct wgmma_context {
private:
  static constexpr uint32_t NRA = 4;

  using ctx_c = wmma_context<NT, It, Ot, is_sparse, NRC_>;
  using ctx_a = wmma_context<NT, It, Ot, is_sparse, NRA>;

  // Direct geometry from NT (no wmma_config_t dependency)
  static constexpr uint32_t clog2(uint32_t x) {
    return (x < 2) ? 0 : (1 + clog2(x / 2));
  }
  static constexpr uint32_t XB = 4;
  static constexpr uint32_t lg_NT = clog2(NT);

  using vreg_t = float;
  enum frag_use_t { matrix_a, matrix_b, accumulator };

  // Type trait for smem descriptors
  template <typename T> struct is_smem_desc : std::false_type {};
  template <> struct is_smem_desc<smem_matrix_desc> : std::true_type {};

  // WGMMA flags encoding (rs2 field):
  //   bit 0     : is_sparse
  //   bits [2:1]: cd_nregs — 0=8, 1=16, 2=32
  //   bit 3     : a_from_smem
  static constexpr int cd_nregs_code = (NRC_ == 8) ? 0 : (NRC_ == 16) ? 1 : 2;

  template <bool a_is_smem>
  static constexpr int wgmma_flags() {
    return (is_sparse ? 1 : 0)
         | (cd_nregs_code << 1)
         | ((a_is_smem ? 1 : 0) << 3);
  }

public:
  // Types
  using input_t  = typename ctx_c::input_t;
  using output_t = typename ctx_c::output_t;

  // Fragments
  using fragment_acc = typename ctx_c::fragment_acc;
  using fragment_a   = typename ctx_a::fragment_a;

  // Block (micro-tile) geometry — derived from NT alone
  static constexpr uint32_t i_ratio = XB / sizeof(typename It::dtype);
  static constexpr uint32_t tcM = 1u << ((lg_NT + 1) / 2);
  static constexpr uint32_t tcN = 1u << (lg_NT / 2);
  static constexpr uint32_t tcK = tcN;

  // Per-warp tile geometry — m_steps = k_steps = 2 always
  static constexpr uint32_t m_steps = 2;
  static constexpr uint32_t k_steps = 2;
  static constexpr uint32_t xtileM  = m_steps * tcM;
  static constexpr uint32_t xtileN  = (NRC_ * NT) / xtileM;
  static constexpr uint32_t tileK   = k_steps * tcK * i_ratio;

  static constexpr uint32_t NRC = NRC_;

  // Sparse metadata constants (WGMMA geometry, NOT wmma geometry)
  static constexpr uint32_t sp_rtl_i_ratio       = 32 / It::bits;
  static constexpr uint32_t wg_meta_banks        = m_steps * (k_steps / 2);
  static constexpr uint32_t wg_meta_row_bits     = tcK * 2 * sp_rtl_i_ratio;
  static constexpr uint32_t wg_meta_stride_words = (tcM * wg_meta_row_bits + 31) / 32;
  static constexpr uint32_t wg_meta_stride_bytes = wg_meta_stride_words * 4;
  static constexpr uint32_t wg_meta_total_bytes  = wg_meta_banks * wg_meta_stride_bytes;
  static constexpr uint32_t a_k_stride_sp        = tileK / 2;

  // ---- Delegated operations ----

  template <typename Frag, typename T>
  static __attribute__((always_inline)) void fill_fragment(Frag &dst, T value) {
    ctx_c::fill_fragment(dst, value);
  }

  // Load A fragment (NRA=4 config) — 3-arg base
  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm) {
    static_assert(Frag::Use == matrix_a, "only matrix_a fragment can be loaded from registers in wgmma context");
    ctx_a::template load_matrix_sync<src_layout>(dst, src, ldm);
  }

  // Load sparse metadata into fragment_a.
  // VX_tcu_meta uses WMMA's per_warp_depth (NR=8) for the thread-to-bank mapping,
  // so we scatter smem data from bank-contiguous to interleaved register layout.
  template <typename Frag>
  static __attribute__((always_inline)) void load_sp_metadata(Frag& frag, const void* meta_sp_ptr) {
    static_assert(Frag::Use == matrix_a, "sparse metadata load is only valid for matrix_a fragment");
    using rtl_cfg = wmma_config_t<NT>;  // default NR=8 matches RTL's PER_WARP_DEPTH
    static constexpr uint32_t RTL_DEPTH = rtl_cfg::per_warp_depth;
    auto meta_base = reinterpret_cast<const float*>(meta_sp_ptr);
    uint32_t lane_id = vx_thread_id();
    uint32_t rtl_bank = lane_id % RTL_DEPTH;
    uint32_t rtl_col  = lane_id / RTL_DEPTH;
    frag.data[ctx_a::sparse_regs] = meta_base[rtl_bank * wg_meta_stride_words + rtl_col];
  }

  // Store accumulator with n-major register layout: r = n * m_steps + m
  template <mem_layout dst_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void store_matrix_sync(void *dst, const Frag &src, size_t ldm) {
    static_assert(Frag::Use == accumulator, "only accumulator fragment can be stored");
    uint32_t lane = vx_thread_id();
    uint32_t base_row = lane / tcN;
    uint32_t base_col = lane % tcN;
    auto base = reinterpret_cast<output_t*>(dst) + base_row * ldm + base_col;
    detail::unroll_for<Frag::NR>([&](auto r) {
      uint32_t block_m = r % m_steps;       // n-major: m is inner
      uint32_t block_n = r / m_steps;       // n-major: n is outer
      uint32_t elem_row = block_m * tcM;
      uint32_t elem_col = block_n * tcN;
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

  // ---- WGMMA sync intrinsic ----
  //   SS: wgmma_sync(fragD, desc_a, desc_b, fragC) — any NRC
  //   RS: wgmma_sync(fragD, fragA,  desc_b, fragC) — NRC <= 16

  template <typename OpA, typename OpB, typename FragD, typename FragC>
  static __attribute__((always_inline)) void wgmma_sync(FragD &frag_d,
                                                         const OpA &op_a,
                                                         const OpB &op_b,
                                                         const FragC &frag_c) {
    static_assert(FragC::NR == NRC_, "C fragment size mismatch");
    static_assert(FragD::NR == NRC_, "D fragment size mismatch");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");
    static_assert(NRC_ == 8 || NRC_ == 16 || NRC_ == 32,
                  "wgmma_sync supports NRC = 8, 16, or 32");

    constexpr bool a_is_smem = is_smem_desc<OpA>::value;
    constexpr bool b_is_smem = is_smem_desc<OpB>::value;

    if constexpr (!a_is_smem) {
      static_assert(static_cast<int>(OpA::Use) == static_cast<int>(matrix_a), "A operand must be matrix_a fragment");
      static_assert(NRC_ <= 16, "A-from-reg requires NRC <= 16");
    }
    static_assert(b_is_smem, "B must be smem_matrix_desc (SR mode is not supported)");

    constexpr int flags = wgmma_flags<a_is_smem>();

    // --- SS path: both from smem ---
    if constexpr (a_is_smem && b_is_smem) {
      register uint32_t ra __asm__("a0") = op_a.value;
      register uint32_t rb __asm__("a1") = op_b.value;

      if constexpr (NRC_ == 32) {
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

    __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
      : "+f"(fd0),  "+f"(fd1),  "+f"(fd2),  "+f"(fd3),
        "+f"(fd4),  "+f"(fd5),  "+f"(fd6),  "+f"(fd7),
        "+f"(fd8),  "+f"(fd9),  "+f"(fd10), "+f"(fd11),
        "+f"(fd12), "+f"(fd13), "+f"(fd14), "+f"(fd15),
        "+f"(fd16), "+f"(fd17), "+f"(fd18), "+f"(fd19),
        "+f"(fd20), "+f"(fd21), "+f"(fd22), "+f"(fd23),
        "+f"(fd24), "+f"(fd25), "+f"(fd26), "+f"(fd27),
        "+f"(fd28), "+f"(fd29), "+f"(fd30), "+f"(fd31)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags), "r"(ra), "r"(rb)
      );

    frag_d.data = {
      fd0,  fd1,  fd2,  fd3,  fd4,  fd5,  fd6,  fd7,
      fd8,  fd9,  fd10, fd11, fd12, fd13, fd14, fd15,
      fd16, fd17, fd18, fd19, fd20, fd21, fd22, fd23,
      fd24, fd25, fd26, fd27, fd28, fd29, fd30, fd31
    };
      } else if constexpr (NRC_ == 16) {
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

        __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0),  "+f"(fd1),  "+f"(fd2),  "+f"(fd3),
            "+f"(fd4),  "+f"(fd5),  "+f"(fd6),  "+f"(fd7),
            "+f"(fd8),  "+f"(fd9),  "+f"(fd10), "+f"(fd11),
            "+f"(fd12), "+f"(fd13), "+f"(fd14), "+f"(fd15)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "r"(ra), "r"(rb)
        );

        frag_d.data = {
          fd0,  fd1,  fd2,  fd3,  fd4,  fd5,  fd6,  fd7,
          fd8,  fd9,  fd10, fd11, fd12, fd13, fd14, fd15
        };
      } else { // NRC == 8
        register float fd0 __asm__("f0") = frag_c.data[0];
        register float fd1 __asm__("f1") = frag_c.data[1];
        register float fd2 __asm__("f2") = frag_c.data[2];
        register float fd3 __asm__("f3") = frag_c.data[3];
        register float fd4 __asm__("f4") = frag_c.data[4];
        register float fd5 __asm__("f5") = frag_c.data[5];
        register float fd6 __asm__("f6") = frag_c.data[6];
        register float fd7 __asm__("f7") = frag_c.data[7];

        __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3),
            "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "r"(ra), "r"(rb)
        );

        frag_d.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      }
    }
    // --- RS path: A from registers, B from smem (NRC <= 16, NRA = 4) ---
    else if constexpr (!a_is_smem && b_is_smem) {
      static_assert(OpA::NR == 4, "WGMMA RS requires NRA=4");
      register uint32_t rb __asm__("a1") = op_b.value;

      if constexpr (NRC_ == 16) {
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

        register float fa0 __asm__("f24") = op_a.data[0];
        register float fa1 __asm__("f25") = op_a.data[1];
        register float fa2 __asm__("f26") = op_a.data[2];
        register float fa3 __asm__("f27") = op_a.data[3];

        __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0),  "+f"(fd1),  "+f"(fd2),  "+f"(fd3),
            "+f"(fd4),  "+f"(fd5),  "+f"(fd6),  "+f"(fd7),
            "+f"(fd8),  "+f"(fd9),  "+f"(fd10), "+f"(fd11),
            "+f"(fd12), "+f"(fd13), "+f"(fd14), "+f"(fd15)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3),
            "r"(rb)
        );

        frag_d.data = {
          fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7,
          fd8, fd9, fd10, fd11, fd12, fd13, fd14, fd15
        };
      } else { // NRC == 8
        register float fd0 __asm__("f0") = frag_c.data[0];
        register float fd1 __asm__("f1") = frag_c.data[1];
        register float fd2 __asm__("f2") = frag_c.data[2];
        register float fd3 __asm__("f3") = frag_c.data[3];
        register float fd4 __asm__("f4") = frag_c.data[4];
        register float fd5 __asm__("f5") = frag_c.data[5];
        register float fd6 __asm__("f6") = frag_c.data[6];
        register float fd7 __asm__("f7") = frag_c.data[7];

        register float fa0 __asm__("f24") = op_a.data[0];
        register float fa1 __asm__("f25") = op_a.data[1];
        register float fa2 __asm__("f26") = op_a.data[2];
        register float fa3 __asm__("f27") = op_a.data[3];

        __asm__ volatile (".insn r %[insn], 1, 2, x%[fmd], x%[fms], x%[flags]"
          : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3),
            "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
          : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
            "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3),
            "r"(rb)
        );

        frag_d.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
      }
    }
  }
};

} // namespace tensor
} // namespace vortex
