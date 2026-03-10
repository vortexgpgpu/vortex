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
        for (uint32_t i = 0; i < count; i++) {
          result_u |= (src_d << (i * bits));
        }
        return *reinterpret_cast<const D*>(&result_u);
      }
    }

    static __attribute__((always_inline)) D pack_row(const Type *base, uint32_t ldm) {
      static_assert(sizeof(D) % sizeof(Type) == 0, "D must be a multiple of Type in size");
      constexpr uint32_t count = sizeof(D) / sizeof(Type);
      constexpr uint32_t bits = 8 * sizeof(Type);
      using US = raw_unsigned_t<Type>;
      using UD = raw_unsigned_t<D>;
      UD result_u(0);
      for (uint32_t i = 0; i < count; ++i) {
        auto src_u = *reinterpret_cast<const US*>(base); // unsigned cast
        auto src_d = static_cast<UD>(src_u); // zero-extend
        result_u |= (src_d << (i * bits));
        base += ldm; // next row
      }
      return *reinterpret_cast<const D*>(&result_u);
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
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
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
      for (uint32_t i = 0; i < count; i++) {
        result_u |= (src_d << (i * 8));
      }
      return *reinterpret_cast<const D*>(&result_u);
    }
  };
}

template <uint32_t NT, // number of threads per warp
          typename It, // input type (A,B)
          typename Ot, // output type (C,D)
          bool is_sparse = false> // sparse mode flag
struct wmma_context {
private:
  using cfg = wmma_config_t<NT>;

  enum frag_use_t { matrix_a, matrix_b, accumulator };

  using vreg_t = float;

  template <frag_use_t U, typename T, uint32_t N>
  struct fragment_t {
    using Type = T;
    static constexpr frag_use_t Use = U;
    static constexpr uint32_t NR = N;
    std::array<vreg_t, N> data;
  };

public:

  using input_t  = typename It::dtype;
  using output_t = typename Ot::dtype;

  using input_acessor_t = detail::data_accessor_t<It, vreg_t>;
  using output_acessor_t = detail::data_accessor_t<Ot, vreg_t>;

  static constexpr uint32_t input_is_subbyte = (It::bits < 8);

  static constexpr uint32_t i_ratio = sizeof(vreg_t) / sizeof(input_t);
  static constexpr uint32_t tileM = cfg::tileM;
  static constexpr uint32_t tileN = cfg::tileN;
  static constexpr uint32_t tileK = cfg::tileK * i_ratio;

  using fragment_a   = fragment_t<matrix_a, input_t, cfg::NRA>;
  using fragment_b   = fragment_t<matrix_b, input_t, cfg::NRB>;
  using fragment_acc = fragment_t<accumulator, output_t, cfg::NRC>;

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
        uint32_t n_stride  = cfg::nt16_sparse ? 2 : (cfg::b_sub_blocks_sp * cfg::tcN);
        uint32_t k_stride  = b_tcK * i_ratio;
        if constexpr (src_layout == col_major) {
          std::swap(block_row, block_col);
        }
        auto base = reinterpret_cast<const input_t*>(src) + block_row * ldm + block_col;
        detail::unroll_for<Frag::NR>([&](auto r) {
          uint32_t block_k, block_n;
          if constexpr (cfg::nt16_sparse) {
            block_k = 0;
            block_n = r;
          } else {
            block_k = r / cfg::b_sub_steps_sp;
            block_n = r % cfg::b_sub_steps_sp;
          }
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

  template <mem_layout src_layout = row_major, typename Frag>
  static __attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm,
                                                               const void *meta_ptr) {
    static_assert(is_sparse, "meta_ptr is only supported in sparse mode");
    static_assert(Frag::Use == matrix_a, "meta_ptr is only supported for matrix_a");

    // Reuse the base function for the standard data load
    load_matrix_sync<src_layout, Frag>(dst, src, ldm);

    // Load metadata into tail registers (fragA.data[sparse_regs..sparse_regs+num_loads-1])
    constexpr uint32_t sparse_k_steps = cfg::k_steps / 2;
    constexpr uint32_t sparse_regs = cfg::m_steps * sparse_k_steps;
    constexpr uint32_t rtl_i_ratio = 32 / It::bits;
    constexpr uint32_t num_cols = (NT * 2 * rtl_i_ratio) / 32;
    constexpr uint32_t PD = cfg::m_steps * (cfg::k_steps / 2);
    constexpr uint32_t cols_per_load = NT / PD;
    constexpr uint32_t num_loads = (num_cols + cols_per_load - 1) / cols_per_load;
    auto meta_base = reinterpret_cast<const float*>(meta_ptr);
    uint32_t lane_id = vx_thread_id();
    dst.data[sparse_regs] = meta_base[lane_id];
    if constexpr (num_loads == 2) {
      dst.data[sparse_regs + 1] = meta_base[NT + lane_id];
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

#ifdef TCU_SPARSE_ENABLE
  template <int FMT_S>
  static __attribute__((always_inline)) void meta_store_expand(float d0, float d1) {
    __asm__ volatile(".insn r 0x0b, 2, 2, x%[fmt], %[d0], %[d1]"
      :: [fmt]"i"(FMT_S), [d0]"f"(d0), [d1]"f"(d1));
  }

  template <typename FragD, typename FragA, typename FragB, typename FragC>
  static __attribute__((always_inline)) void mma_sync(FragD &fragD, const FragA &fragA, const FragB &fragB, const FragC &fragC) {
    static_assert(FragA::Use == matrix_a, "A must be matrix_a");
    static_assert(FragB::Use == matrix_b, "B must be matrix_b");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");

    constexpr int flags = is_sparse ? 1 : 0;

    // fragC initialized into accumulator registers (f0-f7)
    register float fd0 __asm__("f0") = fragC.data[0];
    register float fd1 __asm__("f1") = fragC.data[1];
    register float fd2 __asm__("f2") = fragC.data[2];
    register float fd3 __asm__("f3") = fragC.data[3];
    register float fd4 __asm__("f4") = fragC.data[4];
    register float fd5 __asm__("f5") = fragC.data[5];
    register float fd6 __asm__("f6") = fragC.data[6];
    register float fd7 __asm__("f7") = fragC.data[7];

    // fragA: caller-saved registers (f10-f17)
    register float fa0 __asm__("f10") = fragA.data[0];
    register float fa1 __asm__("f11") = fragA.data[1];
    register float fa2 __asm__("f12") = fragA.data[2];
    register float fa3 __asm__("f13") = fragA.data[3];
    register float fa4 __asm__("f14") = fragA.data[4];
    register float fa5 __asm__("f15") = fragA.data[5];
    register float fa6 __asm__("f16") = fragA.data[6];
    register float fa7 __asm__("f17") = fragA.data[7];

    if constexpr (FragB::NR == 8) {

      // fragB: caller-saved registers (f24-f31)
      register float fb0 __asm__("f24")  = fragB.data[0];
      register float fb1 __asm__("f25")  = fragB.data[1];
      register float fb2 __asm__("f26")  = fragB.data[2];
      register float fb3 __asm__("f27")  = fragB.data[3];
      register float fb4 __asm__("f28")  = fragB.data[4];
      register float fb5 __asm__("f29")  = fragB.data[5];
      register float fb6 __asm__("f30")  = fragB.data[6];
      register float fb7 __asm__("f31")  = fragB.data[7];

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7)
      );
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");

      // fragB: caller-saved registers (f28-f31)
      register float fb0 __asm__("f28") = fragB.data[0];
      register float fb1 __asm__("f29") = fragB.data[1];
      register float fb2 __asm__("f30") = fragB.data[2];
      register float fb3 __asm__("f31") = fragB.data[3];

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x%[flags]"
        : "+f"(fd0), "+f"(fd1), "+f"(fd2), "+f"(fd3), "+f"(fd4), "+f"(fd5), "+f"(fd6), "+f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id), [flags]"i"(flags),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3)
      );
    }

      // Write results to fragD
      fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
    }
};

} // namespace tensor
} // namespace vortex
