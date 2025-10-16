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

    static inline D bit_fill(Type src) {
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

    static inline D pack_row(const Type *base, uint32_t ldm) {
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

    static inline D bit_fill(uint8_t src) {
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

    static inline D bit_fill(uint8_t src) {
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
          typename Ot> // output type (C,D)
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
    } else if constexpr (Frag::Use == matrix_b) {
      // Load column-major matrix B
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
  static __attribute__((always_inline)) void mma_sync(FragD &fragD, const FragA &fragA, const FragB &fragB, const FragC &fragC) {
    static_assert(FragA::Use == matrix_a, "A must be matrix_a");
    static_assert(FragB::Use == matrix_b, "B must be matrix_b");
    static_assert(FragC::Use == accumulator, "C must be accumulator");
    static_assert(FragD::Use == accumulator, "D must be accumulator");

    // fragA: caller-saved registers (f0-f7)
    register float fa0 __asm__("f0")  = fragA.data[0];
    register float fa1 __asm__("f1")  = fragA.data[1];
    register float fa2 __asm__("f2")  = fragA.data[2];
    register float fa3 __asm__("f3")  = fragA.data[3];
    register float fa4 __asm__("f4")  = fragA.data[4];
    register float fa5 __asm__("f5")  = fragA.data[5];
    register float fa6 __asm__("f6")  = fragA.data[6];
    register float fa7 __asm__("f7")  = fragA.data[7];

    if constexpr (FragB::NR == 8) {
      // fragB: caller-saved registers (f10-f17)
      register float fb0 __asm__("f10") = fragB.data[0];
      register float fb1 __asm__("f11") = fragB.data[1];
      register float fb2 __asm__("f12") = fragB.data[2];
      register float fb3 __asm__("f13") = fragB.data[3];
      register float fb4 __asm__("f14") = fragB.data[4];
      register float fb5 __asm__("f15") = fragB.data[5];
      register float fb6 __asm__("f16") = fragB.data[6];
      register float fb7 __asm__("f17") = fragB.data[7];

      // fragC: mix of caller-saved (f28-f31) and callee-saved (f18-f21)
      register float fc0 __asm__("f24") = fragC.data[0];
      register float fc1 __asm__("f25") = fragC.data[1];
      register float fc2 __asm__("f26") = fragC.data[2];
      register float fc3 __asm__("f27") = fragC.data[3];
      register float fc4 __asm__("f28") = fragC.data[4];
      register float fc5 __asm__("f29") = fragC.data[5];
      register float fc6 __asm__("f30") = fragC.data[6];
      register float fc7 __asm__("f31") = fragC.data[7];

      // Force outputs into accumulator registers
      register float fd0 __asm__("f24");
      register float fd1 __asm__("f25");
      register float fd2 __asm__("f26");
      register float fd3 __asm__("f27");
      register float fd4 __asm__("f28");
      register float fd5 __asm__("f29");
      register float fd6 __asm__("f30");
      register float fd7 __asm__("f31");

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x0"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3), "f"(fb4), "f"(fb5), "f"(fb6), "f"(fb7),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      // Write results to fragD
      fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
    } else {
      static_assert(FragB::NR == 4, "Unsupported number of registers for FragB");
      // fragB: caller-saved registers (f28-f31)
      register float fb0 __asm__("f28") = fragB.data[0];
      register float fb1 __asm__("f29") = fragB.data[1];
      register float fb2 __asm__("f30") = fragB.data[2];
      register float fb3 __asm__("f31") = fragB.data[3];

      // fragC: mix of caller-saved (f10-f17)
      register float fc0 __asm__("f10") = fragC.data[0];
      register float fc1 __asm__("f11") = fragC.data[1];
      register float fc2 __asm__("f12") = fragC.data[2];
      register float fc3 __asm__("f13") = fragC.data[3];
      register float fc4 __asm__("f14") = fragC.data[4];
      register float fc5 __asm__("f15") = fragC.data[5];
      register float fc6 __asm__("f16") = fragC.data[6];
      register float fc7 __asm__("f17") = fragC.data[7];

      // Force outputs into accumulator registers
      register float fd0 __asm__("f10");
      register float fd1 __asm__("f11");
      register float fd2 __asm__("f12");
      register float fd3 __asm__("f13");
      register float fd4 __asm__("f14");
      register float fd5 __asm__("f15");
      register float fd6 __asm__("f16");
      register float fd7 __asm__("f17");

      __asm__ volatile (".insn r %[insn], 0, 2, x%[fmd], x%[fms], x0"
        : "=f"(fd0), "=f"(fd1), "=f"(fd2), "=f"(fd3), "=f"(fd4), "=f"(fd5), "=f"(fd6), "=f"(fd7)
        : [insn]"i"(RISCV_CUSTOM0), [fmd]"i"(Ot::id), [fms]"i"(It::id),
          "f"(fa0), "f"(fa1), "f"(fa2), "f"(fa3), "f"(fa4), "f"(fa5), "f"(fa6), "f"(fa7),
          "f"(fb0), "f"(fb1), "f"(fb2), "f"(fb3),
          "f"(fc0), "f"(fc1), "f"(fc2), "f"(fc3), "f"(fc4), "f"(fc5), "f"(fc6), "f"(fc7)
      );

      // Write results to fragD
      fragD.data = {fd0, fd1, fd2, fd3, fd4, fd5, fd6, fd7};
    }
  }
};

} // namespace tensor
} // namespace vortex
