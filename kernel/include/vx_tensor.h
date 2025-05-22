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

#ifndef __VX_TENSOR_H__
#define __VX_TENSOR_H__

#include <stdint.h>
#include <vx_intrinsics.h>
#include <type_traits>
#include <hfloats.h>

#ifndef NUM_LANES
#define NUM_LANES 32
#endif

namespace tensor {

enum frag_use_t { matrix_d, matrix_a, matrix_b, matrix_c };
enum layout_t { row_major, col_major };

template <frag_use_t U, typename T, layout_t L>
struct fragment {
  typedef T Type;
  static const frag_use_t Use = U;
  static const layout_t Layout = L;
  mf32x8_t data;
};

__attribute__((always_inline)) void map_operand_ab_32lanes(int tid, int &row, int &col) {
  int tg = tid / 4;

  // A (row major)
  // Figure 7(a) in paper
  // row  0~ 3: threadgroups 0 and 2
  // row  4~ 7: threadgroups 4 and 6
  // row  8~11: threadgroups 1 and 3
  // row 12~15: threadgroups 5 and 7
  row = tid % 4;
  row += (tg * 8) % 16;
  row += (tg / 4) * 4;

  // B (column major)
  // NOTE: Matrix B mapping in Figure 7(a) is incorrect; below is the
  // corrected mapping:
  // col  0~ 3: threadgroups 0 and 1
  // col  4~ 7: threadgroups 4 and 5
  // col  8~11: threadgroups 2 and 3
  // col 12~15: threadgroups 6 and 7
  col = tid % 4;
  col += ((tg % 4) / 2) * 8;
  col += (tg / 4) * 4;
}

__attribute__((always_inline)) void map_operand_ab_8lanes(int tid, int &row, int &col) {
  int tg = tid / 4;

  // A (row major)
  // row  0~ 3: threadgroup 0
  // row  4~ 7: threadgroup 1
  row = tid % 4;
  row += tg * 4;

  // B (column major)
  // col  0~ 3: threadgroup 0
  // col  4~ 7: threadgroup 1
  col = tid % 4;
  col += tg * 4;
}

__attribute__((always_inline)) void map_operand_c_32lanes(int tid, int &row, int &col) {
  int tg = tid / 4;

  // Figure 7(b), left
  col = ((tg % 4) / 2) * 8;
  row = (tg * 8) % 16;
  row += (tg / 4) * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

__attribute__((always_inline)) void map_operand_c_8lanes(int tid, int &row, int &col) {
  int tg = tid / 4;

  // Figure 7(b), left
  col = 0;
  row = tg * 4;

  // Figure 7(b), right
  row += (tid % 4) % 2;
  col += ((tid % 4) / 2) * 2;
}

__attribute__((always_inline)) void map_operand_ab(int tid, int &row, int &col) {
  if constexpr (NUM_LANES == 32) {
    map_operand_ab_32lanes(tid, row, col);
  } else if constexpr (NUM_LANES == 8) {
    map_operand_ab_8lanes(tid, row, col);
  } else {
    static_assert(NUM_LANES == 32 || NUM_LANES == 8, "NUM_LANES must be 8 or 32");
  }
}

__attribute__((always_inline)) void map_operand_c(int tid, int &row, int &col) {
  if constexpr (NUM_LANES == 32) {
    map_operand_c_32lanes(tid, row, col);
  } else if constexpr (NUM_LANES == 8) {
    map_operand_c_8lanes(tid, row, col);
  } else {
    static_assert(NUM_LANES == 32 || NUM_LANES == 8, "NUM_LANES must be 8 or 32");
  }
}

template <typename Frag>
__attribute__((always_inline)) void fill_fragment(Frag &dst, size_t value) {
  if constexpr (Frag::Use == matrix_d) {
    dst.data = vx_wsetm_d_f32(value);
  } else if constexpr (Frag::Use == matrix_a) {
    dst.data = vx_wsetm_a_f32(value);
  } else if constexpr (Frag::Use == matrix_b) {
    dst.data = vx_wsetm_b_f32(value);
  } else if constexpr (Frag::Use == matrix_c) {
    dst.data = vx_wsetm_c_f32(value);
  }
}

template <layout_t mem_layout, typename Frag>
__attribute__((always_inline)) void load_matrix_sync(Frag &dst, const void *src, size_t ldm) {
  if constexpr (Frag::Use == matrix_a) {
    if constexpr (Frag::Layout == mem_layout) {
      dst.data = vx_wldm_ad_f32(src, ldm);
    } else {
      dst.data = vx_wldm_at_f32(src, ldm);
    }
  } else if constexpr (Frag::Use == matrix_b) {
    if constexpr (Frag::Layout == mem_layout) {
      dst.data = vx_wldm_bd_f32(src, ldm);
    } else {
      dst.data = vx_wldm_bt_f32(src, ldm);
    }
  } else {
    static_assert(false, "Only matrix_a and matrix_b are supported!");
  }
}

template <layout_t mem_layout, typename Frag>
__attribute__((always_inline)) void store_matrix_sync(void *dst, const Frag &src, size_t ldm) {
  static_assert(Frag::Layout == mem_layout, "fragment layout should match memory!");
  if constexpr (Frag::Use == matrix_c) {
    vx_wstm_f32(dst, src.data, ldm);
  } else if constexpr (Frag::Use == matrix_d) {
    vx_wstm_f32(dst, src.data, ldm);
  } else {
    static_assert(false, "Only matrix_c or matrix_c are supported!");
  }
}

template <typename FragD, typename FragA, typename FragB, typename FragC>
__attribute__((always_inline)) void mma_sync(FragD &D, const FragA &A, const FragB &B, const FragC &C) {
  static_assert(FragA::Use == matrix_a, "A must be matrix_a");
  static_assert(FragB::Use == matrix_b, "B must be matrix_b");
  static_assert(FragC::Use == matrix_c, "C must be matrix_c");
  static_assert(FragD::Use == matrix_d || FragD::Use == matrix_c, "D must be matrix_d or matrix_c");
  static_assert(std::is_same_v<typename FragA::Type, typename FragB::Type>, "A and B must have the same type");
  static_assert(std::is_same_v<typename FragC::Type, typename FragD::Type>, "C and D must have the same type");

  if constexpr (std::is_same_v<typename FragC::Type, float>
             && std::is_same_v<typename FragA::Type, float>) {
    if constexpr (FragD::Use == matrix_d) {
      D.data = vx_hmma_844_d_f16_f32(A.data, B.data, C.data);
    } else {
      D.data = vx_hmma_844_c_f16_f32(A.data, B.data, C.data);
    }
  } else {
    static_assert(false, "Unsupported type!");
  }
}

} // namespace wmma

#endif // __VX_TENSOR_H__