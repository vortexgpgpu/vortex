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

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

namespace tensor {

enum frag_layout_t { row_major, col_major };
enum mem_layout_t { mem_row_major, mem_col_major };

template <typename T, frag_layout_t L>
struct fragment {
  typedef T DType;
  static const frag_layout_t Layout = L;
  typedef T VType __attribute__((vector_size(8 * sizeof(void*))));
  VType data;
};

template <typename Frag>
void fill_fragment(Frag &frag, size_t value) {
  // empty skeleton
}

template <typename Frag>
void load_matrix_sync(Frag &frag, const void *ptr, size_t ld) {
  // empty skeleton
}

// Perform the matrix multiply-accumulate: D = A * B + C
template <typename FragD, typename FragA, typename FragB, typename FragC>
void mma_sync(FragD &D, const FragA &A, const FragB &B, const FragC &C) {
  // empty skeleton
}

// Store a fragment result back to global memory
template <typename Type, typename Frag>
void store_matrix_sync(void *ptr, const Frag &frag, size_t ld, mem_layout_t layout) {
  // empty skeleton
}

} // namespace wmma

#endif // __VX_TENSOR_H__