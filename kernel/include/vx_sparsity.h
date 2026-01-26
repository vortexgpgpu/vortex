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

#include <stdint.h>
#include <type_traits>
#include <utility>

namespace vortex {
namespace tensor {

enum mem_layout {
  row_major,
  col_major
};

namespace detail {

  template <typename T>
  inline double abs_to_double(T v) {
    if constexpr (std::is_floating_point_v<T>) {
      return (v < static_cast<T>(0)) ? -static_cast<double>(v) : static_cast<double>(v);
    } else if constexpr (std::is_signed_v<T>) {
      auto wide = static_cast<long long>(v);
      return (wide < 0) ? static_cast<double>(-wide) : static_cast<double>(wide);
    } else {
      return static_cast<double>(v);
    }
  }

  template <typename T>
  inline void select_top2(const T (&vals)[4], uint32_t &keep0, uint32_t &keep1) {
    uint32_t k0 = 0;
    uint32_t k1 = 1;
    double m0 = abs_to_double(vals[0]);
    double m1 = abs_to_double(vals[1]);
    if (m1 > m0) {
      std::swap(m0, m1);
      std::swap(k0, k1);
    }
    for (uint32_t i = 2; i < 4; ++i) {
      double mi = abs_to_double(vals[i]);
      if (mi > m0) {
        m1 = m0;
        k1 = k0;
        m0 = mi;
        k0 = i;
      } else if (mi > m1) {
        m1 = mi;
        k1 = i;
      }
    }
    keep0 = k0;
    keep1 = k1;
  }
}

template <typename T>
inline bool prune_2to4_matrix(const T* dense, uint32_t rows, uint32_t cols, uint32_t ld,
                                  T* pruned, uint32_t ld_pruned, mem_layout layout = row_major) {
  constexpr uint32_t kBlock = 4;
  if (layout != row_major)
    return false;
  if ((cols % kBlock) != 0 || ld < cols || ld_pruned < cols)
    return false;

  // Keep the top-2 magnitudes per 4-wide block, zero the rest.
  for (uint32_t r = 0; r < rows; ++r) {
    const T* row_in = dense + r * ld;
    T* row_out = pruned + r * ld_pruned;
    for (uint32_t c = 0; c < cols; c += kBlock) {
      T vals[kBlock] = {row_in[c + 0], row_in[c + 1], row_in[c + 2], row_in[c + 3]};
      uint32_t keep0, keep1;
      detail::select_top2(vals, keep0, keep1);
      for (uint32_t i = 0; i < kBlock; ++i) {
        row_out[c + i] = ((i == keep0) || (i == keep1)) ? vals[i] : static_cast<T>(0);
      }
    }
  }
  return true;
}

template <typename T>
inline bool compress_2to4_matrix(const T* dense, uint32_t rows, uint32_t cols, uint32_t ld,
                                 T* compressed, uint32_t ld_compressed,
                                 uint8_t* metadata, uint32_t ld_metadata,
                                 mem_layout layout = row_major) {
  constexpr uint32_t kBlock = 4;
  constexpr uint32_t kKeep = 2;
  if (layout != row_major)
    return false;
  if ((cols % kBlock) != 0 || ld < cols)
    return false;

  const uint32_t out_cols = cols / kKeep;
  const uint32_t meta_cols = cols / kBlock;
  if (ld_compressed < out_cols || ld_metadata < meta_cols)
    return false;

  // Assumes input is already 2:4 compliant (at most two non-zeros per 4-wide block).
  for (uint32_t r = 0; r < rows; ++r) {
    const T* row_in = dense + r * ld;
    T* row_out = compressed + r * ld_compressed;
    uint8_t* row_meta = metadata + r * ld_metadata;
    for (uint32_t c = 0; c < cols; c += kBlock) {
      T vals[kBlock] = {row_in[c + 0], row_in[c + 1], row_in[c + 2], row_in[c + 3]};
      uint32_t idx0 = kBlock;
      uint32_t idx1 = kBlock;
      for (uint32_t i = 0; i < kBlock; ++i) {
        if (vals[i] != static_cast<T>(0)) {
          if (idx0 == kBlock) {
            idx0 = i;
          } else if (idx1 == kBlock) {
            idx1 = i;
          } else {
            return false;
          }
        }
      }
      if (idx0 == kBlock) {
        idx0 = 0;
        idx1 = 1;
      } else if (idx1 == kBlock) {
        idx1 = (idx0 == 0) ? 1 : 0;
      }
      if (idx0 > idx1)
        std::swap(idx0, idx1);
      uint32_t out_base = (c / kBlock) * kKeep;
      row_out[out_base + 0] = vals[idx0];
      row_out[out_base + 1] = vals[idx1];
      // Metadata byte encodes which 2 of 4 entries were kept (low 4 bits).
      row_meta[c / kBlock] = static_cast<uint8_t>((1u << idx0) | (1u << idx1));
    }
  }
  return true;
}

} // namespace tensor
} // namespace vortex
