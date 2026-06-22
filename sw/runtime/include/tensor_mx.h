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

#ifndef __VX_TENSOR_MX_HOST_H__
#define __VX_TENSOR_MX_HOST_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>

namespace vortex {
namespace tensor {

namespace detail {

template <typename TensorT>
struct data_accessor_t {
  using Type = typename TensorT::dtype;

  static Type read(const Type* ptr, uint32_t offset) {
    return ptr[offset];
  }

  static void write(Type* ptr, uint32_t offset, Type value) {
    ptr[offset] = value;
  }
};

template <>
struct data_accessor_t<mxfp4> {
  static uint8_t read(const uint8_t* ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f);
  }

  static void write(uint8_t* ptr, uint32_t offset, uint8_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[row_off] = new_value;
  }
};

template <>
struct data_accessor_t<nvfp4> {
  static uint8_t read(const uint8_t* ptr, uint32_t offset) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t value8 = ptr[row_off];
    return odd ? (value8 >> 4) : (value8 & 0x0f);
  }

  static void write(uint8_t* ptr, uint32_t offset, uint8_t value) {
    uint32_t row_off = offset / 2;
    bool odd = offset & 0x1;
    uint8_t old_value = ptr[row_off];
    uint8_t new_value = odd ? ((old_value & 0x0f) | (value << 4))
                            : ((old_value & 0xf0) | (value & 0x0f));
    ptr[row_off] = new_value;
  }
};

enum class mx_block_layout_t {
  row_major,
  col_major
};

template <typename FormatT>
struct mx_format_t;

inline uint8_t select_e8m0_scale(float max_abs, float format_max_abs) {
  if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
    return 127;
  }
  float target = max_abs / format_max_abs;
  int32_t scale_exp = static_cast<int32_t>(std::ceil(std::log2(target)));
  int32_t sf = scale_exp + 127;
  sf = std::max(0, std::min(255, sf));
  return static_cast<uint8_t>(sf);
}

template <>
struct mx_format_t<mxfp8> {
  using storage_type = uint8_t;
  static constexpr bool needs_tensor_scale = false;

  static uint8_t select_scale(float max_abs) {
    return select_e8m0_scale(max_abs, 448.0f);
  }

  static storage_type convert(float v, uint8_t sf) {
    return rv_ftomxfp8_s(bit_cast<uint32_t>(v), sf, 0, nullptr);
  }
};

template <>
struct mx_format_t<mxbf8> {
  using storage_type = uint8_t;
  static constexpr bool needs_tensor_scale = false;

  static uint8_t select_scale(float max_abs) {
    return select_e8m0_scale(max_abs, 57344.0f);
  }

  static storage_type convert(float v, uint8_t sf) {
    return rv_ftomxbf8_s(bit_cast<uint32_t>(v), sf, 0, nullptr);
  }
};

template <>
struct mx_format_t<mxint8> {
  using storage_type = int8_t;
  static constexpr bool needs_tensor_scale = false;

  static uint8_t select_scale(float max_abs) {
    return select_e8m0_scale(max_abs, 127.0f / 64.0f);
  }

  static storage_type convert(float v, uint8_t sf) {
    return static_cast<int8_t>(rv_ftomxint8_s(bit_cast<uint32_t>(v), sf, 0, nullptr));
  }
};

template <>
struct mx_format_t<mxfp4> {
  using storage_type = uint8_t;
  static constexpr bool needs_tensor_scale = false;

  static uint8_t select_scale(float max_abs) {
    return select_e8m0_scale(max_abs, 6.0f);
  }

  static storage_type convert(float v, uint8_t sf) {
    return rv_ftomxfp4_s(bit_cast<uint32_t>(v), sf, 0, nullptr) & 0x0f;
  }
};

template <>
struct mx_format_t<nvfp4> {
  using storage_type = uint8_t;
  static constexpr bool needs_tensor_scale = true;

  static uint8_t select_scale(float max_abs) {
    if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
      return rv_ftoe4m3_s(bit_cast<uint32_t>(1.0f), 0, nullptr);
    }
    float target = max_abs / 6.0f;
    uint8_t sf = rv_ftoe4m3_s(bit_cast<uint32_t>(target), 0, nullptr);
    float sf_val = bit_cast<float>(rv_e4m3tof_s(sf, 0, nullptr));
    if (sf == 0 || !std::isfinite(sf_val) || !(sf_val > 0.0f)) {
      sf = rv_ftoe4m3_s(bit_cast<uint32_t>(1.0f), 0, nullptr);
    }
    return sf;
  }

  static storage_type convert(float v, uint8_t sf) {
    return rv_ftonvfp4_s(bit_cast<uint32_t>(v), sf, 0, nullptr) & 0x0f;
  }
};

template <typename FormatT>
inline void write_mx_value(typename mx_format_t<FormatT>::storage_type* quantized,
                           uint32_t offset,
                           typename mx_format_t<FormatT>::storage_type value) {
  if constexpr (FormatT::bits < 8) {
    data_accessor_t<FormatT>::write(quantized, offset, value);
  } else {
    quantized[offset] = value;
  }
}

template <typename FormatT>
inline bool quantize_mx_blocks(typename mx_format_t<FormatT>::storage_type* quantized,
                               std::vector<uint8_t>& scale_meta,
                               const float* dense,
                               uint32_t rows,
                               uint32_t cols,
                               mx_block_layout_t layout,
                               float tensor_scale = 1.0f) {
  using Traits = mx_format_t<FormatT>;
  if ((rows == 0) || (cols == 0) || (FormatT::ele_block == 0)) {
    return false;
  }
  bool scale_over_rows = (layout == mx_block_layout_t::col_major);
  uint32_t kdim = scale_over_rows ? rows : cols;
  if ((kdim % FormatT::ele_block) != 0) {
    return false;
  }

  if constexpr (FormatT::bits < 8) {
    uint32_t packed_size = (rows * cols + 1) / 2;
    std::fill(quantized, quantized + packed_size, 0);
  }

  uint32_t k_blocks = kdim / FormatT::ele_block;
  scale_meta.resize(scale_over_rows ? (k_blocks * cols) : (rows * k_blocks));

  uint32_t major_count = scale_over_rows ? cols : rows;
  for (uint32_t major = 0; major < major_count; ++major) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * FormatT::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < FormatT::ele_block; ++i) {
        uint32_t row = scale_over_rows ? (k0 + i) : major;
        uint32_t col = scale_over_rows ? major : (k0 + i);
        float v = dense[row * cols + col] / tensor_scale;
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = Traits::select_scale(max_abs);
      if (scale_over_rows) {
        scale_meta[kb * cols + major] = sf;
      } else {
        scale_meta[major * k_blocks + kb] = sf;
      }
      for (uint32_t i = 0; i < FormatT::ele_block; ++i) {
        uint32_t row = scale_over_rows ? (k0 + i) : major;
        uint32_t col = scale_over_rows ? major : (k0 + i);
        uint32_t out_offset = (layout == mx_block_layout_t::col_major) ? (col * rows + row) : (row * cols + col);
        float v = dense[row * cols + col] / tensor_scale;
        write_mx_value<FormatT>(quantized, out_offset, Traits::convert(v, sf));
      }
    }
  }

  return true;
}

inline float select_nvfp4_tensor_scale(float max_abs) {
  if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
    return 1.0f;
  }
  return max_abs;
}

inline float select_tensor_scale(const float* dense, uint32_t rows, uint32_t cols) {
  float tensor_max = 0.0f;
  for (uint32_t i = 0; i < rows * cols; ++i) {
    tensor_max = std::max(tensor_max, std::abs(dense[i]));
  }
  return select_nvfp4_tensor_scale(tensor_max);
}

} // namespace detail

template <typename FormatT>
inline bool quantize_mx_a_rowmajor(typename detail::mx_format_t<FormatT>::storage_type* quantized,
                                   std::vector<uint8_t>& scale_meta,
                                   const float* dense,
                                   uint32_t rows,
                                   uint32_t cols) {
  static_assert(!detail::mx_format_t<FormatT>::needs_tensor_scale,
                "Use the tensor_scale overload for this MX format");
  return detail::quantize_mx_blocks<FormatT>(
      quantized, scale_meta, dense, rows, cols, detail::mx_block_layout_t::row_major);
}

template <typename FormatT>
inline bool quantize_mx_b_colmajor(typename detail::mx_format_t<FormatT>::storage_type* quantized,
                                   std::vector<uint8_t>& scale_meta,
                                   const float* dense_rowmajor,
                                   uint32_t K,
                                   uint32_t N) {
  static_assert(!detail::mx_format_t<FormatT>::needs_tensor_scale,
                "Use the tensor_scale overload for this MX format");
  return detail::quantize_mx_blocks<FormatT>(
      quantized, scale_meta, dense_rowmajor, K, N, detail::mx_block_layout_t::col_major);
}

template <typename FormatT>
inline bool quantize_mx_a_rowmajor(typename detail::mx_format_t<FormatT>::storage_type* quantized,
                                   std::vector<uint8_t>& scale_meta,
                                   float& tensor_scale,
                                   const float* dense,
                                   uint32_t rows,
                                   uint32_t cols) {
  static_assert(detail::mx_format_t<FormatT>::needs_tensor_scale,
                "Use the overload without tensor_scale for this MX format");
  tensor_scale = detail::select_tensor_scale(dense, rows, cols);
  return detail::quantize_mx_blocks<FormatT>(
      quantized, scale_meta, dense, rows, cols, detail::mx_block_layout_t::row_major, tensor_scale);
}

template <typename FormatT>
inline bool quantize_mx_b_colmajor(typename detail::mx_format_t<FormatT>::storage_type* quantized,
                                   std::vector<uint8_t>& scale_meta,
                                   float& tensor_scale,
                                   const float* dense_rowmajor,
                                   uint32_t K,
                                   uint32_t N) {
  static_assert(detail::mx_format_t<FormatT>::needs_tensor_scale,
                "Use the overload without tensor_scale for this MX format");
  tensor_scale = detail::select_tensor_scale(dense_rowmajor, K, N);
  return detail::quantize_mx_blocks<FormatT>(
      quantized, scale_meta, dense_rowmajor, K, N, detail::mx_block_layout_t::col_major, tensor_scale);
}

} // namespace tensor
} // namespace vortex

#endif // __VX_TENSOR_MX_HOST_H__
