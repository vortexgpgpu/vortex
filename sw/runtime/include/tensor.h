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

#ifndef __VX_TENSOR_HOST_H__
#define __VX_TENSOR_HOST_H__

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
struct data_accessor_t<int4> {
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
struct data_accessor_t<uint4> {
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

template <typename TensorT>
inline uint32_t expanded_cols(uint32_t cols) {
  return (TensorT::bits < 8) ? (cols * (8 / TensorT::bits)) : cols;
}

template <typename TensorT>
inline float element_magnitude(const typename TensorT::dtype* data, uint32_t offset);

template <typename TensorT>
inline void select_top2(const typename TensorT::dtype* data,
                        uint32_t base,
                        uint32_t& keep0,
                        uint32_t& keep1) {
  uint32_t k0 = 0;
  uint32_t k1 = 1;
  float m0 = element_magnitude<TensorT>(data, base + 0);
  float m1 = element_magnitude<TensorT>(data, base + 1);
  if (m1 > m0) {
    std::swap(m0, m1);
    std::swap(k0, k1);
  }
  for (uint32_t i = 2; i < 4; ++i) {
    float mi = element_magnitude<TensorT>(data, base + i);
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

template <typename TensorT>
inline float element_magnitude(const typename TensorT::dtype* data, uint32_t offset) {
  auto val = data_accessor_t<TensorT>::read(data, offset);
  if constexpr (std::is_same_v<TensorT, int8> || std::is_same_v<TensorT, mxint8>) {
    return std::abs(static_cast<float>(static_cast<int8_t>(val)));
  } else if constexpr (std::is_same_v<TensorT, uint8>
                    || std::is_same_v<TensorT, fp8>
                    || std::is_same_v<TensorT, bf8>
                    || std::is_same_v<TensorT, mxfp8>) {
    return static_cast<float>(val);
  } else if constexpr (std::is_same_v<TensorT, int4>) {
    int32_t sval = val & 0xF;
    if (sval & 0x8) {
      sval |= ~0xF;
    }
    return std::abs(static_cast<float>(sval));
  } else if constexpr (std::is_same_v<TensorT, uint4>) {
    return static_cast<float>(val & 0xF);
  } else if constexpr (std::is_same_v<TensorT, fp16>) {
    return std::abs(bit_cast<float>(rv_htof_s(val, 0, nullptr)));
  } else if constexpr (std::is_same_v<TensorT, bf16>) {
    return std::abs(bit_cast<float>(rv_btof_s(val, 0, nullptr)));
  } else {
    return std::abs(static_cast<float>(val));
  }
}

} // namespace detail

template <typename TensorT>
inline bool prune_2to4_matrix(typename TensorT::dtype* dense, uint32_t rows, uint32_t cols) {
  constexpr uint32_t kBlock = 4;
  // For types with i_ratio=1 (one element per 32-bit word, e.g. tf32), the gather
  // hardware processes each half-group of 2 independently. Each sparse K-step covers
  // exactly one half-group, so we must guarantee exactly one non-zero per pair of 2
  // consecutive elements (positions {0,1} and {2,3} within each group of 4).
  constexpr bool per_half = (sizeof(typename TensorT::dtype) == sizeof(uint32_t));
  uint32_t cols_expanded = detail::expanded_cols<TensorT>(cols);
  if ((cols_expanded % kBlock) != 0) {
    return false;
  }

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t group = 0; group < (cols_expanded / kBlock); ++group) {
      uint32_t k_start = group * kBlock;
      uint32_t base = row * cols_expanded + k_start;
      if constexpr (per_half) {
        // 1-of-2 pruning per half: keep the larger-magnitude element from each pair
        for (uint32_t h = 0; h < 2; ++h) {
          uint32_t h_base = base + h * 2;
          float m0 = detail::element_magnitude<TensorT>(dense, h_base + 0);
          float m1 = detail::element_magnitude<TensorT>(dense, h_base + 1);
          uint32_t zero_idx = (m0 >= m1) ? 1 : 0;
          detail::data_accessor_t<TensorT>::write(dense, h_base + zero_idx, 0);
        }
      } else {
        uint32_t keep0, keep1;
        detail::select_top2<TensorT>(dense, base, keep0, keep1);
        for (uint32_t i = 0; i < kBlock; ++i) {
          if (i != keep0 && i != keep1) {
            detail::data_accessor_t<TensorT>::write(dense, base + i, 0);
          }
        }
      }
    }
  }

  return true;
}

template <typename TensorT>
inline bool compress_2to4_matrix(typename TensorT::dtype* compressed,
                                 const typename TensorT::dtype* pruned,
                                 std::vector<uint8_t>& metadata,
                                 uint32_t rows,
                                 uint32_t cols) {
  constexpr uint32_t kBlock = 4;
  constexpr uint32_t kKeep = 2;
  uint32_t cols_expanded = detail::expanded_cols<TensorT>(cols);
  if ((cols_expanded % kBlock) != 0) {
    return false;
  }

  uint32_t stride_comp = cols_expanded / kKeep;
  uint32_t meta_cols = cols_expanded / kBlock;
  metadata.resize(rows * meta_cols);

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t group = 0; group < meta_cols; ++group) {
      uint32_t k_start = group * kBlock;
      uint32_t idx0 = kBlock;
      uint32_t idx1 = kBlock;
      for (uint32_t i = 0; i < kBlock; ++i) {
        auto value = detail::data_accessor_t<TensorT>::read(pruned, row * cols_expanded + k_start + i);
        if (value != 0) {
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
      if (idx0 > idx1) {
        std::swap(idx0, idx1);
      }

      uint32_t out_base = group * kKeep;
      auto value0 = detail::data_accessor_t<TensorT>::read(pruned, row * cols_expanded + k_start + idx0);
      auto value1 = detail::data_accessor_t<TensorT>::read(pruned, row * cols_expanded + k_start + idx1);
      detail::data_accessor_t<TensorT>::write(compressed, row * stride_comp + out_base + 0, value0);
      detail::data_accessor_t<TensorT>::write(compressed, row * stride_comp + out_base + 1, value1);
      metadata[row * meta_cols + group] = static_cast<uint8_t>((1u << idx0) | (1u << idx1));
    }
  }

  return true;
}

inline uint8_t select_mxfp8_scale(float max_abs) {
  if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
    return 127;
  }
  constexpr float kE4M3Max = 448.0f;
  float target = max_abs / kE4M3Max;
  int32_t scale_exp = static_cast<int32_t>(std::ceil(std::log2(target)));
  int32_t sf = scale_exp + 127;
  sf = std::max(0, std::min(255, sf));
  return static_cast<uint8_t>(sf);
}

inline bool quantize_mxfp8_a_rowmajor(uint8_t* quantized,
                                      std::vector<uint8_t>& scale_meta,
                                      const float* dense,
                                      uint32_t rows,
                                      uint32_t cols) {
  if ((cols % mxfp8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = cols / mxfp8::ele_block;
  scale_meta.resize(rows * k_blocks);

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * mxfp8::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense[row * cols + k0 + i];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxfp8_scale(max_abs);
      scale_meta[row * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense[row * cols + k0 + i];
        quantized[row * cols + k0 + i] = rv_ftomxfp8_s(bit_cast<uint32_t>(v), sf, 0, nullptr);
      }
    }
  }

  return true;
}

inline bool quantize_mxfp8_b_rowmajor(uint8_t* quantized,
                                      std::vector<uint8_t>& scale_meta,
                                      const float* dense_rowmajor,
                                      uint32_t K,
                                      uint32_t N) {
  if ((K % mxfp8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = K / mxfp8::ele_block;
  scale_meta.resize(k_blocks * N);

  for (uint32_t kb = 0; kb < k_blocks; ++kb) {
    uint32_t k0 = kb * mxfp8::ele_block;
    for (uint32_t col = 0; col < N; ++col) {
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxfp8_scale(max_abs);
      scale_meta[kb * N + col] = sf;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col];
        quantized[(k0 + i) * N + col] = rv_ftomxfp8_s(bit_cast<uint32_t>(v), sf, 0, nullptr);
      }
    }
  }

  return true;
}

inline bool quantize_mxfp8_b_colmajor(uint8_t* quantized_rowmajor,
                                      std::vector<uint8_t>& scale_meta,
                                      const float* dense_colmajor,
                                      uint32_t K,
                                      uint32_t N) {
  if ((K % mxfp8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = K / mxfp8::ele_block;
  scale_meta.resize(N * k_blocks);

  for (uint32_t col = 0; col < N; ++col) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * mxfp8::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxfp8_scale(max_abs);
      scale_meta[col * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < mxfp8::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i];
        quantized_rowmajor[(k0 + i) * N + col] = rv_ftomxfp8_s(bit_cast<uint32_t>(v), sf, 0, nullptr);
      }
    }
  }

  return true;
}

inline uint8_t select_mxint8_scale(float max_abs) {
  if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
    return 127;
  }
  constexpr float kMxInt8Max = 127.0f / 64.0f;
  float target = max_abs / kMxInt8Max;
  int32_t scale_exp = static_cast<int32_t>(std::ceil(std::log2(target)));
  int32_t sf = scale_exp + 127;
  sf = std::max(0, std::min(255, sf));
  return static_cast<uint8_t>(sf);
}

inline bool quantize_mxint8_a_rowmajor(int8_t* quantized,
                                       std::vector<uint8_t>& scale_meta,
                                       const float* dense,
                                       uint32_t rows,
                                       uint32_t cols) {
  if ((cols % mxint8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = cols / mxint8::ele_block;
  scale_meta.resize(rows * k_blocks);

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * mxint8::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense[row * cols + k0 + i];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxint8_scale(max_abs);
      scale_meta[row * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense[row * cols + k0 + i];
        quantized[row * cols + k0 + i] =
            static_cast<int8_t>(rv_ftomxint8_s(bit_cast<uint32_t>(v), sf, 0, nullptr));
      }
    }
  }

  return true;
}

inline bool quantize_mxint8_b_rowmajor(int8_t* quantized,
                                       std::vector<uint8_t>& scale_meta,
                                       const float* dense_rowmajor,
                                       uint32_t K,
                                       uint32_t N) {
  if ((K % mxint8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = K / mxint8::ele_block;
  scale_meta.resize(k_blocks * N);

  for (uint32_t kb = 0; kb < k_blocks; ++kb) {
    uint32_t k0 = kb * mxint8::ele_block;
    for (uint32_t col = 0; col < N; ++col) {
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxint8_scale(max_abs);
      scale_meta[kb * N + col] = sf;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col];
        quantized[(k0 + i) * N + col] =
            static_cast<int8_t>(rv_ftomxint8_s(bit_cast<uint32_t>(v), sf, 0, nullptr));
      }
    }
  }

  return true;
}

inline bool quantize_mxint8_b_colmajor(int8_t* quantized_rowmajor,
                                       std::vector<uint8_t>& scale_meta,
                                       const float* dense_colmajor,
                                       uint32_t K,
                                       uint32_t N) {
  if ((K % mxint8::ele_block) != 0) {
    return false;
  }

  uint32_t k_blocks = K / mxint8::ele_block;
  scale_meta.resize(N * k_blocks);

  for (uint32_t col = 0; col < N; ++col) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * mxint8::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i];
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_mxint8_scale(max_abs);
      scale_meta[col * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < mxint8::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i];
        quantized_rowmajor[(k0 + i) * N + col] =
            static_cast<int8_t>(rv_ftomxint8_s(bit_cast<uint32_t>(v), sf, 0, nullptr));
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

inline uint8_t select_nvfp4_block_scale(float max_abs) {
  if (!(max_abs > 0.0f) || !std::isfinite(max_abs)) {
    return rv_ftoe4m3_s(bit_cast<uint32_t>(1.0f), 0, nullptr);
  }
  constexpr float kE2M1Max = 3.0f;
  float target = max_abs / kE2M1Max;
  uint8_t sf = rv_ftoe4m3_s(bit_cast<uint32_t>(target), 0, nullptr);
  float sf_val = bit_cast<float>(rv_e4m3tof_s(sf, 0, nullptr));
  if (sf == 0 || !std::isfinite(sf_val) || !(sf_val > 0.0f)) {
    sf = rv_ftoe4m3_s(bit_cast<uint32_t>(1.0f), 0, nullptr);
  }
  return sf;
}

inline bool quantize_nvfp4_a_rowmajor(uint8_t* quantized,
                                      std::vector<uint8_t>& scale_meta,
                                      float& tensor_scale,
                                      const float* dense,
                                      uint32_t rows,
                                      uint32_t cols) {
  if ((cols % nvfp4::ele_block) != 0) {
    return false;
  }

  float tensor_max = 0.0f;
  for (uint32_t i = 0; i < rows * cols; ++i) {
    tensor_max = std::max(tensor_max, std::abs(dense[i]));
  }
  tensor_scale = select_nvfp4_tensor_scale(tensor_max);

  uint32_t packed_size = (rows * cols + 1) / 2;
  std::fill(quantized, quantized + packed_size, 0);

  uint32_t k_blocks = cols / nvfp4::ele_block;
  scale_meta.resize(rows * k_blocks);

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * nvfp4::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense[row * cols + k0 + i] / tensor_scale;
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_nvfp4_block_scale(max_abs);
      scale_meta[row * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense[row * cols + k0 + i] / tensor_scale;
        uint8_t q = rv_ftonvfp4_s(bit_cast<uint32_t>(v), sf, 0, nullptr) & 0x0f;
        detail::data_accessor_t<nvfp4>::write(quantized, row * cols + k0 + i, q);
      }
    }
  }

  return true;
}

inline bool quantize_nvfp4_b_rowmajor(uint8_t* quantized,
                                      std::vector<uint8_t>& scale_meta,
                                      float& tensor_scale,
                                      const float* dense_rowmajor,
                                      uint32_t K,
                                      uint32_t N) {
  if ((K % nvfp4::ele_block) != 0) {
    return false;
  }

  float tensor_max = 0.0f;
  for (uint32_t i = 0; i < K * N; ++i) {
    tensor_max = std::max(tensor_max, std::abs(dense_rowmajor[i]));
  }
  tensor_scale = select_nvfp4_tensor_scale(tensor_max);

  uint32_t packed_size = (K * N + 1) / 2;
  std::fill(quantized, quantized + packed_size, 0);

  uint32_t k_blocks = K / nvfp4::ele_block;
  scale_meta.resize(k_blocks * N);

  for (uint32_t kb = 0; kb < k_blocks; ++kb) {
    uint32_t k0 = kb * nvfp4::ele_block;
    for (uint32_t col = 0; col < N; ++col) {
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col] / tensor_scale;
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_nvfp4_block_scale(max_abs);
      scale_meta[kb * N + col] = sf;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense_rowmajor[(k0 + i) * N + col] / tensor_scale;
        uint8_t q = rv_ftonvfp4_s(bit_cast<uint32_t>(v), sf, 0, nullptr) & 0x0f;
        detail::data_accessor_t<nvfp4>::write(quantized, (k0 + i) * N + col, q);
      }
    }
  }

  return true;
}

inline bool quantize_nvfp4_b_colmajor(uint8_t* quantized_rowmajor,
                                      std::vector<uint8_t>& scale_meta,
                                      float& tensor_scale,
                                      const float* dense_colmajor,
                                      uint32_t K,
                                      uint32_t N) {
  if ((K % nvfp4::ele_block) != 0) {
    return false;
  }

  float tensor_max = 0.0f;
  for (uint32_t col = 0; col < N; ++col) {
    for (uint32_t k = 0; k < K; ++k) {
      tensor_max = std::max(tensor_max, std::abs(dense_colmajor[col * K + k]));
    }
  }
  tensor_scale = select_nvfp4_tensor_scale(tensor_max);

  uint32_t packed_size = (K * N + 1) / 2;
  std::fill(quantized_rowmajor, quantized_rowmajor + packed_size, 0);

  uint32_t k_blocks = K / nvfp4::ele_block;
  scale_meta.resize(N * k_blocks);

  for (uint32_t col = 0; col < N; ++col) {
    for (uint32_t kb = 0; kb < k_blocks; ++kb) {
      uint32_t k0 = kb * nvfp4::ele_block;
      float max_abs = 0.0f;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i] / tensor_scale;
        max_abs = std::max(max_abs, std::abs(v));
      }
      uint8_t sf = select_nvfp4_block_scale(max_abs);
      scale_meta[col * k_blocks + kb] = sf;
      for (uint32_t i = 0; i < nvfp4::ele_block; ++i) {
        float v = dense_colmajor[col * K + k0 + i] / tensor_scale;
        uint8_t q = rv_ftonvfp4_s(bit_cast<uint32_t>(v), sf, 0, nullptr) & 0x0f;
        detail::data_accessor_t<nvfp4>::write(quantized_rowmajor, (k0 + i) * N + col, q);
      }
    }
  }

  return true;
}

} // namespace tensor
} // namespace vortex

#endif // __VX_TENSOR_HOST_H__
