#ifndef __VX_SPARSITY_HOST_H__
#define __VX_SPARSITY_HOST_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <rvfloats.h>
#include <tensor_cfg.h>
#include <util.h>

namespace vortex {
namespace sparsity {

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
struct data_accessor_t<tensor::int4> {
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
struct data_accessor_t<tensor::uint4> {
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
struct data_accessor_t<tensor::nvfp4> {
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
  if constexpr (std::is_same_v<TensorT, tensor::int8> || std::is_same_v<TensorT, tensor::mxint8>) {
    return std::abs(static_cast<float>(static_cast<int8_t>(val)));
  } else if constexpr (std::is_same_v<TensorT, tensor::uint8>
                    || std::is_same_v<TensorT, tensor::fp16>
                    || std::is_same_v<TensorT, tensor::bf16>
                    || std::is_same_v<TensorT, tensor::fp8>
                    || std::is_same_v<TensorT, tensor::bf8>
                    || std::is_same_v<TensorT, tensor::mxfp8>) {
    return static_cast<float>(val);
  } else if constexpr (std::is_same_v<TensorT, tensor::int4>) {
    int32_t sval = val & 0xF;
    if (sval & 0x8) {
      sval |= ~0xF;
    }
    return std::abs(static_cast<float>(sval));
  } else if constexpr (std::is_same_v<TensorT, tensor::uint4>) {
    return static_cast<float>(val & 0xF);
  } else {
    return std::abs(static_cast<float>(val));
  }
}

} // namespace detail

template <typename TensorT>
inline bool prune_2to4_matrix(typename TensorT::dtype* dense, uint32_t rows, uint32_t cols) {
  constexpr uint32_t kBlock = 4;
  uint32_t cols_expanded = detail::expanded_cols<TensorT>(cols);
  if ((cols_expanded % kBlock) != 0) {
    return false;
  }

  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t group = 0; group < (cols_expanded / kBlock); ++group) {
      uint32_t k_start = group * kBlock;
      uint32_t base = row * cols_expanded + k_start;
      uint32_t keep0, keep1;
      detail::select_top2<TensorT>(dense, base, keep0, keep1);
      for (uint32_t i = 0; i < kBlock; ++i) {
        if (i != keep0 && i != keep1) {
          detail::data_accessor_t<TensorT>::write(dense, base + i, 0);
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

} // namespace sparsity
} // namespace vortex

#endif // __VX_SPARSITY_HOST_H__
