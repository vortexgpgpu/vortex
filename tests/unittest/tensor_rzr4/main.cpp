#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <vector>

#include <rvfloats.h>
#include <tensor_mx.h>
#include <tensor_sp.h>
#include <util.h>

namespace vt = vortex::tensor;

static int errors = 0;

static void check_impl(bool condition, int line) {
  if (!condition) {
    std::fprintf(stderr, "check failed at line %d\n", line);
    ++errors;
  }
}

#define check(condition) check_impl((condition), __LINE__)

static uint8_t nibble(const uint8_t* data, uint32_t index) {
  return (data[index / 2] >> (4 * (index & 1))) & 0xf;
}

static float decode(uint8_t code, uint8_t scale) {
  return vortex::bit_cast<float>(rv_rzr4tof_s(code, scale, 0, nullptr));
}

static void test_decode() {
  constexpr std::array<float, 16> positive = {
    5.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f
  };
  constexpr std::array<uint8_t, 3> scales = {0x01, 0x38, 0x7e};
  for (uint8_t scale : scales) {
    float scale_value = vortex::bit_cast<float>(
        rv_e4m3tof_s(scale, 0, nullptr));
    for (uint32_t code = 0; code < positive.size(); ++code) {
      check(decode(code, scale) == positive[code] * scale_value);
      float negative_special = (code == 0) ? -5.0f : positive[code];
      check(decode(code, scale | 0x80) == negative_special * scale_value);
    }
  }
}

static void test_quantization() {
  std::array<float, 16> positive{};
  positive[0] = 6.0f;
  positive[1] = 5.0f;
  positive[2] = 4.5f;
  positive[3] = 5.5f;
  std::array<uint8_t, 8> packed{};
  std::vector<uint8_t> scales;
  float tensor_scale = 0.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, positive.data(), 1, 16));
  check(tensor_scale == 6.0f / (6.0f * 448.0f));
  check(scales.size() == 1 && scales[0] == 0x7e);
  check(nibble(packed.data(), 0) == 0x7);
  check(nibble(packed.data(), 1) == 0x0);
  check(nibble(packed.data(), 2) == 0x0);
  check(nibble(packed.data(), 3) == 0x0);
  check(nibble(packed.data(), 4) == 0x8);

  positive.fill(0.0f);
  positive[0] = -0.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, positive.data(), 1, 16));
  check(nibble(packed.data(), 0) == 0x8);

  std::array<float, 16> negative{};
  negative[0] = -6.0f;
  negative[1] = -5.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, negative.data(), 1, 16));
  check(scales[0] == 0xfe);
  check(nibble(packed.data(), 1) == 0x0);

  std::array<float, 16> tie{};
  tie[0] = 6.0f;
  tie[1] = 5.0f;
  tie[2] = -5.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, tie.data(), 1, 16));
  check(scales[0] == 0xfe);
  check(nibble(packed.data(), 1) == 0x6);
  check(nibble(packed.data(), 2) == 0x0);

  std::array<float, 16> zeros{};
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, zeros.data(), 1, 16));
  check(tensor_scale == 0.0f && scales[0] == 0x01);
  for (uint32_t i = 0; i < 16; ++i) {
    check(nibble(packed.data(), i) == 0x8);
  }

  zeros[3] = std::numeric_limits<float>::infinity();
  check(!vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, zeros.data(), 1, 16));
  zeros[3] = std::numeric_limits<float>::quiet_NaN();
  check(!vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, zeros.data(), 1, 16));
}

static void test_layout_and_sparse() {
  std::array<float, 32> matrix{};
  matrix[0] = 6.0f;
  matrix[1] = -6.0f;
  matrix[2] = 5.0f;
  matrix[3] = -5.0f;
  std::array<uint8_t, 16> packed{};
  std::vector<uint8_t> scales;
  float tensor_scale = 0.0f;
  check(vt::quantize_mx_b_colmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, matrix.data(), 16, 2));
  check(scales.size() == 2);
  check((scales[0] & 0x80) == 0);
  check((scales[1] & 0x80) != 0);

  matrix.fill(0.0f);
  matrix[0] = 6.0f;
  matrix[1] = 5.0f;
  matrix[16] = -6.0f;
  matrix[17] = -5.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      packed.data(), scales, tensor_scale, matrix.data(), 2, 16));
  check(scales.size() == 2);
  check((scales[0] & 0x80) == 0);
  check((scales[1] & 0x80) != 0);

  std::array<uint8_t, 2> sparse = {0x80, 0x21};
  check(vt::prune_2to4_matrix<vt::rzr4>(sparse.data(), 1, 2));
  check(nibble(sparse.data(), 0) == 0x0);
  check(nibble(sparse.data(), 1) == 0x8);
  check(nibble(sparse.data(), 2) == 0x8);
  check(nibble(sparse.data(), 3) == 0x2);
  std::array<uint8_t, 1> compressed{};
  std::vector<uint8_t> metadata;
  check(vt::compress_2to4_matrix<vt::rzr4>(
      compressed.data(), sparse.data(), metadata, 1, 2));
  check(nibble(compressed.data(), 0) == 0x0);
  check(nibble(compressed.data(), 1) == 0x2);
  check(metadata.size() == 1 && metadata[0] == 0x9);
}

static void test_mse_dominates_nvfp4() {
  constexpr uint32_t rows = 4;
  constexpr uint32_t cols = 16;
  std::array<float, rows * cols> matrix{};
  for (uint32_t i = 0; i < matrix.size(); ++i) {
    static constexpr float values[] = {
      -6.0f, -5.5f, -5.0f, -4.5f, -3.25f, -2.0f, -0.25f, -0.0f,
       0.0f,  0.25f, 2.0f,  3.25f,  4.5f,  5.0f,  5.5f,  6.0f
    };
    matrix[i] = values[(i * 5 + i / cols) % 16];
  }

  std::array<uint8_t, rows * cols / 2> rzr_data{}, nv_data{};
  std::vector<uint8_t> rzr_scales, nv_scales;
  float rzr_tensor_scale = 0.0f;
  float nv_tensor_scale = 0.0f;
  check(vt::quantize_mx_a_rowmajor<vt::rzr4>(
      rzr_data.data(), rzr_scales, rzr_tensor_scale,
      matrix.data(), rows, cols));
  check(vt::quantize_mx_a_rowmajor<vt::nvfp4>(
      nv_data.data(), nv_scales, nv_tensor_scale,
      matrix.data(), rows, cols));

  float rzr_error = 0.0f;
  float nv_error = 0.0f;
  for (uint32_t i = 0; i < matrix.size(); ++i) {
    uint32_t block = i / cols;
    float rzr_value = decode(nibble(rzr_data.data(), i), rzr_scales[block])
                    * rzr_tensor_scale;
    float nv_value = vortex::bit_cast<float>(rv_nvfp4tof_s(
        nibble(nv_data.data(), i), nv_scales[block], 0, nullptr))
                   * nv_tensor_scale;
    rzr_error += std::pow(matrix[i] - rzr_value, 2.0f);
    nv_error += std::pow(matrix[i] - nv_value, 2.0f);
  }
  check(rzr_error <= nv_error);
}

int main() {
  test_decode();
  test_quantization();
  test_layout_and_sparse();
  test_mse_dominates_nvfp4();
  return errors == 0 ? 0 : 1;
}
