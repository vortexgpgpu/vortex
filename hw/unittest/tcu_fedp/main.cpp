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

#if defined(TCU_DRL)
#include "VVX_tcu_fedp_drl.h"
#define MODULE VVX_tcu_fedp_drl
#elif defined(TCU_BHF)
#include "VVX_tcu_fedp_bhf.h"
#define MODULE VVX_tcu_fedp_bhf
#elif defined(TCU_DSP)
#include "VVX_tcu_fedp_dsp.h"
#define MODULE VVX_tcu_fedp_dsp
#else
#include "VVX_tcu_fedp_dpi.h"
#define MODULE VVX_tcu_fedp_dpi
#endif

#include <verilated.h>

#ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
#endif

#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>
#include "softfloat_ext.h"

bool sim_trace_enabled() {
  return true;
}

#ifndef NUM_REGS
#define NUM_REGS 2
#endif

#ifndef LATENCY
#define LATENCY 4
#endif

#if (NUM_REGS * 32 <= 64)
struct wdata_t {
  uint32_t data[NUM_REGS];
};
#define WRITE_WDATA(dst, i, src) \
  ((wdata_t *)&dst)->data[i] = src
#else
#define WRITE_WDATA(dst, i, src) \
  dst[i] = src
#endif

template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) &&
        std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible_v<To>,
                "This implementation additionally requires "
                "destination type to be trivially constructible");
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

enum RoundingMode {
  RNE=0,
  RTZ=1,
  RDN=2,
  RUP=3,
  RMM=4
};

// Convert RoundingMode to std::string
std::string to_string(RoundingMode mode) {
  switch (mode) {
    case RNE: return "RNE";
    case RTZ: return "RTZ";
    case RDN: return "RDN";
    case RUP: return "RUP";
    case RMM: return "RMM";
    default: return "UNKNOWN";
  }
}

// Convert std::string to RoundingMode
RoundingMode from_string(const std::string &mode_str) {
  if (mode_str == "RNE") return RNE;
  if (mode_str == "RTZ") return RTZ;
  if (mode_str == "RDN") return RDN;
  if (mode_str == "RUP") return RUP;
  if (mode_str == "RMM") return RMM;
  throw std::invalid_argument("Invalid RoundingMode string: " + mode_str);
}

// Test configuration
struct TestConfig {
  uint64_t max_cycles = 10000;
  bool enable_tracing = true;
  uint64_t trace_start = 0;
  uint64_t trace_end = 100;
  unsigned int random_seed = 12345;
  uint32_t fmt_s = 0; // Source format
  uint32_t fmt_d = 0; // Destination format (always 0 for float32)
  std::map<std::string, bool> test_features = {
      {"normals", true},
      {"zeros", true},
      {"infinities", true},
      {"nans", true},
      {"subnormals", true}};
  int exp_bits = 8;               // Exponent bits
  int sig_bits = 23;              // Significand bits
  RoundingMode round_mode = RNE;  // Rounding mode
  uint32_t num_tests = 100;       // Number of tests per feature
};

// Add ostream overload for TestConfig
std::ostream &operator<<(std::ostream &os, const TestConfig &config) {
  os << "{max_cycles: " << config.max_cycles
     << ", enable_tracing: " << (config.enable_tracing ? "true" : "false")
     << ", trace_start: " << config.trace_start
     << ", trace_end: " << config.trace_end
     << ", random_seed: " << config.random_seed
     << ", fmt_s: " << config.fmt_s
     << ", fmt_d: " << config.fmt_d
     << ", exp_bits: " << config.exp_bits
     << ", sig_bits: " << config.sig_bits
     << ", round_mode: " << to_string(config.round_mode)
     << ", num_tests: " << config.num_tests
     << ", test_features: {";
  int i = 0;
  for (const auto &feature : config.test_features) {
    if (i++ > 0) os << ", ";
    os << feature.first << ": " << (feature.second ? "true" : "false");
  }
  os << "}}";
  return os;
}

// Test result enumeration
enum class TestResult {
  PASSED,
  FAILED,
  TIMEOUT
};

// Assertion macro with better error reporting
#define TEST_ASSERT(condition, message)                                                        \
  do {                                                                                         \
    if (!(condition)) {                                                                        \
      std::cout << "FAILED at cycle " << cycle_count_ << ": " << message << std::endl;         \
      std::cout << "    " << #condition << " in " << __FILE__ << ":" << __LINE__ << std::endl; \
      return TestResult::FAILED;                                                               \
    }                                                                                          \
  } while (0)

void print_float(const std::string& prefix, float value, bool newline = true) {
  std::cout << prefix << value << " (0x" << std::hex << bit_cast<uint32_t>(value) << ")" << std::dec;
  if (newline) {
    std::cout << std::endl;
  }
}

void print_float(const std::string& prefix, std::vector<float>& values, bool newline = true) {
  std::cout << prefix << "{";
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0) std::cout << ", ";
    print_float("", values[i], false);
  }
  std::cout << "}";
  if (newline) {
    std::cout << std::endl;
  }
}

void print_format(const std::string& prefix, uint32_t value, bool newline = true) {
  std::cout << prefix << std::hex << "0x" << value << std::dec;
  if (newline) {
    std::cout << std::endl;
  }
}

void print_format(const std::string& prefix, std::vector<uint32_t>& values, bool newline = true) {
  std::cout << prefix << "{";
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0) std::cout << ", ";
    print_format("", values[i], false);
  }
  std::cout << "}";
  if (newline) {
    std::cout << std::endl;
  }
}

inline int calculate_elements_width(int exp_bits, int sig_bits) {
  int element_bits = 1 + exp_bits + sig_bits;
  return 8 * ((element_bits + 7) / 8);
}

// Floating-point dot product testbench
class Testbench {
private:
  std::unique_ptr<MODULE> dut_;
#ifdef VCD_OUTPUT
  std::unique_ptr<VerilatedVcdC> trace_;
#endif
  TestConfig config_;
  uint64_t cycle_count_;
  uint64_t timestamp_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> normal_dist_;
  std::uniform_int_distribution<uint32_t> bits_dist_;

  // Clock generation
  void tick() {
    dut_->clk = 0;
    dut_->eval();
#ifdef VCD_OUTPUT
    if (config_.enable_tracing && timestamp_ >= config_.trace_start && timestamp_ < config_.trace_end) {
      trace_->dump(timestamp_);
    }
#endif
    timestamp_++;

    dut_->clk = 1;
    dut_->eval();
#ifdef VCD_OUTPUT
    if (config_.enable_tracing && timestamp_ >= config_.trace_start && timestamp_ < config_.trace_end) {
      trace_->dump(timestamp_);
    }
#endif
    timestamp_++;

    cycle_count_++;
  }

  // Reset the DUT
  void reset() {
    dut_->reset = 1;
    dut_->enable = 0;
    tick();
    dut_->reset = 0;
  }

  // Pack elements into XLEN words based on format
  void pack_elements(const std::vector<uint32_t> &elements, int exp_bits, int sig_bits, int num_words, uint32_t *packed) {
    int element_bits = calculate_elements_width(exp_bits, sig_bits);
    int elements_per_word = 32 / element_bits;
    int elements_mask = (1 << element_bits) - 1;

    for (int i = 0; i < num_words; i++) {
      uint32_t tmp(0);
      for (int j = 0; j < elements_per_word; j++) {
        int elem_idx = i * elements_per_word + j;
        assert(elem_idx < elements.size());
        tmp |= (elements[elem_idx] & elements_mask) << (j * element_bits);
      }
      packed[i] = tmp;
    }
  }

  // Generate test value for a specific feature
  float generate_test_value(const std::string &feature) {
    if (feature == "zeros") {
      return 0.0f;
    } else if (feature == "normals") {
      return normal_dist_(rng_) * 100.0f;
    } else if (feature == "subnormals") {
      // Generate a subnormal value
      uint32_t subnormal_bits = bits_dist_(rng_) & 0x007FFFFF; // Subnormal bit pattern
      float result;
      std::memcpy(&result, &subnormal_bits, sizeof(float));
      return result;
    } else if (feature == "infinities") {
      return std::numeric_limits<float>::infinity();
    } else if (feature == "nans") {
      return std::numeric_limits<float>::quiet_NaN();
    } else {
      std::cout << "Unknown feature: " << feature << std::endl;
      std::abort();
    }
    return 0.0f;
  }

  // Calculate expected dot product
  float calculate_expected_dot_product(const std::vector<float> &a_values,
                                       const std::vector<float> &b_values,
                                       float c_value) {
    float result(0.0f);
    for (size_t i = 0; i < a_values.size(); i++) {
      result += a_values[i] * b_values[i];
    }
    result += c_value;
    return result;
  }

  // Check if two floats are approximately equal
  bool floats_approximately_equal(float a, float b, int exp_bits, int sig_bits) {
    // Handle NaN
    if (std::isnan(a) && std::isnan(b))
      return true;

    // Handle infinity
    if (std::isinf(a) && std::isinf(b))
      return std::signbit(a) == std::signbit(b);

    // Generalize epsilon calculation based on format precision
    float epsilon;
    int total_bits = 1 + exp_bits + sig_bits;
    if (total_bits <= 8) {
      epsilon = 0.1f; // Very low precision formats
    } else if (total_bits <= 16) {
      epsilon = 0.01f; // Low precision formats
    } else if (total_bits <= 24) {
      epsilon = 0.001f; // Medium precision formats
    } else {
      epsilon = 0.0001f; // High precision formats
    }

    // Check for approximate equality
    if (std::abs(a - b) < epsilon)
      return true;

    // Check for relative error for larger numbers
    if (std::abs(a) > 1.0f || std::abs(b) > 1.0f) {
      float relative_error = std::abs(a - b) / std::max(std::abs(a), std::abs(b));
      return relative_error < epsilon;
    }

    return false;
  }

public:
  Testbench(const TestConfig &cfg)
      : config_(cfg), cycle_count_(0), timestamp_(0), rng_(config_.random_seed),
        normal_dist_(-1.0f, 1.0f), bits_dist_(0, 0xFFFFFFFF) {
    Verilated::traceEverOn(config_.enable_tracing);
    dut_ = std::make_unique<MODULE>();

#ifdef VCD_OUTPUT
    if (config_.enable_tracing) {
      trace_ = std::make_unique<VerilatedVcdC>();
      dut_->trace(trace_.get(), 99);
      trace_->open("trace.vcd");
    }
#endif

    // Initialize inputs
    dut_->clk = 0;
    dut_->reset = 0;
    dut_->enable = 0;
    dut_->fmt_s = config_.fmt_s;
    dut_->fmt_d = config_.fmt_d;
    for (int i = 0; i < NUM_REGS; i++) {
      WRITE_WDATA(dut_->a_row, i, 0);
      WRITE_WDATA(dut_->b_col, i, 0);
    }
    dut_->c_val = 0;

    reset();
  }

  ~Testbench() {
#ifdef VCD_OUTPUT
    if (trace_) {
      trace_->close();
    }
#endif
  }

  // Test a specific feature
  TestResult test_feature(const std::string &feature) {
    std::cout << "Testing with feature: " << feature << std::endl;

    // Calculate how many elements we can fit in NUM_REGS XLEN words
    int elements_per_word = 32 / calculate_elements_width(config_.exp_bits, config_.sig_bits);
    int total_elements = NUM_REGS * elements_per_word;

    std::cout << "  elements_per_word=" << elements_per_word << ", total_elements=" << total_elements << std::endl;

    for (int test_idx = 0; test_idx < config_.num_tests; test_idx++) {
      if (cycle_count_ >= config_.max_cycles) {
        return TestResult::TIMEOUT;
      }

      // Generate test vectors
      std::vector<float> a_values_float(total_elements), b_values_float(total_elements);
      std::vector<uint32_t> a_values_format(total_elements), b_values_format(total_elements);

      bool a_enable = (test_idx % 3) == 0;
      bool b_enable = (test_idx % 3) == 1;
      bool c_enable = (test_idx % 3) == 2;

      for (int i = 0; i < total_elements; i++) {
        a_values_float[i] = (a_enable && i == 0) ? generate_test_value(feature) : generate_test_value("normals");
        b_values_float[i] = (b_enable && i == 0) ? generate_test_value(feature) : generate_test_value("normals");

        a_values_format[i] = cvt_f32_to_custom(a_values_float[i], config_.exp_bits, config_.sig_bits, config_.round_mode, nullptr);
        b_values_format[i] = cvt_f32_to_custom(b_values_float[i], config_.exp_bits, config_.sig_bits, config_.round_mode, nullptr);

        // Convert back to float to account for precision loss
        a_values_float[i] = cvt_custom_to_f32(a_values_format[i], config_.exp_bits, config_.sig_bits, config_.round_mode, nullptr);
        b_values_float[i] = cvt_custom_to_f32(b_values_format[i], config_.exp_bits, config_.sig_bits, config_.round_mode, nullptr);
      }

      // Generate c value
      float c_value_float = c_enable ? generate_test_value(feature) : generate_test_value("normals");
      uint32_t c_value_bits;
      std::memcpy(&c_value_bits, &c_value_float, sizeof(float));

      // Pack into XLEN words
      uint32_t a_packed[NUM_REGS], b_packed[NUM_REGS];
      pack_elements(a_values_format, config_.exp_bits, config_.sig_bits, NUM_REGS, a_packed);
      pack_elements(b_values_format, config_.exp_bits, config_.sig_bits, NUM_REGS, b_packed);

      // Apply to DUT
      for (int i = 0; i < NUM_REGS; i++) {
        WRITE_WDATA(dut_->a_row, i, a_packed[i]);
        WRITE_WDATA(dut_->b_col, i, b_packed[i]);
      }
      dut_->c_val = c_value_bits;
      dut_->fmt_s = config_.fmt_s;
      dut_->enable = 1;

      // Run for latency cycles
      for (int i = 0; i < LATENCY; i++) {
        tick();
      }

      // Check result
      uint32_t dut_result_bits = dut_->d_val;
      float dut_result;
      std::memcpy(&dut_result, &dut_result_bits, sizeof(float));

      // Calculate expected result
      float expected = calculate_expected_dot_product(a_values_float, b_values_float, c_value_float);

      if (!floats_approximately_equal(dut_result, expected, config_.exp_bits, config_.sig_bits)) {
        std::cout << "Test " << feature << ":" << test_idx << " failed:" << std::endl;
        print_float("  af_values=", a_values_float);
        print_format("  ax_values=", a_values_format);
        print_float("  bf_values=", b_values_float);
        print_format("  bx_values=", b_values_format);
        print_float("  c_value=", c_value_float);
        print_float("  Expected=", expected);
        print_float("  Actual=", dut_result);
        return TestResult::FAILED;
      }

      // Add a few idle cycles between tests
      dut_->enable = 0;
      tick();
      tick();
    }

    std::cout << "  " << config_.num_tests << " tests passed" << std::endl;
    return TestResult::PASSED;
  }

  // Run all tests based on configuration
  TestResult run_tests() {
    std::vector<std::string> features_to_test;
    for (const auto &feature : config_.test_features) {
      if (feature.second) {
        features_to_test.push_back(feature.first);
      }
    }

    for (const auto &feature : features_to_test) {
      reset();
      TestResult result = test_feature(feature);
      if (result != TestResult::PASSED) {
        return result;
      }
    }

    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "Simulation completed in " << cycle_count_ << " cycles" << std::endl;
    return TestResult::PASSED;
  }
};

// Parse command line arguments
TestConfig parse_args(int argc, char **argv) {
  TestConfig config_;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--no-trace") {
      config_.enable_tracing = false;
    } else if (arg == "--no-normals") {
      config_.test_features["normals"] = false;
    } else if (arg == "--no-zeros") {
      config_.test_features["zeros"] = false;
    } else if (arg == "--no-infinities") {
      config_.test_features["infinities"] = false;
    } else if (arg == "--no-nans") {
      config_.test_features["nans"] = false;
    } else if (arg == "--no-subnormals") {
      config_.test_features["subnormals"] = false;
    } else if (arg.substr(0, 6) == "--fmt=") {
      config_.fmt_s = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--ext=") {
      config_.exp_bits = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--sig=") {
      config_.sig_bits = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--rnd=") {
      config_.round_mode = from_string(arg.substr(6));
    } else if (arg == "--tests" && i + 1 < argc) {
      config_.num_tests = std::stoul(argv[++i]);
    } else if (arg == "--seed" && i + 1 < argc) {
      config_.random_seed = std::stoul(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --no-trace        Disable VCD tracing" << std::endl;
      std::cout << "  --no-normals      Skip normal number tests" << std::endl;
      std::cout << "  --no-zeros        Skip zero tests" << std::endl;
      std::cout << "  --no-infinities   Skip infinity tests" << std::endl;
      std::cout << "  --no-nans         Skip NaN tests" << std::endl;
      std::cout << "  --no-subnormals   Skip subnormal tests" << std::endl;
      std::cout << "  --fmt=N           Set source format code" << std::endl;
      std::cout << "  --ext=BITS        Exponent bits for custom format" << std::endl;
      std::cout << "  --sig=BITS        Significand bits for custom format" << std::endl;
      std::cout << "  --rnd=MODE        Rounding mode (RNE, RZ, RU, RD, RM)" << std::endl;
      std::cout << "  --seed <value>    Set random seed" << std::endl;
      std::cout << "  --tests <count>   Number of tests to run (default: 100)" << std::endl;
      std::cout << "  --help            Show this help message" << std::endl;
      exit(0);
    }
  }

  return config_;
}

int main(int argc, char **argv) {
  // Initialize Verilator
  Verilated::commandArgs(argc, argv);

  // Parse command line arguments
  TestConfig config = parse_args(argc, argv);
  std::cout << "Using configuration: " << config << std::endl;

  int total_bits = 1 + config.exp_bits + config.sig_bits;

  // Validate parameters
  if (total_bits > 32) {
    std::cout << "Error: Total bits (1 + exp_bits + sig_bits) cannot exceed 32" << std::endl;
    return 1;
  }

  if (32 % total_bits != 0) {
    std::cout << "Warning: Element size " << total_bits << " doesn't evenly divide 32 bits" << std::endl;
    std::cout << "This may result in unused bits in the packed words" << std::endl;
  }

  // Create and run testbench
  Testbench testbench(config);
  TestResult result = testbench.run_tests();

  // Return appropriate exit code
  if (result == TestResult::PASSED) {
    std::cout << "--- TEST PASSED ---" << std::endl;
    return 0;
  } else {
    std::cout << "--- TEST FAILED ---" << std::endl;
    return 1;
  }
}