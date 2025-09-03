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
#include <bitmanip.h>
#include "softfloat_ext.h"
#include "fedp.h"

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

enum class RoundingMode {
  RNE = 0,
  RTZ = 1,
  RDN = 2,
  RUP = 3,
  RMM = 4
};

// Convert RoundingMode to std::string
std::string frm_to_string(RoundingMode frm) {
  switch (frm) {
    case RoundingMode::RNE: return "RNE";
    case RoundingMode::RTZ: return "RTZ";
    case RoundingMode::RDN: return "RDN";
    case RoundingMode::RUP: return "RUP";
    case RoundingMode::RMM: return "RMM";
    default: return "UNKNOWN";
  }
}

// Convert std::string to RoundingMode
RoundingMode frm_from_string(const std::string &frm_str) {
  if (frm_str == "RNE") return RoundingMode::RNE;
  if (frm_str == "RTZ") return RoundingMode::RTZ;
  if (frm_str == "RDN") return RoundingMode::RDN;
  if (frm_str == "RUP") return RoundingMode::RUP;
  if (frm_str == "RMM") return RoundingMode::RMM;
  throw std::invalid_argument("Invalid RoundingMode: " + frm_str);
}

// initialize default fp format based on format code
void init_default_fp_format(uint32_t fmt, int *exp_bits, int *sig_bits) {
  switch (fmt) {
    case 0: // fp32
      *exp_bits = 8;
      *sig_bits = 23;
      break;
    case 1: // fp16
      *exp_bits = 5;
      *sig_bits = 10;
      break;
    case 2: // bf16
      *exp_bits = 8;
      *sig_bits = 7;
      break;
    case 3: // fp8 (E4M3)
      *exp_bits = 4;
      *sig_bits = 3;
      break;
    case 4: // bf8 (E5M2)
      *exp_bits = 5;
      *sig_bits = 2;
      break;
    default:
      break;
  }
}

// Test configuration
struct TestConfig {
  uint64_t max_cycles = 1000;
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
  int exp_bits = 0; // Exponent bits
  int sig_bits = 0; // Significand bits
  RoundingMode frm = RoundingMode::RNE; // Rounding mode
  uint32_t num_tests = 1000; // Number of tests per feature
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
     << ", frm: " << frm_to_string(config.frm)
     << ", num_tests: " << config.num_tests
     << ", test_features: {";
  int i = 0;
  for (const auto &f : config.test_features) {
    if (i++ > 0) os << ", ";
    os << f.first << ": " << (f.second ? "true" : "false");
  }
  os << "}}";
  return os;
}

static void print_float(const std::string& prefix, float value, bool newline = false) {
  std::cout << prefix << value << " (0x" << std::hex << bit_cast<uint32_t>(value) << ")" << std::dec;
  if (newline) {
    std::cout << std::endl;
  }
}

static void print_float(const std::string& prefix, std::vector<float>& values, bool newline = false) {
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

static void print_format(const std::string& prefix, uint32_t value, bool newline = false) {
  std::cout << prefix << std::hex << "0x" << value << std::dec;
  if (newline) {
    std::cout << std::endl;
  }
}

static void print_format(const std::string& prefix, std::vector<uint32_t>& values, bool newline = false) {
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

static int float_fmt_width(int exp_bits, int sig_bits) {
  int element_bits = 1 + exp_bits + sig_bits;
  return 8 * ((element_bits + 7) / 8);
}

static int int_fmt_width(int fmt) {
  switch (fmt) {
  case 8:  return 32; // int32
  case 9:  return 8;  // int8
  case 10: return 8;  // uint8
  case 11: return 4;  // int4
  case 12: return 4;  // uint4
  default:
    std::cerr << "Unsupported integer format: " << fmt << std::endl;
    std::abort();
  }
}

static int int_fmt_sign(int fmt) {
  switch (fmt) {
  case 8:  return true;  // int32
  case 9:  return true;  // int8
  case 10: return false; // uint8
  case 11: return true;  // int4
  case 12: return false; // uint4
  default:
    std::cerr << "Unsupported integer format: " << fmt << std::endl;
    std::abort();
  }
}

static void pack_elements(const std::vector<uint32_t> &elements, int element_bits, int num_words, uint32_t *packed) {
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

// Calculate expected fp dot product
/*static float calculate_fp_dot_product(const std::vector<float> &a_values,
                                      const std::vector<float> &b_values,
                                      float c_value) {
  float result(c_value);
  for (size_t i = 0; i < a_values.size(); i++) {
    result += a_values[i] * b_values[i];
  }
  return result;
}*/

// Calculate expected int dot product
static int32_t calculate_int_dot_product(const std::vector<uint32_t> &a_values,
                                         const std::vector<uint32_t> &b_values,
                                         int32_t c_value) {
  int32_t result(c_value);
  for (size_t i = 0; i < a_values.size(); i++) {
    result += int32_t(a_values[i]) * int32_t(b_values[i]);
  }
  return result;
}

// Check if two floats are approximately equal
static bool approximately_equal(float a, float b, int exp_bits, int sig_bits) {
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

  // Clock generation
  void tick() {
    dut_->clk = 0;
    dut_->eval();
  #ifdef VCD_OUTPUT
    if (config_.enable_tracing && (timestamp_ >= config_.trace_start) && (timestamp_ < config_.trace_end)) {
      trace_->dump(timestamp_);
    }
  #endif
    ++timestamp_;

    dut_->clk = 1;
    dut_->eval();
  #ifdef VCD_OUTPUT
    if (config_.enable_tracing && (timestamp_ >= config_.trace_start) && (timestamp_ < config_.trace_end)) {
      trace_->dump(timestamp_);
    }
  #endif
    ++timestamp_;
    ++cycle_count_;
  }

  // Reset the DUT
  void reset() {
    dut_->reset = 1;
    dut_->enable = 0;
    tick();
    dut_->reset = 0;
  }

  // Generate fp value for a specific feature
  uint32_t generate_int_value(bool is_signed, int element_bits, int index) {
    uint32_t mask = (element_bits < 32) ? ((1u << element_bits) - 1u) : 0xFFFFFFFFu;
    std::uniform_int_distribution<uint32_t> int_dist(0, mask);

    uint32_t value;
    if (index == -1) {
      value = int_dist(rng_);
    } else {
      switch (index) {
      case 0: // zero
        value = 0;
        break;
      case 1: // min
        if (is_signed) {
            value = (element_bits < 32) ? (1u << (element_bits-1)) : 0x80000000u;
        } else {
            value = 0;
        }
        break;
      case 2: // max
        if (is_signed) {
            value = (element_bits < 32) ? ((1u << (element_bits-1))-1u) : 0x7FFFFFFFu;
        } else {
            value = mask;
        }
        break;
      default:
        value = int_dist(rng_);
        break;
      }
    }

    if (is_signed) {
      value = vortex::sext<int32_t>(value, element_bits);
    }
    return value;
  }

  // Generate fp value for a specific feature
  float generate_fp_value(const std::string &feature) {
    static std::uniform_real_distribution<float> normal_dist(-1.0f, 1.0f);
    static std::uniform_int_distribution<uint32_t> bits_dist(0, 0xFFFFFFFF);
    if (feature == "zeros") {
      return 0.0f;
    } else if (feature == "normals") {
      return normal_dist(rng_) * 100.0f;
    } else if (feature == "subnormals") {
      // Generate a subnormal value
      uint32_t subnormal_bits = bits_dist(rng_) & 0x007FFFFF; // Subnormal bit pattern
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

public:
  Testbench(const TestConfig &cfg)
    : config_(cfg)
    , cycle_count_(0)
    , timestamp_(0)
    , rng_(config_.random_seed) {
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
  }

  ~Testbench() {
#ifdef VCD_OUTPUT
    if (trace_) {
      trace_->close();
    }
#endif
  }

  bool test_integers() {
    std::cout << "Testing integer format" << std::endl;

    // Calculate how many elements we can fit in NUM_REGS XLEN words
    bool is_signed = int_fmt_sign(config_.fmt_s);
    int element_bits = int_fmt_width(config_.fmt_s);
    int elements_per_word = 32 / element_bits;
    int total_elements = NUM_REGS * elements_per_word;

    std::cout << "  elements_per_word=" << elements_per_word << ", total_elements=" << total_elements << std::endl;

    for (int test_idx = 0; test_idx < config_.num_tests; test_idx++) {
      // Generate test vectors
      std::vector<uint32_t> a_values(total_elements), b_values(total_elements);

      bool a_enable = (test_idx % 3) == 0;
      bool b_enable = (test_idx % 3) == 1;
      bool c_enable = (test_idx % 3) == 2;

      for (int i = 0; i < total_elements; i++) {
        a_values[i] = (a_enable && i == 0) ? generate_int_value(is_signed, element_bits, test_idx) : generate_int_value(is_signed, element_bits, -1);
        a_values[i] = (b_enable && i == 0) ? generate_int_value(is_signed, element_bits, test_idx) : generate_int_value(is_signed, element_bits, -1);
      }

      // Generate c value
      int32_t c_value = c_enable ? generate_int_value(true, 32, test_idx) : generate_int_value(true, 32, -1);

      // Pack into XLEN words
      uint32_t a_packed[NUM_REGS], b_packed[NUM_REGS];
      pack_elements(a_values, element_bits, NUM_REGS, a_packed);
      pack_elements(b_values, element_bits, NUM_REGS, b_packed);

      // Apply to DUT
      for (int i = 0; i < NUM_REGS; i++) {
        WRITE_WDATA(dut_->a_row, i, a_packed[i]);
        WRITE_WDATA(dut_->b_col, i, b_packed[i]);
      }
      dut_->c_val = c_value;
      dut_->fmt_s = config_.fmt_s;
      dut_->enable = 1;

      // Run for latency cycles
      for (int i = 0; i < LATENCY; i++) {
        tick();
      }

      // Check result
      int32_t dut_result = dut_->d_val;

      // Calculate expected result
      int32_t expected = calculate_int_dot_product(a_values, b_values, c_value);

      if (dut_result != expected) {
        std::cout << "Test:" << test_idx << " failed:" << std::endl;
        print_format("  a_values=", a_values, true);
        print_format("  b_values=", b_values, true);
        print_format("  c_value=", c_value, true);
        print_format("  expected=", expected, true);
        print_format("  actual=", dut_result, true);
        return false;
      }

      // Add one idle cycles between tests
      dut_->enable = 0;
      tick();
    }

    std::cout << "  " << config_.num_tests << " tests passed" << std::endl;
    return true;
  }

  // Test a specific fp feature
  bool test_fp_feature(const std::string &feature) {
    std::cout << "Testing floating-point feature: " << feature << std::endl;

    // Calculate how many elements we can fit in NUM_REGS XLEN words
    int element_bits = float_fmt_width(config_.exp_bits, config_.sig_bits);
    int elements_per_word = 32 / element_bits;
    int total_elements = NUM_REGS * elements_per_word;

    std::cout << "  elements_per_word=" << elements_per_word << ", total_elements=" << total_elements << std::endl;

    for (int test_idx = 0; test_idx < config_.num_tests; test_idx++) {
      // Generate test vectors
      std::vector<float> a_values_float(total_elements), b_values_float(total_elements);
      std::vector<uint32_t> a_values_format(total_elements), b_values_format(total_elements);

      bool a_enable = (test_idx % 3) == 0;
      bool b_enable = (test_idx % 3) == 1;
      bool c_enable = (test_idx % 3) == 2;

      for (int i = 0; i < total_elements; i++) {
        float af = (a_enable && i == 0) ? generate_fp_value(feature) : generate_fp_value("normals");
        float bf = (b_enable && i == 0) ? generate_fp_value(feature) : generate_fp_value("normals");

        a_values_format[i] = cvt_f32_to_custom(af, config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);
        b_values_format[i] = cvt_f32_to_custom(bf, config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);

        // Convert back to float to account for precision loss
        a_values_float[i] = cvt_custom_to_f32(a_values_format[i], config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);
        b_values_float[i] = cvt_custom_to_f32(b_values_format[i], config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);

        /*print_float("  af=", af);
        print_float(", bf=", bf);
        std::cout << std::endl;*/
      }

      // Generate c value
      float c_value_float = c_enable ? generate_fp_value(feature) : generate_fp_value("normals");
      uint32_t c_value_bits;
      std::memcpy(&c_value_bits, &c_value_float, sizeof(float));

      // Pack into XLEN words
      std::vector<uint32_t> a_packed(NUM_REGS), b_packed(NUM_REGS);
      pack_elements(a_values_format, element_bits, NUM_REGS, a_packed.data());
      pack_elements(b_values_format, element_bits, NUM_REGS, b_packed.data());

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
      //float expected = calculate_fp_dot_product(a_values_float, b_values_float, c_value_float);
      FEDP fedp(config_.exp_bits, config_.sig_bits, (int)config_.frm, total_elements);
      float expected = fedp(a_packed, b_packed, c_value_float);

      bool passed = approximately_equal(dut_result, expected, config_.exp_bits, config_.sig_bits);
      if (!passed) {
        std::cout << "Test " << feature << ":" << test_idx << " failed:" << std::endl;
        print_float("  af_values=", a_values_float, true);
        print_format("  ax_values=", a_values_format, true);
        print_float("  bf_values=", b_values_float, true);
        print_format("  bx_values=", b_values_format, true);
        print_float("  c_value=", c_value_float, true);
        print_float("  expected=", expected, true);
        print_float("  actual=", dut_result, true);
        return false;
      }

      // Add one idle cycles between tests
      dut_->enable = 0;
      tick();
    }

    std::cout << "  " << config_.num_tests << " tests passed" << std::endl;
    return true;
  }

  // Run all tests based on configuration
  bool run_tests() {

    // reset device under test
    this->reset();

    if (config_.fmt_s >= 8) {
      // test integer formats
      if (!test_integers()) {
        return false;
      }
    } else {
      // test floating-point formats
      std::vector<std::string> features_to_test;
      for (const auto &feature : config_.test_features) {
        if (feature.second) {
          features_to_test.push_back(feature.first);
        }
      }

      for (const auto &feature : features_to_test) {
        if (!test_fp_feature(feature)) {
          return false;
        }
      }
    }

    std::cout << "All tests PASSED!" << std::endl;
    std::cout << "Simulation completed in " << cycle_count_ << " cycles" << std::endl;
    return true;
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
      config_.frm = frm_from_string(arg.substr(6));
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

  if (config_.fmt_s >= 8) {
    // Integer formats
    if (config_.exp_bits != 0 || config_.sig_bits != 0) {
      std::cerr << "Error: Exponent and significand bits should not be set for integer formats" << std::endl;
      exit(1);
    }
  } else {
    // Floating-point formats
    if (config_.exp_bits == 0 && config_.sig_bits == 0) {
      init_default_fp_format(config_.fmt_s, &config_.exp_bits, &config_.sig_bits);
    }

    // Validate parameters
    int total_bits = 1 + config_.exp_bits + config_.sig_bits;
    if (total_bits > 32) {
      std::cout << "Error: Total bits (1 + exp_bits + sig_bits) cannot exceed 32" << std::endl;
       exit(1);
    }
    if (32 % total_bits != 0) {
      std::cout << "Warning: Element size " << total_bits << " doesn't evenly divide 32 bits" << std::endl;
      std::cout << "This may result in unused bits in the packed words" << std::endl;
      exit(1);
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

  // Create and run testbench
  Testbench testbench(config);
  if (!testbench.run_tests()) {
    return 1;
  }

  return 0;
}