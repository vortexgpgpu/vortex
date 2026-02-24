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

#if defined(TCU_TYPE_TFR)
#include "VVX_tcu_fedp_tfr.h"
#define MODULE VVX_tcu_fedp_tfr
#elif defined(TCU_TYPE_BHF)
#include "VVX_tcu_fedp_bhf.h"
#define MODULE VVX_tcu_fedp_bhf
#elif defined(TCU_TYPE_DSP)
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
#include <string>
#include <bitmanip.h>
#include "softfloat_ext.h"

#ifdef USE_FEDP
#include "fedp.h"
#endif

bool sim_trace_enabled() {
  return true;
}

static uint64_t timestamp = 0;

double sc_time_stamp() {
  return timestamp;
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
    case 5: // tf32
      *exp_bits = 8;
      *sig_bits = 10;
      break;
    default:
      break;
  }
}

// Test configuration
struct TestConfig {
  uint64_t max_cycles = 1000;
  bool enable_tracing = true;
  bool fused = true;
  uint64_t trace_start = 0;
  uint64_t trace_end = -1ull;
  unsigned int random_seed = 12345;
  uint32_t fmt_s = 0; // Source format
  uint32_t fmt_d = 0; // Destination format (always 0 for float32)
  std::vector<std::string> test_features = {
      "infinities",
      "nans",
      "subnormals",
      "normals",
      "zeros"};
  int exp_bits = 0; // Exponent bits
  int sig_bits = 0; // Significand bits
  RoundingMode frm = RoundingMode::RNE; // Rounding mode
  int W = 53; // Accumulator window width
  bool renorm = false; // renormalize product
  uint32_t num_tests = 100000; // Number of tests per feature
  int ulp = 1; // float precision error bits
  int sparsity = 0; // % of zero inputs
  int32_t test_id = -1; // Specific test ID to run (for debugging)
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
     << ", sparsity: " << config.sparsity
     << ", num_tests: " << config.num_tests
     << ", test_id: " << config.test_id
     << ", test_features: {";
  int i = 0;
  for (const auto &f : config.test_features) {
    if (i++ > 0) os << ", ";
    os << f;
  }
  os << "}}";
  return os;
}

static void print_float(const std::string &prefix, float value, bool newline = false) {
  std::cout << prefix << value << " (0x" << std::hex << bit_cast<uint32_t>(value) << ")" << std::dec;
  if (newline) {
    std::cout << std::endl;
  }
}

static void print_float(const std::string &prefix, std::vector<float> &values, bool newline = false) {
  std::cout << prefix << "{";
  for (size_t i = 0; i < values.size(); i++) {
    if (i > 0)
      std::cout << ", ";
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
    if (i > 0)
      std::cout << ", ";
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

#ifndef USE_FEDP
// Calculate expected fp dot product
static float dot_product(const uint32_t* A, const uint32_t* B, uint32_t C, int n, int eb, int sb, bool fused) {
  auto to_float = [&](uint32_t x, int ebits, int sbits) -> long double {
    uint32_t sign = (x >> (ebits + sbits)) & 1u;
    uint32_t exp  = (x >> sbits) & ((ebits == 32) ? 0xFFFFFFFFu : ((1u << ebits) - 1u));
    uint32_t frac = x & ((sbits == 32) ? 0xFFFFFFFFu : ((1u << sbits) - 1u));
    uint32_t all1 = (ebits == 32) ? 0xFFFFFFFFu : ((1u << ebits) - 1u);
    uint32_t bias = (1u << (ebits - 1)) - 1u;

    // Zero (preserve sign)
    if (exp == 0 && frac == 0) {
      long double z = 0.0L;
      return sign ? -z : z;
    }
    // Inf / NaN
    if (ebits == 4 && sbits == 3) {
      // FP8 E4M3 exception
      if (exp == 15 && frac == 7) return std::numeric_limits<long double>::quiet_NaN();
    } else
    if (exp == all1) {
      if (frac) return std::numeric_limits<long double>::quiet_NaN();
      return sign ? -std::numeric_limits<long double>::infinity()
                  :  std::numeric_limits<long double>::infinity();
    }
    long double mant;
    int ex;
    if (exp == 0) {
        mant = static_cast<long double>(frac) / static_cast<long double>(1ULL << sbits);
        ex = 1 - bias;
    } else {
        mant = 1.0L + static_cast<long double>(frac) / static_cast<long double>(1ULL << sbits);
        ex = static_cast<int>(exp) - bias;
    }
    long double val = ldexpl(mant, ex);
    return sign ? -val : val;
  };
  auto fadd = [&](long double a, long double b) -> long double {
    return fused ? (a + b) : float(a) + float(b);
  };
  long double sum = 0.0L;
  auto dc = to_float(C, 8, 23);
  for (size_t i = 0; i < n; ++i) {
    auto da = to_float(A[i], eb, sb);
    auto db = to_float(B[i], eb, sb);
    auto prod = da * db;
    sum = fadd(sum, prod);

  }
  sum = fadd(sum, dc);
  return static_cast<float>(sum);
}
#endif

// Calculate expected int dot product
static int32_t dot_product(const int32_t *a, const int32_t *b, int32_t c, uint32_t n) {
  int32_t acc(0);
  for (size_t i = 0; i < n; i++) {
    acc += a[i] * b[i];
  }
  acc += c;
  return acc;
}

// Check if two floats are approximately equal
static int approximately_equal(float a, float b) {
  // Handle NaN
  if (std::isnan(a) && std::isnan(b))
    return 0;

  // Handle infinity
  if (std::isinf(a) && std::isinf(b))
    return (std::signbit(a) == std::signbit(b)) ? 0 : 1;

  // comparison
  uint32_t xa, xb;
  std::memcpy(&xa, &a, sizeof(a));
  std::memcpy(&xb, &b, sizeof(b));
  uint32_t oa = xa ^ 0x80000000u;
  uint32_t ob = xb ^ 0x80000000u;
  int delta = oa - ob;
  return delta;
}

static inline uint32_t pack_fp_bits(uint32_t sign, uint32_t exp, uint32_t frac, uint32_t exp_bits, uint32_t sig_bits) {
  const uint32_t S = 1u;
  const uint32_t W = S + exp_bits + sig_bits;
  assert(W <= 32);
  const uint32_t sign_mask = 1u;
  const uint32_t exp_mask  = (exp_bits == 32 ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u));
  const uint32_t frac_mask = (sig_bits == 32 ? 0xFFFFFFFFu : ((1u << sig_bits) - 1u));

  sign &= sign_mask;
  exp  &= exp_mask;
  frac &= frac_mask;

  const uint32_t sign_shift = exp_bits + sig_bits;
  const uint32_t exp_shift  = sig_bits;

  return (sign << sign_shift) | (exp << exp_shift) | frac;
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
  std::mt19937 rng_;

  // Clock generation
  void tick() {
    dut_->clk = 0;
    dut_->eval();
  #ifdef VCD_OUTPUT
    if (config_.enable_tracing && (timestamp >= config_.trace_start) && (timestamp < config_.trace_end)) {
      trace_->dump(timestamp);
    }
  #endif
    ++timestamp;

    dut_->clk = 1;
    dut_->eval();
  #ifdef VCD_OUTPUT
    if (config_.enable_tracing && (timestamp >= config_.trace_start) && (timestamp < config_.trace_end)) {
      trace_->dump(timestamp);
    }
  #endif
    ++timestamp;
    ++cycle_count_;
    fflush(stdout);
  }

  // Reset the DUT
  void reset() {
    dut_->reset = 1;
    dut_->enable = 0;
    tick();
    dut_->reset = 0;
  }

  // Generate test int values based on sign, element bits, and test ID
  int32_t generate_int_value(bool is_signed, int element_bits, int test_id) {
    uint32_t mask = (element_bits < 32) ? ((1u << element_bits) - 1u) : 0xFFFFFFFFu;
    std::uniform_int_distribution<uint32_t> int_dist(0, mask);

    int32_t value;
    if (test_id == -1) {
      value = int_dist(rng_);
    } else {
      switch (test_id % 4) {
      case 0: // zero
        value = 0;
        break;
      case 1: // min
        if (is_signed) {
          value = (element_bits < 32) ? (1u << (element_bits - 1)) : 0x80000000u;
        } else {
          value = 0;
        }
        break;
      case 2: // max
        if (is_signed) {
          value = (element_bits < 32) ? ((1u << (element_bits - 1)) - 1u) : 0x7FFFFFFFu;
        } else {
          value = mask;
        }
        break;
      default: // random
        value = int_dist(rng_);
        break;
      }
    }

    if (is_signed) {
      value = vortex::sext<int32_t>(value, element_bits);
    }
    return value;
  }

  // Generate test floating-point values based on feature, format, and test ID
  uint32_t generate_fp_value(const std::string &feature, uint32_t exp_bits, uint32_t sig_bits, uint32_t test_id) {
    const uint32_t all_exp = (exp_bits == 32 ? 0xFFFFFFFFu : ((1u << exp_bits) - 1u));
    const uint32_t max_frac = (sig_bits == 32 ? 0xFFFFFFFFu : ((1u << sig_bits) - 1u));

    std::uniform_int_distribution<uint32_t> bit_dist(0, 0xFFFFFFFFu);
    std::uniform_int_distribution<uint32_t> sign_dist(0, 1);
    std::uniform_int_distribution<uint32_t> exp_norm_dist(1, (all_exp > 0 ? all_exp - 1 : 0)); // [1, all_ones-1]
    std::uniform_int_distribution<uint32_t> frac_dist(0, (max_frac > 0 ? max_frac : 0));
    std::uniform_int_distribution<uint32_t> frac_nz_dist(1, (max_frac > 0 ? max_frac : 1)); // nonzero frac

    auto deterministic_sign = [&](uint32_t salt) -> uint32_t {
      // Mix test_id with salt for reproducible sign selection
      return ((test_id ^ (salt * 0x9E3779B9u)) >> 31) & 1u;
    };

    if (feature == "zeros") {
      // +0 and -0; alternate by test_id for coverage
      uint32_t s = (test_id & 1u) ? 1u : 0u;
      return pack_fp_bits(s, 0u, 0u, exp_bits, sig_bits);

    } else if (feature == "normals") {
      if (exp_bits < 2) {
        // No normal numbers exist; fall back to zero
        return pack_fp_bits(0u, 0u, 0u, exp_bits, sig_bits);
      }
      // Spread some directed cases using test_id to hit edges:
      const uint32_t which = test_id % 5u;
      uint32_t s = (which == 0) ? deterministic_sign(1) : sign_dist(rng_);
      uint32_t e, f;
      switch (which) {
      case 0: // random normal
        e = exp_norm_dist(rng_);
        f = frac_dist(rng_);
        break;
      case 1: // smallest normal: exp=1, frac varies
        e = 1u;
        f = frac_dist(rng_);
        break;
      case 2: // largest finite: exp=all_ones-1, frac=all_ones
        e = (all_exp > 0 ? all_exp - 1 : 0);
        f = max_frac;
        break;
      case 3: // near one: exponent=bias, small fraction
        e = (exp_bits ? ((1u << (exp_bits - 1)) - 1u) : 0u);
        f = (sig_bits ? (frac_dist(rng_) & ((1u << (sig_bits > 4 ? 4 : sig_bits)) - 1u)) : 0u);
        break;
      default: // dense random but still normal
        e = exp_norm_dist(rng_);
        f = frac_dist(rng_);
        break;
      }
      return pack_fp_bits(s, e, f, exp_bits, sig_bits);

    } else if (feature == "subnormals") {
      // exp = 0, frac != 0
      if (sig_bits == 0) {
        // No fraction field -> subnormals don't exist; return zero
        return pack_fp_bits(0u, 0u, 0u, exp_bits, sig_bits);
      }
      uint32_t s = sign_dist(rng_);
      // Bias test_id to sometimes hit smallest and largest subnormals
      const uint32_t which = test_id % 3u;
      uint32_t f;
      if (which == 0)
        f = 1u; // smallest subnormal (lsb)
      else if (which == 1)
        f = max_frac; // largest subnormal
      else
        f = frac_nz_dist(rng_);
      return pack_fp_bits(s, 0u, f, exp_bits, sig_bits);

    } else if (feature == "infinities") {
      // exp = all ones, frac = 0
      uint32_t s = deterministic_sign(2);
      uint32_t e = all_exp;
      uint32_t f = 0u;
      return pack_fp_bits(s, e, f, exp_bits, sig_bits);

    } else if (feature == "nans") {
      // exp = all ones, frac != 0. Prefer quiet NaNs by setting MSB of fraction.
      if (sig_bits == 0) {
        // If there is no fraction field, you can't encode a NaN distinct from Inf; return Inf pattern.
        uint32_t s = deterministic_sign(3);
        return pack_fp_bits(s, all_exp, 0u, exp_bits, sig_bits);
      }
      uint32_t s = sign_dist(rng_);
      uint32_t e = all_exp;
      uint32_t quiet_bit = 1u << (sig_bits - 1); // top frac bit
      uint32_t payload = frac_dist(rng_) & (quiet_bit - 1u);
      if (payload == 0u)
        payload = 1u;                   // ensure nonzero payload
      uint32_t f = quiet_bit | payload; // make it a qNaN
      return pack_fp_bits(s, e, f, exp_bits, sig_bits);

    } else {
      std::cout << "Unknown feature: " << feature << std::endl;
      std::abort();
    }
  }

public:

  Testbench(const TestConfig &cfg)
    : config_(cfg)
    , cycle_count_(0)
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
  #ifdef TCU_TYPE_TFR
    dut_->vld_mask = 0;
  #endif
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

    // Determine bits per element in vld_mask depending on element packing size
    int vld_bits_per_elem = 32 / elements_per_word / 4;
    uint32_t elem_vld_mask = (1 << vld_bits_per_elem) - 1;

    std::cout << "  elements_per_word=" << elements_per_word << ", total_elements=" << total_elements << std::endl;

    std::uniform_int_distribution<int> sparsity_dist(0, 99);

    for (int test_id = 0; test_id < config_.num_tests; test_id++) {
      // Generate test vectors
      std::vector<uint32_t> a_values(total_elements), b_values(total_elements);

      bool a_enable = (test_id % 3) == 0;
      bool b_enable = (test_id % 3) == 1;
      bool c_enable = (test_id % 3) == 2;
      uint32_t current_vld_mask = 0;

      for (int i = 0; i < total_elements; i++) {
        bool is_sparse = (config_.sparsity > 0) && (sparsity_dist(rng_) < config_.sparsity);

        if (is_sparse) {
          a_values[i] = 0;
        } else {
          a_values[i] = generate_int_value(is_signed, element_bits, (a_enable && i == 0) ? test_id : -1);
          current_vld_mask |= (elem_vld_mask << (i * vld_bits_per_elem));
        }
        b_values[i] = generate_int_value(is_signed, element_bits, (b_enable && i == 0) ? test_id : -1);
      }

      // Generate c value
      int32_t c_value = generate_int_value(true, 32, c_enable ? test_id : -1);

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
    #ifdef TCU_TYPE_TFR
      dut_->vld_mask = current_vld_mask;
    #endif

      // Run for latency cycles
      for (int i = 0; i < LATENCY; i++) {
        tick();
      #ifdef TCU_TYPE_TFR
        dut_->vld_mask = 0;
      #endif
      }
      // Add one idle cycles between tests
      dut_->enable = 0;
      tick();

      // Check result
      int32_t dut_result = dut_->d_val;

      // Calculate expected result
      int32_t expected = dot_product((const int32_t *)a_values.data(), (const int32_t *)b_values.data(), c_value, total_elements);

      if (dut_result != expected) {
        std::cout << "Test:" << test_id << " failed:" << std::endl;
        print_format("  a_values=", a_values, true);
        print_format("  b_values=", b_values, true);
        print_format("  c_value=", c_value, true);
        print_format("  expected=", expected, true);
        print_format("  actual=", dut_result, true);
        return false;
      }
    }

    return true;
  }

  bool test_floating_points(const std::vector<std::string> &features_to_test) {
    if (features_to_test.empty()) return true;

    // Calculate how many elements we can fit in NUM_REGS XLEN words
    int element_bits = float_fmt_width(config_.exp_bits, config_.sig_bits);
    int elements_per_word = 32 / element_bits;
    int total_elements = NUM_REGS * elements_per_word;

    // Determine bits per element in vld_mask depending on element packing size
    int vld_bits_per_elem = 32 / elements_per_word / 4;
    uint32_t elem_vld_mask = (1 << vld_bits_per_elem) - 1;

    // std::cout << "  elements_per_word=" << elements_per_word << ", total_elements=" << total_elements << std::endl;

    const uint32_t NT = config_.num_tests;
    const uint32_t NF = features_to_test.size();
    const uint32_t tests_per_feature = (NT + NF - 1) / NF;

  #ifdef USE_FEDP
    FEDP fedp(config_.exp_bits, config_.sig_bits, (int)config_.frm, config_.W, config_.renorm);
  #endif

    uint32_t skipped = 0;
    std::uniform_int_distribution<int> sparsity_dist(0, 99);

    for (int test_id = 0; test_id < NT; test_id++) {
      // select feature to test
      int feature_id = test_id / tests_per_feature;
      if (feature_id >= NF) break; // prevent overflow if division rounding goes out of bounds
      std::string feature = features_to_test[feature_id];

      // Generate test vectors
      std::vector<float> a_values_float(total_elements), b_values_float(total_elements);
      std::vector<uint32_t> a_value_hex(total_elements), b_value_hex(total_elements);

      bool a_enable = (test_id % 3) == 0;
      bool b_enable = (test_id % 3) == 1;
      bool c_enable = (test_id % 3) == 2;
      uint32_t current_vld_mask = 0;

      for (int i = 0; i < total_elements; i++) {
        bool is_sparse = (config_.sparsity > 0) && (sparsity_dist(rng_) < config_.sparsity);

        if (is_sparse) {
          a_value_hex[i] = pack_fp_bits(0u, 0u, 0u, config_.exp_bits, config_.sig_bits);
          a_values_float[i] = 0.0f;
        } else {
          a_value_hex[i] = generate_fp_value((a_enable && (i & 0x1) == 0) ? feature : "normals", config_.exp_bits, config_.sig_bits, test_id);
          a_values_float[i] = cvt_custom_to_f32(a_value_hex[i], config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);
          current_vld_mask |= (elem_vld_mask << (i * vld_bits_per_elem));
        }

        b_value_hex[i] = generate_fp_value((b_enable && (i & 0x1) == 0) ? feature : "normals", config_.exp_bits, config_.sig_bits, test_id);
        b_values_float[i] = cvt_custom_to_f32(b_value_hex[i], config_.exp_bits, config_.sig_bits, (int)config_.frm, nullptr);
      }

      // Generate c value
      float c_value_float = generate_fp_value(c_enable ? feature : "normals", 8, 23, test_id);

      uint32_t c_value_hex;
      std::memcpy(&c_value_hex, &c_value_float, sizeof(float));

      // skip if not in selected test id
      if (config_.test_id >= 0 && test_id != config_.test_id)
        continue;

      if (config_.test_id >= 0 || (test_id % tests_per_feature) == 0) {
        std::cout << "Testing floating-point feature: " << feature << std::endl;
      }

      // Pack into XLEN words
      std::vector<uint32_t> a_packed(NUM_REGS), b_packed(NUM_REGS);
      pack_elements(a_value_hex, element_bits, NUM_REGS, a_packed.data());
      pack_elements(b_value_hex, element_bits, NUM_REGS, b_packed.data());

      // Apply to DUT
      for (int i = 0; i < NUM_REGS; i++) {
        WRITE_WDATA(dut_->a_row, i, a_packed[i]);
        WRITE_WDATA(dut_->b_col, i, b_packed[i]);
      }
      dut_->c_val = c_value_hex;
      dut_->fmt_s = config_.fmt_s;
      dut_->enable = 1;
    #ifdef TCU_TYPE_TFR
      dut_->vld_mask = current_vld_mask;
    #endif

      // Run for latency cycles
      for (int i = 0; i < LATENCY; i++) {
        tick();
      #ifdef TCU_TYPE_TFR
        dut_->vld_mask = 0;
      #endif
      }
      // Add one idle cycles between tests
      dut_->enable = 0;
      tick();

      // Check result
      uint32_t dut_result_bits = dut_->d_val;
      float dut_result;
      std::memcpy(&dut_result, &dut_result_bits, sizeof(float));

      // Calculate expected result
    #ifdef USE_FEDP
      float expected = fedp(a_packed.data(), b_packed.data(), c_value_float, NUM_REGS);
    #else
      float expected = dot_product(a_value_hex.data(), b_value_hex.data(),
        c_value_hex, total_elements, config_.exp_bits, config_.sig_bits, config_.fused);
    #endif

      int delta = approximately_equal(dut_result, expected);
      if (abs(delta) > config_.ulp) {
        std::cout << "Test #" << test_id << " (" << feature << ") failed:" << std::endl;

        std::cout << "  scenario='[";
          for (uint32_t i = 0; i < total_elements; i++) {
            if (i > 0) std::cout << ",";
            print_format("", a_value_hex[i], false);
          }
        std::cout << "];[";
          for (uint32_t i = 0; i < total_elements; i++) {
            if (i > 0) std::cout << ",";
            print_format("", b_value_hex[i], false);
          }
        std::cout << "];" << std::hex << "0x" << c_value_hex << std::dec << "';" << std::endl;

        print_float("  af_values=", a_values_float, true);
        print_format("  ax_values=", a_value_hex, true);
        print_float("  bf_values=", b_values_float, true);
        print_format("  bx_values=", b_value_hex, true);
        print_float("  c_value=", c_value_float, true);
        print_float("  expected=", expected, true);
        print_float("  actual=", dut_result, true);
        std::cout << "  delta=" << delta << std::endl;
        return false;
      }
    }

    if (skipped > 0) {
      std::cout << "  Skipped " << skipped << " invalid test(s) that produced NaN or Inf" << std::endl;
    }

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
      if (!test_floating_points(config_.test_features)) {
        return false;
      }
    }

    if (config_.test_id >= 0) {
      std::cout << "Test #" << config_.test_id << " PASSED!" << std::endl;
    } else {
      std::cout << config_.num_tests << " test(s) PASSED!" << std::endl;
    }

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
    } else if (arg == "--no-fused") {
      config_.fused = false;
    } else if (arg.substr(0, 11) == "--features=") {
      config_.test_features.clear();
      std::string features_str = arg.substr(11);
      std::stringstream ss(features_str);
      std::string str = ss.str();
      size_t pos = 0;
      while ((pos = str.find_first_not_of(" ,;", pos)) != std::string::npos) {
        auto end = str.find_first_of(" ,;", pos);
        auto item = str.substr(pos, end - pos);
        config_.test_features.push_back(item);
        pos = end;
      }
    } else if (arg.substr(0, 6) == "--fmt=") {
      config_.fmt_s = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--ext=") {
      config_.exp_bits = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--sig=") {
      config_.sig_bits = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 6) == "--frm=") {
      config_.frm = frm_from_string(arg.substr(6));
    } else if (arg.substr(0, 4) == "--W=") {
      config_.W = std::stoi(arg.substr(4));
    } else if (arg == "--renorm") {
      config_.renorm = true;
    } else if (arg.substr(0, 6) == "--ulp=") {
      config_.ulp = std::stoi(arg.substr(6));
    } else if (arg.substr(0, 11) == "--sparsity=") {
      config_.sparsity = std::stoi(arg.substr(11));
    } else if (arg.substr(0, 8) == "--tests=") {
      config_.num_tests = std::stoi(arg.substr(8));
    } else if (arg.substr(0, 7) == "--test=") {
      config_.test_id = std::stoi(arg.substr(7));
    } else if (arg.substr(0, 7) == "--seed=") {
      config_.random_seed = std::stoi(arg.substr(7));
    } else {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --no-fused       Enable discrete Mul and Add pipeline" << std::endl;
      std::cout << "  --no-trace       Disable VCD tracing" << std::endl;
      std::cout << "  --features=LIST  Semicolon-separated list of features to test (default: infinities;nans;subnormals;normals;zeros)" << std::endl;
      std::cout << "  --fmt=<id>       Set source format code" << std::endl;
      std::cout << "  --ext=BITS       Exponent bits for custom format" << std::endl;
      std::cout << "  --sig=BITS       Significand bits for custom format" << std::endl;
      std::cout << "  --frm=MODE       Rounding mode (RNE, RZ, RU, RD, RM)" << std::endl;
      std::cout << "  --W <value>      Accumulator window width W" << std::endl;
      std::cout << "  --renorm         Renormalize product" << std::endl;
      std::cout << "  --ulp=<value>    Adjust floats precision error bits" << std::endl;
      std::cout << "  --sparsity=<%>   Percentage of zero inputs (0-100)" << std::endl;
      std::cout << "  --seed <value>   Set random seed" << std::endl;
      std::cout << "  --tests <count>  Number of tests to run (default: 100000)" << std::endl;
      std::cout << "  --test <id>      Run the specified test only" << std::endl;
      std::cout << "  --help           Show this help message" << std::endl;
      exit(0);
    }
  }

  if (config_.fmt_s >= 8) {
    // Integer formats
    config_.fmt_d = 8;
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