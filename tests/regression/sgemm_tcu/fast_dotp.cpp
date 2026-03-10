#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <unistd.h>
#include <vector>

// README
// gcc -std=c++17 -o fast_dotp -O2 ../tests/regression/sgemm_tcu/fast_dotp.cpp -lstdc++

// float32 parameters
static constexpr float EPSILON = 1e-3f;
static constexpr int MANT_W = 23;
static constexpr int EXP_BIAS = 127;

static uint16_t fp32_to_fp16(float x) {
  union { float f; uint32_t u; } in = { x };
  uint32_t f = in.u;

  uint32_t sign = (f >> 16) & 0x8000;
  int32_t exp  = ((f >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = f & 0x7FFFFF;

  if (exp <= 0) {
    // Subnormal or zero
    if (exp < -10)
      return sign; // too small becomes zero
    mant |= 1 << 23;
    uint32_t sub = mant >> (1 - exp + 13);
    if ((mant >> (1 - exp + 12)) & 1)
      sub += ((mant & ((1 << (1 - exp + 12)) - 1)) != 0); // round to nearest
    return sign | sub;
  } else if (exp >= 0x1F) {
    // Overflow → Inf
    return sign | 0x7C00;
  } else {
    // Normal
    mant = mant + 0x1000; // round to nearest (13th bit is G-bit)
    if (mant & 0x800000) {
      mant = 0;
      exp += 1;
    }
    if (exp >= 0x1F)
      return sign | 0x7C00; // Inf
    return sign | (exp << 10) | (mant >> 13);
  }
}

static float fp16_to_fp32(uint16_t x) {
  uint16_t h_exp = (x >> 10) & 0x1F;
  uint16_t h_frac = x & 0x3FF;
  uint16_t h_sign = x >> 15;

  uint32_t f_sign = uint32_t(h_sign) << 31;
  uint32_t f_exp, f_frac;

  if (h_exp == 0) {
    if (h_frac == 0) {
      // Zero
      f_exp = 0;
      f_frac = 0;
    } else {
      // Subnormal
      int shift = 0;
      while ((h_frac & 0x400) == 0) {
        h_frac <<= 1;
        ++shift;
      }
      h_frac &= 0x3FF;
      f_exp = uint32_t(127 - 15 - shift + 1) << 23;
      f_frac = uint32_t(h_frac) << 13;
    }
  } else if (h_exp == 0x1F) {
    // Inf or NaN
    f_exp = 0xFF << 23;
    f_frac = uint32_t(h_frac) << 13;
  } else {
    // Normalized
    f_exp = uint32_t(h_exp + (127 - 15)) << 23;
    f_frac = uint32_t(h_frac) << 13;
  }

  uint32_t f_bits = f_sign | f_exp | f_frac;
  union { uint32_t u; float f; } out = { f_bits };
  return out.f;
}

static uint16_t fp32_to_bf16(float x) {
  union { float f; uint32_t u; } in = { x };
  uint32_t val = in.u;

  // Round to nearest-even (RNE)
  uint32_t round_bit = (val >> 16) & 1;
  uint32_t lsb_bits  = val & 0xFFFF;
  if (lsb_bits > 0x8000 || (lsb_bits == 0x8000 && round_bit)) {
    val += 0x10000;
  }

  return static_cast<uint16_t>(val >> 16);
}

static float bf16_to_fp32(uint16_t x) {
  // Expand 16-bit bf16 into 32-bit float by zero-extending the lower 16 bits
  uint32_t val = static_cast<uint32_t>(x) << 16;
  union { uint32_t u; float f; } out = { val };
  return out.f;
}

static int bit_length(__int128 x) {
  int len = 0;
  while (x) {
    x >>= 1;
    ++len;
  }
  return len;
}

struct unpack_t {
  uint32_t sign : 1;
  uint32_t exp  : 8;
  uint32_t mant : 23;
};

struct prod_t {
  uint32_t exp   : 9;
  __int64_t mant : 49;
};

struct acc_t {
  uint32_t e_max : 9;
  __int128 value;
};

// unpack float32 into sign, exponent, and mantissa
static unpack_t unpack(float f) {
  return *reinterpret_cast<const unpack_t*>(&f);
}

// multiply
static prod_t mult(const unpack_t &a, const unpack_t &b) {
  uint32_t amant = (a.exp == 0) ? a.mant : (1 << MANT_W) | a.mant;
  uint32_t bmant = (b.exp == 0) ? b.mant : (1 << MANT_W) | b.mant;
  uint64_t raw = uint64_t(amant) * uint64_t(bmant);
  prod_t p;
  p.exp = a.exp + b.exp;
  int sign = a.sign ^ b.sign;
  p.mant = sign ? -static_cast<int64_t>(raw) : static_cast<int64_t>(raw);
  return p;
}

// accumulate only left-shifts & adds
static acc_t reduceAdd(const std::vector<prod_t> &ps) {
  if (ps.empty())
    return {0, 0};

  uint32_t e_max = ps[0].exp;
  for (auto &p : ps) {
    if (p.exp > e_max) {
      e_max = p.exp;
    }
  }

  __int128 acc = 0;
  for (auto &p : ps) {
    __int128 val = static_cast<__int128>(p.mant);
    int32_t shift = static_cast<int32_t>(e_max) - static_cast<int32_t>(p.exp);
    if (shift >= 0) {
      val <<= shift;
    } else {
      val >>= (-shift);
    }
    acc += val;
  }
  return acc_t{e_max, acc};
}

// one global right-shift + normalize & round
static float pack(const acc_t &acc) {
  if (acc.value == 0)
    return 0.0f;

  bool neg = (acc.value < 0);
  __int128 mag = neg ? -acc.value : acc.value;

  int p = bit_length(mag) - 1;
  int norm_shift = p - MANT_W;
  __int128 normv;
  bool guard = false, sticky = false, lsb = false;

  if (norm_shift > 0) {
    normv = mag >> (norm_shift - 1);
    guard = normv & 1;
    normv >>= 1;
    __int128 sticky_mask = (static_cast<__int128>(1) << (norm_shift - 1)) - 1;
    sticky = (mag & sticky_mask) != 0;
    lsb = normv & 1;
  } else if (norm_shift < 0) {
    normv = mag << (-norm_shift);
  } else {
    normv = mag;
  }

  uint32_t m24 = static_cast<uint32_t>(normv) & 0xFFFFFF;
  if (norm_shift > 0) {
    bool round_up = guard && (sticky || lsb);
    if (round_up) {
      ++m24;
      if (m24 & (1u << 24)) {
        m24 >>= 1;
        ++p;
      }
    }
  }

  int final_exp = p + static_cast<int>(acc.e_max) - 300 + EXP_BIAS;
  if (final_exp <= 0)
    return 0.0f;
  if (final_exp >= 0xFF)
    return neg ? -std::numeric_limits<float>::infinity()
               : std::numeric_limits<float>::infinity();

  uint32_t sbits = neg ? 1u : 0u;
  uint32_t bits = (sbits << 31) | (static_cast<uint32_t>(final_exp) << MANT_W) | (m24 & ((1u << MANT_W) - 1));
  float result;
  std::memcpy(&result, &bits, sizeof(float));
  return result;
}

// Full exact dot-product pipeline
static float exact_dotp(int format, const std::vector<uint16_t> &A, const std::vector<uint16_t> &B, float C) {
  size_t N = A.size();
  std::vector<prod_t> p(N);

  // multiply
  for (size_t i = 0; i < N; ++i) {
    switch (format) {
    case 0: // fp16
      p[i] = mult_fp16(A[i], B[i]);
      break;
    case 1: // bf16
      p[i] = mult_bf16(A[i], B[i]);
      break;
    default:
      std::abort();
    }
  }

  acc_t acc = accumulate(p);

  raw1_t acc_raw = acc_to_raw(acc);

  raw1_t c_raw = fp32_to_raw(C);

  raw2_t result_raw = add_raw(c_raw, acc);

  raw3_t result = norm_rounding(result_raw);

  return pack(result);
}

// Naive IEEE-754 chain for comparison
static float naive_dotp(int format, const std::vector<uint16_t> &A, const std::vector<uint16_t> &B, float C) {
  float s = 0.0f;
  for (size_t i = 0; i < A.size(); ++i) {
    float a, b;
    switch (format) {
    case 0:
      a = fp16_to_fp32(A[i]);
      b = fp16_to_fp32(B[i]);
      break;
    case 1:
      a = bf16_to_fp32(A[i]);
      b = bf16_to_fp32(B[i]);
      break;
    }
    s = s + a * b;
  }
  return s;
}

int vector_size = 4;
int test_samples = 100;
int format = 0; // 0 = fp16, 1 = bf16

static void show_usage() {
  std::cout << "Dot Product Test\n";
  std::cout << "Usage: [-n <size>] [-s <samples>] [-f <format>] [-h help]\n";
  std::cout << "  <size>:    vector size (default: 4)\n";
  std::cout << "  <samples>: number of samples (default: 100)\n";
  std::cout << "  <format>:  input format (default: 0)\n";
  std::cout << "             0=fp16, 1=bp16\n";
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:s:f:h")) != -1) {
    switch (c) {
    case 'n':
      vector_size = std::atoi(optarg);
      break;
    case 's':
      test_samples = std::atoi(optarg);
      break;
    case 'f':
      format = std::atoi(optarg);
      break;
    case 'h':
      show_usage();
      std::exit(0);
    default:
      show_usage();
      std::exit(1);
    }
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  // verify that vector_size is multiple of 2 and less than or equal to 16
  if (vector_size <= 0 || vector_size > 16 || vector_size % 2 != 0) {
    std::cerr << "Error: Vector size must be a positive even number <= 16." << std::endl;
    return -1;
  }

  std::cout << "Vector size: " << vector_size << ", Samples: " << test_samples << std::endl;

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

  int num_errors = 0;

  for (int i = 0; i < test_samples; ++i) {
    std::vector<uint16_t> A(vector_size), B(vector_size);
    float C = dist(rng);

    for (int j = 0; j < vector_size; ++j) {
      switch (format) {
      case 0:
        A[j] = fp32_to_fp16(dist(rng));
        B[j] = fp32_to_fp16(dist(rng));
        break;
      case 1:
        A[j] = fp32_to_bf16(dist(rng));
        B[j] = fp32_to_bf16(dist(rng));
        break;
      default:
        std::cerr << "Error: Invalid format specified." << std::endl;
        return -2;
      }
    }

    float result_naive = naive_dotp(format, A, B, C);
    float result_exact = exact_dotp(format, A, B, C);

    if (std::fabs(result_naive - result_exact) > EPSILON) {
      ++num_errors;
    }
  }

  if (num_errors == 0) {
    std::cout << "✅ PASSED" << std::endl;
  } else {
    std::cout << "❌ FAILED with " << num_errors << " errors!" << std::endl;
  }

  return num_errors;
}
