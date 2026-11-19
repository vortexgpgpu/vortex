#pragma once

#include <iostream>
#include <math.h>
#include <limits>
#include <assert.h>
#include "common.h"

void cleanup();

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

union Float_t {
    float f;
    int   i;
    struct {
        uint32_t man  : 23;
        uint32_t exp  : 8;
        uint32_t sign : 1;
    } parts;
};

inline float fround(float x, int32_t precision = 8) {
  auto power_of_10 = std::pow(10, precision);
  return std::round(x * power_of_10) / power_of_10;
}

inline bool almost_equal_precision(float a, float b, int precision = 4) {
  auto power_of_10 = std::pow(10, precision);
  auto ap = std::round(a * power_of_10) / power_of_10;
  auto bp = std::round(b * power_of_10) / power_of_10;
  auto eps = std::numeric_limits<float>::epsilon();
  auto d = fabs(ap - bp);
  if (d > eps) {
    std::cout << "*** almost_equal_precision: d=" << d << ", precision=" << precision << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal_eps(float a, float b, int ulp = 128) {
  auto eps = std::numeric_limits<float>::epsilon() * (std::max(fabs(a), fabs(b)) * ulp);
  auto d = fabs(a - b);
  if (d > eps) {
    std::cout << "*** almost_equal_eps: d=" << d << ", eps=" << eps << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal_ulp(float a, float b, int32_t ulp = 6) {
  Float_t fa{a}, fb{b};
  auto d = std::abs(fa.i - fb.i);
  if (d > ulp) {
    std::cout << "*** almost_equal_ulp: a=" << a << ", b=" << b << ", ulp=" << d << ", ia=" << std::hex << fa.i << ", ib=" << fb.i << std::endl;
    return false;
  }
  return true;
}

inline bool almost_equal(float a, float b) {
  if (a == b)
    return true;
  /*if (almost_equal_eps(a, b))
    return true;*/
  return almost_equal_ulp(a, b);
}

class ITestCase;

class TestSuite {
public:
  TestSuite(vx_device_h device);
  ~TestSuite();

  ITestCase* get_test(int testid) const;

  void add_test(ITestCase* test);

  size_t size() const;

  vx_device_h device() const;

private:
  std::vector<ITestCase*> _tests;
  vx_device_h device_;
};

class ITestCase {
public:
  ITestCase(TestSuite* suite, const char* name)
    : suite_(suite)
    , name_(name)
  {}

  virtual ~ITestCase() {}

  TestSuite* suite() const {
    return suite_;
  }

  const char* name() const {
    return name_;
  }

  virtual int setup(uint32_t n, void* src1, void* src2) = 0;

  virtual int verify(uint32_t n, void* dst, const void* src1, const void* src2) = 0;

protected:
  TestSuite*  suite_;
  const char* const name_;
};

class Test_IADD : public ITestCase {
public:
  Test_IADD(TestSuite* suite) : ITestCase(suite, "iadd") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] + b[i];
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IMUL : public ITestCase {
public:
  Test_IMUL(TestSuite* suite) : ITestCase(suite, "imul") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] * b[i];
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IDIV : public ITestCase {
public:
  Test_IDIV(TestSuite* suite) : ITestCase(suite, "idiv") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] / b[i];
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_IDIV_MUL : public ITestCase {
public:
  Test_IDIV_MUL(TestSuite* suite) : ITestCase(suite, "idiv-mul") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 + i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] / b[i];
      auto y = a[i] * b[i];
      auto ref = x + y;
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FADD : public ITestCase {
public:
  Test_FADD(TestSuite* suite) : ITestCase(suite, "fadd") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] + b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FSUB : public ITestCase {
public:
  Test_FSUB(TestSuite* suite) : ITestCase(suite, "fsub") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] - b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMUL : public ITestCase {
public:
  Test_FMUL(TestSuite* suite) : ITestCase(suite, "fmul") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] * b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMADD : public ITestCase {
public:
  Test_FMADD(TestSuite* suite) : ITestCase(suite, "fmadd") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = fmaf(a[i], b[i], b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FMSUB : public ITestCase {
public:
  Test_FMSUB(TestSuite* suite) : ITestCase(suite, "fmsub") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = fmaf(a[i], b[i], -b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMADD : public ITestCase {
public:
  Test_FNMADD(TestSuite* suite) : ITestCase(suite, "fnmadd") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = fmaf(-a[i], b[i], -b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMSUB : public ITestCase {
public:
  Test_FNMSUB(TestSuite* suite) : ITestCase(suite, "fnmsub") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = fmaf(-a[i], b[i], b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FNMADD_MADD : public ITestCase {
public:
  Test_FNMADD_MADD(TestSuite* suite) : ITestCase(suite, "fnmadd-madd") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = fmaf(-a[i], b[i], -b[i]);
      auto y = fmaf(a[i], b[i], b[i]);
      auto ref = x + y;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FDIV : public ITestCase {
public:
  Test_FDIV(TestSuite* suite) : ITestCase(suite, "fdiv") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] / b[i];
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FDIV2 : public ITestCase {
public:
  Test_FDIV2(TestSuite* suite) : ITestCase(suite, "fdiv2") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] / b[i];
      auto y = b[i] / a[i];
      auto ref = x + y;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FSQRT : public ITestCase {
public:
  Test_FSQRT(TestSuite* suite) : ITestCase(suite, "fsqrt") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      float q = 1.0f + (i % 64);
      a[i] = q;
      b[i] = q;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = sqrt(a[i] * b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FDIV_SPECIAL : public ITestCase {
public:
  Test_FDIV_SPECIAL(TestSuite* suite) : ITestCase(suite, "fdiv-special") {}

  // Zero, infinity, NaN and negative operands take the divide and square-root
  // early-out paths; the hash scatters them across the threads of a warp.
  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      float x = 1.0f + fround((n - i) * (1.0f/n));
      float y = fround((n + i) * (1.0f/n));
      switch ((i * 0x9E3779B1u) >> 28) {
      case 0: y = 0.0f; break;
      case 1: x = 0.0f; y = 0.0f; break;
      case 2: x = INFINITY; break;
      case 3: x = NAN; break;
      case 4: x = 0.0f; break;
      case 5: x = -x; break;
      default: break;
      }
      a[i] = x;
      b[i] = y;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = sqrtf(a[i]) / b[i];
      bool match = std::isnan(ref) ? std::isnan(c[i]) : almost_equal(c[i], ref);
      if (!match) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FP_NANBOX : public ITestCase {
public:
  Test_FP_NANBOX(TestSuite* suite) : ITestCase(suite, "fp-nanbox") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = 1.0f + fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  // With 64-bit FP registers, the kernel leaves one input of five elements in
  // six improperly NaN-boxed, which must read as the canonical NaN.
  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t k = i % 6;
    #ifdef VX_CFG_FLEN_64
      if (k != 5) {
        Float_t r{c[i]};
        if (r.i != 0x7fc00000) {
          std::cout << "error at result #" << std::dec << i << " (case " << k << ")" << std::hex << ": expected=0x7fc00000, actual=0x" << r.i << ", a=" << a[i] << ", b=" << b[i] << std::endl;
          ++errors;
        }
        continue;
      }
    #endif
      float ref;
      switch (k) {
      case 2:  ref = fmaf(a[i], b[i], b[i]); break;
      case 3:  ref = fmaf(a[i], b[i], a[i]); break;
      case 4:  ref = fmaf(a[i], b[i], b[i]); break;
      default: ref = sqrtf(a[i]) / b[i]; break;
      }
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << " (case " << k << ")" << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FFLAGS_LANES : public ITestCase {
public:
  Test_FFLAGS_LANES(TestSuite* suite) : ITestCase(suite, "fflags-lanes") {}

  enum { FL_NX = 0x01, FL_OF = 0x04, FL_DZ = 0x08, FL_NV = 0x10 };
  enum { K_NONE, K_NX, K_X1, K_X2 };

  // A warp divides or adds by the parity of its iteration plus its index, so
  // warps running side by side mix the two; X1 is DZ or OF, X2 is NV.
  static uint32_t recipe(bool is_add, int kind, float* a, float* b) {
    static const float div_a[] = {1.0f, 1.0f, 1.0f, 0.0f};
    static const float div_b[] = {2.0f, 3.0f, 0.0f, 0.0f};
    static const uint32_t div_f[] = {0, FL_NX, FL_DZ, FL_NV};
    static const float add_a[] = {1.0f, 1.0f, std::numeric_limits<float>::max(), INFINITY};
    static const float add_b[] = {2.0f, 0x1p-30f, std::numeric_limits<float>::max(), -INFINITY};
    static const uint32_t add_f[] = {0, FL_NX, FL_OF | FL_NX, FL_NV};
    *a = is_add ? add_a[kind] : div_a[kind];
    *b = is_add ? add_b[kind] : div_b[kind];
    return is_add ? add_f[kind] : div_f[kind];
  }

  // Flags raised by one lane; only non-zero lanes raise any, and a masked lane
  // gets operands raising a flag no active lane raises.
  uint32_t lane_flags(uint32_t k, uint32_t n, float* a, float* b) const {
    uint32_t L = num_threads_;
    uint32_t count = n / (num_cores_ * num_warps_ * L);
    uint32_t t = k / count, i = k % count;
    uint32_t lane = t % L, wid = t / L;
    uint32_t sc = ((i >> 2) + wid) % 4;
    bool is_add = ((i + wid) & 1) != 0;
    int kind = K_NONE;
    if ((sc == 0 && lane == 2) || (sc == 2 && lane == L - 1)) {
      kind = K_X1;
    } else if (sc == 1 && lane == 3) {
      kind = K_X2;
    } else if ((sc == 2 && lane == 2) || (sc == 3 && lane == 0)) {
      kind = K_NX;
    }
    if (fflags_lane_masked(lane, i, L)) {
      kind = (sc == 1) ? K_X1 : K_X2;
    }
    return recipe(is_add, kind, a, b);
  }

  int setup(uint32_t n, void* src1, void* src2) override {
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_CORES, &num_cores_));
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_WARPS, &num_warps_));
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_THREADS, &num_threads_));
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t k = 0; k < n; ++k) {
      lane_flags(k, n, &a[k], &b[k]);
    }
    return 0;
  }

  // fflags is per warp: every active lane reads the OR over the active lanes.
  int verify(uint32_t n, void* dst, const void* /*src1*/, const void* /*src2*/) override {
    int errors = 0;
    auto c = (uint32_t*)dst;
    uint32_t L = num_threads_;
    uint32_t count = n / (num_cores_ * num_warps_ * L);
    for (uint32_t k = 0; k < n; ++k) {
      uint32_t t = k / count, i = k % count;
      if (fflags_lane_masked(t % L, i, L))
        continue;
      uint32_t ref = 0;
      for (uint32_t l = 0; l < L; ++l) {
        if (fflags_lane_masked(l, i, L))
          continue;
        float a, b;
        ref |= lane_flags(((t / L) * L + l) * count + i, n, &a, &b);
      }
      if (c[k] != ref) {
        std::cout << "error at result #" << std::dec << k << " (lane " << (t % L) << ", iteration " << i << ")" << std::hex << ": expected fflags=0x" << ref << ", actual=0x" << c[k] << std::endl;
        ++errors;
      }
    }
    return errors;
  }

  uint64_t num_cores_;
  uint64_t num_warps_;
  uint64_t num_threads_;
};

class Test_FTOI : public ITestCase {
public:
  Test_FTOI(TestSuite* suite) : ITestCase(suite, "ftoi") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      float q = fround(float(n/2) - i + (float(i) / n));
      a[i] = q;
      b[i] = q;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (int32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (int32_t)x;
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FTOU : public ITestCase {
public:
  Test_FTOU(TestSuite* suite) : ITestCase(suite, "ftou") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      float q = fround(i + (float(i) / n));
      a[i] = q;
      b[i] = q;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (uint32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (uint32_t)x;
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_ITOF : public ITestCase {
public:
  Test_ITOF(TestSuite* suite) : ITestCase(suite, "itof") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 - i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int32_t*)src1;
    auto b = (int32_t*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (float)x;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_UTOF : public ITestCase {
public:
  Test_UTOF(TestSuite* suite) : ITestCase(suite, "utof") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (uint32_t*)src1;
    auto b = (uint32_t*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = i;
      b[i] = i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (uint32_t*)src1;
    auto b = (uint32_t*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto x = a[i] + b[i];
      auto ref = (float)x;
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_FCLAMP : public ITestCase {
public:
  Test_FCLAMP(TestSuite* suite) : ITestCase(suite, "fclamp") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround((n - i) * (1.0f/n));
      b[i] = fround((n + i) * (1.0f/n));
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = fmin(fmax(1.0f, a[i]), b[i]);
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_ICLAMP : public ITestCase {
public:
  Test_ICLAMP(TestSuite* suite) : ITestCase(suite, "iclamp") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (int*)src1;
    auto b = (int*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = n/2 - i;
      b[i] = n/2 - i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (int*)src1;
    auto b = (int*)src2;
    auto c = (int*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = std::min(std::max(1, a[i]), b[i]);
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_TRIGO : public ITestCase {
public:
  Test_TRIGO(TestSuite* suite) : ITestCase(suite, "trig") {}

  int setup(uint32_t n, void* src1, void* src2) override {
    auto a = (float*)src1;
    auto b = (float*)src2;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = fround(int(2*i-n) * (1.0f/n) * 3.1416);
      b[i] = fround(int(2*i-n) * (1.0f/n) * 3.1416);
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* src2) override {
    int errors = 0;
    auto a = (float*)src1;
    auto b = (float*)src2;
    auto c = (float*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      auto ref = a[i] * b[i];
      if ((i % 4) == 0) {
        ref = sinf(ref);
      }
      if (!almost_equal(c[i], ref)) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }
};

class Test_BAR : public ITestCase {
public:
  Test_BAR(TestSuite* suite) : ITestCase(suite, "bar") {}

  int setup(uint32_t n, void* src1, void* /*src2*/) override {
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_WARPS, &num_warps_));
    if (num_warps_ == 1) {
      std::cout << "Error: multiple warps configuration required!" << std::endl;
      return -1;
    }
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_THREADS, &num_threads_));
    auto a = (uint32_t*)src1;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* /*src2*/) override {
    int errors = 0;
    auto a = (uint32_t*)src1;
    auto c = (uint32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t ref = a[i] + 1;
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << std::hex << ref << ", actual=" << c[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }

  uint64_t num_warps_;
  uint64_t num_threads_;
};

class Test_GBAR : public ITestCase {
public:
  Test_GBAR(TestSuite* suite) : ITestCase(suite, "gbar") {}

  int setup(uint32_t n, void* src1, void* /*src2*/) override {
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_CORES, &num_cores_));
    if (num_cores_ == 1) {
      std::cout << "Error: multiple cores configuration required!" << std::endl;
      return -1;
    }
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_WARPS, &num_warps_));
    RT_CHECK(vx_dev_caps(suite_->device(), VX_CAPS_NUM_THREADS, &num_threads_));
    auto a = (uint32_t*)src1;
    for (uint32_t i = 0; i < n; ++i) {
      a[i] = i;
    }
    return 0;
  }

  int verify(uint32_t n, void* dst, const void* src1, const void* /*src2*/) override {
    int errors = 0;
    auto a = (uint32_t*)src1;
    auto c = (uint32_t*)dst;
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t ref = a[i] + 1;
      if (c[i] != ref) {
        std::cout << "error at result #" << std::dec << i << std::hex << ": expected=" << std::hex << ref << ", actual=" << c[i] << std::endl;
        ++errors;
      }
    }
    return errors;
  }

  uint64_t num_cores_;
  uint64_t num_warps_;
  uint64_t num_threads_;
};

///////////////////////////////////////////////////////////////////////////////

TestSuite::TestSuite(vx_device_h device)
  : device_(device) {
  this->add_test(new Test_IADD(this));
  this->add_test(new Test_IMUL(this));
  this->add_test(new Test_IDIV(this));
  this->add_test(new Test_IDIV_MUL(this));
  this->add_test(new Test_FADD(this));
  this->add_test(new Test_FSUB(this));
  this->add_test(new Test_FMUL(this));
  this->add_test(new Test_FMADD(this));
  this->add_test(new Test_FMSUB(this));
  this->add_test(new Test_FNMADD(this));
  this->add_test(new Test_FNMSUB(this));
  this->add_test(new Test_FNMADD_MADD(this));
  this->add_test(new Test_FDIV(this));
  this->add_test(new Test_FDIV2(this));
  this->add_test(new Test_FSQRT(this));
  this->add_test(new Test_FTOI(this));
  this->add_test(new Test_FTOU(this));
  this->add_test(new Test_ITOF(this));
  this->add_test(new Test_UTOF(this));
  this->add_test(new Test_FCLAMP(this));
  this->add_test(new Test_ICLAMP(this));
  this->add_test(new Test_TRIGO(this));
  this->add_test(new Test_BAR(this));
  this->add_test(new Test_GBAR(this));
  this->add_test(new Test_FDIV_SPECIAL(this));
  this->add_test(new Test_FP_NANBOX(this));
  this->add_test(new Test_FFLAGS_LANES(this));
}

TestSuite::~TestSuite() {
  for (size_t i = 0; i < _tests.size(); ++i) {
    delete _tests[i];
  }
}

ITestCase* TestSuite::get_test(int testid) const {
  return _tests.at(testid);
}

void TestSuite::add_test(ITestCase* test) {
  _tests.push_back(test);
}

size_t TestSuite::size() const {
  return _tests.size();
}

vx_device_h TestSuite::device() const {
  return device_;
}