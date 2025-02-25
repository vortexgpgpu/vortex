#pragma once

#include <iostream>
#include <math.h>
#include <limits>
#include <assert.h>

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
      auto ref = a[i] * b[i] + b[i];
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
      auto ref = a[i] * b[i] - b[i];
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
      auto ref = -a[i] * b[i] - b[i];
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
      auto ref = -a[i] * b[i] + b[i];
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
      auto x = -a[i] * b[i] - b[i];
      auto y =  a[i] * b[i] + b[i];
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